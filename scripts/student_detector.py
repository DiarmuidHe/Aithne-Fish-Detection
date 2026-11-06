"""Part 3 stages of ``scripts/sam3_botsort.py``: a small student detector, RF-DETR Nano
(Apache-2.0), trained on SAM 3's reliable "fish" labels and run in SAM 3's place.

PowerShell, from the repository root:

    $env:FISHIAL_PREPROCESS='none'; $env:YOLO_OFFLINE='true'
    .venv\\Scripts\\python.exe -m scripts.sam3_botsort student --video-id VIDEO_UUID

Stages (``--video-id`` or ``--video``, as elsewhere):

- ``student-data``: the time split (train first, test last, 5 s gaps; validation cut from
  the end of train) and a COCO dataset of the video's frames with SAM 3's reliable labels,
  which stay unreviewed suggestions, never ground truth.
- ``student-train``: RF-DETR Nano from its COCO checkpoint (size, MD5 and SHA256 checked
  before any load), with the epoch count sized from a measured one-epoch probe so that the
  whole run fits the 2 h training budget, which Lightning's ``max_time`` also enforces.
- ``student-eval``: the trained student on every frame of any video, timed from 1080p
  decode to boxes; a detection threshold picked on the validation segment only; BoT-SORT
  identities; static suppression; then agreement on the held-out segment with SAM 3's
  labels, SAM 3's tracked boxes and VIAME. On another video the whole video is held out.
- ``student``: all three on the training video.

With one training video this is a pipeline smoke test: agreement with SAM 3 (the teacher),
never precision or recall, and nothing here decides whether VIAME stays. The student is
derived from SAM 3's labels, so Meta's SAM License ships with it; RF-DETR (code and Nano
COCO weights) is Apache-2.0; BoT-SORT is Ultralytics (AGPL-3.0).

Rounds (Parts 6 and 7 of ``docs/smartbay-rolling-training-prompt.md``) train one student
over a labelling round's windows, split by day role, on the instance:

    /opt/pytorch/bin/python -m scripts.student_detector round --root data/outputs/smartbay-rounds/round-1

- ``round-data``: a COCO dataset of every 5th frame of each window kept by
  ``scripts/cloud_labelling/round_checks.py``. Test days are the test split; validation is
  a seeded 10% of the training days, recorded before any training and kept by later rounds
  (``--validation-from`` an earlier round's journal); expert and unused days never appear.
- ``round-train``: RF-DETR Nano as in ``student-train``, sized to the round's budget.
- ``round-eval``: the student on every frame of the validation and test windows; the
  threshold is chosen on validation only; agreement with SAM 3's labels on test days, per
  stratum; student boxes per frame on *none* and *murky* windows; speed from 1080p decode.

``local-eval`` runs a trained student (``--root``: a round, or the smoke test's folder) on
videos SAM 3 labelled on the PC and scores it the way ``round-eval`` does; ``--render``
adds an MP4 per video with matched, student-only and SAM 3-only boxes, and ``--render alone``
one with the student's boxes and scores only:

    .venv\\Scripts\\python.exe -m scripts.student_detector local-eval --render `
      --root data/outputs/smartbay-rounds/round-1/cloud `
      --video-dir data/outputs/sam3-botsort-20261014/videos/SmartBayCam1_2025-08-01_0000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from app.services.external_datasets import sha256
from scripts.sam3_autolabel import SAM3_LICENCE, UnverifiedCheckpoint
from scripts.sam3_botsort import (
    OFFLINE,
    BudgetExceeded,
    Journal,
    agreement,
    by_frame,
    file_id,
    proxies,
    read,
    read_rows,
    save,
    settings_for,
    static_stage,
    time_split,
    track_rows,
    video_dir,
    video_record,
    write_rows,
)

RFDETR_CHECKPOINT = Path("data/models/rfdetr/rf-detr-nano.pth")
RFDETR_URL = "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
RFDETR_MD5 = "fb6504cce7fbdc783f7a46991f07639f"  # rfdetr 1.11.2's own registry value
RFDETR_SHA256 = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"
RFDETR_BYTES = 366_287_238
RFDETR_LICENCE = "Apache-2.0 (rfdetr 1.11.2; RF-DETR Nano and its COCO weights)"
NAME = "rfdetr-nano"  # Per-video outputs land in student/<NAME>/, read as "student-<NAME>"
LABELS = "fish"  # The recorded SAM 3 prompt choice (decision sam3_prompt_choice)
SPLIT = {"train": .75, "test": .20, "gap_seconds": 5.0, "validation": .10}
SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}  # RF-DETR's folder names
# Fixed before training, from SAM 3's label sizes alone: Nano's native 384 px input would
# shrink 30% of the label boxes below 8 px on their short side, 640 px leaves 5%.
RESOLUTION = 640
TRAIN = {"batch_size": 4, "grad_accum_steps": 4, "lr": 1e-4, "lr_encoder": 1.5e-4, "num_workers": 2,
         "seed": 0, "checkpoint_interval": 25, "tensorboard": False, "progress_bar": None}
EPOCHS = {"min": 10, "max": 50}
MARGIN_SECONDS = 600  # Training budget held back for start-up, the last validation and saving
FLOOR = .05  # Inference keeps boxes from here; the operating threshold is chosen later
# RF-DETR maps the one COCO category (id 1) to label 0. Its spare output slot (label 1)
# never trains positive: on the probe it scored at most 0.06, sometimes on a fish box.
FISH_LABEL = 0
THRESHOLDS = [round(.05 * i, 2) for i in range(2, 17)]  # 0.10 .. 0.80
WARMUP_FRAMES = 10
LIVE = {"fps": 4.0, "frame_budget_ms": 250.0}


# ------------------------------------------------------------- checkpoint --

def verify_rfdetr(path: Path = RFDETR_CHECKPOINT) -> dict:
    """Size, RF-DETR's published MD5 and the pinned SHA256 of the file that will load."""
    path = Path(path).resolve()
    if not path.is_file():
        raise UnverifiedCheckpoint(f"RF-DETR Nano checkpoint not found at {path}; download {RFDETR_URL} "
                                   f"({RFDETR_BYTES:,} bytes, MD5 {RFDETR_MD5}) to {RFDETR_CHECKPOINT.as_posix()}")
    size = path.stat().st_size
    if size != RFDETR_BYTES:
        raise UnverifiedCheckpoint(f"{path} is {size:,} bytes, expected {RFDETR_BYTES:,}; refusing to load")
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            md5.update(block)
    digest = sha256(path)
    if md5.hexdigest() != RFDETR_MD5 or digest != RFDETR_SHA256:
        raise UnverifiedCheckpoint(f"{path} MD5 {md5.hexdigest()} / SHA256 {digest} are not the pinned values; "
                                   "refusing to load")
    return {"path": str(path), "bytes": size, "md5": RFDETR_MD5, "sha256": digest, "url": RFDETR_URL,
            "licence": RFDETR_LICENCE}


# ------------------------------------------------------------------- data --

def segment_of(frame: int, split: dict) -> str | None:
    for name, (start, end) in split["segments"].items():
        if start <= frame < end:
            return name
    return None  # A gap frame


def coco_split(rows: list[dict], split: dict, width: int, height: int) -> dict[str, dict]:
    """A COCO dict per segment: every frame of the segment is an image (frames without a
    label are negatives), boxes clipped to the frame, ones under 2 px dropped."""
    grouped = by_frame(rows)
    result = {}
    for name, (start, end) in split["segments"].items():
        images, annotations = [], []
        for frame in range(start, end):
            images.append({"id": frame, "file_name": f"{frame:06d}.jpg", "width": width, "height": height})
            for row in grouped.get(frame, []):
                x1, y1 = max(0.0, row["box"][0]), max(0.0, row["box"][1])
                x2, y2 = min(float(width), row["box"][2]), min(float(height), row["box"][3])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                annotations.append({"id": len(annotations) + 1, "image_id": frame, "category_id": 1,
                                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                    "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0})
        result[name] = {"images": images, "annotations": annotations,
                        "categories": [{"id": 1, "name": "fish", "supercategory": "none"}]}
    return result


def labels_path(directory: Path) -> Path:
    return directory / "labels" / LABELS / "rows.jsonl"


def data_stage(args) -> dict:
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts)
    labels = labels_path(directory)
    if not labels.exists():
        raise SystemExit(f"No SAM 3 labels at {labels}; run the labels stage first")
    target = directory / "student" / "dataset"

    def step(heartbeat):
        import cv2

        split = time_split(facts["file"]["decoded_frames"], facts["file"]["fps"], **SPLIT)
        coco = coco_split(read_rows(labels), split, facts["file"]["width"], facts["file"]["height"])
        wanted = {image["id"]: name for name, data in coco.items() for image in data["images"]}
        capture, index = cv2.VideoCapture(facts["file"]["path"]), 0
        try:
            while capture.grab():
                if index in wanted:
                    _, image = capture.retrieve()
                    path = target / SPLIT_DIRS[wanted[index]] / f"{index:06d}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if index % 200 == 0:
                        heartbeat(frame=index)
                index += 1
        finally:
            capture.release()
        for name, data in coco.items():
            save(target / SPLIT_DIRS[name] / "_annotations.coco.json", data)
        save(directory / "student" / "split.json", split)
        return {"split": split, "dataset": target.as_posix(),
                "counts": {name: {"images": len(d["images"]), "boxes": len(d["annotations"])} for name, d in coco.items()},
                "review_status": "unreviewed SAM 3 suggestions; never ground truth", "licence": SAM3_LICENCE}

    return journal.run(f"{facts['key']}/student/data", "cpu", {"video": facts["file"]["sha256"],
                                                               "labels": file_id(labels)},
                       SPLIT | {"jpeg_quality": 92}, step)


# --------------------------------------------------------------- training --

def size_epochs(probe_seconds: float, remaining: float, *, margin: float = MARGIN_SECONDS,
                bounds: dict = EPOCHS) -> int:
    """As many epochs as the measured probe epoch says fit in what is left of the training
    budget after the margin, within ``bounds``; refused if even the minimum does not fit."""
    epochs = min(bounds["max"], math.floor((remaining - margin) / probe_seconds))
    if epochs < bounds["min"]:
        raise BudgetExceeded(f"Only {epochs} epochs of {probe_seconds:.0f} s fit in the {remaining:.0f} s "
                             f"training budget left (minimum {bounds['min']})")
    return epochs


def _train(dataset: Path, output: Path, epochs: int, max_seconds: float, heartbeat) -> dict:
    """RF-DETR Nano's own training loop with two additions on its Lightning trainer: a
    ``max_time`` stop and a callback that beats the journal after every epoch."""
    from datetime import timedelta

    from pytorch_lightning import Callback
    from rfdetr import RFDETRNano, training

    seen = {"epochs": 0, "metrics": {}}

    class Beat(Callback):
        def on_train_epoch_end(self, trainer, module):
            metrics = {}
            for key, value in trainer.callback_metrics.items():
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError, RuntimeError):
                    pass
            seen.update(epochs=trainer.current_epoch + 1, metrics=metrics)
            heartbeat(epoch=trainer.current_epoch + 1, of=epochs, metrics=metrics)

    original = training.build_trainer

    def build(train_config, model_config, **kwargs):
        trainer = original(train_config, model_config, **kwargs | {"max_time": timedelta(seconds=max_seconds)})
        trainer.callbacks.append(Beat())
        return trainer

    training.build_trainer = build
    started = time.monotonic()
    try:
        model = RFDETRNano(pretrain_weights=str(Path(verify_rfdetr()["path"])), resolution=RESOLUTION)
        model.train(dataset_dir=str(dataset), output_dir=str(output), epochs=epochs, resolution=RESOLUTION,
                    lr_scheduler="step", lr_scheduler_kwargs={"lr_drop": max(1, round(.8 * epochs))}, **TRAIN)
    finally:
        training.build_trainer = original
    best = output / "checkpoint_best_total.pth"
    return {"seconds": time.monotonic() - started, "epochs_asked": epochs, "epochs_run": seen["epochs"],
            "stopped_by_max_time": seen["epochs"] < epochs, "last_metrics": seen["metrics"],
            "best": file_id(best) if best.exists() else None,
            "files": sorted(p.name for p in output.iterdir() if p.is_file())}


def train_stage(args) -> dict:
    root, facts = video_record(args)
    journal = Journal(root)
    data = data_stage(args)
    checkpoint = verify_rfdetr()
    models = root / "models" / f"{NAME}-{facts['key']}"
    inputs = {"dataset": data["counts"], "split": data["split"]["segments"], "checkpoint": checkpoint["sha256"],
              "labels": journal.data["entries"][f"{facts['key']}/student/data"]["inputs"]["labels"]["sha256"]}
    settings = TRAIN | {"resolution": RESOLUTION, "variant": "nano"}
    dataset = Path(data["dataset"])

    probe = journal.run(f"{facts['key']}/student/probe", "training", inputs, settings | {"epochs": 1},
                        lambda beat: _train(dataset, models / "probe", 1, journal.remaining("training"), beat),
                        estimate=600)
    epochs = size_epochs(probe["seconds"], journal.remaining("training"))

    def step(heartbeat):
        result = _train(dataset, models / "train", epochs, journal.remaining("training") - 120, heartbeat)
        if result["best"] is None:
            raise RuntimeError(f"Training finished without {models / 'train' / 'checkpoint_best_total.pth'}")
        return result | {"probe_epoch_seconds": probe["seconds"], "licences": [RFDETR_LICENCE, SAM3_LICENCE]}

    result = journal.run(f"{facts['key']}/student/train", "training", inputs, settings | {"epochs": epochs}, step,
                         estimate=epochs * probe["seconds"])
    journal.decide("student_model", {"checkpoint": result["best"], "trained_on": facts["key"], "name": NAME,
                                     "resolution": RESOLUTION, "epochs_run": result["epochs_run"],
                                     "licences": [RFDETR_LICENCE, SAM3_LICENCE]})
    return result


# -------------------------------------------------------------- inference --

def percentiles(values: list[float]) -> dict:
    values = sorted(values)
    at = lambda q: values[min(len(values) - 1, int(q * len(values)))] if values else None
    return {"n": len(values), "mean": sum(values) / len(values) if values else None,
            "p50": at(.5), "p95": at(.95), "max": values[-1] if values else None}


def detect_video(model, video: Path, *, floor: float = FLOOR, heartbeat=None) -> tuple[list[dict], dict]:
    """The student on every frame, one at a time as live mode would, from the decoded
    1080p frame to boxes in source pixels. Decode and model time are measured apart; the
    first frames (CUDA warm-up) are left out of the timings."""
    import cv2
    import torch

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    torch.cuda.reset_peak_memory_stats()
    frames, decode, infer, index, size = [], [], [], 0, None
    try:
        while True:
            started = time.perf_counter()
            ok, image = capture.read()
            decoded = time.perf_counter()
            if not ok:
                break
            size = [image.shape[1], image.shape[0]]
            detections = model.predict(image[:, :, ::-1].copy(), threshold=floor, include_source_image=False)
            detections = detections[detections.class_id == FISH_LABEL]
            done = time.perf_counter()
            frames.append({"frame": index, "boxes": detections.xyxy.round(2).tolist(), "ids": [],
                           "scores": [round(float(s), 5) for s in detections.confidence]})
            if index >= WARMUP_FRAMES:
                decode.append(1000 * (decoded - started))
                infer.append(1000 * (done - decoded))
            index += 1
            if heartbeat and index % 200 == 0:
                heartbeat(frame=index)
    finally:
        capture.release()
    total = [a + b for a, b in zip(decode, infer)]
    speed = {"frame_size": size, "input_resolution": RESOLUTION,
             "gpu": torch.cuda.get_device_name(0), "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
             "warmup_frames_excluded": WARMUP_FRAMES, "decode_ms": percentiles(decode),
             "model_ms": percentiles(infer), "total_ms": percentiles(total),
             "fps": 1000 / (sum(total) / len(total)) if total else None}
    speed["keeps_up_with_live"] = bool(total) and speed["total_ms"]["p95"] <= LIVE["frame_budget_ms"]
    return frames, speed


def raw_rows(frames: list[dict], fps: float, threshold: float) -> list[dict]:
    return [{"id": None, "frame": f["frame"], "t": f["frame"] / fps, "box": box, "confidence": score}
            for f in frames for box, score in zip(f["boxes"], f["scores"]) if score >= threshold]


def choose_threshold(frames: list[dict], labels: list[dict], segment: list[int], fps: float,
                     thresholds: list[float] = THRESHOLDS) -> dict:
    """The operating threshold: the one whose boxes best agree with SAM 3's labels on the
    validation segment (matched share of both, IoU >= 0.5); ties go to the higher one."""
    frames_in = [f for f in frames if segment[0] <= f["frame"] < segment[1]]
    analysed = range(*segment)
    sweep = {}
    for threshold in thresholds:
        summary, _ = agreement(raw_rows(frames_in, fps, threshold), labels, analysed, analysed)
        matched = summary["matched"]
        sweep[str(threshold)] = {"student_boxes": summary["left_boxes"], "label_boxes": summary["right_boxes"],
                                 "matched": matched,
                                 "f1": 2 * matched / max(1, summary["left_boxes"] + summary["right_boxes"])}
    best = max(thresholds, key=lambda t: (round(sweep[str(t)]["f1"], 6), t))
    return {"threshold": best, "segment": segment, "sweep": sweep,
            "rule": "max matched share of student and SAM 3 label boxes (IoU >= 0.5) on the validation "
                    "segment only; ties to the higher threshold"}


def student_tracker(journal: Journal, threshold: float, fps: float):
    """BoT-SORT with the BrackishMOT settings, its score gates moved to the student's
    threshold: every box kept can start or extend a track."""
    settings = settings_for(journal.data["decisions"].get("brackishmot_settings"), "botsort", fps)
    return replace(settings, track_high_thresh=threshold, new_track_thresh=threshold, track_low_thresh=threshold / 2)


def load_student(checkpoint: dict):
    """The trained student, after its file matches the SHA256 recorded when it was trained."""
    path = Path(checkpoint["path"])
    if not path.is_file() or sha256(path) != checkpoint["sha256"]:
        raise UnverifiedCheckpoint(f"{path} is missing or changed since training; refusing to load")
    from rfdetr import RFDETRNano

    return RFDETRNano.from_checkpoint(str(path))


def in_segment(rows: list[dict], segment) -> list[dict]:
    return [r for r in rows if segment[0] <= r["frame"] < segment[1]]


def segment_agreement(left: list[dict], right: list[dict], left_frames, right_frames, segment) -> dict:
    inside = lambda frames: [f for f in frames if segment[0] <= f < segment[1]]
    summary, _ = agreement(in_segment(left, segment), in_segment(right, segment),
                           inside(left_frames), inside(right_frames))
    return {k: v for k, v in summary.items() if k != "frame_shift_check"}


def eval_stage(args) -> dict:
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts)
    model = journal.data["decisions"].get("student_model")
    if not model:
        raise SystemExit("No trained student recorded (run student-train first)")
    fps, name = facts["file"]["fps"], f"student-{NAME}"
    target = directory / "student" / NAME
    trained_here = model["trained_on"] == facts["key"]
    split = read(directory / "student" / "split.json") if trained_here else None

    def infer(heartbeat):
        frames, speed = detect_video(load_student(model["checkpoint"]), Path(facts["file"]["path"]),
                                     heartbeat=heartbeat)
        if len(frames) != facts["file"]["decoded_frames"]:
            raise RuntimeError(f"The student saw {len(frames)} frames, the video has {facts['file']['decoded_frames']}")
        write_rows(frames, target / "frames.jsonl")
        return {"speed": speed, "frames": len(frames), "boxes_at_floor": sum(len(f["boxes"]) for f in frames),
                "floor": FLOOR, "frames_file": file_id(target / "frames.jsonl")}

    inference = journal.run(f"{facts['key']}/student/infer", "gpu", {"video": facts["file"]["sha256"],
                                                                     "model": model["checkpoint"]["sha256"]},
                            {"floor": FLOOR, "resolution": RESOLUTION}, infer)
    frames = read_rows(target / "frames.jsonl")
    labels = read_rows(labels_path(directory))

    if "student_threshold" not in journal.data["decisions"]:
        if not trained_here:
            raise SystemExit("Evaluate the training video first: its validation segment sets the threshold")
        journal.decide("student_threshold", choose_threshold(frames, labels, split["segments"]["validation"], fps)
                       | {"chosen_before": "any held-out (test segment or other video) agreement was computed"})
    threshold = journal.data["decisions"]["student_threshold"]["threshold"]
    tracker = student_tracker(journal, threshold, fps)

    def track(heartbeat):
        tracked = track_rows(raw_rows(frames, fps, threshold), list(range(len(frames))), tracker, fps)
        for row in tracked:
            row["id"] = f"student-{row['id']}"
        write_rows(tracked, target / "detections.jsonl")
        save(target / "analysed.json", {"frames": list(range(len(frames)))})
        return {"threshold": threshold, "tracker": tracker.__dict__, "tracks": len({r["id"] for r in tracked}),
                "boxes": len(tracked)}

    tracking = journal.run(f"{facts['key']}/student/track", "cpu", {"frames": inference["frames_file"]["sha256"]},
                           {"threshold": threshold, "tracker": tracker.tag()}, track)
    static_stage(args)  # The student's tracks go through the same static suppression as SAM 3's

    def compare(heartbeat):
        everything = [0, facts["file"]["decoded_frames"]]
        segments = dict(split["segments"]) if trained_here else {"held_out": everything}
        kept = read_rows(directory / "static" / name / "kept.jsonl")
        analysed = list(range(facts["file"]["decoded_frames"]))
        references = {"sam3_labels": (labels, analysed),
                      "sam3_tracked_after_static": (read_rows(directory / "static" / f"sam3-{LABELS}" / "kept.jsonl"),
                                                    analysed)}
        viame_kept = directory / "static" / "viame" / "kept.jsonl"
        if viame_kept.exists():
            references["viame"] = (read_rows(viame_kept), read(directory / "viame" / "analysed.json")["frames"])
        result = {"note": "Agreement with SAM 3 (the teacher) and VIAME, never precision or recall. "
                          + ("Train and validation segments were seen in training; only test is held out."
                             if trained_here else "The student never saw this video."),
                  "threshold": threshold, "segments": segments, "agreement": {}, "proxies": {}}
        for segment_name, segment in segments.items():
            result["agreement"][segment_name] = {
                f"student_vs_{ref}": segment_agreement(kept, rows, analysed, frames_of, segment)
                for ref, (rows, frames_of) in references.items()}
            if "viame" in references:  # The teacher's own agreement with VIAME, for scale
                result["agreement"][segment_name]["sam3_labels_vs_viame"] = segment_agreement(
                    labels, references["viame"][0], analysed, references["viame"][1], segment)
            result["proxies"][segment_name] = {"student": proxies(in_segment(kept, segment), facts, fps),
                                               "sam3_labels": proxies(in_segment(labels, segment), facts, fps)}
        save(target / "comparison.json", result)
        return result

    comparison = journal.run(f"{facts['key']}/student/compare", "cpu",
                             {"kept": file_id(directory / "static" / name / "kept.jsonl"),
                              "labels": file_id(labels_path(directory))}, {"iou": .5}, compare)
    return {"speed": inference["speed"], "tracking": tracking, "comparison": comparison}


def student_stage(args) -> dict:
    return {"data": data_stage(args), "train": train_stage(args), "eval": eval_stage(args)}


STAGES = {"student-data": data_stage, "student-train": train_stage, "student-eval": eval_stage,
          "student": student_stage}


# ----------------------------------------------------------------- rounds --

ROUND_ROOT = Path("data/outputs/smartbay-rounds/round-1")
ROUND_STRIDE = 5  # Every 5th frame of each window: neighbours are near-duplicates
VALIDATION_PERCENT = 10
ROUND_SEED = 17
# A round holds about 70 times the single video's frames, so an epoch is long; a student of
# even one epoch beats none from an unattended night, so only a probe epoch too long to repeat
# within the budget is refused. The probe epoch still sizes the run.
ROUND_EPOCHS = {"min": 1, "max": 50}
STRATA = ("strong", "weak", "none", "murky")


def choose_validation(train_days: list[str], frozen: dict | None = None, seed: int = ROUND_SEED) -> dict:
    """Validation days: an earlier round's (kept), plus a seeded VALIDATION_PERCENT of the
    training days it never considered, rounded half up, one pick from each consecutive
    stretch so they spread over the season."""
    frozen = frozen or {"days": [], "considered": []}
    new = [d for d in sorted(train_days) if d not in frozen["considered"]]
    count = (VALIDATION_PERCENT * len(new) + 50) // 100
    rng = random.Random(seed)
    picks = [rng.choice(new[round(i * len(new) / count):round((i + 1) * len(new) / count)]) for i in range(count)]
    return {"days": sorted(set(frozen["days"]) & set(train_days) | set(picks)),
            "considered": sorted(set(frozen["considered"]) | set(train_days)), "seed": seed,
            "percent_of_new_training_days": VALIDATION_PERCENT}


def split_of(role: str, day: str, validation: set[str]) -> str | None:
    """A window's split from its day: test days test, validation days validation, other
    training days train; expert and unused days never enter a dataset."""
    if role == "test":
        return "test"
    if role == "train":
        return "validation" if day in validation else "train"
    return None


def round_windows(root: Path) -> list[dict]:
    """Every window of the round's ``to_label.csv`` as checked by ``round_checks``, with
    the cut file it was labelled from, its start frame in the source clip, its size and
    whether it is kept. A window whose role is not its day's role is refused."""
    report = read(root / "checks" / "quality_report.json")
    roles = read(root / "selection" / "day_roles.json")["roles"]
    with (root / "selection" / "to_label.csv").open(newline="", encoding="utf-8") as stream:
        rows = {r["window_id"]: r for r in csv.DictReader(stream)}
    windows = []
    for checked in report["per_window"]:
        row, key = rows[checked["window_id"]], checked["labelled_as"]
        if roles.get(row["day"]) != row["role"]:
            raise ValueError(f"{row['window_id']}: role {row['role']} but day {row['day']} is {roles.get(row['day'])}")
        window = {"window_id": key or row["window_id"], "planned_window_id": row["window_id"], "day": row["day"],
                  "role": row["role"], "evidence": row["evidence"], "round": row.get("round"),
                  "excluded": checked["excluded"]}
        if key and not checked["excluded"]:
            facts = read(root / "videos" / key / "video.json")["file"]
            record = root / "windows" / f"{key}.json"
            start = read(record)["start_frame"] if record.exists() else int(
                row["start_frame"] if key == row["window_id"] else row["reserve_start_frame"])
            window |= {"start_frame": start, "frames": facts["decoded_frames"], "width": facts["width"],
                       "height": facts["height"], "video": facts["path"], "video_sha256": facts["sha256"],
                       "fps": facts["fps"]}
        windows.append(window)
    return windows


def round_coco(windows: list[dict], labels: dict[str, list[dict]], validation: set[str],
               stride: int = ROUND_STRIDE) -> dict[str, dict]:
    """A COCO dict per split over the kept windows: every ``stride``-th frame of a window is
    an image carrying its window, its frame there and its source frame (frames without a
    label are negatives); boxes clipped to the frame, ones under 2 px dropped."""
    result = {name: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "fish", "supercategory": "none"}]}
              for name in SPLIT_DIRS}
    for window in windows:
        split = None if window["excluded"] else split_of(window["role"], window["day"], validation)
        if split is None:
            continue
        data, width, height = result[split], window["width"], window["height"]
        grouped = by_frame(labels.get(window["window_id"], []))
        for frame in range(0, window["frames"], stride):
            image_id = len(data["images"]) + 1
            data["images"].append({"id": image_id, "file_name": f"{window['window_id']}_{frame:03d}.jpg",
                                   "width": width, "height": height, "window_id": window["window_id"], "frame": frame,
                                   "source_frame": window["start_frame"] + frame, "day": window["day"],
                                   "evidence": window["evidence"], "round": window["round"]})
            for row in grouped.get(frame, []):
                x1, y1 = max(0.0, row["box"][0]), max(0.0, row["box"][1])
                x2, y2 = min(float(width), row["box"][2]), min(float(height), row["box"][3])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                data["annotations"].append({"id": len(data["annotations"]) + 1, "image_id": image_id,
                                            "category_id": 1, "iscrowd": 0,
                                            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                            "area": round((x2 - x1) * (y2 - y1), 2)})
    return result


def coco_counts(coco: dict[str, dict]) -> dict:
    """Images and boxes per split, and per stratum within each split."""
    counts = {}
    for name, data in coco.items():
        stratum = {image["id"]: image["evidence"] for image in data["images"]}
        boxes = Counter(stratum[a["image_id"]] for a in data["annotations"])
        images = Counter(stratum.values())
        counts[name] = {"images": len(data["images"]), "boxes": len(data["annotations"]),
                        "windows": len({i["window_id"] for i in data["images"]}),
                        "negative_images": len(data["images"]) - len({a["image_id"] for a in data["annotations"]}),
                        "by_stratum": {s: {"images": images[s], "boxes": boxes[s]} for s in STRATA}}
    return counts


def round_data(root: Path = ROUND_ROOT, validation_from: Path | None = None) -> dict:
    """Part 6: the day-split COCO dataset of a round, under ``<root>/dataset``."""
    journal = Journal(root)
    windows = round_windows(root)
    if "validation_days" not in journal.data["decisions"]:
        roles = read(root / "selection" / "day_roles.json")["roles"]
        frozen = read(validation_from)["decisions"]["validation_days"] if validation_from else None
        journal.decide("validation_days", choose_validation([d for d, r in roles.items() if r == "train"], frozen)
                       | {"chosen_before": "any training", "kept_from": validation_from.as_posix()
                          if validation_from else None})
    validation = set(journal.data["decisions"]["validation_days"]["days"])
    target = root / "dataset"
    report_path = root / "checks" / "quality_report.json"

    def step(heartbeat):
        import cv2

        kept = [w for w in windows if not w["excluded"] and split_of(w["role"], w["day"], validation)]
        labels = {w["window_id"]: read_rows(root / "videos" / w["window_id"] / "labels" / LABELS / "rows.jsonl")
                  for w in kept}
        coco = round_coco(windows, labels, validation)
        (root / "student").mkdir(parents=True, exist_ok=True)
        for name, data in coco.items():
            wanted = {}
            for image in data["images"]:
                wanted.setdefault(image["window_id"], {})[image["frame"]] = image["file_name"]
            folder = target / SPLIT_DIRS[name]
            folder.mkdir(parents=True, exist_ok=True)
            for n, window in enumerate(w for w in kept if w["window_id"] in wanted):
                if sha256(Path(window["video"])) != window["video_sha256"]:
                    raise RuntimeError(f"{window['video']} changed since it was labelled")
                capture, index = cv2.VideoCapture(window["video"]), 0
                try:
                    while capture.grab():
                        if index in wanted[window["window_id"]]:
                            _, image = capture.retrieve()
                            cv2.imwrite(str(folder / wanted[window["window_id"]][index]), image,
                                        [cv2.IMWRITE_JPEG_QUALITY, 92])
                        index += 1
                finally:
                    capture.release()
                if index != window["frames"]:
                    raise RuntimeError(f"{window['video']}: decoded {index} frames, expected {window['frames']}")
                heartbeat(split=name, window=n + 1)
            save(target / SPLIT_DIRS[name] / "_annotations.coco.json", data)
        result = {"dataset": target.as_posix(), "counts": coco_counts(coco), "stride": ROUND_STRIDE,
                  "validation_days": sorted(validation),
                  "windows": [{k: w.get(k) for k in ("window_id", "planned_window_id", "day", "role", "evidence",
                                                     "round", "excluded", "video", "video_sha256", "fps", "frames")}
                              | {"split": None if w["excluded"] else split_of(w["role"], w["day"], validation)}
                              for w in windows],
                  "review_status": "unreviewed SAM 3 suggestions; never ground truth", "licence": SAM3_LICENCE}
        save(root / "student" / "round_dataset.json", result)
        return result

    return journal.run("round/student/data", "cpu", {"quality_report": file_id(report_path)},
                       {"stride": ROUND_STRIDE, "validation_days": sorted(validation), "jpeg_quality": 92}, step)


def round_train(root: Path = ROUND_ROOT, validation_from: Path | None = None) -> dict:
    """Part 7: RF-DETR Nano on the round's dataset, epochs sized by a probe epoch to fit
    what is left of the journal's training budget."""
    journal = Journal(root)
    data = round_data(root, validation_from)
    checkpoint = verify_rfdetr()
    models = root / "models" / f"{NAME}-{root.name}"
    inputs = {"dataset": data["counts"], "validation_days": data["validation_days"], "checkpoint": checkpoint["sha256"]}
    settings = TRAIN | {"resolution": RESOLUTION, "variant": "nano"}
    dataset = Path(data["dataset"])
    probe = journal.run("round/student/probe", "training", inputs, settings | {"epochs": 1},
                        lambda beat: _train(dataset, models / "probe", 1, journal.remaining("training"), beat),
                        estimate=600)
    epochs = size_epochs(probe["seconds"], journal.remaining("training"), bounds=ROUND_EPOCHS)

    def step(heartbeat):
        result = _train(dataset, models / "train", epochs, journal.remaining("training") - 120, heartbeat)
        if result["best"] is None:
            raise RuntimeError(f"Training finished without {models / 'train' / 'checkpoint_best_total.pth'}")
        return result | {"probe_epoch_seconds": probe["seconds"], "licences": [RFDETR_LICENCE, SAM3_LICENCE]}

    result = journal.run("round/student/train", "training", inputs, settings | {"epochs": epochs}, step,
                         estimate=epochs * probe["seconds"])
    journal.decide("student_model", {"checkpoint": result["best"], "trained_on": f"round {root.name}", "name": NAME,
                                     "resolution": RESOLUTION, "epochs_run": result["epochs_run"],
                                     "dataset": data["counts"], "licences": [RFDETR_LICENCE, SAM3_LICENCE]})
    return result


def pool(summaries: list[dict]) -> dict:
    """Agreement summed over windows: matched boxes against both sides' boxes."""
    left, right = sum(s["left_boxes"] for s in summaries), sum(s["right_boxes"] for s in summaries)
    matched = sum(s["matched"] for s in summaries)
    return {"windows": len(summaries), "student_boxes": left, "sam3_label_boxes": right, "matched": matched,
            "agreement": round(matched / max(1, left + right - matched), 4),
            "share_of_student_matched": round(matched / max(1, left), 4),
            "share_of_sam3_matched": round(matched / max(1, right), 4)}


def round_eval(root: Path = ROUND_ROOT) -> dict:
    """Part 7: the student on every frame of the validation and test windows. The threshold
    is chosen on validation windows only, before any test-day agreement is computed."""
    journal = Journal(root)
    model = journal.data["decisions"].get("student_model")
    if not model or not model["trained_on"].startswith("round"):
        raise SystemExit("No round-trained student recorded (run round-train first)")
    data = read(root / "student" / "round_dataset.json")
    loaded = {}

    def infer(window: dict) -> list[dict]:
        target = root / "student" / NAME / window["window_id"] / "frames.jsonl"

        def step(heartbeat):
            if "model" not in loaded:
                loaded["model"] = load_student(model["checkpoint"])
            frames, speed = detect_video(loaded["model"], Path(window["video"]), heartbeat=heartbeat)
            if len(frames) != window["frames"]:
                raise RuntimeError(f"The student saw {len(frames)} frames of {window['window_id']}, not {window['frames']}")
            write_rows(frames, target)
            return {"speed": speed, "frames": len(frames), "frames_file": file_id(target)}

        result = journal.run(f"round/student/infer/{window['window_id']}", "gpu",
                             {"video": window["video_sha256"], "model": model["checkpoint"]["sha256"]},
                             {"floor": FLOOR, "resolution": RESOLUTION}, step)
        window["speed"] = result["speed"]
        return read_rows(target)

    def labels_of(window: dict) -> list[dict]:
        return read_rows(root / "videos" / window["window_id"] / "labels" / LABELS / "rows.jsonl")

    def summary(window: dict, frames: list[dict], threshold: float) -> dict:
        analysed = range(window["frames"])
        result, _ = agreement(raw_rows(frames, window["fps"], threshold), labels_of(window), analysed, analysed)
        return result

    validation = [w for w in data["windows"] if w["split"] == "validation"]
    test = [w for w in data["windows"] if w["split"] == "test"]
    frames = {w["window_id"]: infer(w) for w in validation + test}
    if "student_threshold" not in journal.data["decisions"]:
        sweep = {str(t): pool([summary(w, frames[w["window_id"]], t) for w in validation]) for t in THRESHOLDS}
        f1 = lambda t: 2 * sweep[str(t)]["matched"] / max(1, sweep[str(t)]["student_boxes"]  # noqa: E731
                                                          + sweep[str(t)]["sam3_label_boxes"])
        journal.decide("student_threshold", {
            "threshold": max(THRESHOLDS, key=lambda t: (round(f1(t), 6), t)), "sweep": sweep,
            "validation_windows": len(validation), "chosen_before": "any test-day agreement was computed",
            "rule": "max matched share of student and SAM 3 label boxes (IoU >= 0.5) over the validation windows; "
                    "ties to the higher threshold"})
    threshold = journal.data["decisions"]["student_threshold"]["threshold"]
    per_window = {w["window_id"]: summary(w, frames[w["window_id"]], threshold) for w in test}
    empty_water = {}
    for stratum in ("none", "murky"):
        chosen = [w for w in test if w["evidence"] == stratum]
        boxes = [len(raw_rows([f], w["fps"], threshold)) for w in chosen for f in frames[w["window_id"]]]
        label_boxes = sum(len(labels_of(w)) for w in chosen)
        empty_water[stratum] = {"windows": len(chosen), "frames": len(boxes),
                                "student_boxes_per_frame": round(sum(boxes) / max(1, len(boxes)), 4),
                                "frames_with_a_student_box": round(sum(b > 0 for b in boxes) / max(1, len(boxes)), 4),
                                "sam3_label_boxes_per_frame": round(label_boxes / max(1, len(boxes)), 4)}
    speeds = [w["speed"] for w in validation + test]
    result = {
        "note": "Agreement with SAM 3's unreviewed labels on test days only, never precision, recall or accuracy. "
                "Test days were never trained on; the threshold came from validation days.",
        "threshold": threshold, "test_days": sorted({w["day"] for w in test}),
        "agreement": {"all": pool(list(per_window.values())),
                      **{s: pool([per_window[w["window_id"]] for w in test if w["evidence"] == s]) for s in STRATA}},
        "empty_water_proxy": empty_water,
        "speed_1080p": {"windows": len(speeds), "gpu": speeds[0]["gpu"] if speeds else None,
                        "total_ms_p50_median": percentiles([s["total_ms"]["p50"] for s in speeds])["p50"],
                        "total_ms_p95_max": max((s["total_ms"]["p95"] for s in speeds), default=None),
                        "fps_median": percentiles([s["fps"] for s in speeds])["p50"],
                        "keeps_up_with_live_everywhere": all(s["keeps_up_with_live"] for s in speeds)},
        "per_window": {k: {x: v[x] for x in ("left_boxes", "right_boxes", "matched", "agreement")}
                       for k, v in per_window.items()},
        "licences": [RFDETR_LICENCE, SAM3_LICENCE]}
    save(root / "student" / "round_eval.json", result)
    return result


ROUND_STAGES = {"round-data": round_data, "round-train": round_train, "round-eval": round_eval}


# ------------------------------------------------------------ local check --

LOCAL_DIR = "local-eval"  # A student's outputs land in <root>/local-eval/<video key>/
LOCAL_COLOURS = {"matched": (80, 220, 80), "student_only": (0, 165, 255), "sam3_only": (255, 0, 255),
                 "student": (0, 255, 255)}
LOCAL_RENDERS = {"compare": "student-vs-sam3-labels.mp4", "alone": "student-alone.mp4"}


def local_checkpoint(root: Path, checkpoint: dict) -> dict:
    """The checkpoint record with its path found on this PC: a round's journal records the
    instance's path, and the pulled copy sits at the same place under ``root``."""
    path = Path(checkpoint["path"])
    if not path.is_file() and "models" in path.parts:
        path = root.joinpath(*path.parts[path.parts.index("models"):])
    return checkpoint | {"path": str(path)}


def training_overlap(root: Path, model: dict, key: str) -> dict:
    """How near a video is to the student's training data: the video itself, or for a
    round, its day's role and whether a window was cut from the same source clip."""
    result = {"trained_on_this_video": model["trained_on"] == key}
    roles, windows = root / "selection" / "day_roles.json", root / "selection" / "to_label.csv"
    clip = re.match(r"(SmartBayCam\d+)_(\d{4}-\d{2}-\d{2})_\d{4}", key)
    if clip and roles.exists() and windows.exists():
        with windows.open(newline="", encoding="utf-8") as stream:
            stems = {Path(r["source_mp4_url"]).stem for r in csv.DictReader(stream)}
        result |= {"camera_in_round": any(s.startswith(clip[1] + "_") for s in stems),
                   "day_role": read(roles)["roles"].get(clip[2]), "same_clip_in_round": clip[0] in stems}
    return result


def render_local(video: Path, frames: list[dict], labels: list[dict] | None, fps: float, threshold: float,
                 target: Path, title: str = "student", size: tuple[int, int] = (1280, 720)) -> dict:
    """The student's boxes and SAM 3's labels on one panel, paired per frame at IoU >= 0.5;
    without labels, the student's boxes alone with their scores and a count per frame."""
    import cv2
    import numpy as np

    from scripts.sam3_botsort import h264_profile, h264_writer, pair_frames

    student, sam3 = by_frame(raw_rows(frames, fps, threshold)), by_frame(labels or [])
    legend = ([(f"{title} (score >= {threshold})", "student")] if labels is None else
              [("both (IoU>=0.5)", "matched"), (f"student only (>= {threshold})", "student_only"),
               ("SAM 3 label only", "sam3_only")])
    writer = h264_writer(target, fps, (size[0], size[1] + 50))
    capture, index, counts = cv2.VideoCapture(str(video)), 0, Counter()
    scale = None
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            scale = scale or size[0] / image.shape[1]
            view = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            if labels is None:
                drawn = [(r, "student") for r in student.get(index, [])]
            else:
                pairs = pair_frames([index], student, sam3)
                drawn = ([(m["left"], "matched") for m in pairs["matches"]]
                         + [(r, "student_only") for r in pairs["left_unique"]]
                         + [(r, "sam3_only") for r in pairs["right_unique"]])
            for row, kind in drawn:
                x1, y1, x2, y2 = (int(v * scale) for v in row["box"])
                cv2.rectangle(view, (x1, y1), (x2, y2), LOCAL_COLOURS[kind], 2)
                if labels is None:
                    cv2.putText(view, f"{row['confidence']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                .4, LOCAL_COLOURS[kind], 1)
                counts[kind] += 1
            canvas = np.zeros((size[1] + 50, size[0], 3), np.uint8)
            canvas[:size[1]] = view
            x = 12
            for label, kind in legend:
                cv2.rectangle(canvas, (x, size[1] + 14), (x + 26, size[1] + 36), LOCAL_COLOURS[kind], -1)
                cv2.putText(canvas, label, (x + 34, size[1] + 32), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                x += 60 + 11 * len(label)
            status = f"{len(drawn)} boxes  " * (labels is None) + f"frame {index}  t={index / fps:.1f}s"
            cv2.putText(canvas, status, (size[0] - 12 - 11 * len(status), size[1] + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            writer.write(canvas)
            index += 1
    finally:
        capture.release()
        writer.release()
    return {"video": file_id(target), "frames": index, "encoding": h264_profile(target), "boxes_drawn": dict(counts)}


def local_eval(root: Path, videos: list[Path], render: str | None = None) -> dict:
    """A trained student on videos SAM 3 labelled on the PC, scored as ``round-eval`` scores
    test days: its boxes at the journal's threshold against SAM 3's labels (IoU >= 0.5) on
    every frame, plus speed on this GPU. A video with ``student/split.json`` also reports its
    test segment, the part a student trained on that video never saw. Inference is reused
    while the model and video SHA256s match. ``render`` (a ``LOCAL_RENDERS`` key) also writes
    an MP4 per video, against SAM 3's labels or the student alone."""
    decisions = read(root / "journal.json")["decisions"]
    model, threshold = decisions.get("student_model"), decisions.get("student_threshold", {}).get("threshold")
    if not model or threshold is None:
        raise SystemExit(f"{root / 'journal.json'} records no trained student with a threshold")
    checkpoint = local_checkpoint(root, model["checkpoint"])
    loaded, results = {}, {}
    for directory in videos:
        facts = read(directory / "video.json")
        key, fps, count = facts["key"], facts["file"]["fps"], facts["file"]["decoded_frames"]
        target = root / LOCAL_DIR / key
        record = target / "inference.json"
        inputs = {"model_sha256": checkpoint["sha256"], "video_sha256": facts["file"]["sha256"]}
        if not (record.exists() and all(read(record).get(k) == v for k, v in inputs.items())):
            video = Path(facts["file"]["path"])
            if sha256(video) != facts["file"]["sha256"]:
                raise RuntimeError(f"{video} changed since SAM 3 labelled it")
            if "model" not in loaded:
                loaded["model"] = load_student(checkpoint)
            frames, speed = detect_video(loaded["model"], video)
            if len(frames) != count:
                raise RuntimeError(f"The student saw {len(frames)} frames of {key}, not {count}")
            write_rows(frames, target / "frames.jsonl")
            save(record, inputs | {"speed": speed, "floor": FLOOR, "frames_file": file_id(target / "frames.jsonl")})
        frames, labels = read_rows(target / "frames.jsonl"), read_rows(labels_path(directory))
        rows = raw_rows(frames, fps, threshold)
        segments = {"all": [0, count]}
        if (directory / "student" / "split.json").exists():
            segments["test_segment"] = read(directory / "student" / "split.json")["segments"]["test"]
        result = {"video": facts["file"]["path"], "frames": count, "fps": fps,
                  "overlap": training_overlap(root, model, key), "speed": read(record)["speed"], "agreement": {}}
        for name, segment in segments.items():
            span = range(*segment)
            pooled = pool([segment_agreement(rows, labels, span, span, segment)])
            per_frame = Counter(r["frame"] for r in in_segment(rows, segment))
            result["agreement"][name] = pooled | {
                "segment": segment, "student_boxes_per_frame": round(pooled["student_boxes"] / len(span), 3),
                "sam3_boxes_per_frame": round(pooled["sam3_label_boxes"] / len(span), 3),
                "frames_with_a_student_box": round(len(per_frame) / len(span), 4)}
        if render:
            result["render"] = render_local(Path(facts["file"]["path"]), frames, labels if render == "compare" else None,
                                            fps, threshold, target / LOCAL_RENDERS[render],
                                            title=f"{model.get('name', NAME)} ({model['trained_on']})")
        results[key] = result
    summary = {"note": "Agreement with SAM 3's unreviewed labels, never precision, recall or accuracy.",
               "root": root.as_posix(), "checkpoint": checkpoint, "trained_on": model["trained_on"],
               "threshold": threshold, "videos": results, "licences": [RFDETR_LICENCE, SAM3_LICENCE]}
    (root / LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    save(root / LOCAL_DIR / "local_eval.json", summary)
    return summary


def main() -> None:
    os.environ.update(OFFLINE)  # Training and inference need no network
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*ROUND_STAGES, "round", "local-eval"])
    parser.add_argument("--root", type=Path, default=ROUND_ROOT, help="Round folder holding journal.json")
    parser.add_argument("--validation-from", type=Path, help="An earlier round's journal.json: keep its validation days")
    parser.add_argument("--video-dir", type=Path, action="append", default=[],
                        help="local-eval: a SAM 3 output folder holding video.json and labels/ (repeatable)")
    parser.add_argument("--render", nargs="?", const="compare", choices=list(LOCAL_RENDERS),
                        help="local-eval: also write an MP4 per video, against SAM 3's labels (compare, the default) "
                             "or the student's boxes alone (alone)")
    args = parser.parse_args()
    if args.stage == "local-eval":
        if not args.video_dir:
            parser.error("local-eval needs at least one --video-dir")
        result = local_eval(args.root, args.video_dir, args.render)
    elif args.stage == "round-eval":
        result = round_eval(args.root)
    elif args.stage == "round":
        result = {"train": round_train(args.root, args.validation_from), "eval": round_eval(args.root)}
    else:
        result = ROUND_STAGES[args.stage](args.root, args.validation_from)
    print(json.dumps(result, indent=2, default=str)[:20000])


if __name__ == "__main__":
    main()
