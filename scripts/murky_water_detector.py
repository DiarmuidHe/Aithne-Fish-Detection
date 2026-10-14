"""Murky-water Part 2: a Brackish-trained YOLO detector against production VIAME.

Stages and where they run:

- ``viame-test`` (VIAME worker image, like murky_water_tracker.py): production batch
  profile on the 10 Brackish **test** recordings' AVIs.
- ``train`` (host .venv with CUDA PyTorch): runs A (default augmentation) and B (plus the
  murk copies) through fish_model.py with the same seed, epochs and image size.
- ``evaluate`` (host): fish_model.py evaluate for A and B on the test split and on the
  synthetic darkened test copy, then ``score``.
- ``alignment`` (host): VIAME's test boxes against the labels at frame shifts -2..2, with
  overlays, proving the AVI-to-label frame and pixel mapping.
- ``score`` (host): VIAME, A and B on the same test frames and truth at IoU 0.5.
- ``smartbay`` (host): the better model through fish_model.py track on the fixed SmartBay
  clips, compared with Part 1's production VIAME runs. Proxies only, never accuracy.
- ``brackishmot`` (host): A, B and production VIAME tracking on the BrackishMOT sequences
  the leakage guard allows.

VIAME passes are journalled in part2/viame-journal.json and charged against the one GPU
budget shared with Parts 1 and 3; training time has its own journal. Nothing here touches production settings or data.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from scripts.viame_zero_label_benchmark import align, digest, metrics, observations, read, save

IN_CONTAINER = Path("/opt/noaa/viame").exists()
DATA = Path("/data") if IN_CONTAINER else Path("data")
OUTPUT = DATA / "outputs/murky-water-20261011"
ROOT = OUTPUT / "part2"
ARCHIVE = DATA / "external/brackish-v4.zip"
DATASET = DATA / "training/brackish-fish-v4"
MURK = DATA / "training/brackish-fish-v4-murk"
DARK_TEST = DATA / "training/brackish-fish-v4-dark-test"
MODELS = DATA / "models/murky-water-20261011"
SMARTBAY_MANIFEST = DATA / "outputs/viame-zero-label-benchmark/manifest.json"
PART1 = OUTPUT / "part1"
NATIVE = (1920, 1080)
LABEL_SPACE = (960, 540)
TRAINING_BUDGET_SECONDS = 4 * 3600
SEED = 20261011
DARK_SEED = 7
ACCEPT = .60  # Production MIN_FISH_CONFIDENCE: a track is shown when its best box reaches it.
# Four training and two validation dataloader workers: on Windows each spawned worker holds
# about 1 GB of host memory, and Ultralytics' default (twice as many for validation) thrashed.
TRAIN = {"weights": str(DATA / "models/ultralytics/yolo11s.pt"), "imgsz": 640, "batch": 16, "seed": 0,
         "workers": 4, "val_workers": 2}
# Ultralytics stays offline: no analytics, and its AMP self-check cannot fetch yolo26n.pt
# (not an approved download), so that check is skipped and AMP stays on.
OFFLINE = {"YOLO_OFFLINE": "true", "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
           "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9"}


def ultralytics(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(command, env=os.environ | OFFLINE, **kwargs)


# ------------------------------------------------------------ VIAME on test --

def test_recordings() -> list[str]:
    report = read(DATASET / "report.json")
    return sorted(s for s, split in report["sequences"].items() if split == "test")


def viame_test() -> None:
    """Production VIAME (10 FPS video path) on each test recording's original AVI."""
    from scripts.murky_water_tracker import PRODUCTION_PIPE, VIAME_JOURNALS, Journal, run_pass

    journal = Journal(VIAME_JOURNALS[1])
    metadata = [json.loads(line) for line in (DATASET / "metadata.jsonl").read_text().splitlines()]
    expected = {row["source_sequence"]: row["source_video_sha256"] for row in metadata}
    members = {row["source_sequence"]: row["source_video"] for row in metadata}
    videos = ROOT / "test-videos"
    videos.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for recording in test_recordings():
            video = videos / f"{recording}.avi"
            if not video.exists() or digest(video) != expected[recording]:
                with bundle.open(members[recording]) as source, video.open("wb") as target:
                    shutil.copyfileobj(source, target)
            if digest(video) != expected[recording]:
                raise ValueError(f"Extracted {recording} does not match the imported video hash")
            output = ROOT / "viame" / recording
            run_pass(journal, f"part2/viame/{recording}", "gpu",
                     [video, output, "--fps", 10, "--pipeline", PRODUCTION_PIPE], output, timeout=900)


def infer_analysed(observed: set[int], probe: list[int], frames: int, rate: float = 10 / 15) -> dict:
    """VIAME's analysed source frames on a video, from the frames that have detections.

    The image-list probe pattern is tried at small phase shifts first. The video path's
    timestamps occasionally break that pattern, so when no phase fits, the observed frames
    are accepted only if their count matches the downsampler's rate (every analysed frame
    then had at least one detection); anything else is an error.
    """
    options = []
    for shift in range(-3, 4):
        pattern = {p + shift for p in probe if 0 <= p + shift < frames}
        if observed <= pattern:
            options.append((len(pattern - observed), abs(shift), shift, pattern))
    if options:
        options.sort(key=lambda o: o[:2])
        missing, _, shift, pattern = options[0]
        ambiguous = len(options) > 1 and options[1][0] == missing
        return {"frames": sorted(pattern), "shift": shift, "unobserved": missing, "ambiguous": ambiguous,
                "method": "probe phase"}
    if observed and max(observed) < frames and abs(len(observed) - frames * rate) <= 1:
        return {"frames": sorted(observed), "shift": None, "unobserved": 0, "ambiguous": False,
                "method": "observed frames; count matches the sampling rate"}
    raise ValueError("Observed analysed frames fit no downsampler phase or sampling rate")


def viame_predictions(recording: str, frames: int, probe: list[int]) -> dict:
    """VIAME tracked boxes on one recording, in the 960x540 label space, by one-based frame."""
    from app.services.external_datasets import viame_tracks_to_mot

    directory = ROOT / "viame" / recording
    observed = set()
    for row in csv.reader((directory / "computed_detections.csv").open(encoding="utf-8")):
        if not row or row[0].startswith("#"):
            continue
        match = re.fullmatch(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", row[1])
        seconds = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
        index = round(seconds * 15)
        if abs(seconds * 15 - index) > .01:
            raise ValueError(f"{recording}: detection time {row[1]} is not on a 15 FPS frame")
        observed.add(index)
    analysed = infer_analysed(observed, probe, frames)
    rows = viame_tracks_to_mot(directory / "viame_tracks.csv", frame_offset=-1)
    best = defaultdict(float)
    for row in rows:
        best[row["track_id"]] = max(best[row["track_id"]], row["score"])
    sx, sy = LABEL_SPACE[0] / NATIVE[0], LABEL_SPACE[1] / NATIVE[1]
    boxes = defaultdict(list)
    for row in rows:
        if row["frame"] - 1 not in analysed["frames"]:
            raise ValueError(f"{recording}: track row on unanalysed frame {row['frame']}")
        boxes[row["frame"]].append({"box": (row["x"] * sx, row["y"] * sy, (row["x"] + row["w"]) * sx,
                                            (row["y"] + row["h"]) * sy),
                                    "score": row["score"], "accepted": best[row["track_id"]] >= ACCEPT})
    return {"analysed": [f + 1 for f in analysed["frames"]], "phase": analysed, "boxes": boxes}


# ------------------------------------------------------------------ scoring --

def truth_frames() -> dict:
    """Test truth keyed by (recording, one-based frame), in 960x540 pixels."""
    rows = [json.loads(line) for line in (DATASET / "metadata.jsonl").read_text().splitlines()]
    truth = {}
    for row in rows:
        if row["split"] != "test":
            continue
        boxes = []
        for line in (DATASET / row["output_label"]).read_text().splitlines():
            _, xc, yc, w, h = map(float, line.split())
            boxes.append((0, ((xc - w / 2) * 960, (yc - h / 2) * 540, (xc + w / 2) * 960, (yc + h / 2) * 540)))
        truth[(row["source_sequence"], row["source_frame"])] = {
            "boxes": boxes, "unverified_empty": row["unverified_empty"], "stem": Path(row["output_image"]).stem}
    return truth


def yolo_predictions(path: Path, truth: dict, minimum: float) -> dict:
    by_stem = {v["stem"]: k for k, v in truth.items()}
    predictions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        predictions[by_stem[row["image"]]] = [(int(c), tuple(b), float(s)) for c, b, s in row["predictions"]
                                              if s >= minimum]
    return predictions


def score_frames(truth: dict, predictions: dict, frames) -> dict:
    """TP/FP/FN at IoU 0.5 over the given frames. Predictions on placeholder-empty frames
    are counted apart: those frames' emptiness was never verified."""
    from app.services.detection_metrics import count_matches

    counts = {"tp": 0, "fp": 0, "fn": 0, "frames": 0, "predicted_boxes": 0, "truth_boxes": 0,
              "unverified_empty_frames": 0, "boxes_on_unverified_empty_frames": 0}
    for key in frames:
        entry, predicted = truth[key], predictions.get(key, [])
        if entry["unverified_empty"]:
            counts["unverified_empty_frames"] += 1
            counts["boxes_on_unverified_empty_frames"] += len(predicted)
            continue
        observed = count_matches(entry["boxes"], predicted)
        for name in ("tp", "fp", "fn"):
            counts[name] += observed[name]
        counts["frames"] += 1
        counts["predicted_boxes"] += len(predicted)
        counts["truth_boxes"] += len(entry["boxes"])
    counts["precision"] = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    counts["recall"] = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    return counts


def alignment(sheet_frames: int = 6) -> dict:
    """Proof that VIAME's boxes from the 1920x1080 AVIs line up with the 960x540 test labels.

    Matched boxes at IoU 0.5 are counted with VIAME's frames shifted against the truth by
    -2..2; the peak must be at 0. Overlays draw truth (green) and VIAME (magenta) on the
    dataset's own test images for the frames with the most VIAME boxes."""
    import cv2
    import numpy as np

    from app.services.detection_metrics import count_matches

    truth = truth_frames()
    frames_per_recording = defaultdict(int)
    for recording, frame in truth:
        frames_per_recording[recording] = max(frames_per_recording[recording], frame)
    probe = read(PART1 / "manifest.json")["analysed_positions"]["batch-10fps"]
    viame = {}
    for recording, count in frames_per_recording.items():
        for frame, boxes in viame_predictions(recording, count, probe)["boxes"].items():
            viame[(recording, frame)] = [(0, b["box"], b["score"]) for b in boxes]
    shifts = {}
    for shift in range(-2, 3):
        matched = 0
        for (recording, frame), predicted in viame.items():
            entry = truth.get((recording, frame + shift))
            if entry and not entry["unverified_empty"]:
                matched += count_matches(entry["boxes"], predicted)["tp"]
        shifts[str(shift)] = matched
    chosen = sorted((k for k in viame if truth.get(k) and truth[k]["boxes"]),
                    key=lambda k: (-len(viame[k]), k))[:sheet_frames]
    tiles = []
    for key in chosen:
        image = cv2.imread(str(DATASET / "images" / "test" / f"{truth[key]['stem']}.jpg"))
        sx, sy = image.shape[1] / LABEL_SPACE[0], image.shape[0] / LABEL_SPACE[1]
        for colour, boxes in (((0, 255, 0), truth[key]["boxes"]), ((255, 0, 255), viame[key])):
            for _, (x1, y1, x2, y2), *_ in boxes:
                cv2.rectangle(image, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), colour, 2)
        cv2.putText(image, f"{key[0]} frame {key[1]}: truth green, VIAME magenta", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        tiles.append(cv2.resize(image, LABEL_SPACE))
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    if tiles:
        cv2.imwrite(str(ROOT / "viame-test-alignment.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    peak = max(shifts, key=shifts.get)
    result = {"matched_boxes_by_frame_shift": shifts, "peak_shift": int(peak), "aligned": peak == "0",
              "overlay": (ROOT / "viame-test-alignment.jpg").as_posix(), "overlay_frames": [list(k) for k in chosen]}
    save(ROOT / "viame-test-alignment.json", result)
    return result


def score(runs: dict[str, Path]) -> dict:
    """VIAME and the trained runs on identical test frames and truth."""
    truth = truth_frames()
    frames_per_recording = defaultdict(int)
    for recording, frame in truth:
        frames_per_recording[recording] = max(frames_per_recording[recording], frame)
    probe = read(PART1 / "manifest.json")["analysed_positions"]["batch-10fps"]
    viame, analysed, phases = {}, [], {}
    for recording, count in frames_per_recording.items():
        found = viame_predictions(recording, count, probe)
        phases[recording] = found["phase"] | {"frames": len(found["analysed"])}
        analysed += [(recording, f) for f in found["analysed"]]
        for frame, boxes in found["boxes"].items():
            viame[(recording, frame)] = boxes
    as_rows = lambda keep: {k: [(0, b["box"], b["score"]) for b in v if keep(b)] for k, v in viame.items()}
    models = {"viame_tracked": as_rows(lambda b: True),
              "viame_accepted_tracks": as_rows(lambda b: b["accepted"])}
    for name, run in runs.items():
        models[name] = yolo_predictions(run / "test" / "predictions.jsonl", truth, .25)
    result = {"truth": "Brackish v4 published test annotations (10 recordings); reviewed_for_smartbay false",
              "iou": .5, "label_space": "960x540 (VIAME boxes scaled from the 1920x1080 AVI)",
              "viame_phases": phases,
              "same_frames": {name: score_frames(truth, preds, analysed) for name, preds in models.items()},
              "all_test_frames": {name: score_frames(truth, preds, truth) for name, preds in models.items()
                                  if not name.startswith("viame")},
              "note": "same_frames: VIAME's analysed test frames (10 of 15 FPS), identical for every model. "
                      "all_test_frames is not available for VIAME, which never analyses a third of frames."}
    save(ROOT / "test-comparison.json", result)
    return result


# ------------------------------------------------------------------ training --

class TrainingJournal:
    def __init__(self):
        self.path = ROOT / "training-journal.json"
        self.data = read(self.path) if self.path.exists() else {"runs": {}}

    def seconds(self) -> float:
        return sum(r.get("seconds", 0) for r in self.data["runs"].values())

    def record(self, name, **values):
        self.data["runs"].setdefault(name, {}).update(values)
        save(self.path, self.data)


def train(epochs: int, runs: list[str], murk_fraction: float = 1.0) -> None:
    journal = TrainingJournal()
    for name in runs:
        entry = journal.data["runs"].get(name, {})
        if entry.get("status") == "complete":
            continue
        if journal.seconds() >= TRAINING_BUDGET_SECONDS:
            raise RuntimeError("Training budget exhausted")
        command = [sys.executable, "scripts/fish_model.py", "train", str(DATASET), "--models", str(MODELS / name),
                   "--weights", TRAIN["weights"], "--epochs", str(epochs), "--imgsz", str(TRAIN["imgsz"]),
                   "--batch", str(TRAIN["batch"]), "--seed", str(TRAIN["seed"]),
                   "--workers", str(TRAIN["workers"]), "--val-workers", str(TRAIN["val_workers"]),
                   "--require-cuda"]
        if name == "B":
            command += ["--extra-train", str(MURK), "--extra-train-fraction", str(murk_fraction)]
        journal.record(name, status="running", command=command, started_unix=time.time(), epochs=epochs,
                       murk_fraction=murk_fraction if name == "B" else None)
        started = time.monotonic()
        with (ROOT / f"train-{name}.log").open("w", encoding="utf-8") as log:
            completed = ultralytics(command, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
        if completed.returncode:
            journal.record(name, status="failed", seconds=elapsed)
            raise RuntimeError(f"Training {name} failed; see train-{name}.log")
        best = next((MODELS / name).glob("fish-*/weights/best.pt"))
        journal.record(name, status="complete", seconds=elapsed, best=str(best), best_sha256=digest(best))


def evaluate() -> dict:
    journal = TrainingJournal()
    runs = {}
    for name in ("A", "B"):
        best = Path(journal.data["runs"][name]["best"])
        directory = ROOT / "eval" / name
        for label, dataset in (("test", DATASET), ("dark-test", DARK_TEST)):
            if not (directory / label / "evaluation.json").exists():
                if (directory / label).exists():
                    shutil.rmtree(directory / label)
                ultralytics([sys.executable, "scripts/fish_model.py", "evaluate", str(best), str(dataset),
                             str(directory / label), "--save-predictions", "--workers", "2"], check=True)
        runs[f"yolo_{name}"] = directory
    result = score(runs)
    summary = {name: {label: read(directory / label / "evaluation.json") for label in ("test", "dark-test")}
               for name, directory in runs.items()}
    save(ROOT / "evaluations.json", summary)
    return {"comparison": result, "evaluations": summary}


# ------------------------------------------------------------------ SmartBay --

def yolo_rows(path: Path, timestamps: list[float]) -> list[dict]:
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        frame = int(row["frame"])
        box = [float(row[k]) for k in ("tl_x", "tl_y", "br_x", "br_y")]
        rows.append({"id": row["track_id"], "frame": frame, "t": timestamps[frame], "box": box,
                     "confidence": float(row["confidence"])})
    return sorted(rows, key=lambda r: (r["t"], r["id"]))


def smartbay(model: Path, name: str) -> dict:
    """The better trained model against production VIAME on the fixed SmartBay clips."""
    from scripts.murky_water_tracker import PRODUCTION, tag

    clips = [c for c in read(SMARTBAY_MANIFEST)["clips"] if c["id"] in ("0", "1", "2")]
    origins = read(PART1 / "smartbay-clips.json")  # ffprobe'd in the worker image by Part 1
    for clip in clips:
        clip["pts_origin"] = origins[clip["id"]]["pts_origin"]
    production = tag(dict(PRODUCTION))
    result = {"model": str(model), "model_sha256": digest(model), "run": name, "clips": {}}
    sheets = []
    for clip in clips:
        path = DATA / Path(clip["path"]).relative_to("/data")
        if digest(path) != clip["sha256"]:
            raise RuntimeError(f"SmartBay clip changed: {path}")
        output = ROOT / "smartbay" / f"clip-{clip['id']}"
        if not (output / "report.json").exists():
            if output.exists():
                shutil.rmtree(output)
            ultralytics([sys.executable, "scripts/fish_model.py", "track", str(model), str(path), str(output),
                         "--stride", "3", "--conf", ".25"], check=True)
        viame_dir = PART1 / "smartbay" / "batch-10fps" / production / f"clip-{clip['id']}"
        viame, _ = observations(viame_dir / "viame_tracks.csv", clip["timestamps"], -1, clip["pts_origin"])
        yolo = yolo_rows(output / "tracks.csv", clip["timestamps"])
        pairing = align(viame, yolo)
        aligned_viame = len(viame) - len(pairing["left_unaligned"])
        aligned_yolo = len(yolo) - len(pairing["right_unaligned"])
        m = metrics(yolo, yolo, clip["width"], clip["height"], clip["fps"] / 3)
        v = metrics(viame, viame, clip["width"], clip["height"], 10.0)
        result["clips"][clip["id"]] = {
            "viame_boxes": len(viame), "yolo_boxes": len(yolo), "matched": pairing["matched"],
            "viame_only": len(pairing["left_unique"]), "yolo_only": len(pairing["right_unique"]),
            "unaligned_viame": len(pairing["left_unaligned"]), "unaligned_yolo": len(pairing["right_unaligned"]),
            "agreement": pairing["matched"] / max(1, aligned_viame + aligned_yolo - pairing["matched"]),
            "proxies": {label: {"tracks": x["tracks"], "fragmentation": x["fragmentation_rate"],
                                "duplicates": x["duplicate_pair_rate"],
                                "singleton_tracks": x["length_fractions"]["1"],
                                "acceleration": x["center_acceleration"]["median"],
                                "accepted_tracks": {c: x["accepted_tracks"][c] for c in ("0.55", "0.60", "0.65")}}
                        for label, x in (("viame", v), ("yolo", m))}}
        save(output / "disagreements.json", pairing)
        sheets.append((path, pairing))
    contact_sheet(sheets, ROOT / "smartbay" / "only-one-model.jpg")
    result["note"] = ("Unique boxes are suggestions for review, not misses or false positives. "
                      "Agreement uses only frames both models analysed (matched timestamps, IoU >= 0.5).")
    save(ROOT / "smartbay-comparison.json", result)
    return result


def brackishmot() -> dict:
    """Tracking on the BrackishMOT sequences the leakage guard allows: VIAME production
    (Part 1 replays) against A and B through fish_model.py track, on VIAME's analysed
    10 FPS frames. Each model's tracker sees exactly those frames, in order."""
    from app.services.external_datasets import evaluate_mot, leakage_guard, write_mot
    from scripts.murky_water_tracker import PRODUCTION, sequence_to_video, tag

    manifest = read(PART1 / "manifest.json")
    overlap = read(OUTPUT / "brackishmot-overlap.json")["sequences"]
    splits = read(DATASET / "report.json")["sequences"]
    guard = leakage_guard(overlap, {r for r, s in splits.items() if s in ("train", "val")},
                          sorted(manifest["sequences"]))
    positions = manifest["analysed_positions"]["batch-10fps"]
    journal = TrainingJournal()
    result = {"guard": {k: (len(v) if k != "scorable" else v) for k, v in guard.items()}, "runs": {}}
    sources = {"viame_production": PART1 / "mot" / "batch-10fps" / tag(dict(PRODUCTION))}
    for name in ("A", "B"):
        model = Path(journal.data["runs"][name]["best"])
        target = ROOT / "brackishmot" / f"yolo_{name}" / "mot"
        for sequence in guard["scorable"]:
            mot = target / f"{sequence}.txt"
            if mot.exists():
                continue
            images = sorted((DATA / "external/brackishmot" / sequence / "img1").glob("*.jpg"))
            frames = [i for i in positions if i < manifest["sequences"][sequence]["frames"]]
            video = ROOT / "brackishmot" / "videos" / f"{sequence.replace('/', '__')}.mkv"
            if not video.exists():
                video.parent.mkdir(parents=True, exist_ok=True)
                sequence_to_video([images[i] for i in frames], video, 10.0)
            output = ROOT / "brackishmot" / f"yolo_{name}" / "tracks" / sequence.replace("/", "__")
            if output.exists():
                shutil.rmtree(output)
            ultralytics([sys.executable, "scripts/fish_model.py", "track", str(model), str(video), str(output),
                         "--conf", ".25"], check=True, capture_output=True)
            rows = [{"frame": frames[int(r["frame"])] + 1, "track_id": int(r["track_id"]),
                     "x": float(r["tl_x"]), "y": float(r["tl_y"]),
                     "w": float(r["br_x"]) - float(r["tl_x"]), "h": float(r["br_y"]) - float(r["tl_y"]),
                     "score": float(r["confidence"])}
                    for r in csv.DictReader((output / "tracks.csv").open(encoding="utf-8"))]
            write_mot(sorted(rows, key=lambda r: (r["frame"], r["track_id"])), mot)
        sources[f"yolo_{name}"] = target
    for name, source in sources.items():
        directory = ROOT / "brackishmot" / "eval" / name
        if not (directory / "evaluation.json").exists():
            if directory.exists():
                shutil.rmtree(directory)
            analysed = {}
            for sequence in guard["scorable"]:
                (directory / sequence).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / f"{sequence}.txt", directory / f"{sequence}.txt")
                analysed[sequence] = [i + 1 for i in positions if i < manifest["sequences"][sequence]["frames"]]
            (directory / "analysed_frames.json").write_text(json.dumps(analysed), encoding="utf-8")
            evaluate_mot(DATA / "training/brackishmot-v1", directory, directory / "evaluation.json")
        report = read(directory / "evaluation.json")
        result["runs"][name] = {"identity": report["identity"], "clear": report["clear"]}
    result["note"] = ("Only sequences the leakage guard allows: cut from Brackish test recordings or "
                      "matching no archive recording. Public-data tracking scores, not SmartBay accuracy.")
    save(ROOT / "brackishmot-comparison.json", result)
    return result


def contact_sheet(sheets: list, path: Path, per_clip: int = 2, other: str = "YOLO") -> None:
    """Frames with the most single-model boxes: VIAME-only magenta, other-only cyan, both green."""
    import cv2
    import numpy as np

    tiles = []
    for video, pairing in sheets:
        counts = defaultdict(int)
        for row in pairing["left_unique"] + pairing["right_unique"]:
            counts[row["frame"]] += 1
        capture = cv2.VideoCapture(str(video))
        for frame in sorted(counts, key=lambda f: (-counts[f], f))[:per_clip]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                continue
            for colour, rows in (((255, 0, 255), [r for r in pairing["left_unique"] if r["frame"] == frame]),
                                 ((255, 255, 0), [r for r in pairing["right_unique"] if r["frame"] == frame]),
                                 ((0, 255, 0), [m["left"] for m in pairing["matches"] if m["left"]["frame"] == frame])):
                for row in rows:
                    x1, y1, x2, y2 = map(int, row["box"])
                    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(image, f"{video.name} frame {frame}: VIAME-only magenta, {other}-only cyan, both green",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            tiles.append(cv2.resize(image, (960, 540)))
        capture.release()
    if tiles:
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["viame-test", "alignment", "train", "evaluate", "score", "smartbay",
                                          "brackishmot"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--runs", default="A,B")
    parser.add_argument("--murk-fraction", type=float, default=1.0,
                        help="Stable-hash share of the murk copies used by run B")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "viame-test":
        viame_test()
    elif args.stage == "train":
        train(args.epochs, args.runs.split(","), args.murk_fraction)
    elif args.stage == "evaluate":
        print(json.dumps(evaluate(), indent=2))
    elif args.stage == "alignment":
        print(json.dumps(alignment(), indent=2))
    elif args.stage == "score":
        print(json.dumps(score({}), indent=2))
    elif args.stage == "brackishmot":
        print(json.dumps(brackishmot(), indent=2))
    else:
        print(json.dumps(smartbay(args.model, args.name), indent=2))


if __name__ == "__main__":
    main()
