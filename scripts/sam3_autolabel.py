"""SAM 3 ("fish" text prompt) as an automatic labeller, through Ultralytics' SAM 3 code.

The checkpoint is Meta's ``sam3.pt`` from Hugging Face ``facebook/sam3`` at a pinned
commit, placed by hand at ``data/models/sam3/sam3.pt`` after accepting Meta's SAM License
(its ``LICENSE`` goes beside it). This module never downloads it and never asks for a
token. The file is a pickle: ``build_predictor`` checks its size and SHA256 first, refuses
any other file, and only then imports the model code, which loads that verified path with
``torch.load(weights_only=True)`` (plain tensors only, no code execution).

Ultralytics' SAM 3 builder needs two packages the project does not ship: ``timm`` and
Ultralytics' fork of OpenAI CLIP (``clip``, which carries the BPE vocabulary). Ultralytics
would try to install them itself; ``missing_requirements`` reports them instead so they are
installed only with approval.

Output rows are unreviewed suggestions, never ground truth. Licences: SAM 3 weights are
under Meta's SAM License (free, commercial use allowed, no military, weapons, nuclear or
espionage use; its text ships with anything derived from SAM 3). The Ultralytics code that
runs them is AGPL-3.0.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from app.services.external_datasets import sha256

SAM3_PAGE = "https://huggingface.co/facebook/sam3"
SAM3_REPO = "facebook/sam3"
SAM3_COMMIT = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_FILE = "sam3.pt"
SAM3_BYTES = 3_450_062_241
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_CHECKPOINT = Path("data/models/sam3") / SAM3_FILE
SAM3_LICENCE = ("Meta SAM License: free, commercial use allowed; forbids military, weapons, nuclear "
                "and espionage uses; its text must ship with anything derived from SAM 3")
REQUIREMENTS = {"timm": "timm", "clip": "Ultralytics CLIP fork (git+https://github.com/ultralytics/CLIP.git)"}
CANDIDATE_HEADER = ["track_id", "video_id", "frame", "tl_x", "tl_y", "br_x", "br_y",
                    "confidence", "target_length", "class", "score"]


class UnverifiedCheckpoint(RuntimeError):
    pass


def missing_message(path: Path) -> str:
    return (f"SAM 3 checkpoint not found at {path}. Accept Meta's SAM License on {SAM3_PAGE}, "
            f"download {SAM3_FILE} at commit {SAM3_COMMIT} ({SAM3_BYTES:,} bytes, SHA256 "
            f"{SAM3_SHA256}) and place it at {SAM3_CHECKPOINT.as_posix()}, with Meta's LICENSE "
            f"beside it. No token is needed by this code.")


def verify_checkpoint(path: Path, *, expected_sha256: str = SAM3_SHA256,
                      expected_bytes: int | None = SAM3_BYTES) -> dict:
    """Size, then SHA256, of the exact file that will be loaded. Raises before any load."""
    path = Path(path).resolve()
    if not path.is_file():
        raise UnverifiedCheckpoint(missing_message(path))
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise UnverifiedCheckpoint(f"{path} is {size:,} bytes, expected {expected_bytes:,}; refusing to load")
    started = time.monotonic()
    digest = sha256(path)
    if digest != expected_sha256:
        raise UnverifiedCheckpoint(f"{path} SHA256 {digest} is not the pinned {expected_sha256}; "
                                   "refusing to load")
    licence = path.parent / "LICENSE"
    return {"path": str(path), "bytes": size, "sha256": digest, "repository": SAM3_REPO,
            "commit": SAM3_COMMIT, "verify_seconds": time.monotonic() - started,
            "licence_file": str(licence) if licence.is_file() else None, "licence": SAM3_LICENCE}


def missing_requirements() -> dict[str, str]:
    return {module: what for module, what in REQUIREMENTS.items()
            if importlib.util.find_spec(module) is None}


def _verified_loader(verified: Path):
    """torch.load for the SAM 3 builder: only the verified path, only plain tensors."""
    import torch

    def load(file, *args, **kwargs):
        name = Path(getattr(file, "name", file)).resolve()
        if name != verified:
            raise UnverifiedCheckpoint(f"Refusing to load {name}; only {verified} was verified")
        kwargs["weights_only"] = True
        return torch.load(file, *args, **kwargs)
    return load


def _import_predictor(mode: str = "video"):
    from ultralytics.models.sam import build_sam3
    from ultralytics.models.sam.predict import SAM3SemanticPredictor, SAM3VideoSemanticPredictor
    return build_sam3, SAM3VideoSemanticPredictor if mode == "video" else SAM3SemanticPredictor


def build_predictor(checkpoint: Path = SAM3_CHECKPOINT, *, mode: str = "video", imgsz: int = 1008,
                    fp16: bool = True, conf: float = .25, device: str = "0", **video_options):
    """A SAM 3 predictor on the verified checkpoint; nothing loads before the check.

    ``mode`` "video" is SAM3VideoSemanticPredictor (detector plus SAM 3's mask tracker);
    "image" is SAM3SemanticPredictor, the detector alone, set up for ``detect_video``.
    ``imgsz`` must be a multiple of SAM 3's 14-pixel patch; ``fp16`` is Ultralytics'
    ``quantize=16`` (8.4 replaced the old ``half`` argument)."""
    provenance = verify_checkpoint(checkpoint)
    missing = missing_requirements()
    if missing:
        raise RuntimeError(f"SAM 3 needs packages that are not installed (approval required): {missing}")
    if imgsz % 14:
        raise ValueError("SAM 3 input size must be a multiple of 14")
    build_sam3, predictor_class = _import_predictor(mode)
    build_sam3.torch_load = _verified_loader(Path(provenance["path"]))
    overrides = {"conf": conf, "task": "segment", "mode": "predict", "imgsz": imgsz,
                 "quantize": 16 if fp16 else 32, "model": provenance["path"], "device": device,
                 "save": False, "verbose": False}
    predictor = predictor_class(overrides=overrides, **video_options)
    if mode == "image":
        predictor.setup_model(model=None, verbose=False)
        predictor.imgsz = [imgsz, imgsz]
    return predictor, provenance


def detect_video(predictor, video: Path, prompt: str, *, max_frames: int | None = None,
                 on_frame: Callable[[dict], None] | None = None) -> Iterator[dict]:
    """SAM 3's detector on every frame, boxes only: the same scores, threshold and NMS as
    Ultralytics' SAM3SemanticPredictor.postprocess, without upscaling a full-size mask per
    box. Yields ``{"frame", "boxes", "ids": [], "scores", "tracker_scores", "seconds"}``
    with zero-based source frames and source-pixel boxes; identities come from a tracker."""
    import cv2
    import torch
    import torchvision
    from ultralytics.utils import ops

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    frame, started = 0, time.monotonic()
    try:
        while max_frames is None or frame < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            with torch.inference_mode():
                tensor = predictor.preprocess([image])
                preds = predictor._inference_features(predictor.get_im_features(tensor), text=[prompt])
                scores = (preds["pred_logits"].sigmoid()
                          * preds["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
                keep = scores > predictor.args.conf
                boxes = ops.xywh2xyxy(preds["pred_boxes"][keep].float())
                scores = scores[keep].float()
                order = torchvision.ops.nms(boxes, scores, predictor.args.iou)
                boxes, scores = boxes[order].cpu().numpy(), scores[order].cpu().numpy()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(0, width)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(0, height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            row = {"frame": frame, "boxes": boxes[valid].round(2).tolist(), "ids": [],
                   "scores": [round(float(s), 5) for s in scores[valid]],
                   "tracker_scores": [None] * int(valid.sum()), "seconds": time.monotonic() - started}
            if on_frame:
                on_frame(row)
            yield row
            frame += 1
    finally:
        capture.release()


def infer_video(predictor, video: Path, prompt: str | list[str], *, exemplar: list | None = None,
                max_frames: int | None = None,
                on_frame: Callable[[dict], None] | None = None) -> Iterator[dict]:
    """Per-frame ``{"frame", "boxes", "ids", "scores", "tracker_scores", "seconds"}`` in
    source pixels, zero-based source frames. ``scores`` is the object's detection score
    (SAM 3 keeps the score of the frame that found it); ``tracker_scores`` is the
    per-frame tracker score when the predictor exposes it."""
    original = predictor.postprocess
    captured = {}

    def postprocess(preds, img, orig_imgs):
        captured["tracker"] = dict(preds.get("obj_id_to_tracker_score") or {})
        return original(preds, img, orig_imgs)

    predictor.postprocess = postprocess
    kwargs = {"text": [prompt] if isinstance(prompt, str) else list(prompt)}
    if exemplar is not None:
        kwargs |= {"bboxes": [exemplar], "labels": [1]}
    frame = 0
    started = time.monotonic()
    try:
        for result in predictor(source=str(video), stream=True, **kwargs):
            data = result.boxes.data.detach().float().cpu().numpy() if result.boxes is not None else []
            tracker = captured.get("tracker", {})
            row = {"frame": frame, "boxes": [[float(v) for v in d[:4]] for d in data],
                   "ids": [int(d[4]) for d in data], "scores": [float(d[5]) for d in data],
                   "tracker_scores": [float(tracker[int(d[4])]) if int(d[4]) in tracker else None
                                      for d in data],
                   "seconds": time.monotonic() - started,
                   "source_frame": getattr(getattr(predictor, "dataset", None), "frame", frame + 1) - 1}
            if row["source_frame"] != frame:
                raise RuntimeError(f"SAM 3 result {frame} came from source frame {row['source_frame']}")
            if on_frame:
                on_frame(row)
            yield row
            frame += 1
            if max_frames is not None and frame >= max_frames:
                break
    finally:
        predictor.postprocess = original


def observation_rows(frames: list[dict], fps: float, prefix: str = "sam3") -> list[dict]:
    """Per-box observation rows (the static filter's and the proxies' format). The
    confidence is SAM 3's detection score for the object; the per-frame tracker score,
    when known, rides along as ``tracker_score``."""
    rows = []
    for item in frames:
        ids = item["ids"] or [None] * len(item["boxes"])  # Detector-only frames carry no identities
        trackers = item.get("tracker_scores") or [None] * len(ids)
        for box, track_id, score, tracker in zip(item["boxes"], ids, item["scores"], trackers):
            rows.append({"id": None if track_id is None else f"{prefix}-{track_id}", "frame": item["frame"],
                         "t": item["frame"] / fps, "box": box, "confidence": score, "tracker_score": tracker})
    return rows


def reliable_labels(rows: list[dict], *, min_track_frames: int, min_track_median_score: float,
                    min_box_score: float) -> tuple[list[dict], dict]:
    """Keep tracks seen on enough frames with a confident median score, and within them
    the boxes whose own (tracker, else detection) score clears the floor."""
    from collections import defaultdict
    from statistics import median

    tracks = defaultdict(list)
    for row in rows:
        tracks[row["id"]].append(row)
    kept, reasons = [], {"short": 0, "low_score": 0, "kept": 0}
    dropped_boxes = 0
    for members in tracks.values():
        if len(members) < min_track_frames:
            reasons["short"] += 1
            continue
        if median(r["confidence"] for r in members) < min_track_median_score:
            reasons["low_score"] += 1
            continue
        reasons["kept"] += 1
        for row in members:
            score = row["tracker_score"] if row.get("tracker_score") is not None else row["confidence"]
            if score >= min_box_score:
                kept.append(row)
            else:
                dropped_boxes += 1
    return kept, {"tracks": reasons, "boxes_in": len(rows), "boxes_kept": len(kept),
                  "low_score_boxes_dropped": dropped_boxes}


def write_candidates(rows: list[dict], video_name: str, path: Path, label: str = "fish") -> int:
    """Rows as the project's VIAME-style candidate CSV (zero-based source frames), which
    ``scripts/external_fish.py cvat-suggestions`` loads into a separate CVAT task."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r["frame"], str(r["id"])))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CANDIDATE_HEADER)
        for row in ordered:
            x1, y1, x2, y2 = row["box"]
            writer.writerow([row["id"], video_name, row["frame"], f"{x1:.2f}", f"{y1:.2f}",
                             f"{x2:.2f}", f"{y2:.2f}", f"{row['confidence']:.6f}", -1, label,
                             f"{row['confidence']:.6f}"])
    return len(ordered)


def save_frames(frames: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for item in frames:
            stream.write(json.dumps(item) + "\n")
    temporary.replace(path)


def load_frames(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
