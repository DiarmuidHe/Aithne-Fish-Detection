"""Optional external dataset and Fishial detector commands; never changes VIAME production."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import shutil
import urllib.request
import uuid
import zipfile
from pathlib import Path

from app.services.external_datasets import (
    BRACKISH_ARCHIVE_SHA256_V4,
    brackish_mot_overlap,
    evaluate_mot,
    extract_brackish_mot,
    import_brackish,
    import_brackish_archive,
    import_brackish_mot,
    make_brackish_manifest,
    sha256,
)

FISHIAL_URL = "https://storage.googleapis.com/fishial-ml-resources/detector_v10_m5.zip"
FISHIAL_ARCHIVE_SHA256 = "54839fa0c688c42c4114b251aff036a9b8a1288cf61db7d2956786a0238fa47a"
FISHIAL_WEIGHTS_SHA256 = "78b7c1b2eabd563e7a5d98135e791e4e6fa75c0dd88e119133c3a3f5260bdbbd"
BRACKISH_V4_URL = "https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/brackish-dataset?datasetVersionNumber=4"
BRACKISH_MOT_V4_URL = "https://www.kaggle.com/api/v1/datasets/download/maltepedersen/brackishmot?datasetVersionNumber=4"
BRACKISH_MOT_ARCHIVE_SHA256_V4 = "e717dc1438aab65eaccb59b135b58f50ca7af09043ef7effbf9fd2825410c3ff"
# Requested as FathomNet/megalodon; Hugging Face redirects that name to the renamed
# FathomNet/megalodon-2023-yolov8 repository. The commit pins the exact file.
MEGALODON_REPO = "FathomNet/megalodon"
MEGALODON_COMMIT = "f312604d2f59dcccde74499cf0e0ae0647331f01"
MEGALODON_FILE = "mbari-megalodon-yolov8x.pt"
MEGALODON_URL = f"https://huggingface.co/{MEGALODON_REPO}/resolve/{MEGALODON_COMMIT}/{MEGALODON_FILE}"
MEGALODON_SHA256 = "24fe6d9808fdd44b64d2fbfc23bbe6455520eb7b392d3664c92cbcdf88a09196"


def fetch_brackish_mot(output: Path) -> dict:
    """Pinned Kaggle v4 BrackishMOT ZIP (all 98 real sequences plus synthetic sets)."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(BRACKISH_MOT_V4_URL, output)
        if sha256(output) != BRACKISH_MOT_ARCHIVE_SHA256_V4:
            raise ValueError("BrackishMOT v4 archive SHA256 mismatch")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"source": BRACKISH_MOT_V4_URL, "version": 4, "sha256": BRACKISH_MOT_ARCHIVE_SHA256_V4,
            "archive": str(output), "bytes": output.stat().st_size,
            "license": "CC BY-SA 4.0 in Kaggle metadata; the package itself states no licence"}


def fetch_megalodon(output: Path) -> dict:
    """FathomNet Megalodon YOLOv8x weights at a pinned commit, verified before anything loads."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    weights = output / MEGALODON_FILE
    try:
        urllib.request.urlretrieve(MEGALODON_URL, weights)
        if sha256(weights) != MEGALODON_SHA256:
            raise ValueError("Megalodon weight SHA256 mismatch")
        report = {"source": MEGALODON_URL, "repository": MEGALODON_REPO,
                  "resolved_repository": "FathomNet/megalodon-2023-yolov8",
                  "commit": MEGALODON_COMMIT, "file": MEGALODON_FILE, "sha256": MEGALODON_SHA256,
                  "bytes": weights.stat().st_size, "model": "Ultralytics YOLOv8x, MBARI FathomNet Megalodon",
                  "license": "CC BY 4.0 (model card metadata 'license: cc-by-4.0' at the pinned commit)",
                  "code_license": "Ultralytics AGPL-3.0 (Enterprise licence otherwise)",
                  "class_mapping": {"0": "object"},
                  "note": "One generic 'object' class from all FathomNet localizations: it boxes "
                          "fish, crabs, jellyfish and other animals alike. Never label its output fish.",
                  "security": "The .pt file is a pickle; load it only after this SHA256 check"}
        (output / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except BaseException:
        weights.unlink(missing_ok=True)
        (output / "provenance.json").unlink(missing_ok=True)
        output.rmdir()
        raise
    return report


def fetch_brackish(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(BRACKISH_V4_URL, output)
        if sha256(output) != BRACKISH_ARCHIVE_SHA256_V4:
            raise ValueError("Brackish v4 archive SHA256 mismatch")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {"source": BRACKISH_V4_URL, "version": 4, "license": "CC BY-SA 4.0",
            "sha256": BRACKISH_ARCHIVE_SHA256_V4, "archive": str(output)}


def fetch_fishial(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    archive = output / "detector_v10_m5.zip"
    try:
        urllib.request.urlretrieve(FISHIAL_URL, archive)
        if sha256(archive) != FISHIAL_ARCHIVE_SHA256:
            raise ValueError("Fishial archive SHA256 mismatch")
        with zipfile.ZipFile(archive) as bundle, bundle.open("model.ts") as source, (
            output / "model.ts"
        ).open("wb") as target:
            shutil.copyfileobj(source, target)
        if sha256(output / "model.ts") != FISHIAL_WEIGHTS_SHA256:
            raise ValueError("Fishial weight SHA256 mismatch")
    except Exception:
        (output / "model.ts").unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        output.rmdir()
        raise
    report = {"source": FISHIAL_URL, "archive_sha256": FISHIAL_ARCHIVE_SHA256,
              "weights_sha256": FISHIAL_WEIGHTS_SHA256, "model": "Fishial YOLOv12 medium TorchScript",
              "license": "Fishial repository MIT; checkpoint has no separate explicit license statement",
              "class_mapping": {"0": "fish"}}
    (output / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _letterbox(frame, size: int):
    import cv2
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (resized_w, resized_h))
    left, top = (size-resized_w)//2, (size-resized_h)//2
    canvas = cv2.copyMakeBorder(resized, top, size-resized_h-top, left,
                                size-resized_w-left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas, scale, left, top


def _fishial_boxes(prediction, confidence: float, nms_iou: float = .3):
    """Fishial v12 export returns [5,8400] xywh+score, despite its archive name."""
    import numpy as np
    if prediction.ndim != 2 or prediction.shape[0] != 5:
        raise ValueError(f"Unexpected Fishial output shape: {prediction.shape}")
    centers = prediction[:4].T
    scores = prediction[4]
    selected = scores >= confidence
    centers, scores = centers[selected], scores[selected]
    if not len(scores):
        return []
    boxes = np.column_stack((centers[:, 0]-centers[:, 2]/2,
                             centers[:, 1]-centers[:, 3]/2,
                             centers[:, 0]+centers[:, 2]/2,
                             centers[:, 1]+centers[:, 3]/2))
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = int(order[0])
        keep.append([*boxes[i], float(scores[i])])
        if len(order) == 1:
            break
        other = order[1:]
        tl = np.maximum(boxes[i, :2], boxes[other, :2])
        br = np.minimum(boxes[i, 2:], boxes[other, 2:])
        intersection = np.prod(np.maximum(0, br-tl), axis=1)
        area_i = np.prod(boxes[i, 2:]-boxes[i, :2])
        area_other = np.prod(boxes[other, 2:]-boxes[other, :2], axis=1)
        iou = intersection / np.maximum(area_i+area_other-intersection, 1e-9)
        order = other[iou <= nms_iou]
    return keep


def _candidate_csv(video: Path, output: Path, detect, *, max_frames: int, stride: int,
                   prefix: str, label: str) -> tuple[int, int]:
    """Shared frame loop: ``detect(frame)`` returns [x1, y1, x2, y2, score] rows in source
    pixels; rows are clipped to the frame and written as one-frame VIAME-style suggestions."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    output.mkdir(parents=True)
    frames, boxes, frame_number = 0, 0, 0
    try:
        with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["track_id", "video_id", "frame", "tl_x", "tl_y", "br_x", "br_y",
                             "confidence", "target_length", "class", "score"])
            while frames < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_number % stride == 0:
                    height, width = frame.shape[:2]
                    for index, (x1, y1, x2, y2, score) in enumerate(detect(frame)):
                        x1, x2 = max(0, x1), min(width, x2)
                        y1, y2 = max(0, y1), min(height, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        # The detector supplies boxes, not identities. Each suggestion is a one-frame track.
                        writer.writerow([f"{prefix}-{frame_number}-{index}", video.name, frame_number,
                                         x1, y1, x2, y2, score, -1, label, score])
                        boxes += 1
                    frames += 1
                frame_number += 1
    finally:
        capture.release()
    return frames, boxes


def infer_fishial(weights: Path, video: Path, output: Path, *, max_frames: int = 60,
                  stride: int = 1, confidence: float = .25, device: str = "auto") -> dict:
    import cv2
    import numpy as np
    import torch

    if output.exists():
        raise FileExistsError(output)
    if sha256(weights) != FISHIAL_WEIGHTS_SHA256:
        raise ValueError("Fishial weight SHA256 mismatch")
    if max_frames < 1 or stride < 1 or not 0 <= confidence <= 1:
        raise ValueError("Invalid frame limit, stride or confidence")
    chosen = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = torch.jit.load(str(weights), map_location=chosen).eval()

    def detect(frame):
        canvas, scale, left, top = _letterbox(frame, 640)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(chosen).float()[None] / 255
        with torch.inference_mode():
            prediction = model(tensor)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        prediction = prediction.detach().float().cpu().numpy()
        if prediction.ndim == 3:
            prediction = prediction[0]
        return [((float(row[0])-left)/scale, (float(row[1])-top)/scale,
                 (float(row[2])-left)/scale, (float(row[3])-top)/scale, float(row[4]))
                for row in _fishial_boxes(prediction, confidence)]

    frames, boxes = _candidate_csv(video, output, detect, max_frames=max_frames, stride=stride,
                                   prefix="fishial", label="fish")
    report = {"source": FISHIAL_URL, "weights_sha256": sha256(weights), "video": str(video),
              "video_sha256": sha256(video), "device": chosen, "input_size": 640,
              "model": "Fishial YOLOv12 medium TorchScript; archive info version 2",
              "torch_version": torch.__version__, "opencv_version": cv2.__version__,
              "numpy_version": np.__version__, "nms_iou": .3,
              "confidence": confidence, "stride": stride, "processed_frames": frames,
              "detections": boxes, "review_status": "unreviewed suggestions"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


_DILL_TYPES = {"set": set, "frozenset": frozenset, "list": list, "dict": dict, "tuple": tuple,
               "NoneType": type(None)}


@contextlib.contextmanager
def _dill_types_only():
    """The pinned Megalodon pickle references one dill function, ``_load_type('set')``,
    which only returns a builtin type. Without dill installed (it is not a project
    dependency), provide just that lookup for allow-listed builtin names while loading."""
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("dill") is not None:
        yield
        return
    package, inner = types.ModuleType("dill"), types.ModuleType("dill._dill")

    def load_type(name):
        if name not in _DILL_TYPES:
            raise ValueError(f"Refusing dill type {name!r} outside the builtin allow-list")
        return _DILL_TYPES[name]

    inner._load_type = load_type
    package._dill = inner
    sys.modules.update({"dill": package, "dill._dill": inner})
    try:
        yield
    finally:
        sys.modules.pop("dill", None)
        sys.modules.pop("dill._dill", None)


def _megalodon_rows(boxes) -> list[tuple[float, float, float, float, float]]:
    """Ultralytics result boxes as [x1, y1, x2, y2, score] in source pixels."""
    return [(*map(float, xyxy), float(score))
            for xyxy, score in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())]


def infer_megalodon(weights: Path, video: Path, output: Path, *, max_frames: int = 60,
                    stride: int = 1, confidence: float = .25, device: str = "auto",
                    imgsz: int | None = None) -> dict:
    """Megalodon 'object' suggestions in the same candidates.csv format as infer-fishial."""
    import cv2
    import numpy as np
    import torch

    if output.exists():
        raise FileExistsError(output)
    if max_frames < 1 or stride < 1 or not 0 <= confidence <= 1:
        raise ValueError("Invalid frame limit, stride or confidence")
    weights = weights.resolve()
    # Loading a .pt unpickles it, which can run code: verify this exact path first.
    if sha256(weights) != MEGALODON_SHA256:
        raise ValueError("Megalodon weight SHA256 mismatch; refusing to load")
    from ultralytics import YOLO
    import ultralytics

    chosen = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    with _dill_types_only():
        model = YOLO(str(weights))
    names = dict(model.names)
    if names != {0: "object"}:
        raise ValueError(f"Unexpected Megalodon classes: {names}")
    size = imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))

    def detect(frame):
        result = model.predict(frame, conf=confidence, imgsz=size, device=chosen, verbose=False)[0]
        return _megalodon_rows(result.boxes)

    frames, boxes = _candidate_csv(video, output, detect, max_frames=max_frames, stride=stride,
                                   prefix="megalodon", label="object")
    report = {"source": MEGALODON_URL, "commit": MEGALODON_COMMIT, "weights_sha256": MEGALODON_SHA256,
              "video": str(video), "video_sha256": sha256(video), "device": chosen, "input_size": size,
              "model": "FathomNet Megalodon (Ultralytics YOLOv8x), single class 'object'",
              "license": "CC BY 4.0 weights; Ultralytics AGPL-3.0 code",
              "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
              "opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "confidence": confidence, "stride": stride, "processed_frames": frames,
              "detections": boxes, "class": "object (generic marine object; not fish)",
              "review_status": "unreviewed suggestions"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch-brackish")
    p.add_argument("output", type=Path)
    p = sub.add_parser("fetch-brackish-mot")
    p.add_argument("output", type=Path)
    p = sub.add_parser("extract-brackish-mot")
    p.add_argument("archive", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("import-brackish-archive")
    p.add_argument("archive", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--max-sequences", type=int)
    p.add_argument("--max-frames-per-sequence", type=int)
    p = sub.add_parser("manifest-brackish")
    p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--sequence-regex", required=True,
                   help="Regex on image stem with named sequence and frame groups")
    p = sub.add_parser("import-brackish")
    p.add_argument("root", type=Path); p.add_argument("manifest", type=Path)
    p.add_argument("class_map", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("import-brackish-mot")
    p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("brackish-mot-overlap")
    p.add_argument("archive", type=Path); p.add_argument("mot_root", type=Path)
    p.add_argument("imported", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("evaluate-mot")
    p.add_argument("imported", type=Path); p.add_argument("candidates", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--iou", type=float, default=.5)
    p = sub.add_parser("fetch-fishial")
    p.add_argument("output", type=Path)
    p = sub.add_parser("infer-fishial")
    p.add_argument("weights", type=Path); p.add_argument("video", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--max-frames", type=int, default=60); p.add_argument("--stride", type=int, default=1)
    p.add_argument("--confidence", type=float, default=.25)
    p.add_argument("--device", choices=["auto", "cpu", "cuda:0"], default="auto")
    p = sub.add_parser("fetch-megalodon")
    p.add_argument("output", type=Path)
    p = sub.add_parser("infer-megalodon")
    p.add_argument("weights", type=Path); p.add_argument("video", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--max-frames", type=int, default=60); p.add_argument("--stride", type=int, default=1)
    p.add_argument("--confidence", type=float, default=.25)
    p.add_argument("--imgsz", type=int, help="Defaults to the checkpoint's training size")
    p.add_argument("--device", choices=["auto", "cpu", "cuda:0"], default="auto")
    p = sub.add_parser("cvat-suggestions")
    p.add_argument("video_id", type=uuid.UUID); p.add_argument("name"); p.add_argument("candidates", type=Path)
    args = parser.parse_args()
    if args.command == "fetch-brackish":
        report = fetch_brackish(args.output)
    elif args.command == "fetch-brackish-mot":
        report = fetch_brackish_mot(args.output)
    elif args.command == "extract-brackish-mot":
        report = extract_brackish_mot(args.archive, args.output, BRACKISH_MOT_ARCHIVE_SHA256_V4)
    elif args.command == "import-brackish-archive":
        report = import_brackish_archive(args.archive, args.output,
                                         max_sequences=args.max_sequences,
                                         max_frames_per_sequence=args.max_frames_per_sequence)
    elif args.command == "manifest-brackish":
        report = make_brackish_manifest(args.root, args.output, args.sequence_regex)
    elif args.command == "import-brackish":
        report = import_brackish(args.root, args.manifest, args.class_map, args.output)
    elif args.command == "import-brackish-mot":
        report = import_brackish_mot(args.root, args.output)
    elif args.command == "brackish-mot-overlap":
        if args.output.exists():
            raise FileExistsError(args.output)
        sequences = sorted(json.loads((args.imported / "report.json").read_text())["sequences"])
        report = brackish_mot_overlap(args.archive, args.mot_root, sequences)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report = {k: v for k, v in report.items() if k != "sequences"} | {
            "status_counts": {s: sum(v["status"] == s for v in report["sequences"].values())
                              for s in ("overlap", "none", "undetermined")}}
    elif args.command == "evaluate-mot":
        report = evaluate_mot(args.imported, args.candidates, args.output, args.iou)
    elif args.command == "fetch-fishial":
        report = fetch_fishial(args.output)
    elif args.command == "infer-fishial":
        report = infer_fishial(args.weights, args.video, args.output, max_frames=args.max_frames,
                               stride=args.stride, confidence=args.confidence, device=args.device)
    elif args.command == "fetch-megalodon":
        report = fetch_megalodon(args.output)
    elif args.command == "infer-megalodon":
        report = infer_megalodon(args.weights, args.video, args.output, max_frames=args.max_frames,
                                 stride=args.stride, confidence=args.confidence, device=args.device,
                                 imgsz=args.imgsz)
    else:
        from scripts.import_model_suggestions import import_suggestions
        report = import_suggestions(args.video_id, args.name, args.candidates)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
