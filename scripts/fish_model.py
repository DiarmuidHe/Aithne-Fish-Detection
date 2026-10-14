"""Local VIAME-to-YOLO dataset, training and tracked inference commands."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path


def _ultralytics():
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the optional local training dependencies: pip install ultralytics") from exc
    return YOLO, torch


def combined_data_yaml(dataset: Path, extra_train: list[Path], output: Path,
                       fraction: float = 1.0) -> Path:
    """Dataset YAML whose train split adds training-only folders (for example murk copies).

    Each extra folder holds images/train and labels/train plus metadata.jsonl, and every
    metadata row must be a train row: extra data never reaches validation or test. With
    ``fraction`` below 1, a stable-hash subset of each extra folder is listed instead.
    """
    import hashlib

    if not 0 < fraction <= 1:
        raise SystemExit("Extra training fraction must be in (0, 1]")
    names = json.loads((dataset / "classes.json").read_text(encoding="utf-8"))
    train = [str(dataset / "images" / "train")]
    for extra in extra_train:
        rows = [json.loads(line) for line in (extra / "metadata.jsonl").read_text().splitlines()]
        if not rows or any(row["split"] != "train" for row in rows):
            raise SystemExit(f"Extra training data must contain only train rows: {extra}")
        if json.loads((extra / "classes.json").read_text(encoding="utf-8")) != names:
            raise SystemExit(f"Extra training data has different classes: {extra}")
        if fraction == 1:
            train.append(str(extra / "images" / "train"))
            continue
        images = sorted(extra.glob("images/train/*.jpg"),
                        key=lambda image: hashlib.sha256(image.name.encode()).hexdigest())
        listing = output.with_name(f"{output.stem}.{extra.name}.txt")
        listing.write_text("".join(f"{image.as_posix()}\n" for image in images[:round(len(images) * fraction)]),
                           encoding="utf-8")
        train.append(str(listing))
    lines = [f"path: '{dataset.as_posix()}'", "train:"] + [f"  - '{Path(t).as_posix()}'" for t in train]
    lines += ["val: images/val", "test: images/test", "names:"]
    lines += [f"  {key}: {value}" for key, value in names.items()]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _validation_workers_trainer(workers: int):
    """Detection trainer whose validation loader uses ``workers`` processes.

    Ultralytics gives validation twice the training workers, and on Windows each spawned
    worker holds about 1 GB of host memory for the whole run."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class Trainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            training, self.args.workers = self.args.workers, workers // 2  # Doubled for validation
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = training

    return Trainer


def train(args):
    YOLO, torch = _ultralytics()
    if args.val_workers is not None and (args.val_workers < 0 or args.val_workers % 2):
        raise SystemExit("--val-workers must be an even number (Ultralytics doubles the setting)")
    dataset = Path(args.dataset).resolve()
    if not list((dataset / "images" / "train").glob("*.jpg")):
        raise SystemExit("No train images; review the manifest and labels first")
    if not list((dataset / "images" / "val").glob("*.jpg")):
        raise SystemExit("A reviewed validation split is required")
    rows = [json.loads(line) for line in (dataset / "metadata.jsonl").read_text().splitlines()]
    if any(not row["reviewed"] for row in rows if row["split"] != "train"):
        raise SystemExit("Validation and test images must be reviewed")
    root = Path(args.models).resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device == "cpu":
        raise SystemExit("CUDA is unavailable in this Python environment")
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        return
    run_name = datetime.now(UTC).strftime("fish-%Y%m%dT%H%M%SZ")
    run = root / run_name
    if run.exists():
        raise FileExistsError(run)
    model = YOLO(args.weights)
    data = dataset / "dataset.yaml"
    if args.extra_train:
        data = combined_data_yaml(dataset, [Path(x).resolve() for x in args.extra_train],
                                  root / f"{run_name}.data.yaml", args.extra_train_fraction)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                lr0=args.lr, device=device, project=str(root), name=run_name, exist_ok=False,
                seed=args.seed, deterministic=True, workers=args.workers,
                trainer=None if args.val_workers is None else _validation_workers_trainer(args.val_workers))
    shutil.copy2(dataset / "classes.json", run / "classes.json")
    if data != dataset / "dataset.yaml":
        for listing in data.parent.glob(f"{data.stem}.*.txt"):
            shutil.move(listing, run / listing.name)
        shutil.move(data, run / "data.yaml")
    (run / "fish_model_config.json").write_text(json.dumps(vars(args) | {"device": device}, indent=2), encoding="utf-8")
    print(json.dumps({"run": str(run), "best": str(run / "weights" / "best.pt"), "device": device}))


def track(args):
    YOLO, torch = _ultralytics()
    import cv2

    model_path, video_path, output = (Path(x).resolve() for x in (args.model, args.video, args.output))
    if output.exists():
        raise FileExistsError(output)
    if not model_path.is_file() or not video_path.is_file():
        raise FileNotFoundError("Model weights and video must exist")
    output.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (width, height))
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    # With --stride N only every Nth source frame is tracked; CSV frames stay source indices.
    count, processed, detections = 0, 0, 0
    start = time.monotonic()
    try:
        with (output / "tracks.csv").open("w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["track_id", "video_id", "frame", "tl_x", "tl_y", "br_x", "br_y",
                                 "confidence", "target_length", "class", "score"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if count % args.stride:
                    count += 1
                    continue
                result = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf,
                                     iou=args.iou, device=device, verbose=False)[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                    for i, (xyxy, confidence, cls) in enumerate(zip(
                            boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.int().cpu().tolist())):
                        track_id = ids[i]
                        if track_id is None:
                            continue
                        label = model.names[cls]
                        csv_writer.writerow([track_id, video_path.name, count, *xyxy, confidence, -1, label, confidence])
                        detections += 1
                writer.write(result.plot())
                count += 1
                processed += 1
    finally:
        cap.release()
        writer.release()
    report = {"model": str(model_path), "video": str(video_path), "device": device,
              "frames": count, "processed_frames": processed, "stride": args.stride,
              "detections": detections,
              "fps": processed / max(time.monotonic() - start, 1e-9), "confidence_threshold": args.conf,
              "tracker": args.tracker, "review_status": "unreviewed predictions"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def evaluate(args):
    YOLO, torch = _ultralytics()
    import cv2

    from app.services.detection_metrics import count_matches

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((dataset / "report.json").read_text(encoding="utf-8"))
    if not report["frames"]["test"]:
        raise SystemExit("A nonempty reviewed test split is required")
    metadata = [json.loads(line) for line in (dataset / "metadata.jsonl").read_text().splitlines()]
    if any(not row["reviewed"] for row in metadata if row["split"] == "test"):
        raise SystemExit("Test images must be reviewed")
    model = YOLO(str(Path(args.model).resolve()))
    device = 0 if torch.cuda.is_available() else "cpu"
    output.mkdir(parents=True)
    # A frame whose empty label is only a placeholder (unverified_empty) cannot show that a
    # prediction there is false. Those frames leave mAP and the headline counts; their
    # predictions are reported separately.
    unverified = {Path(row["output_image"]).stem for row in metadata
                  if row["split"] == "test" and row.get("unverified_empty")}
    images = sorted((dataset / "images" / "test").glob("*.jpg"))
    data = dataset / "dataset.yaml"
    if unverified:
        (output / "test-images.txt").write_text(
            "".join(f"{p.as_posix()}\n" for p in images if p.stem not in unverified), encoding="utf-8")
        data = output / "test.data.yaml"
        names = json.loads((dataset / "classes.json").read_text(encoding="utf-8"))
        data.write_text("\n".join([f"path: '{dataset.as_posix()}'", "train: images/train",
                                   "val: images/val", f"test: '{(output / 'test-images.txt').as_posix()}'",
                                   "names:", *[f"  {k}: {v}" for k, v in names.items()]]) + "\n",
                        encoding="utf-8")
    metrics = model.val(data=str(data), split="test", device=device,
                        imgsz=args.imgsz, project=str(output), name="ultralytics", exist_ok=True,
                        plots=True, workers=args.workers)
    counts = {"tp": 0, "fp": 0, "fn": 0}
    per_class = {}
    unverified_fp = 0
    started = time.monotonic()
    frames = 0
    saved = (output / "predictions.jsonl").open("w", encoding="utf-8") if args.save_predictions else None
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truth = []
        for line in (dataset / "labels" / "test" / f"{image_path.stem}.txt").read_text().splitlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            truth.append((int(cls), ((xc-bw/2)*width, (yc-bh/2)*height,
                                     (xc+bw/2)*width, (yc+bh/2)*height)))
        result = model.predict(image, conf=args.conf, device=device, imgsz=args.imgsz,
                               verbose=False)[0]
        predictions = [(int(cls), tuple(box), float(score)) for box, score, cls in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist())]
        if saved:
            saved.write(json.dumps({"image": image_path.stem, "width": width, "height": height,
                                    "predictions": predictions}) + "\n")
        if image_path.stem in unverified:
            if truth:
                raise ValueError(f"{image_path.name} is marked unverified_empty but has labels")
            unverified_fp += len(predictions)
            frames += 1
            continue
        for cls in set([row[0] for row in truth] + [row[0] for row in predictions]):
            observed = count_matches([row for row in truth if row[0] == cls],
                                     [row for row in predictions if row[0] == cls])
            bucket = per_class.setdefault(str(cls), {"tp": 0, "fp": 0, "fn": 0})
            for key, value in observed.items():
                bucket[key] += value
                counts[key] += value
        if frames < 20:
            cv2.imwrite(str(output / f"{image_path.stem}-prediction.jpg"), result.plot())
        frames += 1
    elapsed = time.monotonic() - started
    if saved:
        saved.close()
    for bucket in [counts, *per_class.values()]:
        bucket["precision"] = bucket["tp"] / max(1, bucket["tp"] + bucket["fp"])
        bucket["recall"] = bucket["tp"] / max(1, bucket["tp"] + bucket["fn"])
    result = {"counts_at_iou50": counts, "per_class": per_class, "map50": float(metrics.box.map50),
              "map50_95": float(metrics.box.map), "fps": frames / max(elapsed, 1e-9),
              "device": device, "frames": frames, "confidence": args.conf,
              "unverified_empty_frames": len(unverified),
              "predictions_on_unverified_empty_frames": unverified_fp,
              "note": ("Counts cover reviewed test frames, including verified empty frames when "
                       "the dataset export contains them. Frames whose empty label is an unverified "
                       "placeholder are excluded from mAP and counts; their predictions are listed "
                       "separately and are not called false positives.")}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--jobs", default="data/jobs")
    inventory.add_argument("--uploads", default="data/uploads")
    inventory.add_argument("--output", default="data/training/inventory.json")
    data = commands.add_parser("dataset")
    data.add_argument("manifest")
    data.add_argument("output")
    data.add_argument("--include-pseudo", action="store_true")
    data.add_argument("--low-confidence", type=float, default=.6)
    preview = commands.add_parser("preview")
    preview.add_argument("manifest")
    preview.add_argument("output")
    preview.add_argument("--limit", type=int, default=20)
    preview.add_argument("--low-confidence", type=float, default=.6)
    training = commands.add_parser("train")
    training.add_argument("dataset")
    training.add_argument("--models", default="data/models")
    training.add_argument("--weights", default="yolo11n.pt")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--imgsz", type=int, default=640)
    training.add_argument("--batch", type=int, default=8)
    training.add_argument("--lr", type=float, default=.01)
    training.add_argument("--require-cuda", action="store_true")
    training.add_argument("--resume")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--workers", type=int, default=8)
    training.add_argument("--val-workers", type=int,
                          help="Validation loader workers (even; default twice --workers)")
    training.add_argument("--extra-train-fraction", type=float, default=1.0,
                          help="Use a stable-hash share of each --extra-train folder")
    training.add_argument("--extra-train", action="append", default=[],
                          help="Training-only folder (images/train, labels/train, metadata.jsonl)")
    tracking = commands.add_parser("track")
    tracking.add_argument("model")
    tracking.add_argument("video")
    tracking.add_argument("output")
    tracking.add_argument("--tracker", choices=["bytetrack.yaml", "botsort.yaml"], default="bytetrack.yaml")
    tracking.add_argument("--conf", type=float, default=.25)
    tracking.add_argument("--iou", type=float, default=.7)
    tracking.add_argument("--stride", type=int, default=1,
                          help="Track every Nth source frame (3 of ~30 FPS matches VIAME's 10 FPS)")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("model")
    evaluation.add_argument("dataset")
    evaluation.add_argument("output")
    evaluation.add_argument("--imgsz", type=int, default=640)
    evaluation.add_argument("--conf", type=float, default=.25)
    evaluation.add_argument("--workers", type=int, default=8, help="Dataloader workers for the mAP pass")
    evaluation.add_argument("--save-predictions", action="store_true",
                            help="Write predictions.jsonl for scoring on other frame subsets")
    args = parser.parse_args()
    if args.command == "inventory":
        from app.services.training_dataset import inventory_jobs
        print(json.dumps(inventory_jobs(Path(args.jobs), Path(args.uploads), Path(args.output)), indent=2))
    elif args.command == "dataset":
        from app.services.training_dataset import build_dataset
        print(json.dumps(build_dataset(Path(args.manifest), Path(args.output),
                                       include_pseudo=args.include_pseudo,
                                       low_confidence=args.low_confidence), indent=2))
    elif args.command == "preview":
        from app.services.training_dataset import preview_manifest
        print(json.dumps(preview_manifest(Path(args.manifest), Path(args.output),
                                          limit=args.limit, low_confidence=args.low_confidence), indent=2))
    elif args.command == "train":
        train(args)
    elif args.command == "track":
        track(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
