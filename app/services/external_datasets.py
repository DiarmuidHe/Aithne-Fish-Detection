"""Strict, sequence-aware adapters for optional Brackish datasets."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

BRACKISH_CLASS_NAMES = {"crab", "fish", "jellyfish", "shrimp", "small_fish", "starfish"}
MOT_CLASSES = {1: "fish", 2: "crab", 3: "shrimp", 4: "starfish",
               5: "small_fish", 6: "jellyfish"}
FISH_CLASSES = {"fish", "small_fish"}
SPLITS = {"train", "val", "test"}
BRACKISH_ARCHIVE_SHA256_V4 = "95de9551dd3792e4c690b59ea8dfe9c4ab59dae4c7ad32d994bd6820a75630da"


def _sequence_split(sequence: str) -> str:
    bucket = int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def _official_yolo_lines(raw: bytes, source: str, names: dict[int, str]) -> tuple[list[str], list[dict]]:
    """AAU v4 YOLO files use 1-based IDs and xywh pixels at 960x540."""
    fish, audit = [], []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        cells = line.split()
        if len(cells) != 5:
            raise ValueError(f"{source}:{line_no}: expected 5 pixel YOLO columns")
        try:
            class_id = int(cells[0])
            x, y, w, h = map(float, cells[1:])
        except ValueError as exc:
            raise ValueError(f"{source}:{line_no}: invalid numeric value") from exc
        if class_id not in names or not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
            raise ValueError(f"{source}:{line_no}: invalid class or pixel box")
        left, top, right, bottom = x-w/2, y-h/2, x+w/2, y+h/2
        clipped = left < 0 or top < 0 or right > 960 or bottom > 540
        left, top, right, bottom = max(0, left), max(0, top), min(960, right), min(540, bottom)
        if right <= left or bottom <= top:
            raise ValueError(f"{source}:{line_no}: box is entirely outside the frame")
        name = names[class_id]
        audit.append({"source_class_id": class_id, "source_class": name,
                      "target_class": "fish" if name in FISH_CLASSES else None,
                      "clipped_to_frame": clipped})
        if name in FISH_CLASSES:
            fish.append(f"0 {(left+right)/1920:.8f} {(top+bottom)/1080:.8f} "
                        f"{(right-left)/960:.8f} {(bottom-top)/540:.8f}")
    return fish, audit


def import_brackish_archive(archive: Path, output: Path, *, max_sequences: int | None = None,
                            max_frames_per_sequence: int | None = None) -> dict:
    """Import the official Kaggle v4 ZIP directly, retaining full recording boundaries."""
    import cv2

    archive, output = archive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if max_sequences is not None and max_sequences < 1 or (
        max_frames_per_sequence is not None and max_frames_per_sequence < 1
    ):
        raise ValueError("Smoke limits must be positive")
    archive_hash = sha256(archive)
    if archive_hash != BRACKISH_ARCHIVE_SHA256_V4:
        raise ValueError("Expected the official Brackish Kaggle v4 ZIP SHA256")
    with zipfile.ZipFile(archive) as bundle:
        names_file = "scripts/Brackish.names"
        names_bytes = bundle.read(names_file)
        class_names = [x.strip().lower() for x in names_bytes.decode("utf-8-sig").splitlines() if x.strip()]
        if set(class_names) != BRACKISH_CLASS_NAMES or len(class_names) != 6:
            raise ValueError("Unexpected Brackish.names class list")
        classes = dict(enumerate(class_names, 1))
        videos = {}
        for member in bundle.namelist():
            if member.startswith("dataset/videos/") and member.lower().endswith(".avi"):
                stem = Path(member).stem
                if stem in videos:
                    raise ValueError(f"Duplicate recording name: {stem}")
                videos[stem] = member
        labels = defaultdict(list)
        for member in bundle.namelist():
            if not member.startswith("annotations/annotations_YOLO/") or not member.endswith(".txt"):
                continue
            stem, separator, number = Path(member).stem.rpartition("-")
            if not separator or not number.isdigit() or stem not in videos:
                raise ValueError(f"Unmatched annotation member: {member}")
            frame = int(number)
            if frame < 1:
                raise ValueError(f"Invalid source frame: {member}")
            fish, audit = _official_yolo_lines(bundle.read(member), member, classes)
            labels[stem].append((frame, member, fish, audit))
        if set(labels) != set(videos):
            raise ValueError("Every recording must have annotation files")
        sequence_names = sorted(videos)
        if max_sequences is not None:
            sequence_names = sequence_names[:max_sequences]
        planned = {}
        for sequence in sequence_names:
            frames = sorted(labels[sequence])
            if len({item[0] for item in frames}) != len(frames):
                raise ValueError(f"Duplicate source frame in {sequence}")
            if max_frames_per_sequence is not None:
                frames = frames[:max_frames_per_sequence]
            planned[sequence] = frames
        output.mkdir(parents=True)
        metadata = []
        try:
            for sequence, frames in planned.items():
                member = videos[sequence]
                staged = output / "_current_video.avi"
                with bundle.open(member) as source, staged.open("wb") as target:
                    shutil.copyfileobj(source, target)
                cap = cv2.VideoCapture(str(staged))
                if not cap.isOpened():
                    raise ValueError(f"Cannot decode archive video: {member}")
                split = _sequence_split(sequence)
                source_frame = 0
                video_hash = sha256(staged)
                try:
                    for frame_no, label_member, fish, audit in frames:
                        while source_frame < frame_no:
                            ok, frame = cap.read()
                            source_frame += 1
                            if not ok:
                                raise ValueError(f"Video ends before {sequence} frame {frame_no}")
                        # Match the author's extraction size and bicubic interpolation.
                        resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)
                        name = hashlib.sha256(f"{sequence}:{frame_no}".encode()).hexdigest()[:20]
                        image_out = output / "images" / split / (name + ".jpg")
                        label_out = output / "labels" / split / (name + ".txt")
                        image_out.parent.mkdir(parents=True, exist_ok=True)
                        label_out.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(image_out), resized):
                            raise ValueError(f"Could not write {image_out}")
                        label_out.write_text("\n".join(fish) + ("\n" if fish else ""), encoding="utf-8")
                        metadata.append({"split": split, "source_sequence": sequence,
                                         "source_frame": frame_no, "source_video": member,
                                         "source_video_sha256": video_hash,
                                         "source_label": label_member,
                                         "source_label_sha256": hashlib.sha256(bundle.read(label_member)).hexdigest(),
                                         "output_image": str(image_out.relative_to(output)),
                                         "output_label": str(label_out.relative_to(output)),
                                         "source_classes": audit, "reviewed": True,
                                         # v4 ships an empty label file for every frame without
                                         # boxes (scripts/createDummyYOLOAnnotations.py); nothing
                                         # in the package says those frames were checked as empty.
                                         "unverified_empty": not audit,
                                         "review_scope": "published Brackish v4 annotations",
                                         "reviewed_for_smartbay": False})
                finally:
                    cap.release()
                    staged.unlink(missing_ok=True)
        finally:
            (output / "_current_video.avi").unlink(missing_ok=True)
    (output / "metadata.jsonl").write_text("".join(json.dumps(x) + "\n" for x in metadata), encoding="utf-8")
    (output / "classes.json").write_text(json.dumps({"0": "fish"}, indent=2), encoding="utf-8")
    (output / "dataset.yaml").write_text(
        f"path: '{output.as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fish\n",
        encoding="utf-8")
    report = {"source": "https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset",
              "version": 4, "license": "CC BY-SA 4.0 (Kaggle API metadata)",
              "archive_sha256": archive_hash, "class_names_sha256": hashlib.sha256(names_bytes).hexdigest(),
              "class_mapping": {str(k): {"source": v, "target": "fish" if v in FISH_CLASSES else None}
                                for k, v in classes.items()},
              "sequences": {s: _sequence_split(s) for s in planned},
              "frames": {split: sum(1 for m in metadata if m["split"] == split)
                         for split in ("train", "val", "test")},
              "frame_count": len(metadata),
              "fish_boxes": sum(len(x[2]) for f in planned.values() for x in f),
              "unverified_empty_frames": {split: sum(1 for m in metadata if m["split"] == split
                                                     and m["unverified_empty"])
                                          for split in ("train", "val", "test")},
              "source_frame_base": 1, "resize": "960x540 bicubic",
              "review_status": "published external annotations; no SmartBay review"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def make_brackish_manifest(root: Path, output: Path, sequence_regex: str) -> dict:
    """Inventory YOLO image/label pairs and assign whole recordings by stable hash."""
    if output.exists():
        raise FileExistsError(output)
    pattern = re.compile(sequence_regex)
    if not {"sequence", "frame"} <= pattern.groupindex.keys():
        raise ValueError("Regex needs named sequence and frame groups")
    root = root.resolve()
    images = sorted(path for path in root.rglob("*") if path.parent.name == "images"
                    and path.suffix.lower() == ".jpg")
    if not images:
        raise ValueError("No images found under an images/ folder")
    rows = []
    seen = set()
    for image in images:
        match = pattern.search(image.stem)
        if match is None:
            raise ValueError(f"Cannot identify sequence/frame from {image.name}")
        sequence, frame = match.group("sequence"), int(match.group("frame"))
        if not sequence or frame < 0 or (sequence, frame) in seen:
            raise ValueError(f"Duplicate or invalid sequence/frame: {sequence}/{frame}")
        seen.add((sequence, frame))
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        if not label.is_file():
            raise ValueError(f"Missing label for {image}")
        split = _sequence_split(sequence)
        rows.append({"image": image.relative_to(root).as_posix(),
                     "label": label.relative_to(root).as_posix(), "sequence": sequence,
                     "frame": frame, "split": split})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label", "sequence", "frame", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return {"frames": len(rows), "sequences": len({row["sequence"] for row in rows}),
            "manifest": str(output), "sha256": sha256(output)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-root source file: {relative}")
    return path


def load_brackish_class_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Class map must be a JSON object of source ID to class name")
    try:
        mapping = {int(key): str(value).lower().replace(" ", "_") for key, value in raw.items()}
    except ValueError as exc:
        raise ValueError("Class IDs must be integers") from exc
    if set(mapping.values()) != BRACKISH_CLASS_NAMES or len(mapping) != 6:
        raise ValueError("Class map must explicitly cover the six Brackish classes once each")
    return mapping


def _yolo_lines(path: Path, class_map: dict[int, str]) -> tuple[list[str], list[dict]]:
    fish, audit = [], []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cells = line.split()
        if len(cells) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO columns")
        try:
            class_id = int(cells[0])
            values = [float(value) for value in cells[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid numeric value") from exc
        if class_id not in class_map:
            raise ValueError(f"{path}:{line_no}: unknown class {class_id}")
        x, y, w, h = values
        if not all(math.isfinite(v) for v in values) or w <= 0 or h <= 0 or not (
            0 <= x-w/2 < x+w/2 <= 1 and 0 <= y-h/2 < y+h/2 <= 1
        ):
            raise ValueError(f"{path}:{line_no}: invalid normalized box")
        name = class_map[class_id]
        audit.append({"source_class_id": class_id, "source_class": name,
                      "target_class": "fish" if name in FISH_CLASSES else None})
        if name in FISH_CLASSES:
            fish.append("0 " + " ".join(cells[1:]))
    return fish, audit


def import_brackish(root: Path, manifest: Path, class_map_path: Path, output: Path) -> dict:
    """Manifest rows explicitly name image, label, sequence, frame and split."""
    root, output = root.resolve(), output.resolve()
    class_map = load_brackish_class_map(class_map_path)
    if output.exists():
        raise FileExistsError(output)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    if not rows or not {"image", "label", "sequence", "frame", "split"} <= rows[0].keys():
        raise ValueError("Manifest needs image,label,sequence,frame,split columns")
    sequences: dict[str, str] = {}
    seen: set[tuple[str, int]] = set()
    content_splits: dict[str, str] = {}
    planned = []
    for row in rows:
        split = {"valid": "val", "validation": "val"}.get(row["split"], row["split"])
        sequence = row["sequence"].strip()
        if not sequence or split not in SPLITS:
            raise ValueError("Every frame needs a source sequence and train/val/test split")
        if sequence in sequences and sequences[sequence] != split:
            raise ValueError(f"Split leakage: sequence {sequence} crosses {sequences[sequence]}/{split}")
        sequences[sequence] = split
        frame = int(row["frame"])
        if frame < 0 or (sequence, frame) in seen:
            raise ValueError(f"Duplicate or negative source frame: {sequence}/{frame}")
        seen.add((sequence, frame))
        image = _safe_source(root, row["image"])
        if image.suffix.lower() != ".jpg":
            raise ValueError(f"Project training format requires .jpg frames: {image}")
        label = _safe_source(root, row["label"])
        fish, audit = _yolo_lines(label, class_map)
        image_hash, label_hash = sha256(image), sha256(label)
        if image_hash in content_splits and content_splits[image_hash] != split:
            raise ValueError(f"Split leakage: identical image appears in {content_splits[image_hash]}/{split}")
        content_splits[image_hash] = split
        planned.append((split, sequence, frame, image, label, fish, audit, image_hash, label_hash))
    output.mkdir(parents=True)
    metadata = []
    for split, sequence, frame, image, label, fish, audit, image_hash, label_hash in planned:
        name = hashlib.sha256(f"{sequence}:{frame}".encode()).hexdigest()[:20]
        image_out = output / "images" / split / (name + image.suffix.lower())
        label_out = output / "labels" / split / (name + ".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)
        label_out.write_text("\n".join(fish) + ("\n" if fish else ""), encoding="utf-8")
        metadata.append({"split": split, "source_sequence": sequence, "source_frame": frame,
                         "source_image": str(image), "source_label": str(label),
                         "source_image_sha256": image_hash, "source_label_sha256": label_hash,
                         "output_image": str(image_out.relative_to(output)),
                         "output_label": str(label_out.relative_to(output)),
                         "source_classes": audit, "reviewed_for_smartbay": False})
    (output / "metadata.jsonl").write_text("".join(json.dumps(x) + "\n" for x in metadata),
                                            encoding="utf-8")
    (output / "classes.json").write_text(json.dumps({"0": "fish"}, indent=2), encoding="utf-8")
    (output / "dataset.yaml").write_text(
        f"path: '{output.as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fish\n",
        encoding="utf-8")
    report = {"source": "https://vbn.aau.dk/en/datasets/the-brackish-dataset/",
              "release": "2019; exact Kaggle package revision must be recorded on download",
              "license": "Official portal does not state a license; verify Kaggle package terms",
              "manifest_sha256": sha256(manifest), "class_map_sha256": sha256(class_map_path),
              "class_mapping": {str(k): {"source": v, "target": "fish" if v in FISH_CLASSES else None}
                                for k, v in class_map.items()},
              "frames": len(metadata), "fish_boxes": sum(len(p[5]) for p in planned),
              "sequences": sequences, "review_status": "external dataset; no SmartBay review"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_mot(path: Path, *, source: bool = True, unidentified: list | None = None) -> list[dict]:
    """Strict MOT rows. Source rows with track ID -1 (a box with no identity) go to
    ``unidentified`` when a list is supplied, and are otherwise rejected."""
    result = []
    for line_no, cells in enumerate(csv.reader(path.open(encoding="utf-8-sig", newline="")), 1):
        if not cells or not any(cells):
            continue
        if len(cells) < 6:
            raise ValueError(f"{path}:{line_no}: expected MOT columns")
        frame, track_id = int(cells[0]), int(cells[1])
        if source and track_id == -1 and unidentified is not None:
            unidentified.append({"line": line_no, "row": cells})
            continue
        x, y, w, h = map(float, cells[2:6])
        class_id = int(float(cells[7])) if source and len(cells) > 7 else 1
        if frame < 1 or track_id < 1 or class_id not in MOT_CLASSES or not all(
            math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_no}: invalid MOT row")
        if source and len(cells) > 6 and float(cells[6]) <= 0:
            continue
        result.append({"frame": frame, "track_id": track_id, "x": x, "y": y,
                       "w": w, "h": h, "class_id": class_id})
    return result


def extract_brackish_mot(archive: Path, output: Path, expected_sha256: str) -> dict:
    """Extract only the real BrackishMOT/{train,test} sequences, never the synthetic sets."""
    if output.exists():
        raise FileExistsError(output)
    if sha256(archive) != expected_sha256:
        raise ValueError("BrackishMOT archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        planned = []
        for member in bundle.infolist():
            parts = member.filename.split("/")
            if parts[0] != "BrackishMOT" or member.is_dir():
                continue
            if parts[1] not in ("train", "test") or len(parts) < 4 or any(
                    p in ("", ".", "..") or ":" in p or "\\" in p for p in parts):
                raise ValueError(f"Unexpected BrackishMOT member: {member.filename}")
            planned.append((member, output.joinpath(*parts[1:])))
        if not planned:
            raise ValueError("Archive has no BrackishMOT/ sequences")
        output.mkdir(parents=True)
        for member, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream)
    sequences = sorted({target.relative_to(output).parts[:2] for _, target in planned})
    return {"archive_sha256": expected_sha256, "files": len(planned),
            "sequences": len(sequences), "output": str(output),
            "excluded": "brackishMOTSynth and background_videos are not extracted"}


def import_brackish_mot(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    sources = sorted(root.rglob("gt/gt.txt"))
    if not sources:
        raise ValueError("No <sequence>/gt/gt.txt files found")
    planned = []
    for source in sources:
        sequence = source.parent.parent.relative_to(root).as_posix()
        unidentified: list = []
        rows = [x for x in parse_mot(source, unidentified=unidentified)
                if MOT_CLASSES[x["class_id"]] in FISH_CLASSES]
        planned.append((sequence, source, rows, unidentified))
    output.mkdir(parents=True)
    with (output / "ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sequence", "frame", "track_id", "x", "y", "w", "h", "source_class_id"])
        writer.writeheader()
        for sequence, source, rows, _ in planned:
            for row in rows:
                writer.writerow({"sequence": sequence, "frame": row["frame"],
                                 "track_id": row["track_id"], "x": row["x"], "y": row["y"],
                                 "w": row["w"], "h": row["h"], "source_class_id": row["class_id"]})
    report = {"source": "https://vbn.aau.dk/en/datasets/brackishmot/",
              "release": "2023-02-22; DOI 10.34740/kaggle/ds/2695511",
              "kaggle_metadata_observed": "version 4 on 2026-10-11",
              "license": "CC BY-SA 4.0 in Kaggle v4 metadata; verify downloaded package terms",
              "class_mapping": {str(k): ("fish" if v in FISH_CLASSES else None) for k, v in MOT_CLASSES.items()},
              "sequences": {sequence: {"gt": str(source), "sha256": sha256(source),
                                       "fish_boxes": len(rows),
                                       "excluded_unidentified_rows": unidentified}
                            for sequence, source, rows, unidentified in planned}}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def viame_tracks_to_mot(path: Path, *, input_names: list[str] | None = None,
                        frame_of: dict[str, int] | None = None,
                        frame_offset: int | None = None) -> list[dict]:
    """VIAME track CSV rows as one-based MOT rows (frame, track_id, x, y, w, h, score).

    Image-list runs: column 3 is the one-based position in the list VIAME read and
    column 2 that entry's file name. ``input_names`` (the list's file names in order)
    cross-checks both, and ``frame_of`` maps the name to its one-based MOT frame.
    Video runs: column 3 is a one-based source frame; production's ``frame_offset``
    of -1 makes it zero-based, and MOT adds one back.
    """
    from app.services.viame_parser import parse_viame_csv

    if (input_names is None) == (frame_offset is None) or (input_names is None) != (frame_of is None):
        raise ValueError("Pass input_names with frame_of for image lists, or frame_offset for video")
    rows, seen = [], set()
    for detection in parse_viame_csv(path, strict=True).detections:
        if frame_offset is not None:
            zero_based = detection.frame_number + frame_offset
            if zero_based < 0:
                raise ValueError(f"{path}: frame {detection.frame_number} is before the first frame")
            frame = zero_based + 1
        else:
            position = detection.frame_number
            if not 1 <= position <= len(input_names) or input_names[position - 1] != detection.source:
                raise ValueError(f"{path}: frame {position} is not {detection.source} in the input list")
            frame = frame_of[detection.source]
        track_id = int(detection.track_id)
        width = detection.bbox_right - detection.bbox_left
        height = detection.bbox_bottom - detection.bbox_top
        if track_id < 1 or width <= 0 or height <= 0 or (frame, track_id) in seen:
            raise ValueError(f"{path}: invalid or duplicate track {track_id} on frame {frame}")
        seen.add((frame, track_id))
        rows.append({"frame": frame, "track_id": track_id, "x": detection.bbox_left,
                     "y": detection.bbox_top, "w": width, "h": height,
                     "score": detection.confidence})
    return sorted(rows, key=lambda row: (row["frame"], row["track_id"]))


def write_mot(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow([row["frame"], row["track_id"], f"{row['x']:.3f}", f"{row['y']:.3f}",
                             f"{row['w']:.3f}", f"{row['h']:.3f}", f"{row['score']:.6f}"])


def _iou(a: dict, b: dict) -> float:
    left, top = max(a["x"], b["x"]), max(a["y"], b["y"])
    right, bottom = min(a["x"] + a["w"], b["x"] + b["w"]), min(a["y"] + a["h"], b["y"] + b["h"])
    area = max(0, right-left) * max(0, bottom-top)
    return area / (a["w"]*a["h"] + b["w"]*b["h"] - area) if area else 0.0


def min_cost_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Rectangular minimum-cost assignment (Hungarian, shortest augmenting paths).

    Every row or column of the smaller side is assigned; O(n^2 m). A small,
    dependency-free stand-in for scipy's linear_sum_assignment.
    """
    rows = len(cost)
    columns = len(cost[0]) if rows else 0
    if not rows or not columns:
        return []
    transpose = rows > columns
    if transpose:
        cost = [list(column) for column in zip(*cost)]
        rows, columns = columns, rows
    infinity = float("inf")
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    owner, way = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        owner[0], current = row, 0
        slack, used = [infinity] * (columns + 1), [False] * (columns + 1)
        while True:
            used[current] = True
            row_now, delta, following = owner[current], infinity, 0
            values = cost[row_now - 1]
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = values[column - 1] - u[row_now] - v[column]
                if reduced < slack[column]:
                    slack[column], way[column] = reduced, current
                if slack[column] < delta:
                    delta, following = slack[column], column
            for column in range(columns + 1):
                if used[column]:
                    u[owner[column]] += delta
                    v[column] -= delta
                else:
                    slack[column] -= delta
            current = following
            if owner[current] == 0:
                break
        while current:
            previous = way[current]
            owner[current] = owner[previous]
            current = previous
    pairs = [(owner[column] - 1, column - 1) for column in range(1, columns + 1) if owner[column]]
    return sorted((b, a) for a, b in pairs) if transpose else sorted(pairs)


def _identity_counts(gt_frames: dict, pred_frames: dict, threshold: float) -> dict:
    """IDF1 as in TrackEval's Identity metric: one global ID assignment per sequence.

    A pair's weight is the number of frames where those two boxes overlap at the
    IoU threshold; the assignment maximizes the total (IDTP).
    """
    potential, gt_total, pred_total = defaultdict(int), 0, 0
    for frame in set(gt_frames) | set(pred_frames):
        gt, pred = gt_frames.get(frame, []), pred_frames.get(frame, [])
        gt_total += len(gt)
        pred_total += len(pred)
        for a in gt:
            for b in pred:
                if _iou(a, b) >= threshold:
                    potential[a["track_id"], b["track_id"]] += 1
    gt_ids = sorted({g for g, _ in potential})
    pred_ids = sorted({p for _, p in potential})
    cost = [[-potential.get((g, p), 0) for p in pred_ids] for g in gt_ids]
    idtp = sum(-cost[i][j] for i, j in min_cost_assignment(cost))
    return {"idtp": idtp, "idfp": pred_total - idtp, "idfn": gt_total - idtp}


def _clear_counts(gt_frames: dict, pred_frames: dict, threshold: float) -> dict:
    """CLEAR MOT matching as in TrackEval: per-frame optimal assignment that prefers
    continuing the previous frame's pairing; an ID switch is a GT identity matched to
    a different prediction than at its last match."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "idsw": 0}
    last_match, previous_frame_match = {}, {}
    for frame in sorted(set(gt_frames) | set(pred_frames)):
        gt, pred = gt_frames.get(frame, []), pred_frames.get(frame, [])
        score = [[0.0] * len(pred) for _ in gt]
        for i, a in enumerate(gt):
            for j, b in enumerate(pred):
                overlap = _iou(a, b)
                if overlap >= threshold:
                    continuing = previous_frame_match.get(a["track_id"]) == b["track_id"]
                    score[i][j] = overlap + (1000 if continuing else 0)
        matched = [(i, j) for i, j in min_cost_assignment([[-x for x in row] for row in score])
                   if score[i][j] > 0]
        previous_frame_match = {}
        for i, j in matched:
            identity, prediction = gt[i]["track_id"], pred[j]["track_id"]
            if identity in last_match and last_match[identity] != prediction:
                counts["idsw"] += 1
            last_match[identity] = previous_frame_match[identity] = prediction
        counts["tp"] += len(matched)
        counts["fn"] += len(gt) - len(matched)
        counts["fp"] += len(pred) - len(matched)
    return counts


def _identity_scores(counts: dict) -> dict:
    idtp, idfp, idfn = counts["idtp"], counts["idfp"], counts["idfn"]
    return counts | {"idf1": 2 * idtp / (2 * idtp + idfp + idfn) if idtp + idfp + idfn else None,
                     "idp": idtp / (idtp + idfp) if idtp + idfp else None,
                     "idr": idtp / (idtp + idfn) if idtp + idfn else None}


def _clear_scores(counts: dict, gt_tracks: int) -> dict:
    gt_boxes = counts["tp"] + counts["fn"]
    return counts | {"mota": (counts["tp"] - counts["fp"] - counts["idsw"]) / gt_boxes if gt_boxes else None,
                     "gt_tracks": gt_tracks,
                     "idsw_per_gt_track": counts["idsw"] / gt_tracks if gt_tracks else None}


METRIC_STATUS = {
    "identity": "official-style: TrackEval Identity rules (IDF1, global min-cost ID assignment), "
                "re-implemented here; not TrackEval itself",
    "clear": "official-style: TrackEval CLEAR MOT rules (per-frame optimal assignment preferring "
             "continued pairs), re-implemented here; not TrackEval itself",
    "counts": "diagnostic: greedy per-frame IoU pairing kept for continuity with earlier reports",
    "hota": "not computed: TrackEval is not an approved dependency or download for this task"}


def evaluate_mot(imported: Path, candidates: Path, output: Path, threshold: float = .5) -> dict:
    """Score one-based MOT candidates against the imported fish ground truth.

    If ``candidates/analysed_frames.json`` maps sequences to the one-based frames the
    tracker actually analysed (for example 10 or 4 FPS sampling of 15 FPS video), only
    those sequences and frames are scored: unanalysed frames are not misses.
    """
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    if output.exists():
        raise FileExistsError(output)
    with (imported / "ground_truth.csv").open(encoding="utf-8", newline="") as stream:
        truth = list(csv.DictReader(stream))
    imported_report = json.loads((imported / "report.json").read_text(encoding="utf-8"))
    analysed_path = candidates / "analysed_frames.json"
    analysed = None
    if analysed_path.is_file():
        analysed = {sequence: set(frames) for sequence, frames in
                    json.loads(analysed_path.read_text(encoding="utf-8")).items()}
        unknown = set(analysed) - set(imported_report["sequences"])
        if unknown:
            raise ValueError(f"Analysed frames name unknown sequences: {sorted(unknown)[:3]}")
    groups = defaultdict(lambda: defaultdict(list))
    for sequence in (analysed or imported_report["sequences"]):
        groups[sequence]
    for row in truth:
        frame = int(row["frame"])
        if analysed is not None and frame not in analysed.get(row["sequence"], ()):
            continue
        groups[row["sequence"]][frame].append(
            {key: float(row[key]) for key in ("x", "y", "w", "h")} | {"track_id": int(row["track_id"])})
    counts = {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0}
    identity_total = {"idtp": 0, "idfp": 0, "idfn": 0}
    clear_total = {"tp": 0, "fp": 0, "fn": 0, "idsw": 0}
    gt_tracks_total = 0
    sequence_results = {}
    for sequence, gt_frames in groups.items():
        candidate = candidates / (sequence + ".txt")
        if not candidate.is_file():
            raise ValueError(f"Missing candidate track file: {candidate}")
        pred_frames = defaultdict(list)
        for row in parse_mot(candidate, source=False):
            if analysed is not None and row["frame"] not in analysed[sequence]:
                raise ValueError(f"{candidate}: prediction on unanalysed frame {row['frame']}")
            pred_frames[row["frame"]].append(row)
        local = dict.fromkeys(counts, 0)
        prior = {}
        for frame in sorted(set(gt_frames) | set(pred_frames)):
            gt, pred = gt_frames[frame], pred_frames[frame]
            edges = sorted(((i, j, _iou(a, b)) for i, a in enumerate(gt)
                            for j, b in enumerate(pred) if _iou(a, b) >= threshold),
                           key=lambda edge: -edge[2])
            matched_g, matched_p = set(), set()
            for i, j, _ in edges:
                if i in matched_g or j in matched_p:
                    continue
                matched_g.add(i); matched_p.add(j)
                identity = (sequence, gt[i]["track_id"])
                if identity in prior and prior[identity] != pred[j]["track_id"]:
                    local["id_switches"] += 1
                prior[identity] = pred[j]["track_id"]
            local["tp"] += len(matched_g)
            local["fn"] += len(gt) - len(matched_g)
            local["fp"] += len(pred) - len(matched_p)
        identity = _identity_counts(gt_frames, pred_frames, threshold)
        clear = _clear_counts(gt_frames, pred_frames, threshold)
        gt_tracks = len({box["track_id"] for boxes in gt_frames.values() for box in boxes})
        sequence_results[sequence] = local | {
            "identity": _identity_scores(identity), "clear": _clear_scores(clear, gt_tracks),
            "gt_boxes": sum(map(len, gt_frames.values())),
            "pred_boxes": sum(map(len, pred_frames.values())),
            "analysed_frames": len(analysed[sequence]) if analysed is not None else None}
        for key in counts:
            counts[key] += local[key]
        for key in identity_total:
            identity_total[key] += identity[key]
        for key in clear_total:
            clear_total[key] += clear[key]
        gt_tracks_total += gt_tracks
    total = counts["tp"] + counts["fn"]
    report = {"counts": counts, "mota": 1 - (counts["fn"]+counts["fp"]+counts["id_switches"])/total if total else None,
              "precision": counts["tp"] / max(1, counts["tp"]+counts["fp"]),
              "recall": counts["tp"] / max(1, total),
              "identity": _identity_scores(identity_total),
              "clear": _clear_scores(clear_total, gt_tracks_total),
              "sequences": sequence_results,
              "analysed_frames": ("all frames" if analysed is None else
                                  f"only {analysed_path.name} ({sum(map(len, analysed.values()))} frames)"),
              "metric_status": METRIC_STATUS,
              "iou_threshold": threshold, "note": "Greedy IoU matching; diagnostic counts, not official MOTChallenge scores"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def frame_fingerprint(image):
    """64x36 grey thumbnail as float32."""
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (64, 36), interpolation=cv2.INTER_AREA).astype("float32")


def best_alignment(sequence_prints, recordings: dict) -> list[dict]:
    """Rank (recording, offset) pairs by the cosine similarity of frame-to-frame changes.

    Differencing consecutive thumbnails removes a static camera's background, leaving
    moving animals and sensor noise, which are shared only by the same source frames.
    ``offset`` maps sequence frame k (one-based) to recording frame k + offset.
    """
    import numpy as np

    z = np.diff(np.stack(sequence_prints), axis=0).reshape(len(sequence_prints) - 1, -1)
    z_norm = (z * z).sum(axis=1)
    ranked = []
    for name, prints in recordings.items():
        a = np.diff(prints, axis=0).reshape(len(prints) - 1, -1)
        a_norm = (a * a).sum(axis=1)
        gram = z @ a.T
        for offset in range(-(len(z) - 1), len(a)):
            rows = np.arange(max(0, -offset), min(len(z), len(a) - offset))
            if len(rows) < max(5, len(z) // 2):
                continue
            numerator = gram[rows, rows + offset].sum()
            denominator = np.sqrt(z_norm[rows].sum() * a_norm[rows + offset].sum())
            ranked.append({"recording": name, "offset": int(offset),
                           "similarity": float(numerator / denominator) if denominator else 0.0,
                           "overlap_frames": int(len(rows) + 1)})
    return sorted(ranked, key=lambda r: -r["similarity"])


def classify_alignment(ranked: list[dict], *, accept: float = .6, margin: float = .2,
                       reject: float = .2) -> dict:
    """Overlap needs a strong, unambiguous best alignment; none needs every candidate weak."""
    if not ranked:
        return {"status": "undetermined", "recording": None, "offset": None, "best": None}
    best = ranked[0]
    rival = next((r for r in ranked[1:] if r["recording"] != best["recording"]
                  or abs(r["offset"] - best["offset"]) > 2), None)
    second = rival["similarity"] if rival else 0.0
    if best["similarity"] >= accept and best["similarity"] - second >= margin:
        status = "overlap"
    elif best["similarity"] < reject:
        status = "none"
    else:
        status = "undetermined"
    return {"status": status, "recording": best["recording"] if status == "overlap" else None,
            "offset": best["offset"] if status == "overlap" else None, "best": best,
            "runner_up": rival}


def brackish_mot_overlap(archive: Path, mot_root: Path, sequences: list[str]) -> dict:
    """Which Brackish detection recording, and at what frame offset, each BrackishMOT
    sequence was cut from, by temporal alignment over every frame of both."""
    import tempfile

    import cv2
    import numpy as np

    if sha256(archive) != BRACKISH_ARCHIVE_SHA256_V4:
        raise ValueError("Expected the official Brackish Kaggle v4 ZIP SHA256")
    recordings = {}
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as scratch:
        for member in sorted(n for n in bundle.namelist()
                             if n.startswith("dataset/videos/") and n.lower().endswith(".avi")):
            staged = Path(scratch) / "recording.avi"
            with bundle.open(member) as source, staged.open("wb") as target:
                shutil.copyfileobj(source, target)
            capture, prints = cv2.VideoCapture(str(staged)), []
            try:
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    prints.append(frame_fingerprint(image))
            finally:
                capture.release()
            recordings[Path(member).stem] = np.stack(prints)
    names = list(recordings)
    owners = np.concatenate([[i] * len(recordings[n]) for i, n in enumerate(names)])
    stack = np.concatenate([recordings[n] for n in names])
    result = {}
    for sequence in sequences:
        images = sorted((mot_root / sequence / "img1").glob("*.jpg"))
        prints = [frame_fingerprint(cv2.imread(str(image))) for image in images]
        # Shortlist: recordings holding any of the ten nearest frames to the first, middle
        # or last sequence frame. Only those get the full temporal alignment.
        shortlist = set()
        for probe in (prints[0], prints[len(prints) // 2], prints[-1]):
            distance = np.abs(stack - probe).mean(axis=(1, 2))
            shortlist |= {names[owners[i]] for i in np.argsort(distance)[:10]}
        ranked = best_alignment(prints, {n: recordings[n] for n in sorted(shortlist)})
        result[sequence] = classify_alignment(ranked) | {"frames": len(images),
                                                        "shortlist": sorted(shortlist)}
    return {"archive_sha256": BRACKISH_ARCHIVE_SHA256_V4,
            "method": "cosine similarity of consecutive-frame differences of 64x36 grey thumbnails",
            "thresholds": {"accept": .6, "margin": .2, "reject": .2},
            "archive_frames": sum(len(v) for v in recordings.values()), "sequences": result}


def leakage_guard(overlap: dict, training_recordings: set[str], sequences: list[str]) -> dict:
    """BrackishMOT sequences a Brackish-trained detector may be scored on.

    Sequences cut from a training (or validation) recording are excluded, and so are
    sequences whose origin could not be determined: unknown is not treated as clean.
    """
    scorable, excluded, undetermined = [], {}, []
    for sequence in sequences:
        entry = overlap.get(sequence)
        if entry is None or entry["status"] == "undetermined":
            undetermined.append(sequence)
        elif entry["status"] == "overlap" and entry["recording"] in training_recordings:
            excluded[sequence] = entry["recording"]
        elif entry["status"] in ("overlap", "none"):
            scorable.append(sequence)
        else:
            raise ValueError(f"Unknown overlap status for {sequence}: {entry['status']}")
    return {"scorable": scorable, "excluded_training_overlap": excluded, "undetermined": undetermined}
