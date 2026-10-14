"""Build auditable YOLO data from explicitly paired VIAME CSV and source videos."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from app.services.viame_parser import parse_viame_csv


def _split(group: str) -> str:
    value = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if value < 0.7 else "val" if value < 0.85 else "test"


def _box(d, width: int, height: int) -> str:
    x1, y1, x2, y2 = d.bbox_left, d.bbox_top, d.bbox_right, d.bbox_bottom
    if not all(map(math.isfinite, (x1, y1, x2, y2))):
        raise ValueError("nonfinite box")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box outside frame or inverted")
    return f"{(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"


def build_dataset(manifest_path: Path, output: Path, *, include_pseudo: bool = False,
                  low_confidence: float = 0.6) -> dict:
    """Manifest paths are relative to the manifest. Existing output is never overwritten."""
    import cv2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = [source for source in manifest["sources"] if source.get("enabled", True)]
    mapping = manifest["class_mapping"]
    names = sorted(set(mapping.values()))
    if not sources or not names or not 0 <= low_confidence <= 1:
        raise ValueError("sources, class_mapping and a valid confidence threshold are required")
    if output.exists():
        raise FileExistsError(output)
    base = manifest_path.parent
    prepared = []
    group_splits = {}
    video_groups = {}
    seen_videos = set()
    for source in sources:
        video = (base / source["video"]).resolve()
        csv_path = (base / source["annotations"]).resolve()
        if not video.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"Missing video or annotation: {video}, {csv_path}")
        if video in seen_videos:
            raise ValueError(f"Video {video} has multiple active annotation files; select or merge one")
        seen_videos.add(video)
        group = source.get("group") or str(video)
        if video in video_groups and video_groups[video] != group:
            raise ValueError(f"Video {video} assigned to multiple groups")
        video_groups[video] = group
        split = source.get("split") or _split(group)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Group {group} spans multiple splits")
        group_splits[group] = split
        reviewed = source.get("reviewed", False)
        if type(reviewed) is not bool:
            raise ValueError("reviewed must be a boolean")
        if not reviewed and split != "train":
            raise ValueError("Pseudo-labels may only be assigned to train")
        parsed = parse_viame_csv(csv_path, strict=False,
                                 frame_number_offset=int(source.get("frame_offset", 0)))
        prepared.append((source, video, csv_path, split, reviewed, parsed))
    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels", "previews"):
            (output / kind / split).mkdir(parents=True)
    issues, metadata = [], []
    counts = defaultdict(int)
    for source, video, csv_path, split, reviewed, parsed in prepared:
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                issues.append({"video": str(video), "reason": "cannot open video"})
                continue
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            annotation_fps = source.get("annotation_fps")
            if annotation_fps is not None and (annotation_fps <= 0 or fps <= 0):
                raise ValueError("annotation_fps and source fps must be positive")
            frames = defaultdict(list)
            uncertain_frames = set()
            seen = set()
            labels_by_track = defaultdict(set)
            for row in parsed.skipped_rows:
                issues.append({"video": str(video), "annotation": str(csv_path),
                               "line": row.line_number, "reason": row.reason})
            for d in parsed.detections:
                frame = round(d.frame_number * fps / annotation_fps) if annotation_fps else d.frame_number
                if not math.isfinite(d.confidence) or not 0 <= d.confidence <= 1:
                    issues.append({"video": str(video), "frame": frame, "reason": "invalid confidence"})
                    uncertain_frames.add(frame)
                    continue
                key = (frame, d.track_id, d.bbox_left, d.bbox_top, d.bbox_right, d.bbox_bottom)
                if key in seen:
                    issues.append({"video": str(video), "frame": frame, "reason": "duplicate", "track_id": d.track_id})
                    continue
                seen.add(key)
                if frame < 0 or frame >= frame_count:
                    issues.append({"video": str(video), "frame": frame, "reason": "frame outside video"})
                    continue
                try:
                    box = _box(d, width, height)
                except ValueError as exc:
                    issues.append({"video": str(video), "frame": frame, "reason": str(exc)})
                    uncertain_frames.add(frame)
                    continue
                label = d.class_name or "unknown"
                labels_by_track[d.track_id].add(label)
                if d.confidence < low_confidence:
                    issues.append({"video": str(video), "frame": frame, "reason": "low confidence",
                                   "track_id": d.track_id, "confidence": d.confidence})
                    uncertain_frames.add(frame)
                if label not in mapping:
                    issues.append({"video": str(video), "frame": frame, "reason": "unmapped label", "label": label})
                    uncertain_frames.add(frame)
                    continue
                if not reviewed and (not include_pseudo or d.confidence < low_confidence):
                    continue
                frames[frame].append((d, names.index(mapping[label]), box))
            for track_id, labels in labels_by_track.items():
                if len(labels) > 1:
                    issues.append({"video": str(video), "reason": "inconsistent track labels",
                                   "track_id": track_id, "labels": sorted(labels)})
            if not reviewed:
                for frame in uncertain_frames:
                    frames.pop(frame, None)
            for frame_number, entries in sorted(frames.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, image = cap.read()
                if not ok:
                    issues.append({"video": str(video), "frame": frame_number, "reason": "frame decode failed"})
                    continue
                # Actual decoded dimensions take precedence over container metadata.
                if image.shape[1] != width or image.shape[0] != height:
                    issues.append({"video": str(video), "frame": frame_number, "reason": "dimension mismatch"})
                    continue
                stem = hashlib.sha256(str(video).encode()).hexdigest()[:12] + f"_{frame_number:08d}"
                image_path = output / "images" / split / f"{stem}.jpg"
                label_path = output / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(image_path), image)
                label_path.write_text("\n".join(f"{cls} {box}" for _, cls, box in entries) + "\n", encoding="utf-8")
                if counts[f"preview_{split}"] < 20:
                    preview = image.copy()
                    for d, cls, _ in entries:
                        cv2.rectangle(preview, (round(d.bbox_left), round(d.bbox_top)),
                                      (round(d.bbox_right), round(d.bbox_bottom)), (0, 255, 0), 2)
                        cv2.putText(preview, f"{names[cls]} {d.confidence:.2f}",
                                    (round(d.bbox_left), max(12, round(d.bbox_top)-4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
                    cv2.imwrite(str(output / "previews" / split / f"{stem}.jpg"), preview)
                    counts[f"preview_{split}"] += 1
                for d, cls, _ in entries:
                    metadata.append({"image": str(image_path.relative_to(output)), "video": str(video),
                                     "annotation": str(csv_path), "source": d.source,
                                     "frame": frame_number, "viame_frame": d.frame_number,
                                     "track_id": d.track_id, "original_label": d.class_name,
                                     "class_id": cls, "confidence": d.confidence,
                                     "reviewed": reviewed, "split": split})
                counts[split] += 1
        finally:
            cap.release()
    (output / "metadata.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metadata), encoding="utf-8")
    (output / "review_issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    (output / "classes.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    # YAML subset avoids a hard PyYAML dependency in the API package.
    (output / "dataset.yaml").write_text(
        f"path: {json.dumps(str(output.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: {json.dumps(names)}\n", encoding="utf-8")
    report = {"frames": {s: counts[s] for s in ("train", "val", "test")},
              "annotations": len(metadata), "issues": len(issues), "classes": names,
              "pseudo_included": include_pseudo, "evaluation_ready": bool(counts["val"] and counts["test"])}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def inventory_jobs(job_root: Path, upload_root: Path, output: Path) -> dict:
    """Create a reviewable pairing draft from VIAME job input lists; never certify labels."""
    if output.exists():
        raise FileExistsError(output)
    sources, missing, labels = [], [], set()
    seen_videos = set()
    alternatives = 0
    for csv_path in sorted(job_root.glob("*/*/viame_tracks.csv")):
        input_list = csv_path.with_name("input_list.txt")
        if not input_list.is_file():
            missing.append({"annotation": str(csv_path.resolve()), "reason": "input list missing"})
            continue
        entries = [line.strip() for line in input_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(entries) != 1:
            missing.append({"annotation": str(csv_path.resolve()), "reason": "expected one source video"})
            continue
        video = upload_root / Path(entries[0]).name
        if not video.is_file():
            missing.append({"annotation": str(csv_path.resolve()), "video": str(video.resolve()),
                            "reason": "source video missing"})
            continue
        parsed = parse_viame_csv(csv_path)
        labels.update(d.class_name for d in parsed.detections if d.class_name and d.class_name != "unknown")
        header = csv_path.read_text(encoding="utf-8-sig")[:500]
        enabled = video.resolve() not in seen_videos
        seen_videos.add(video.resolve())
        alternatives += not enabled
        sources.append({"video": str(video.resolve()), "annotations": str(csv_path.resolve()),
                        "group": str(video.resolve()), "split": "train", "reviewed": False,
                        "enabled": enabled,
                        "frame_offset": -1 if "write_object_track_set_viame_csv" in header else 0})
    result = {"class_mapping": {label: label for label in sorted(labels)},
              "sources": sources, "missing": missing,
              "note": "Draft only. Inspect pairs, frame offsets, classes, and review status before building."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"paired": len(sources), "active": len(sources) - alternatives,
            "alternate_runs": alternatives, "missing": len(missing), "labels": sorted(labels),
            "manifest": str(output)}


def preview_manifest(manifest_path: Path, output: Path, *, limit: int = 20,
                     low_confidence: float = .6) -> dict:
    """Render VIAME boxes for human inspection, including rejected pseudo labels."""
    import cv2

    if output.exists():
        raise FileExistsError(output)
    if limit < 1:
        raise ValueError("limit must be positive")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    saved = 0
    for source in data["sources"]:
        if not source.get("enabled", True):
            continue
        video = (manifest_path.parent / source["video"]).resolve()
        annotation = (manifest_path.parent / source["annotations"]).resolve()
        if not video.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Missing video or annotation: {video}, {annotation}")
        parsed = parse_viame_csv(annotation,
                                 frame_number_offset=int(source.get("frame_offset", 0)))
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            annotation_fps = source.get("annotation_fps")
            if annotation_fps is not None and (annotation_fps <= 0 or fps <= 0):
                raise ValueError("annotation_fps and source fps must be positive")
            frames = defaultdict(list)
            for d in parsed.detections:
                frame = round(d.frame_number * fps / annotation_fps) if annotation_fps else d.frame_number
                frames[frame].append(d)
            for frame_number, detections in sorted(frames.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, image = cap.read()
                if not ok:
                    continue
                height, width = image.shape[:2]
                for d in detections:
                    try:
                        _box(d, width, height)
                    except ValueError:
                        continue
                    color = (0, 255, 0) if d.confidence >= low_confidence else (0, 0, 255)
                    cv2.rectangle(image, (round(d.bbox_left), round(d.bbox_top)),
                                  (round(d.bbox_right), round(d.bbox_bottom)), color, 2)
                    cv2.putText(image, f"{d.class_name or 'unknown'} {d.confidence:.2f}",
                                (round(d.bbox_left), max(12, round(d.bbox_top)-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                stem = hashlib.sha256(str(video).encode()).hexdigest()[:12]
                cv2.imwrite(str(output / f"{stem}_{frame_number:08d}.jpg"), image)
                saved += 1
                if saved >= limit:
                    return {"previews": saved, "directory": str(output)}
        finally:
            cap.release()
    return {"previews": saved, "directory": str(output)}
