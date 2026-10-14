"""Bounded human review of CVAT video annotations and exportable truth."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import uuid
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from app.services.detection_metrics import iou

LATEST_PILOT_VIDEO = uuid.UUID("85ea19d4-db7b-4ed7-90a0-a8acccba2b0c")


def proposal_regions(video_id: uuid.UUID, fps: float, frame_count: int) -> list[dict]:
    """A short review budget: distributed frames and two complete track windows."""
    if fps <= 0 or frame_count <= 0:
        return []
    duration = frame_count / fps
    if video_id == LATEST_PILOT_VIDEO:
        singles = list(range(0, 120, 10))
        windows = [(48, 51), (88, 91)]
    else:
        singles = [round(i * max(0, duration - 1) / 11) for i in range(12)]
        windows = [(round(duration * .42), round(duration * .42) + 3),
                   (round(duration * .77), round(duration * .77) + 3)]
    result = []
    for second in singles:
        frame = min(frame_count - 1, round(second * fps))
        result.append({"start_frame": frame, "end_frame": frame, "kind": "frame",
                       "start_seconds": frame / fps, "end_seconds": frame / fps})
    for start, end in windows:
        first = min(frame_count - 1, round(start * fps))
        last = min(frame_count - 1, round(end * fps) - 1)
        if last > first:
            result.append({"start_frame": first, "end_frame": last, "kind": "track_window",
                           "start_seconds": first / fps, "end_seconds": last / fps})
    return result


def candidate_audit(track_csv: Path, detector_csv: Path | None, fps: float,
                    *, frame_offset: int = -1, limit: int = 12) -> dict:
    """Rank review candidates; neither proxy is a verified error."""
    if fps <= 0:
        raise ValueError("source fps must be positive")
    tracks: dict[int, list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
    track_frames: dict[str, set[int]] = defaultdict(set)
    with track_csv.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(line for line in file if not line.startswith("#")):
            try:
                frame = int(row[2]) + frame_offset
                box = tuple(float(value) for value in row[3:7])
                if frame >= 0 and _valid_box(box):
                    tracks[frame].append((row[0], box))
                    track_frames[row[0]].add(frame)
            except (ValueError, IndexError):
                continue
    overlap = defaultdict(int)
    for frame, boxes in tracks.items():
        for index, (left_id, left) in enumerate(boxes):
            for right_id, right in boxes[index + 1:]:
                if left_id != right_id and iou(left, right) >= .5:
                    overlap[frame] += 1
    gaps = Counter(
        later-earlier for ordered in (sorted(frames) for frames in track_frames.values())
        for earlier, later in pairwise(ordered)
        if 0 < later-earlier <= 30
    )
    step = gaps.most_common(1)[0][0] if gaps else 1
    long_gaps = defaultdict(int)
    for frames in track_frames.values():
        ordered = sorted(frames)
        for earlier, later in pairwise(ordered):
            if later-earlier > 2*step:
                long_gaps[earlier+step] += 1
    unmatched = defaultdict(int)
    if detector_csv and detector_csv.is_file():
        with detector_csv.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.reader(line for line in file if not line.startswith("#")):
                try:
                    timestamp = _timestamp_seconds(row[1])
                    frame = round(timestamp * fps)
                    box = tuple(float(value) for value in row[3:7])
                    score = float(row[7])
                except (ValueError, IndexError):
                    continue
                if score < .35 or not _valid_box(box):
                    continue
                nearby = [candidate for number in (frame - 1, frame, frame + 1)
                          for _, candidate in tracks.get(number, ())]
                if not any(iou(box, candidate) >= .3 for candidate in nearby):
                    unmatched[frame] += 1
    ranked = sorted(set(overlap) | set(unmatched) | set(long_gaps),
                    key=lambda frame: (-(3 * overlap.get(frame, 0) + unmatched.get(frame, 0)
                                         + 3 * long_gaps.get(frame, 0)), frame))
    return {"overlap_frames": len(overlap), "unmatched_candidate_frames": len(unmatched),
            "long_gap_frames": len(long_gaps),
            "hotspots": [{"frame": frame, "seconds": round(frame / fps, 2),
                          "overlapping_ids": overlap.get(frame, 0),
                          "unmatched_detector_boxes": unmatched.get(frame, 0),
                          "long_track_gaps": long_gaps.get(frame, 0)}
                         for frame in ranked[:limit]]}


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _valid_box(box) -> bool:
    return len(box) == 4 and all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3]


def corrected_rows(annotations, first: int, last: int, width: int, height: int,
                   fish_label_ids: set[int]) -> dict[int, list[dict]]:
    """Interpolate CVAT keyframes only within explicit, human-reviewed bounds."""
    by_frame: dict[int, list[dict]] = {frame: [] for frame in range(first, last + 1)}
    for track in getattr(annotations, "tracks", ()):
        if int(track.label_id) not in fish_label_ids:
            continue
        track_id = getattr(track, "id", None)
        if track_id is None:
            raise ValueError("CVAT track has no persisted ID")
        shapes = sorted((shape for shape in track.shapes if str(shape.type).lower().endswith("rectangle")),
                        key=lambda shape: shape.frame)
        for index, shape in enumerate(shapes):
            if shape.outside:
                continue
            following = shapes[index + 1] if index + 1 < len(shapes) else None
            stop = min(last, following.frame - 1 if following else last)
            for frame in range(max(first, shape.frame), stop + 1):
                ratio = ((frame - shape.frame) / (following.frame - shape.frame)
                         if following and not following.outside and following.frame > shape.frame else 0)
                points = [float(a) + ratio * (float(b) - float(a))
                          for a, b in zip(shape.points, following.points if following else shape.points)]
                if not _valid_box(points) or not (0 <= points[0] < points[2] <= width
                                                  and 0 <= points[1] < points[3] <= height):
                    raise ValueError(f"Invalid corrected CVAT box at frame {frame}")
                by_frame[frame].append({"track_id": int(track_id), "box": points})
    for shape in getattr(annotations, "shapes", ()):
        if int(shape.label_id) not in fish_label_ids or not first <= shape.frame <= last:
            continue
        if not str(shape.type).lower().endswith("rectangle"):
            continue
        points = [float(value) for value in shape.points]
        if not _valid_box(points) or not (0 <= points[0] < points[2] <= width
                                          and 0 <= points[1] < points[3] <= height):
            raise ValueError(f"Invalid corrected CVAT shape at frame {shape.frame}")
        by_frame[shape.frame].append({"track_id": None, "box": points})
    for rows in by_frame.values():
        rows.sort(key=lambda row: (row["track_id"] is None, row["track_id"] or -1, row["box"]))
    return by_frame


def annotation_sha256(rows: dict[int, list[dict]], species: dict[int, str] | None = None) -> str:
    payload: object = rows
    if species is not None:
        track_ids = {row["track_id"] for entries in rows.values() for row in entries
                     if row["track_id"] is not None}
        payload = {"rows": rows, "species": {str(track_id): species.get(track_id)
                                              for track_id in sorted(track_ids)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fish_label_ids(task) -> set[int]:
    ids = {int(label.id) for label in task.get_labels() if label.name.lower() == "fish"}
    if not ids:
        raise ValueError("CVAT task has no fish label")
    return ids


def region_rows(task, first: int, last: int, width: int, height: int) -> dict[int, list[dict]]:
    return corrected_rows(task.get_annotations(), first, last, width, height, fish_label_ids(task))
