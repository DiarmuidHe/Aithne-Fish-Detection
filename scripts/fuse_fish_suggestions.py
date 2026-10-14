"""Build an optional VIAME + Fishial CVAT candidate without changing the baseline.

Fishial-only boxes must persist across sampled frames before they become tracks.
The output is unreviewed suggestions, never training truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from app.services.external_datasets import sha256
from app.services.viame_parser import VIAMEDetection, parse_viame_csv

HEADER = ["track_id", "video_id", "frame", "tl_x", "tl_y", "br_x", "br_y",
          "confidence", "target_length", "class", "score"]


def _box(row: VIAMEDetection) -> tuple[float, float, float, float]:
    return row.bbox_left, row.bbox_top, row.bbox_right, row.bbox_bottom


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def _valid(row: VIAMEDetection) -> bool:
    return (all(math.isfinite(value) for value in (*_box(row), row.confidence))
            and row.bbox_left < row.bbox_right and row.bbox_top < row.bbox_bottom
            and 0 <= row.confidence <= 1)


def _aspect(row: VIAMEDetection) -> float:
    width = row.bbox_right-row.bbox_left
    height = row.bbox_bottom-row.bbox_top
    return max(width/height, height/width)


def _csv_row(row: VIAMEDetection, track_id: str) -> list:
    return [track_id, row.source, row.frame_number, *_box(row), row.confidence,
            row.target_length if row.target_length is not None else -1,
            row.class_name or "fish", row.class_confidence if row.class_confidence is not None
            else row.confidence]


def fuse(baseline_csv: Path, fishial_csv: Path, output: Path, *, baseline_offset: int = -1,
         min_score: float = .35, match_iou: float = .3, link_iou: float = .3,
         match_frame_window: int = 1, max_gap: int = 6, min_support: int = 2,
         max_aspect: float = 5.0) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= min_score <= 1 or not 0 < match_iou <= 1 or not 0 < link_iou <= 1:
        raise ValueError("Invalid score or IoU threshold")
    if match_frame_window < 0 or max_gap < 1 or min_support < 2 or max_aspect < 1:
        raise ValueError("Invalid frame window, gap or support")
    baseline_result = parse_viame_csv(baseline_csv, frame_number_offset=baseline_offset)
    baseline = [row for row in baseline_result.detections if _valid(row)]
    fishial_result = parse_viame_csv(fishial_csv, strict=True)
    fishial = fishial_result.detections
    if any(not _valid(row) for row in fishial):
        raise ValueError("Invalid Fishial geometry or confidence")
    baseline_frames: dict[int, list[VIAMEDetection]] = defaultdict(list)
    for row in baseline:
        baseline_frames[row.frame_number].append(row)
    unmatched = []
    matched = below_score = implausible_shape = 0
    for row in sorted(fishial, key=lambda x: (x.frame_number, -x.confidence)):
        if row.confidence < min_score:
            below_score += 1
            continue
        if _aspect(row) > max_aspect:
            implausible_shape += 1
            continue
        nearby = (candidate for frame in range(max(0, row.frame_number-match_frame_window),
                                              row.frame_number+match_frame_window+1)
                  for candidate in baseline_frames.get(frame, ()))
        if any(_iou(_box(row), _box(candidate)) >= match_iou for candidate in nearby):
            matched += 1
            continue
        unmatched.append(row)
    # Link only observations with geometric continuity. Never fabricate missing boxes.
    tracks: list[list[VIAMEDetection]] = []
    unmatched_frames: dict[int, list[VIAMEDetection]] = defaultdict(list)
    for row in unmatched:
        unmatched_frames[row.frame_number].append(row)
    for frame, rows in sorted(unmatched_frames.items()):
        edges = sorted(((-_iou(_box(track[-1]), _box(row)), track_index, row_index)
                        for track_index, track in enumerate(tracks)
                        if 0 < frame-track[-1].frame_number <= max_gap
                        for row_index, row in enumerate(rows)
                        if _iou(_box(track[-1]), _box(row)) >= link_iou))
        used_tracks, used_rows = set(), set()
        for _, track_index, row_index in edges:
            if track_index not in used_tracks and row_index not in used_rows:
                tracks[track_index].append(rows[row_index])
                used_tracks.add(track_index)
                used_rows.add(row_index)
        tracks.extend([[row] for index, row in enumerate(rows) if index not in used_rows])
    accepted = [track for track in tracks if len(track) >= min_support]
    output.mkdir(parents=True)
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for row in baseline:
            writer.writerow(_csv_row(row, f"viame-{row.track_id}"))
        for index, track in enumerate(accepted, 1):
            for row in track:
                writer.writerow(_csv_row(row, f"fishial-extra-{index}"))
    report = {
        "baseline_csv": str(baseline_csv), "baseline_sha256": sha256(baseline_csv),
        "fishial_csv": str(fishial_csv), "fishial_sha256": sha256(fishial_csv),
        "candidate_csv_sha256": sha256(output / "candidates.csv"),
        "settings": {"baseline_offset": baseline_offset, "min_score": min_score,
                     "match_iou": match_iou, "link_iou": link_iou,
                     "match_frame_window": match_frame_window, "max_gap": max_gap,
                     "min_support": min_support, "max_aspect": max_aspect},
        "baseline_rows": len(baseline), "baseline_skipped_rows": len(baseline_result.skipped_rows)
                         + len(baseline_result.detections)-len(baseline),
        "fishial_rows": len(fishial), "fishial_below_score": below_score,
        "fishial_implausible_shape": implausible_shape,
        "fishial_matched_to_viame": matched, "fishial_unmatched": len(unmatched),
        "fishial_added_tracks": len(accepted),
        "fishial_added_boxes": sum(map(len, accepted)),
        "fishial_unconfirmed_boxes": len(unmatched)-sum(map(len, accepted)),
        "review_status": "unreviewed candidate; no accuracy or time saving established",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("fishial_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-offset", type=int, default=-1)
    parser.add_argument("--min-score", type=float, default=.35)
    parser.add_argument("--match-iou", type=float, default=.3)
    parser.add_argument("--link-iou", type=float, default=.3)
    parser.add_argument("--match-frame-window", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=6)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-aspect", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(fuse(args.baseline_csv, args.fishial_csv, args.output,
                          baseline_offset=args.baseline_offset, min_score=args.min_score,
                          match_iou=args.match_iou, link_iou=args.link_iou,
                          match_frame_window=args.match_frame_window,
                          max_gap=args.max_gap, min_support=args.min_support,
                          max_aspect=args.max_aspect), indent=2))


if __name__ == "__main__":
    main()
