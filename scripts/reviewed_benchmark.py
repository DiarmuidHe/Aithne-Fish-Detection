"""Score VIAME CSV candidates only on certified CVAT source frames.

Candidates JSON:
  {"candidates":[{"name":"baseline","videos":{"<video UUID>":{"csv":"...csv",
    "frame_offset":-1}},"review_minutes":28.0,"processing_seconds":330.0}]}
The first candidate is the unchanged baseline. Missing review minutes prevent promotion.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from app.services.detection_metrics import count_matches, iou
from app.services.viame_parser import parse_viame_csv


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _truth(dataset: Path, metadata: dict) -> list[tuple[int, tuple]]:
    image = cv2.imread(str(dataset / metadata["image"]))
    if image is None:
        raise RuntimeError(f"Missing reviewed image {metadata['image']}")
    height, width = image.shape[:2]
    label = dataset / "labels" / metadata["split"] / f"{Path(metadata['image']).stem}.txt"
    truth = []
    for line in label.read_text(encoding="utf-8").splitlines():
        cls, x, y, w, h = map(float, line.split())
        truth.append((int(cls), ((x-w/2)*width, (y-h/2)*height,
                                 (x+w/2)*width, (y+h/2)*height)))
    return truth


def _switches(track_truth: list[dict], predicted: dict[tuple[str, int], list[tuple]]) -> int:
    observations = defaultdict(list)
    for frame in track_truth:
        key = (frame["video_id"], frame["frame"])
        for fish in frame["tracks"]:
            if fish["track_id"] is None:
                continue
            best = max(((iou(fish["box"], box), track_id)
                        for track_id, box, _ in predicted.get(key, [])), default=(0, None))
            observations[(frame["video_id"], fish["track_id"])].append(
                (frame["frame"], best[1] if best[0] >= .5 else None))
    switches = 0
    for series in observations.values():
        series.sort()
        for (previous_frame, previous_id), (frame, current_id) in zip(series, series[1:]):
            if frame == previous_frame + 1 and previous_id is not None and current_id is not None:
                switches += previous_id != current_id
    return switches


def benchmark(dataset: Path, candidates_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    reviewed = _jsonl(dataset / "metadata.jsonl")
    identity = _jsonl(dataset / "tracks_truth.jsonl")
    if not reviewed or not all(row.get("reviewed") for row in reviewed):
        raise ValueError("A nonempty, fully reviewed dataset is required")
    spec = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = spec["candidates"]
    if not candidates or candidates[0]["name"] != "baseline":
        raise ValueError("The first candidate must be the unchanged baseline")
    report = {"dataset": str(dataset), "candidates": []}
    reviewed_videos = {row["video_id"] for row in reviewed}
    for candidate in candidates:
        if set(candidate["videos"]) != reviewed_videos:
            raise ValueError(f"{candidate['name']} must provide a CSV for every reviewed video")
        predicted: dict[tuple[str, int], list[tuple]] = defaultdict(list)
        for video_id, source in candidate["videos"].items():
            parsed = parse_viame_csv(Path(source["csv"]), strict=True,
                                     frame_number_offset=int(source.get("frame_offset", 0)))
            for row in parsed.detections:
                predicted[(video_id, row.frame_number)].append((
                    row.track_id, (row.bbox_left, row.bbox_top, row.bbox_right,
                                   row.bbox_bottom), row.confidence))
        counts = {"tp": 0, "fp": 0, "fn": 0}
        test_counts = {"tp": 0, "fp": 0, "fn": 0}
        per_video = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for row in reviewed:
            key = (row["video_id"], row["frame"])
            actual = _truth(dataset, row)
            boxes = [(0, box, score) for _, box, score in predicted.get(key, [])]
            found = count_matches(actual, boxes)
            for name, value in found.items():
                counts[name] += value
                per_video[row["video_id"]][name] += value
                if row["split"] == "test":
                    test_counts[name] += value
        counts["recall"] = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        counts["precision"] = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        test_counts["recall"] = test_counts["tp"] / max(1, test_counts["tp"] + test_counts["fn"])
        test_counts["precision"] = test_counts["tp"] / max(1, test_counts["tp"] + test_counts["fp"])
        test_identity = [row for row in identity if row["split"] == "test"]
        report["candidates"].append({
            "name": candidate["name"], "counts_at_iou50": counts,
            "test_counts_at_iou50": test_counts,
            "per_video": dict(per_video),
            "id_switches_on_checked_consecutive_frames": _switches(identity, predicted),
            "test_id_switches": _switches(test_identity, predicted),
            "review_minutes": candidate.get("review_minutes"),
            "processing_seconds": candidate.get("processing_seconds"),
        })
    baseline = report["candidates"][0]
    test_groups = {row["group"] for row in reviewed if row["split"] == "test"}
    all_groups = {row["group"] for row in reviewed}
    for candidate in report["candidates"][1:]:
        baseline_time = baseline["review_minutes"]
        candidate_time = candidate["review_minutes"]
        candidate["promotion_ready"] = bool(
            len(all_groups) >= 6 and test_groups
            and candidate["test_counts_at_iou50"]["tp"] +
                candidate["test_counts_at_iou50"]["fn"] > 0
            and candidate["test_counts_at_iou50"]["recall"] -
                baseline["test_counts_at_iou50"]["recall"] >= .05
            and baseline_time is not None and candidate_time is not None
            and candidate_time <= .8 * baseline_time
            and candidate["test_id_switches"] <= baseline["test_id_switches"]
        )
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.dataset, args.candidates, args.output), indent=2))
