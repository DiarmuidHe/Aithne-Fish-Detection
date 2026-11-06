"""Part 5 of ``docs/smartbay-rolling-training-prompt.md``: automatic checks of a labelled
round, with no human gate (decided 2026-10-30).

On the instance, from the repository root, after ``run_labelling.sh``:

    PYTHONPATH=.labelling-python-packages /opt/pytorch/bin/python -m scripts.cloud_labelling.round_checks

The exclusion rules are recorded in the round journal as decision ``auto_exclusions``
before any window is measured. A window is excluded if a stage failed for it, it has more
than 150 label boxes in a frame (the prompt said 40; the user raised it on 2026-10-30 to
keep dense schools), static suppression removed more than half its boxes, or it is a
*strong* window with no labels (so likely-fish footage never teaches "empty"). *None* and
*murky* windows with labels are flagged but kept: the screening may have been wrong.

Writes ``checks/quality_report.json`` and ``checks/windows.csv`` (per window: label boxes,
tracks, the share removed as static, score percentiles, median box size, flags) and, for an
optional skim, ``checks/spot_check_<n>.jpg``: about 100 random labelled test-day frames with
SAM 3's labels drawn. Every number is agreement with SAM 3, never accuracy; the labels stay
unreviewed suggestions. Exits 1 when fewer than ``--min-labelled`` of the windows have labels,
so a broken run never goes on to training.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.cloud_labelling.run_windows import PROMPT, ROOT, choices
from scripts.sam3_botsort import Journal, read, read_rows, save

# The prompt's cap of 40 boxes in a frame caught real dense schools (74-94 boxes at peak in 4
# of the first 9 strong windows); the user raised it to 150 on 2026-10-30, which still catches
# runaway texture boxing and fits RF-DETR Nano's 300 queries.
RULES = {"max_label_boxes_in_a_frame": 150, "max_static_removed_share": .5,
         "strong_needs_labels": True, "stage_failed": True}
STAGE_ENTRIES = (f"sam3/{PROMPT}", f"static/sam3-{PROMPT}", f"labels/{PROMPT}")
SPOT_CHECK = {"frames": 100, "per_sheet": 20, "columns": 4, "tile": (480, 270), "seed": 17}


def labelled_as(row: dict, root: Path, entries: dict) -> str | None:
    """The window ID the labels are under: the window, or its reserve if that was cut."""
    for _, window_id, _, _ in choices(row):
        if (root / "windows" / f"{window_id}.json").exists() or f"{window_id}/sam3/{PROMPT}" in entries:
            return window_id
    return None


def percentile(values: list[float], q: float):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 4) if values else None


def measure(window_id: str | None, root: Path, entries: dict) -> dict:
    """A window's computed facts: stage status, label boxes and tracks, boxes per frame,
    scores, box sizes and the share static suppression removed."""
    if window_id is None:
        return {"stages": {name: "missing" for name in STAGE_ENTRIES}}
    stages = {name: entries.get(f"{window_id}/{name}", {}).get("status", "missing") for name in STAGE_ENTRIES}
    facts = {"stages": stages}
    directory = root / "videos" / window_id
    labels = directory / "labels" / PROMPT / "rows.jsonl"
    static = directory / "static" / f"sam3-{PROMPT}" / "report.json"
    if labels.exists():
        rows = read_rows(labels)
        per_frame = Counter(r["frame"] for r in rows)
        sides = [((r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])) ** .5 for r in rows]
        facts |= {"label_boxes": len(rows), "tracks": len({r["id"] for r in rows}),
                  "frames_with_labels": len(per_frame), "max_boxes_in_a_frame": max(per_frame.values(), default=0),
                  "score_p10": percentile([r["confidence"] for r in rows], .1),
                  "score_p50": percentile([r["confidence"] for r in rows], .5),
                  "score_p90": percentile([r["confidence"] for r in rows], .9),
                  "median_box_side_px": round(statistics.median(sides), 1) if sides else None}
    if static.exists():
        counts = read(static)["counts"]
        facts["static_removed_share"] = round(counts["removed_boxes"] / counts["boxes"], 4) if counts["boxes"] else 0.0
    return facts


def judge(row: dict, facts: dict, rules: dict = RULES) -> tuple[dict[str, str], list[str]]:
    """The window's exclusion reasons ({rule: detail}) and its keep-but-flag notes."""
    reasons, flags = {}, []
    if rules["stage_failed"] and any(status != "complete" for status in facts["stages"].values()):
        reasons["stage_failed"] = ", ".join(f"{name} {status}" for name, status in facts["stages"].items()
                                            if status != "complete")
    elif rules["stage_failed"] and ("label_boxes" not in facts or "static_removed_share" not in facts):
        reasons["stage_failed"] = "a stage is recorded complete but its output file is missing"
    labels = facts.get("label_boxes", 0)
    if facts.get("max_boxes_in_a_frame", 0) > rules["max_label_boxes_in_a_frame"]:
        reasons["max_label_boxes_in_a_frame"] = f"{facts['max_boxes_in_a_frame']} label boxes in one frame"
    if facts.get("static_removed_share", 0) > rules["max_static_removed_share"]:
        reasons["max_static_removed_share"] = f"static suppression removed {facts['static_removed_share']:.0%}"
    if rules["strong_needs_labels"] and row["evidence"] == "strong" and not labels and not reasons:
        reasons["strong_needs_labels"] = "strong window with no labels"
    if row["evidence"] in ("none", "murky") and labels:
        flags.append(f"{row['evidence']} window with {labels} label boxes (kept: the screening may be wrong)")
    return reasons, flags


def spot_check(windows: list[dict], root: Path, target: Path, settings: dict = SPOT_CHECK) -> list[str]:
    """Contact sheets of random labelled frames from kept test-day windows, labels drawn."""
    import cv2
    import numpy as np

    pool = []
    for window in windows:
        if window["role"] == "test" and not window["excluded"] and window.get("label_boxes"):
            rows = read_rows(root / "videos" / window["labelled_as"] / "labels" / PROMPT / "rows.jsonl")
            grouped = defaultdict(list)
            for r in rows:
                grouped[r["frame"]].append(r["box"])
            pool += [(window["labelled_as"], frame, boxes) for frame, boxes in sorted(grouped.items())]
    picks = sorted(random.Random(settings["seed"]).sample(pool, min(settings["frames"], len(pool))))
    width, height = settings["tile"]
    tiles = []
    for window_id, frame, boxes in picks:
        capture = cv2.VideoCapture(str(root / "windows" / f"{window_id}.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        capture.release()
        if not ok:
            continue
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 3)
        tile = cv2.resize(image, (width, height))
        cv2.rectangle(tile, (0, 0), (width, 22), (30, 30, 30), -1)
        cv2.putText(tile, f"{window_id.removeprefix('SmartBayCam1_')} f{frame} ({len(boxes)})", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    written = []
    for n, start in enumerate(range(0, len(tiles), settings["per_sheet"]), start=1):
        sheet = tiles[start:start + settings["per_sheet"]]
        while len(sheet) % settings["columns"]:
            sheet.append(np.zeros_like(sheet[0]))
        cols = settings["columns"]
        path = target / f"spot_check_{n}.jpg"
        cv2.imwrite(str(path), np.vstack([np.hstack(sheet[i:i + cols]) for i in range(0, len(sheet), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(path.as_posix())
    return written


def check(root: Path, to_label: Path, *, sheets: bool = True) -> dict:
    journal = Journal(root)
    if "auto_exclusions" not in journal.data["decisions"]:  # The rules come before any measurement
        journal.decide("auto_exclusions", {"rules": RULES, "label_boxes": "reliable SAM 3 labels after static "
                                           "suppression (labels/fish/rows.jsonl), the boxes a dataset would train on",
                                           "flag_only": "none and murky windows with labels are kept",
                                           "human_review": "none (decided 2026-10-30); an optional spot check only"})
    rules = journal.data["decisions"]["auto_exclusions"]["rules"]
    entries = journal.data["entries"]
    with to_label.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    windows = []
    for row in rows:
        window_id = labelled_as(row, root, entries)
        facts = measure(window_id, root, entries)
        reasons, flags = judge(row, facts, rules)
        windows.append({"window_id": row["window_id"], "labelled_as": window_id, "round": row.get("round"),
                        "day": row["day"], "role": row["role"], "evidence": row["evidence"], "light": row["light"],
                        "hour_bin": row["hour_bin"], **facts, "excluded": bool(reasons), "exclusion_reasons": reasons,
                        "flags": flags})
    labelled = sum(w["stages"][f"labels/{PROMPT}"] == "complete" for w in windows)
    target = root / "checks"
    target.mkdir(parents=True, exist_ok=True)
    kept = [w for w in windows if not w["excluded"]]
    report = {
        "rules": rules, "windows": len(windows), "labelled": labelled, "labelled_share": round(labelled / len(windows), 4),
        "excluded": len(windows) - len(kept), "kept": len(kept),
        "exclusion_reasons": dict(Counter(rule for w in windows for rule in w["exclusion_reasons"])),
        "flagged_kept": sum(bool(w["flags"]) for w in kept),
        "kept_by": {key: dict(Counter(w[key] for w in kept)) for key in ("role", "evidence", "light", "hour_bin")},
        "label_boxes_kept_by_evidence": {e: sum(w.get("label_boxes", 0) for w in kept if w["evidence"] == e)
                                         for e in ("strong", "weak", "none", "murky")},
        "note": "Computed flags only: agreement with SAM 3, never accuracy; labels are unreviewed suggestions.",
        "per_window": windows}
    if sheets:
        report["spot_check_sheets"] = spot_check(windows, root, target)
    save(target / "quality_report.json", report)
    columns = ["window_id", "labelled_as", "round", "day", "role", "evidence", "light", "hour_bin", "label_boxes",
               "tracks", "frames_with_labels", "max_boxes_in_a_frame", "static_removed_share", "score_p10",
               "score_p50", "score_p90", "median_box_side_px", "excluded", "exclusion_reasons", "flags"]
    with (target / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(w | {"exclusion_reasons": "; ".join(w["exclusion_reasons"].values()),
                              "flags": "; ".join(w["flags"])} for w in windows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--to-label", type=Path, help="Default <root>/selection/to_label.csv")
    parser.add_argument("--min-labelled", type=float, default=.9, help="Exit 1 below this share of labelled windows")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()
    report = check(args.root, args.to_label or args.root / "selection" / "to_label.csv", sheets=not args.no_sheets)
    print(json.dumps({k: v for k, v in report.items() if k != "per_window"}, indent=2))
    if report["labelled_share"] < args.min_labelled:
        print(f"only {report['labelled_share']:.0%} of the windows have labels (need {args.min_labelled:.0%})",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
