"""Hard negatives for a round's student: places it keeps boxing that never move and that SAM 3
did not find, such as debris, muck on the lens or a fixture. The runbook for a new instance is
``docs/smartbay-hard-negatives.md``. On the instance, from the repository root:

    /opt/pytorch/bin/python -m scripts.hard_negatives STAGE

Stages, in order:

- ``restore``: the round's cut windows from their public source clips. A new instance has no
  MP4s: only the labels and the records came with the bundle. Each source's SHA256 must match
  the round's record, so the frames are the ones SAM 3 labelled, whatever the encoder does.
- ``mine``: the round's student on every window of a training day, then the rule below on each.
  Validation and test windows are mined for the report only and never reach a dataset. Writes
  ``hard_negatives.json``, ``hard_negatives.csv`` and ``review/*.jpg``, then stops for a person
  to look. Add a zone's ID to ``review/rejected_zones.txt`` to leave it out.
- ``data``: round 1's dataset, unchanged, plus frames where the student boxed a hard-negative
  zone and SAM 3 did not. Its validation and test splits are identical to round 1's.
- ``train``: RF-DETR Nano from the COCO checkpoint for round 1's epoch count.
- ``eval``: ``round_eval`` on the same days, then both students compared in ``comparison.json``.

**The rule.** A zone is a box that recurs in one place: at the operating threshold, IoU >= 0.5,
in at least 5% of a window's frames and across at least half of its time. It is a hard negative
when SAM 3's final labels overlap it on at most 20% of the frames where the student fired there,
and the crops on those frames stay the same object (median NCC with their own median crop >= 0.9,
the static filter's background threshold). The final labels are the test, not SAM 3's raw boxes:
on the local clip SAM 3 fired on a stuck piece of debris on 39% of those frames, but too weakly or
briefly to ever become a label. The raw share is recorded for each zone.

**What "labelled as a hard negative" means in training.** RF-DETR has no such label: a region
with no box is already "not fish". So a hard negative is an image with no box on that spot, added
to give the student more practice there, and each image lists its ``hard_negative_boxes`` so it
can be traced. All of this is agreement with SAM 3's unreviewed labels, never accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from app.services.external_datasets import sha256
from app.services.static_filter import DEFAULT_SETTINGS, _pairwise_iou, background_ncc, static_zones
from scripts.cloud_labelling.run_windows import DownloadFailed, cut_command, download, frame_count
from scripts.sam3_autolabel import SAM3_LICENCE
from scripts.sam3_botsort import (
    OFFLINE,
    Journal,
    by_frame,
    file_id,
    pair_frames,
    read,
    read_rows,
    save,
    write_rows,
)
from scripts.student_detector import (
    FLOOR,
    LABELS,
    NAME,
    RESOLUTION,
    RFDETR_LICENCE,
    ROUND_STRIDE,
    SPLIT_DIRS,
    TRAIN,
    _train,
    add_coco_image,
    coco_counts,
    detect_video,
    extract_frames,
    load_student,
    raw_rows,
    round_coco,
    round_eval,
    round_windows,
    split_of,
    verify_rfdetr,
)

ROUND = Path("data/outputs/smartbay-rounds/round-1")
OUT = Path("data/outputs/smartbay-rounds/round-1-hn")
# Set from the validation windows and the two local clips, never from test windows. Loose on
# purpose: this also finds spots dense schools keep crossing, and the SAM 3 and stillness checks
# do the separating (on the local clip: no labels and stillness 0.98 on stuck debris, against
# stillness of at most 0.94, 0.67 at the median, on the 68 spots dense schools kept crossing).
ZONES = replace(DEFAULT_SETTINGS, zone_iou=.5, min_zone_frame_share=.05, min_zone_seconds=1.0, min_zone_bin_share=.5)
RULE = {"max_sam3_share": .2, "sam3_iou": .3, "sam3_covered": .5, "min_stillness": .9, "stillness_crops": 12,
        "zones": {k: getattr(ZONES, k) for k in ("zone_iou", "min_zone_frame_share", "min_zone_seconds",
                                                 "zone_time_bins", "min_zone_bin_share")}}
EXTRA_PER_ZONE = 45  # At most half a window's 90 images, so one spot can't crowd out the fish
TRAINING_BUDGET = 3.5 * 3600
MIN_FREE_GB = 25  # About 7 GB of windows and 9 GB of JPEGs
REVIEWED_VERDICTS = ("hard_negative", "moving_content", "not_checked")
UNSUPPORTED_SAMPLE = 12  # Boxes per window kept for the sheet of boxes on no zone
REJECTED = "review/rejected_zones.txt"


class RestoreFailed(RuntimeError):
    pass


# ------------------------------------------------------------------- rule --

def _as_arrays(rows: list[dict]):
    import numpy as np

    return (np.array([r["box"] for r in rows], dtype=float).reshape(-1, 4),
            np.array([r["frame"] for r in rows], dtype=int))


def _overlaps(box, boxes):
    """Which ``boxes`` overlap ``box``: IoU >= sam3_iou, or covering at least sam3_covered of
    it, so a larger box around the same object still counts as found."""
    import numpy as np

    w = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    h = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = w * h
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    union = area + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return (inter / np.maximum(union, 1e-9) >= RULE["sam3_iou"]) | (inter / area >= RULE["sam3_covered"])


def _firing(box, student) -> list[int]:
    """Frames where a student box sits on the zone, at the zone search's own IoU."""
    import numpy as np

    boxes, frames = student
    hit = _pairwise_iou(np.array(box, dtype=float)[None], boxes)[0] >= ZONES.zone_iou
    return sorted(set(frames[hit].tolist()))


def verdict(shares: dict, stillness: float | None) -> str:
    """A zone's verdict from SAM 3's labels on its firing frames and its stillness."""
    if shares["labels"] > RULE["max_sam3_share"]:
        return "sam3_labelled"
    if stillness is None:
        return "not_checked"
    return "hard_negative" if stillness >= RULE["min_stillness"] else "moving_content"


def zone_stillness(video: Path, box: list[float], firing: list[int]) -> float | None:
    """Median NCC of up to ``stillness_crops`` crops at the firing frames with their own median
    crop: the same object stays near 1, different fish passing the same spot do not. None for
    an unreadable video, a tiny box or flat crops."""
    import cv2
    import numpy as np

    wanted = set(firing[::max(1, len(firing) // RULE["stillness_crops"])][:RULE["stillness_crops"]])
    capture, index, crops = cv2.VideoCapture(str(video)), 0, []
    try:
        while wanted and index <= max(wanted) and capture.grab():
            if index in wanted:
                _, image = capture.retrieve()
                x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
                x2, y2 = min(image.shape[1], round(box[2])), min(image.shape[0], round(box[3]))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    return None
                crops.append(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))
            index += 1
    finally:
        capture.release()
    if len(crops) < 2:
        return None
    median = np.median(np.stack(crops), axis=0)
    scores = [s for s in (background_ncc(crop, median) for crop in crops) if s is not None]
    return float(np.median(scores)) if scores else None


def mine_window(window: dict, frames: list[dict], labels: list[dict], raw: list[dict], removed: list[dict],
                threshold: float, video: Path | None = None) -> dict:
    """Every zone of one window with its verdict, and the student's boxes with no SAM 3 box near
    them on their frame, per kind of zone they fall on. Those are the boxes that could be wrong:
    most boxes SAM 3's labels don't match are on the same fish, just drawn differently, so they
    are counted but not split. ``video`` None leaves stillness unchecked."""
    fps, analysed = window["fps"], list(range(window["frames"]))
    rows = raw_rows(frames, fps, threshold)
    zones = static_zones(rows, analysed, fps, ZONES)
    student = _as_arrays(rows)
    against = {"labels": _as_arrays(labels), "raw": _as_arrays(raw), "removed": _as_arrays(removed)}
    found = []
    for n, zone in enumerate(zones):
        firing = _firing(zone["box"], student)
        hits = {name: set(found_frames[_overlaps(zone["box"], boxes)].tolist())
                for name, (boxes, found_frames) in against.items()}
        shares = {name: round(len(hits[name] & set(firing)) / len(firing), 4) for name in hits}
        still = zone_stillness(video, zone["box"], firing) if video and shares["labels"] <= RULE["max_sam3_share"] else None
        found.append({"zone_id": f"{window['window_id']}#{n}", "window_id": window["window_id"],
                      "box": [round(v, 2) for v in zone["box"]], "frame_share": round(zone["frame_share"], 4),
                      "time_bin_share": zone["time_bin_share"], "firing_frames": len(firing),
                      "sam3_share_of_firing_frames": shares,
                      "stillness": None if still is None else round(still, 4), "verdict": verdict(shares, still),
                      "clean_frames": sorted(set(firing) - hits["labels"])})
    raw_on = by_frame(raw)
    unsupported = [r for r in rows if not (raw_on.get(r["frame"]) and _overlaps(r["box"], _as_arrays(raw_on[r["frame"]])[0]).any())]
    by_zone, off_zones = Counter(), unsupported
    if unsupported and found:  # Each box counts for the zone it overlaps most
        overlap = _pairwise_iou(_as_arrays(unsupported)[0], _as_arrays([{"box": z["box"], "frame": 0} for z in found])[0])
        best = overlap.argmax(axis=1)
        off_zones = [r for i, r in enumerate(unsupported) if overlap[i, best[i]] < ZONES.zone_iou]
        by_zone.update(found[best[i]]["verdict"] for i in range(len(unsupported)) if overlap[i, best[i]] >= ZONES.zone_iou)
    if off_zones:
        by_zone["no_zone"] = len(off_zones)
    sample = off_zones[::max(1, len(off_zones) // UNSUPPORTED_SAMPLE)][:UNSUPPORTED_SAMPLE]
    return {"window_id": window["window_id"], "threshold": threshold, "student_boxes": len(rows),
            "label_boxes": len(labels),
            "student_only_boxes": len(pair_frames(analysed, by_frame(rows), by_frame(labels))["left_unique"]),
            "unsupported_boxes": len(unsupported), "unsupported_by_zone": dict(by_zone),
            "unsupported_sample": [{k: r[k] for k in ("frame", "box", "confidence")} for r in sample], "zones": found}


# ----------------------------------------------------------------- windows --

def all_windows(root: Path) -> list[dict]:
    """The round's windows with their facts, also for excluded ones (``round_windows`` leaves
    those without): a window excluded on a training day has never been trained on."""
    windows = round_windows(root)
    for window in windows:
        facts, record = root / "videos" / window["window_id"] / "video.json", root / "windows" / f"{window['window_id']}.json"
        if window["excluded"] and facts.exists() and record.exists():
            file = read(facts)["file"]
            window |= {"start_frame": read(record)["start_frame"], "frames": file["decoded_frames"],
                       "width": file["width"], "height": file["height"], "video": file["path"],
                       "video_sha256": file["sha256"], "fps": file["fps"]}
    return windows


def use_of(window: dict, validation: set[str]) -> str | None:
    """What mining a window is for: ``train`` feeds a dataset; ``report`` only counts, either
    because the day is held out or because the frozen exclusions left the window out; None is
    never touched, including an excluded window on a test day."""
    if window["role"] == "train" and window["day"] not in validation:
        return "report_excluded" if window["excluded"] else "train"
    if window["excluded"]:
        return None
    return "report" if split_of(window["role"], window["day"], validation) else None


def restore_window(window: dict, row: dict, record: dict | None, *, fetch=download, run=subprocess.run,
                   frames=frame_count) -> dict:
    """Rebuild a window's cut from its public source. The SHA256 of the source must match the
    round's record; the cut's bytes may differ if the encoder did, and the frame count may not.
    The source and the partial cut use names of their own, as windows can share a clip."""
    target = Path(window["video"])
    target.parent.mkdir(parents=True, exist_ok=True)
    source, part = target.with_name(f"{target.stem}.source.mp4"), target.with_name(f"{target.stem}.part.mp4")
    if record:
        url, expected = record["source"]["url"], record["source"]["sha256"]
        command = list(record["cut"]["command"])
        command[command.index("-i") + 1], command[-1] = source.name, part.name
    else:  # Four round 0 windows have no record: their source can't be checked
        reserve = window["window_id"] != row["window_id"]
        url, expected = row["reserve_url"] if reserve else row["source_mp4_url"], None
        start = int(row["reserve_start_frame"] if reserve else row["start_frame"])
        command = cut_command(Path(source.name), Path(part.name), start, window["frames"])
    try:
        fetched = fetch(url, source)
        if expected and fetched["sha256"] != expected:
            raise RestoreFailed(f"{window['window_id']}: {url} changed since it was labelled "
                                f"({fetched['sha256']}, was {expected})")
        run(command, cwd=target.parent, check=True, capture_output=True, text=True)
        got = frames(part)
        if got != window["frames"]:
            raise RestoreFailed(f"{window['window_id']}: the cut has {got} frames, expected {window['frames']}")
        part.replace(target)
    except (DownloadFailed, subprocess.CalledProcessError, OSError, ValueError) as error:
        raise RestoreFailed(f"{window['window_id']}: {type(error).__name__}: {str(error)[-300:]}") from error
    finally:
        source.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    digest = sha256(target)
    return {"status": "cut", "sha256": digest, "original_sha256": window["video_sha256"],
            "byte_identical": digest == window["video_sha256"], "source_verified": bool(expected),
            "source_sha256": fetched["sha256"]}


def restore(root: Path = ROUND, out: Path = OUT, workers: int = 2, limit: int | None = None, **injected) -> dict:
    """Every cut a later stage reads, skipping those already in place. Progress is kept in
    ``restored.json``, so a rerun repeats only what failed."""
    journal, record_path = Journal(out, {"training": TRAINING_BUDGET}), out / "restored.json"
    validation = set(read(root / "journal.json")["decisions"]["validation_days"]["days"])
    windows = [w for w in all_windows(root) if use_of(w, validation) and "video" in w][:limit]
    with (root / "selection" / "to_label.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row_of = {r["window_id"]: r for r in rows} | {r["reserve_window_id"]: r for r in rows if r.get("reserve_window_id")}
    if shutil.disk_usage(out).free < MIN_FREE_GB * 2**30:
        raise SystemExit(f"Under {MIN_FREE_GB} GB free for the windows and dataset")
    state = read(record_path) if record_path.exists() else {"windows": {}}

    def one(window: dict) -> dict:
        target, done = Path(window["video"]), state["windows"].get(window["window_id"])
        if target.exists():
            digest = sha256(target)
            if digest == window["video_sha256"] or (done and digest == done["sha256"]):
                return done or {"status": "present", "sha256": digest, "original_sha256": window["video_sha256"],
                                "byte_identical": True, "source_verified": None}
        path = root / "windows" / f"{window['window_id']}.json"
        return restore_window(window, row_of[window["window_id"]], read(path) if path.exists() else None, **injected)

    def step(heartbeat):
        failures = {}
        with ThreadPoolExecutor(workers) as pool:
            jobs = {pool.submit(one, w): w["window_id"] for w in windows}
            for n, job in enumerate(as_completed(jobs), start=1):
                try:
                    state["windows"][jobs[job]] = job.result()
                except RestoreFailed as error:
                    failures[jobs[job]] = str(error)
                save(record_path, state)
                heartbeat(done=n, of=len(jobs), failed=len(failures))
        if failures:
            raise RestoreFailed(f"{len(failures)} windows failed: {json.dumps(failures)[:2000]}")
        restored = state["windows"].values()
        return {"windows": len(state["windows"]), "byte_identical": sum(bool(r["byte_identical"]) for r in restored),
                "source_not_verifiable": sorted(k for k, r in state["windows"].items() if r["source_verified"] is False)}

    return journal.run(f"restore/{limit or 'all'}", "cpu", {"windows": [w["window_id"] for w in windows]},
                       {"workers": workers}, step)


def restored_windows(root: Path, out: Path) -> dict:
    """Window ID to its cut's current SHA256, which ``restore`` recorded when it differs."""
    path = out / "restored.json"
    return {k: v["sha256"] for k, v in read(path)["windows"].items()} if path.exists() else {}


def sam3_rows(root: Path, window_id: str) -> tuple[list[dict], list[dict], list[dict]]:
    """SAM 3's labels, its raw boxes before any filter and those its static filter removed."""
    directory = root / "videos" / window_id
    raw = directory / "sam3" / LABELS / "frames.jsonl"
    removed = directory / "static" / f"sam3-{LABELS}" / "removed.jsonl"
    return (read_rows(directory / "labels" / LABELS / "rows.jsonl"),
            [{"frame": f["frame"], "box": b} for f in read_rows(raw) for b in f["boxes"]] if raw.exists() else [],
            read_rows(removed) if removed.exists() else [])


# ------------------------------------------------------------------- mine --

def _existing_frames(root: Path, entries: dict, window: dict, model_sha: str) -> Path | None:
    """The round's own inference for a window, when it came from this student."""
    entry = entries.get(f"round/student/infer/{window['window_id']}")
    path = root / "student" / NAME / window["window_id"] / "frames.jsonl"
    done = entry and entry["status"] == "complete" and entry["inputs"]["model"] == model_sha
    return path if done and path.exists() else None


def mine(root: Path = ROUND, out: Path = OUT, limit: int | None = None, existing_only: bool = False) -> dict:
    """The round's student on training-day windows, then the rule on every window it has seen.
    ``existing_only`` needs no GPU: only windows the round already ran the student on, with
    stillness where the video is present."""
    journal = Journal(out, {"training": TRAINING_BUDGET})
    source = read(root / "journal.json")
    model, threshold = source["decisions"]["student_model"], source["decisions"]["student_threshold"]["threshold"]
    validation = set(source["decisions"]["validation_days"]["days"])
    restored = restored_windows(root, out)
    windows = [w | {"use": use_of(w, validation)} for w in all_windows(root)]
    windows = [w for w in windows if w["use"] and "video" in w]
    results, loaded = {}, {}
    if limit is not None:
        windows = windows[:limit]

    def frames_of(window: dict) -> Path | None:
        found = _existing_frames(root, source["entries"], window, model["checkpoint"]["sha256"])
        if found or existing_only:
            return found
        target = out / "mined" / "frames" / f"{window['window_id']}.jsonl"

        def step(heartbeat):
            if "model" not in loaded:
                loaded["model"] = load_student(model["checkpoint"])
            frames, speed = detect_video(loaded["model"], Path(window["video"]), heartbeat=heartbeat)
            if len(frames) != window["frames"]:
                raise RuntimeError(f"The student saw {len(frames)} frames of {window['window_id']}, not {window['frames']}")
            write_rows(frames, target)
            return {"speed": speed, "frames": len(frames), "frames_file": file_id(target)}

        journal.run(f"mine/infer/{window['window_id']}", "gpu",
                    {"video": restored.get(window["window_id"], window["video_sha256"]),
                     "model": model["checkpoint"]["sha256"]}, {"floor": FLOOR, "resolution": RESOLUTION}, step)
        return target

    for n, window in enumerate(windows, start=1):
        frames_path = frames_of(window)
        if frames_path is None:
            continue
        video = Path(window["video"]) if Path(window["video"]).is_file() else None
        labels, raw, removed = sam3_rows(root, window["window_id"])
        target = out / "mined" / "zones" / f"{window['window_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        def step(heartbeat, window=window, labels=labels, raw=raw, removed=removed, frames_path=frames_path, video=video,
                 target=target):
            result = mine_window(window, read_rows(frames_path), labels, raw, removed, threshold, video)
            save(target, result | {key: window.get(key) for key in ("use", "day", "role", "evidence")})
            return {"zones": len(result["zones"]), "file": target.as_posix()}

        journal.run(f"mine/zones/{window['window_id']}", "cpu",
                    {"frames": file_id(frames_path)["sha256"], "video_seen": video is not None},
                    {"rule": RULE, "threshold": threshold}, step)
        results[window["window_id"]] = read(target)
        kinds = Counter(z["verdict"] for z in results[window["window_id"]]["zones"])
        print(f"{n}/{len(windows)} {window['window_id']} {window['use']}: {dict(kinds)}", flush=True)
    summary = report(out, results, threshold, model)
    sheets = review_sheets(out, results, {w["window_id"]: w for w in windows})
    return summary | {"review_sheets": sheets}


def report(out: Path, results: dict[str, dict], threshold: float, model: dict) -> dict:
    """``hard_negatives.json`` and ``.csv``: every zone that isn't simply fish SAM 3 labelled,
    and per use how many of the boxes SAM 3 has nothing near those zones explain."""
    per_use = defaultdict(lambda: defaultdict(Counter))
    rows = []
    for result in results.values():
        use = result["use"]
        counts = per_use[use]
        counts["windows"]["windows"] += 1
        counts["boxes"]["student"] += result["student_boxes"]
        counts["boxes"]["student_only"] += result["student_only_boxes"]
        counts["boxes"]["no_sam3_box_near"] += result["unsupported_boxes"]
        counts["no_sam3_box_near_by_zone"].update(result["unsupported_by_zone"])
        for zone in result["zones"]:
            counts["zones"][zone["verdict"]] += 1
            if zone["verdict"] != "sam3_labelled":
                shares = zone["sam3_share_of_firing_frames"]
                rows.append({"zone_id": zone["zone_id"], "use": use, "day": result["day"], "evidence": result["evidence"],
                             "verdict": zone["verdict"], "box": zone["box"], "frame_share": zone["frame_share"],
                             "time_bin_share": zone["time_bin_share"], "firing_frames": zone["firing_frames"],
                             "clean_frames": len(zone["clean_frames"]), "stillness": zone["stillness"],
                             "sam3_labels": shares["labels"], "sam3_raw": shares["raw"], "sam3_removed": shares["removed"]})
    summary = {"note": ("Student against SAM 3's unreviewed output, never accuracy. no_sam3_box_near counts boxes with no "
                        "SAM 3 box at all near them on their frame: some are real fish SAM 3 missed."),
               "threshold": threshold, "checkpoint": model["checkpoint"], "rule": RULE,
               "by_use": {use: {name: dict(counter) for name, counter in counts.items()} for use, counts in per_use.items()},
               "zones": rows, "licences": [RFDETR_LICENCE, SAM3_LICENCE]}
    save(out / "hard_negatives.json", summary)
    with (out / "hard_negatives.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["zone_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {k: summary[k] for k in ("threshold", "by_use")}


def _captioned(image, lines: list[str], scale: float = .55):
    """``image`` under a black band with ``lines`` of white text, so no caption hides a zone."""
    import cv2
    import numpy as np

    height = round(46 * scale * len(lines)) + 6
    band = np.zeros((height, image.shape[1], 3), np.uint8)
    for n, line in enumerate(lines):
        cv2.putText(band, line, (6, round(40 * scale) + round(46 * scale * n)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1)
    return np.vstack([band, image])


def _crop_around(image, box: list[float], size: tuple[int, int], colour: tuple[int, int, int]):
    """A view of ``box`` with some of its surroundings, drawn and scaled to ``size``."""
    import cv2

    x1, y1, x2, y2 = (round(v) for v in box)
    pad = max(60, x2 - x1, y2 - y1)
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    crop = image[top:y2 + pad, left:x2 + pad].copy()
    cv2.rectangle(crop, (x1 - left, y1 - top), (x2 - left, y2 - top), colour, 2)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def _read_frames(video: Path, wanted: set[int]) -> dict:
    """The ``wanted`` frames of a video, by frame number, in one sequential pass."""
    import cv2

    images, capture, index = {}, cv2.VideoCapture(str(video)), 0
    try:
        while wanted and index <= max(wanted) and capture.grab():
            if index in wanted:
                images[index] = capture.retrieve()[1]
            index += 1
    finally:
        capture.release()
    return images


def unsupported_sheet(out: Path, results: dict[str, dict], windows: dict[str, dict], limit: int = 96) -> str | None:
    """Crops of boxes SAM 3 had nothing near and no zone explains, first from the windows with the
    most of them. These are what mining can't reach: real fish SAM 3 missed, or drifting muck."""
    import cv2
    import numpy as np

    tiles = []
    for result in sorted(results.values(), key=lambda r: -r["unsupported_by_zone"].get("no_zone", 0)):
        video = Path(windows[result["window_id"]]["video"])
        wanted = by_frame(result["unsupported_sample"])
        if not wanted or not video.is_file() or len(tiles) >= limit:
            continue
        images = _read_frames(video, set(wanted))
        for frame, rows in wanted.items():
            for row in rows:
                caption = f"{result['window_id'][13:28]} {result['evidence']} f{frame} {row['confidence']:.2f}"
                tiles.append(_captioned(_crop_around(images[frame], row["box"], (240, 180), (0, 165, 255)), [caption], .34))
    if not tiles:
        return None
    tiles = tiles[:limit]
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    path = out / "review" / "no_zone_boxes.jpg"
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path.as_posix()


def review_sheets(out: Path, results: dict[str, dict], windows: dict[str, dict], per_sheet: int = 8) -> list[str]:
    """One row per zone for a person to look at: the whole frame, then the zone close up on three
    frames from across its time. Then a sheet of boxes on no zone. A zone whose video can't be read
    is skipped."""
    import cv2
    import numpy as np

    size, written = (360, 270), []
    (out / "review").mkdir(parents=True, exist_ok=True)
    for name in REVIEWED_VERDICTS:
        strips = []
        for result in results.values():
            video = Path(windows[result["window_id"]]["video"])
            for zone in (z for z in result["zones"] if z["verdict"] == name):
                if not video.is_file():
                    continue
                frames = zone["clean_frames"] or [0]
                chosen = [frames[0], frames[len(frames) // 2], frames[-1]]
                images = _read_frames(video, set(chosen))
                whole = images[chosen[1]].copy()
                x1, y1, x2, y2 = (round(v) for v in zone["box"])
                cv2.rectangle(whole, (x1, y1), (x2, y2), (0, 255, 255), 4)
                tiles = [cv2.resize(whole, (480, 270), interpolation=cv2.INTER_AREA)]
                tiles += [_crop_around(images[f], zone["box"], size, (0, 255, 255)) for f in chosen]
                shares = zone["sam3_share_of_firing_frames"]
                strips.append(_captioned(np.hstack(tiles), [
                    f"{zone['zone_id']}  {result['use']}  {result['evidence']}  {name}",
                    f"student on {zone['frame_share']:.0%} of frames; on those, SAM 3 labels {shares['labels']:.0%}, "
                    f"raw {shares['raw']:.0%}; stillness {zone['stillness']}; close-ups at frames {chosen}"]))
        for n in range(0, len(strips), per_sheet):
            path = out / "review" / f"{name}_{n // per_sheet + 1}.jpg"
            cv2.imwrite(str(path), np.vstack(strips[n:n + per_sheet]), [cv2.IMWRITE_JPEG_QUALITY, 85])
            written.append(path.as_posix())
    sheet = unsupported_sheet(out, results, windows)
    return written + ([sheet] if sheet else [])


# ------------------------------------------------------------------- data --

def chosen_zones(out: Path, use_excluded: bool) -> tuple[list[dict], list[str]]:
    """The zones a dataset may use: a training window's hard negatives, minus any the reviewer
    rejected. A rejected ID that matches nothing stops the build, as it is probably a typo."""
    uses = {"train", *(["report_excluded"] if use_excluded else [])}
    rejected_path = out / REJECTED
    rejected = ({line.strip() for line in rejected_path.read_text(encoding="utf-8").splitlines() if line.strip()}
                if rejected_path.exists() else set())
    zones, seen = [], set()
    for path in sorted((out / "mined" / "zones").glob("*.json")):
        result = read(path)
        seen |= {z["zone_id"] for z in result["zones"]}
        if result["use"] in uses:
            zones += [z for z in result["zones"] if z["verdict"] == "hard_negative" and z["zone_id"] not in rejected]
    if rejected - seen:
        raise SystemExit(f"{rejected_path} names unknown zones: {sorted(rejected - seen)}")
    return zones, sorted(rejected)


def add_hard_negatives(coco: dict[str, dict], windows: dict[str, dict], labels: dict[str, list[dict]], zones: list[dict],
                       stride: int = ROUND_STRIDE, limit: int = EXTRA_PER_ZONE) -> dict:
    """Add each zone's frames to the training split. Only frames where the student fired and no
    SAM 3 label touched the zone qualify, so no label is ever contradicted. Frames already in the
    dataset are marked; up to ``limit`` others, evenly spaced, are added with the frame's labels."""
    train = coco["train"]
    existing = {(i["window_id"], i["frame"]): i for i in train["images"]}
    added, marked = Counter(), Counter()
    for zone in sorted(zones, key=lambda z: z["zone_id"]):
        window = windows[zone["window_id"]]
        clean = zone["clean_frames"]
        known = [f for f in clean if (zone["window_id"], f) in existing]
        new = [f for f in clean if (zone["window_id"], f) not in existing and f % stride]
        if window["excluded"]:  # Its grid frames aren't in the dataset either
            new = [f for f in clean if (zone["window_id"], f) not in existing]
        for frame in known:
            image = existing[(zone["window_id"], frame)]
            image.setdefault("hard_negative_boxes", []).append(zone["box"])
            marked[zone["zone_id"]] += 1
        grouped = by_frame(labels.get(zone["window_id"], []))
        for frame in new[::max(1, len(new) // limit)][:limit]:
            add_coco_image(train, window, frame, grouped.get(frame, []), hard_negative_boxes=[zone["box"]])
            existing[(zone["window_id"], frame)] = train["images"][-1]
            added[zone["zone_id"]] += 1
    return {"zones": len(zones), "images_added": sum(added.values()), "images_marked": sum(marked.values()),
            "per_zone": {z["zone_id"]: {"added": added[z["zone_id"]], "already_in_dataset": marked[z["zone_id"]]}
                         for z in zones}}


def build(root: Path = ROUND, out: Path = OUT, approval: str | None = None, use_excluded: bool = False) -> dict:
    """Round 1's dataset plus the hard negatives. Refuses until a person's approval of the review
    sheets is recorded, and if the base differs from what round 1 trained on."""
    journal = Journal(out, {"training": TRAINING_BUDGET})
    if not approval and "hard_negatives_approved" not in journal.data["decisions"]:
        raise SystemExit("Look at review/*.jpg first, then pass --approval with the reviewer's words")
    if approval:
        journal.decide("hard_negatives_approved", {"words": approval, "use_excluded": use_excluded})
    source = read(root / "journal.json")
    validation = set(source["decisions"]["validation_days"]["days"])
    restored = restored_windows(root, out)
    windows = {w["window_id"]: w | {"video_sha256": restored.get(w["window_id"], w.get("video_sha256"))}
               for w in all_windows(root)}
    zones, rejected = chosen_zones(out, use_excluded)
    if not zones:
        raise SystemExit("No hard negatives to add, so training would only repeat round 1; nothing was built")
    target = out / "dataset"
    inputs = {"zones": [z["zone_id"] for z in zones], "rejected": rejected,
              "round_dataset": file_id(root / "student" / "round_dataset.json")["sha256"]}

    def step(heartbeat):
        wanted_windows = {w["window_id"] for w in windows.values()
                          if not w["excluded"] and split_of(w["role"], w["day"], validation)} | {z["window_id"] for z in zones}
        labels = {k: read_rows(root / "videos" / k / "labels" / LABELS / "rows.jsonl") for k in wanted_windows}
        coco = round_coco(list(windows.values()), labels, validation)
        base = read(root / "student" / "round_dataset.json")
        if coco_counts(coco) != base["counts"]:
            raise RuntimeError(f"The base dataset differs from {root}'s: {coco_counts(coco)} against {base['counts']}")
        added = add_hard_negatives(coco, windows, labels, zones)
        for name, data in coco.items():
            folder = target / SPLIT_DIRS[name]
            folder.mkdir(parents=True, exist_ok=True)
            frames = defaultdict(dict)
            for image in data["images"]:
                frames[image["window_id"]][image["frame"]] = image["file_name"]
            for n, (key, wanted) in enumerate(frames.items(), start=1):
                extract_frames(windows[key], wanted, folder)
                heartbeat(split=name, window=n, of=len(frames))
            save(folder / "_annotations.coco.json", data)
        result = {"dataset": target.as_posix(), "counts": coco_counts(coco), "base_counts": base["counts"],
                  "stride": ROUND_STRIDE, "validation_days": sorted(validation), "hard_negatives": added,
                  "windows": [{k: window.get(k) for k in ("window_id", "planned_window_id", "day", "role", "evidence",
                                                          "round", "excluded", "video", "video_sha256", "fps", "frames")}
                              | {"split": None if window["excluded"] else split_of(window["role"], window["day"], validation)}
                              for window in windows.values()],
                  "review_status": "unreviewed SAM 3 suggestions; never ground truth", "licence": SAM3_LICENCE}
        save(out / "student" / "round_dataset.json", result)
        return result

    (out / "student").mkdir(parents=True, exist_ok=True)
    return journal.run("round/student/data", "cpu", inputs, {"extra_per_zone": EXTRA_PER_ZONE, "jpeg_quality": 92}, step)


# ------------------------------------------------------------- train, eval --

def train(root: Path = ROUND, out: Path = OUT) -> dict:
    """RF-DETR Nano from its COCO checkpoint, as round 1 was and for the same number of epochs,
    so the two students differ only in the hard negatives."""
    journal = Journal(out, {"training": TRAINING_BUDGET})
    data = read(out / "student" / "round_dataset.json")
    epochs = read(root / "journal.json")["entries"]["round/student/train"]["result"]["epochs_run"]
    models = out / "models" / f"{NAME}-{out.name}"
    inputs = {"dataset": data["counts"], "checkpoint": verify_rfdetr()["sha256"]}

    def step(heartbeat):
        result = _train(Path(data["dataset"]), models / "train", epochs, journal.remaining("training") - 120, heartbeat)
        if result["best"] is None:
            raise RuntimeError(f"Training finished without {models / 'train' / 'checkpoint_best_total.pth'}")
        return result | {"licences": [RFDETR_LICENCE, SAM3_LICENCE]}

    result = journal.run("round/student/train", "training", inputs,
                         TRAIN | {"resolution": RESOLUTION, "variant": "nano", "epochs": epochs}, step)
    journal.decide("student_model", {"checkpoint": result["best"], "trained_on": f"round {out.name}", "name": NAME,
                                     "resolution": RESOLUTION, "epochs_run": result["epochs_run"],
                                     "dataset": data["counts"], "licences": [RFDETR_LICENCE, SAM3_LICENCE]})
    return result


def zone_totals(root: Path, windows: list[dict], frames_of, threshold: float) -> dict:
    """Hard-negative zones found by applying the rule to one student's own boxes on ``windows``.
    Each student is mined on its own: counting boxes on the first student's zones would flatter
    the second, as those spots were picked for being where the first went wrong."""
    totals = Counter()
    for window in windows:
        video = Path(window["video"])
        result = mine_window(window, read_rows(frames_of(window["window_id"])), *sam3_rows(root, window["window_id"]),
                             threshold, video if video.is_file() else None)
        totals["windows"] += 1
        totals["frames"] += window["frames"]
        totals["student_boxes"] += result["student_boxes"]
        totals["student_only_boxes"] += result["student_only_boxes"]
        totals["hard_negative_zones"] += sum(z["verdict"] == "hard_negative" for z in result["zones"])
        totals["no_sam3_box_near"] += result["unsupported_boxes"]
        totals["no_sam3_box_near_on_hard_negative_zones"] += result["unsupported_by_zone"].get("hard_negative", 0)
    return dict(totals)


def evaluate(root: Path = ROUND, out: Path = OUT) -> dict:
    """``round_eval`` on the same validation and test windows, with the threshold again chosen
    on validation only, then both students side by side."""
    after = round_eval(out, labels_root=root)
    before = read(root / "student" / "round_eval.json")
    windows = read(out / "student" / "round_dataset.json")["windows"]
    thresholds = {"before": before["threshold"], "after": after["threshold"]}
    frames = {"before": lambda key: root / "student" / NAME / key / "frames.jsonl",
              "after": lambda key: out / "student" / NAME / key / "frames.jsonl"}
    static = {split: {side: zone_totals(root, [w for w in windows if w["split"] == split], frames[side], thresholds[side])
                      for side in frames} for split in ("validation", "test")}
    result = {"note": "Agreement with SAM 3's unreviewed labels on days neither student trained on, never accuracy.",
              "thresholds": thresholds,
              "agreement": {"before": before["agreement"], "after": after["agreement"]},
              "empty_water_proxy": {"before": before["empty_water_proxy"], "after": after["empty_water_proxy"]},
              "static_zones_mined_from_each_student": static}
    save(out / "comparison.json", result)
    return result


# -------------------------------------------------------------------- CLI --

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["restore", "mine", "data", "train", "eval"])
    parser.add_argument("--root", type=Path, default=ROUND, help="The round whose student and labels are used")
    parser.add_argument("--out", type=Path, default=OUT, help="Folder for everything this writes")
    parser.add_argument("--limit", type=int, help="restore, mine: only this many windows, as a quick check")
    parser.add_argument("--workers", type=int, default=2, help="restore: cuts at a time")
    parser.add_argument("--existing-only", action="store_true",
                        help="mine: only windows the round already ran the student on; needs no GPU")
    parser.add_argument("--approval", help="data: the reviewer's words approving review/*.jpg, kept in the journal")
    parser.add_argument("--use-excluded", action="store_true",
                        help="data: also use windows the frozen exclusions left out, on training days")
    args = parser.parse_args()
    if args.stage != "restore":
        os.environ.update(OFFLINE)  # Only restore needs the network
    if args.stage == "restore":
        result = restore(args.root, args.out, args.workers, args.limit)
    elif args.stage == "mine":
        result = mine(args.root, args.out, args.limit, args.existing_only)
    elif args.stage == "data":
        result = build(args.root, args.out, args.approval, args.use_excluded)
    elif args.stage == "train":
        result = train(args.root, args.out)
    else:
        result = evaluate(args.root, args.out)
    print(json.dumps(result, indent=2, default=str)[:20000])


if __name__ == "__main__":
    main()
