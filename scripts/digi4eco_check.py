"""Score round students against people's boxes on DIGI4ECO's SmartBayCam1 sequence (DS2).

DS2 is one hand-labelled SmartBayCam1 clip (CVAT 1.1 video tracks, one class, Fish) from the
DIGI4ECO Underwater Fish Tracking Dataset, Zenodo record 20701225, CC BY 4.0 (cite
González-Villà et al. 2026, doi:10.5281/zenodo.20701225). It is a check set only: never
trained on, never used to tune a threshold or a setting.

PowerShell, from the repository root:

    $env:FISHIAL_PREPROCESS='none'
    .venv\\Scripts\\python.exe -m scripts.digi4eco_check all

Stages: ``inspect`` (labels, video facts, alignment sheet), ``detect`` (each student on every
frame), ``score`` (precision, recall and F1 against people's boxes), ``track`` (BoT-SORT,
IDF1 and MOTA against people's tracks), ``render`` (comparison MP4), ``audit`` (numbered
contact sheets of model boxes with no human box, for a person to mark), ``audit-score``
(precision adjusted by those marks). Each student keeps the threshold its round chose on
its own validation days.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import stat
import xml.etree.ElementTree as ElementTree
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DATASET = Path("data/external/digi4eco-20701225/DIGI4ECO_tracking_dataset")
SEQUENCE = "SmartBay2"
ZIP = {"path": "data/external/digi4eco-20701225-zip/DIGI4ECO_tracking_dataset.zip", "bytes": 138_667_050,
       "md5": "e91d8df3e9f6a18527bcb854f98df962",
       "sha256": "037c709c2ce059277c669430b83e9b1fad521c98773be90adc874c91cd0e910d"}
OUT = Path("data/outputs/digi4eco-check-20261115")
MODELS = {"round-2": Path("data/outputs/smartbay-rounds/round-2-train"),
          "round-1": Path("data/outputs/smartbay-rounds/round-1/cloud")}
IOU, IOU_LOOSE = .5, .3
SIZE_BINS = [(0, 16), (16, 32), (32, 64), (64, math.inf)]  # Short side of people's box, source px
EDGE_PX = 20
SWEEP = [round(.05 * i, 2) for i in range(2, 17)]  # Descriptive only: the threshold never moves
AUDIT_PER_MODEL, AUDIT_SEED, AUDIT_LETTERS = 60, 17, {"round-2": "A", "round-1": "B"}
LICENCES = ["DIGI4ECO Underwater Fish Tracking Dataset: CC BY 4.0, González-Villà et al. (2026), Zenodo, "
            "doi:10.5281/zenodo.20701225",
            "SmartBay footage: CC BY 4.0, Marine Institute",
            "RF-DETR 1.11.2 and its COCO weights: Apache-2.0",
            "The students: Meta SAM License (trained on SAM 3's labels)",
            "BoT-SORT (Ultralytics): AGPL-3.0"]


# ---------------------------------------------------------------- helpers --

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def log(out: Path, step: str, **fields) -> None:
    """Append one step to the check's journal."""
    path = out / "journal.json"
    journal = read(path) if path.exists() else {"created_utc": now(), "zip": ZIP, "licences": LICENCES, "steps": []}
    journal["steps"].append({"step": step, "utc": now()} | fields)
    save(path, journal)


def unsafe_members(archive: Path) -> list[str]:
    """Zip entries that are absolute, climb out with ``..``, name a drive or are links."""
    bad = []
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = info.filename.replace("\\", "/")
            if (name.startswith("/") or ":" in name or ".." in PurePosixPath(name).parts
                    or stat.S_ISLNK(info.external_attr >> 16)):
                bad.append(info.filename)
    return bad


def short_side(box) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def near_edge(box, width: int, height: int, margin: int = EDGE_PX) -> bool:
    return box[0] < margin or box[1] < margin or box[2] > width - margin or box[3] > height - margin


def size_bin(box) -> str:
    side = short_side(box)
    for low, high in SIZE_BINS:
        if low <= side < high:
            return f"{low}-{high}" if high != math.inf else f"{low}+"
    return "?"


def prf(matched: int, predicted: int, truth: int) -> dict:
    precision = matched / predicted if predicted else None
    recall = matched / truth if truth else None
    f1 = 2 * matched / (predicted + truth) if predicted + truth else None
    return {"matched": matched, "predicted": predicted, "people": truth,
            "precision": precision, "recall": recall, "f1": f1}


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


# ----------------------------------------------------------------- labels --

def parse_cvat(path: Path) -> tuple[dict, list[dict]]:
    """CVAT 1.1 video tracks: the job and sequence meta, and every visible box (``outside``
    is 0) with its track, ``occluded`` and ``keyframe`` flags. Frames are offset by the
    job's ``start_frame`` and divided by any ``frame_filter`` step, so they index the video."""
    root = ElementTree.parse(path).getroot()
    job = root.find("meta/job")
    if job is None:
        job = root.find("meta/task")
    start = int(job.findtext("start_frame", "0")) if job is not None else 0
    step = 1
    if job is not None and (job.findtext("frame_filter") or "").startswith("step="):
        step = int(job.findtext("frame_filter").split("=")[1])
    meta = {"job_size": int(job.findtext("size", "0")) if job is not None else None,
            "start_frame": start, "stop_frame": int(job.findtext("stop_frame", "-1")) if job is not None else None,
            "frame_step": step, "created": job.findtext("created") if job is not None else None,
            "updated": job.findtext("updated") if job is not None else None,
            "dumped": root.findtext("meta/dumped"),
            "original_size": [int(root.findtext("meta/original_size/width", "0")),
                              int(root.findtext("meta/original_size/height", "0"))]}
    info = root.find("meta/sequence_info")
    if info is not None:
        meta["sequence_info"] = {child.tag: child.text for child in info if len(child) == 0}
    rows, tracks = [], set()
    for track in root.iter("track"):
        track_id = int(track.get("id"))
        tracks.add(track_id)
        for box in track.iter("box"):
            if box.get("outside") == "1":
                continue
            frame = (int(box.get("frame")) - start) // step
            rows.append({"frame": frame, "track": track_id,
                         "box": [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")],
                         "occluded": box.get("occluded") == "1", "keyframe": box.get("keyframe") == "1"})
    meta["tracks"] = len(tracks)
    return meta, sorted(rows, key=lambda r: (r["frame"], r["track"]))


def video_facts(video: Path) -> dict:
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(video=0))
        return {"path": video.as_posix(), "bytes": video.stat().st_size, "codec": stream.codec_context.name,
                "profile": stream.codec_context.profile, "fps": float(stream.average_rate),
                "declared_frames": stream.frames, "decoded_frames": decoded, "width": stream.width,
                "height": stream.height, "bit_rate": stream.bit_rate, "format_tags": dict(container.metadata),
                "stream_tags": {k: v.strip("\x00\x17") for k, v in stream.metadata.items()},
                "bytes_per_frame": round(video.stat().st_size / max(1, decoded))}


def people_rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "people.jsonl").read_text(encoding="utf-8").splitlines() if line]


def inspect_stage(out: Path) -> dict:
    """Parse the labels, read the video, check both against the record, draw alignment."""
    import cv2

    from app.services.external_datasets import sha256

    folder = DATASET / SEQUENCE
    bad = unsafe_members(Path(ZIP["path"]))
    if bad:
        raise SystemExit(f"Unsafe zip entries: {bad}")
    meta, rows = parse_cvat(folder / "annotations.xml")
    facts = video_facts(folder / "video.mp4") | {"sha256": sha256(folder / "video.mp4")}
    frames = facts["decoded_frames"]
    per_frame = len(rows) / frames
    checks = {"frames_match_video": meta["job_size"] == frames and max(r["frame"] for r in rows) < frames,
              "tracks_301": meta["tracks"] == 301, "boxes_per_frame": round(per_frame, 2),
              "boxes_per_frame_near_26_2": abs(per_frame - 26.2) < .1,
              "occluded_share": round(sum(r["occluded"] for r in rows) / len(rows), 4),
              "keyframe_share": round(sum(r["keyframe"] for r in rows) / len(rows), 4),
              "size_matches_video": meta["original_size"] == [facts["width"], facts["height"]]}
    if not (checks["frames_match_video"] and checks["tracks_301"] and checks["boxes_per_frame_near_26_2"]
            and checks["size_matches_video"]):
        raise SystemExit(f"Labels don't match the record: {checks}")
    with (out / "people.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    # Alignment: people's boxes on first, middle, last and three seeded random frames
    picks = sorted({0, frames // 2, frames - 1, *random.Random(AUDIT_SEED).sample(range(frames), 3)})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["frame"]].append(row)
    capture, tiles, index = cv2.VideoCapture(str(folder / "video.mp4")), [], 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index in picks:
            for row in grouped[index]:
                x1, y1, x2, y2 = map(int, row["box"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255) if row["occluded"] else (0, 255, 0), 2)
            cv2.putText(image, f"frame {index}: {len(grouped[index])} boxes", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (255, 255, 255), 3)
            tiles.append(cv2.resize(image, (960, 540)))
        index += 1
    capture.release()
    import numpy as np

    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    (out / "inspect").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "inspect" / "alignment.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result = {"meta": meta, "video": facts, "checks": checks, "alignment_frames": picks,
              "people_boxes": len(rows)}
    save(out / "inspect" / "inspect.json", result)
    log(out, "inspect", checks=checks, video_sha256=facts["sha256"],
        recording=meta.get("sequence_info", {}).get("video_filename"))
    return result


# --------------------------------------------------------------- detection --

def student_record(root: Path) -> tuple[dict, float]:
    from scripts.student_detector import local_checkpoint

    decisions = read(root / "journal.json")["decisions"]
    return local_checkpoint(root, decisions["student_model"]["checkpoint"]), decisions["student_threshold"]["threshold"]


def detect_stage(out: Path) -> dict:
    """Each student on every DS2 frame at its inference floor; reused while the model and
    video SHA256s match."""
    from scripts.sam3_botsort import file_id, write_rows
    from scripts.student_detector import FLOOR, detect_video, load_student

    facts = read(out / "inspect" / "inspect.json")["video"]
    results = {}
    for name, root in MODELS.items():
        checkpoint, threshold = student_record(root)
        target = out / "detect" / name
        record = target / "inference.json"
        inputs = {"model_sha256": checkpoint["sha256"], "video_sha256": facts["sha256"]}
        if not (record.exists() and all(read(record).get(k) == v for k, v in inputs.items())):
            frames, speed = detect_video(load_student(checkpoint), Path(facts["path"]))
            if len(frames) != facts["decoded_frames"]:
                raise RuntimeError(f"{name} saw {len(frames)} frames, the video has {facts['decoded_frames']}")
            write_rows(frames, target / "frames.jsonl")
            save(record, inputs | {"root": root.as_posix(), "checkpoint": checkpoint, "threshold": threshold,
                                   "floor": FLOOR, "speed": speed, "frames_file": file_id(target / "frames.jsonl")})
        results[name] = read(record)
        log(out, f"detect/{name}", model_sha256=checkpoint["sha256"], threshold=threshold,
            fps=results[name]["speed"]["fps"])
    return results


def model_rows(out: Path, name: str, threshold: float | None = None) -> tuple[list[dict], float]:
    from scripts.sam3_botsort import read_rows
    from scripts.student_detector import raw_rows

    record = read(out / "detect" / name / "inference.json")
    fps = read(out / "inspect" / "inspect.json")["video"]["fps"]
    threshold = record["threshold"] if threshold is None else threshold
    return raw_rows(read_rows(out / "detect" / name / "frames.jsonl"), fps, threshold), threshold


# ----------------------------------------------------------------- scoring --

def match(frames, predicted: list[dict], people: list[dict], iou: float = IOU) -> dict:
    """One-to-one greedy highest-IoU pairs per frame (``pair_frames``), keyed back to rows."""
    from scripts.sam3_botsort import by_frame, pair_frames

    return pair_frames(frames, by_frame(predicted), by_frame(people), iou)


def average_precision(predicted: list[dict], people: list[dict], iou: float = IOU) -> dict:
    """All-point interpolated AP: detections taken in score order, each matched to the
    best still-unmatched person box on its frame. Descriptive only."""
    import numpy as np

    truth = defaultdict(list)
    for row in people:
        truth[row["frame"]].append(row["box"])
    truth = {f: np.array(b, dtype=float) for f, b in truth.items()}
    used = {f: np.zeros(len(b), bool) for f, b in truth.items()}
    hits = []
    for row in sorted(predicted, key=lambda r: -r["confidence"]):
        boxes, hit = truth.get(row["frame"]), False
        if boxes is not None:
            b = np.array(row["box"], dtype=float)
            x1, y1 = np.maximum(boxes[:, 0], b[0]), np.maximum(boxes[:, 1], b[1])
            x2, y2 = np.minimum(boxes[:, 2], b[2]), np.minimum(boxes[:, 3], b[3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            union = (b[2] - b[0]) * (b[3] - b[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
            overlap = np.where(used[row["frame"]], -1, inter / np.maximum(union, 1e-9))
            best = int(overlap.argmax())
            if overlap[best] >= iou:
                used[row["frame"]][best], hit = True, True
        hits.append((row["confidence"], hit))
    tp = np.cumsum([h for _, h in hits])
    precision = tp / np.arange(1, len(hits) + 1)
    recall = tp / max(1, len(people))
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.concatenate([[0], recall]))
    curve = [{"score": round(float(hits[i][0]), 3), "precision": round(float(precision[i]), 4),
              "recall": round(float(recall[i]), 4)} for i in range(0, len(hits), max(1, len(hits) // 200))]
    return {"ap50" if iou == .5 else f"ap{int(iou * 100)}": float((steps * envelope).sum()), "curve": curve}


def score_model(people: list[dict], predicted: list[dict], frames: int, width: int, height: int) -> dict:
    span = range(frames)
    strict, loose = match(span, predicted, people, IOU), match(span, predicted, people, IOU_LOOSE)
    found = {id(m["right"]) for m in strict["matches"]}
    result = {"iou_0.5": prf(strict["matched"], len(predicted), len(people)),
              "iou_0.3_reference_only": prf(loose["matched"], len(predicted), len(people)),
              "boxes_per_frame": round(len(predicted) / frames, 2),
              "people_boxes_per_frame": round(len(people) / frames, 2)}
    groups = {"size": lambda r: size_bin(r["box"]), "occluded": lambda r: "occluded" if r["occluded"] else "clear",
              "edge": lambda r: f"within {EDGE_PX} px of edge" if near_edge(r["box"], width, height) else "inside",
              "keyframe": lambda r: "keyframe" if r["keyframe"] else "interpolated"}
    result["recall_by"] = {}
    for group, key in groups.items():
        counts, hits = Counter(), Counter()
        for row in people:
            counts[key(row)] += 1
            hits[key(row)] += id(row) in found
        result["recall_by"][group] = {k: {"people": counts[k], "found": hits[k], "recall": hits[k] / counts[k]}
                                      for k in sorted(counts)}
    per_track = defaultdict(lambda: [0, 0])
    for row in people:
        per_track[row["track"]][0] += 1
        per_track[row["track"]][1] += id(row) in found
    result["tracks"] = {"people_tracks": len(per_track),
                        "found_at_least_once": sum(1 for v, h in per_track.values() if h),
                        "found_in_half_their_frames": sum(1 for v, h in per_track.values() if h >= v / 2)}
    from scripts.sam3_botsort import by_frame, pair_frames

    people_by = by_frame(people)
    result["frame_shift_check"] = {str(s): pair_frames(span, by_frame(predicted),
                                                       {f + s: v for f, v in people_by.items()}, IOU)["matched"]
                                   for s in (-2, -1, 0, 1, 2)}
    return result, strict


def score_stage(out: Path) -> dict:
    facts = read(out / "inspect" / "inspect.json")["video"]
    frames, width, height = facts["decoded_frames"], facts["width"], facts["height"]
    people = people_rows(out)
    log(out, "score/rules", match_rule="greedy one-to-one per frame, IoU >= 0.5 (pair_frames); IoU 0.3 reported only",
        people_boxes="outside = 0, occluded kept", frames="all", thresholds={n: student_record(r)[1] for n, r in MODELS.items()})
    result = {"note": "Precision against people's boxes is a lower bound: people skip fish in dense schools.",
              "frames": frames, "people_boxes": len(people), "models": {}}
    found_by = {}
    for name in MODELS:
        predicted, threshold = model_rows(out, name)
        scored, strict = score_model(people, predicted, frames, width, height)
        found_by[name] = {(m["right"]["frame"], m["right"]["track"]) for m in strict["matches"]}
        floor_rows, _ = model_rows(out, name, threshold=0.0)
        ap = average_precision(floor_rows, people)
        sweep = {}
        for t in SWEEP:
            rows = [r for r in floor_rows if r["confidence"] >= t]
            sweep[str(t)] = prf(match(range(frames), rows, people)["matched"], len(rows), len(people))
        speed = read(out / "detect" / name / "inference.json")["speed"]
        result["models"][name] = {"threshold": threshold, **scored, "ap50": ap["ap50"],
                                  "sweep_descriptive_only": sweep, "pr_curve": ap["curve"],
                                  "speed_fps_rtx5060": speed["fps"]}
    a, b = list(MODELS)
    keys = {(r["frame"], r["track"]) for r in people}
    split = {"both": len(found_by[a] & found_by[b]), f"{a}_only": len(found_by[a] - found_by[b]),
             f"{b}_only": len(found_by[b] - found_by[a]), "neither": len(keys - found_by[a] - found_by[b])}
    result["people_boxes_found_by"] = {k: {"boxes": v, "share": v / len(people)} for k, v in split.items()}
    save(out / "score" / "score.json", result)
    log(out, "score", summary={n: {k: m["iou_0.5"][k] for k in ("precision", "recall", "f1")}
                               for n, m in result["models"].items()})
    return result


# ---------------------------------------------------------------- tracking --

def track_stage(out: Path) -> dict:
    """BoT-SORT over each student's boxes with the round's BrackishMOT settings at DS2's fps,
    gates moved to its threshold (as ``student_tracker``), scored against people's tracks."""
    from app.services.external_datasets import evaluate_mot, write_mot
    from scripts.sam3_botsort import settings_for, track_rows

    facts = read(out / "inspect" / "inspect.json")["video"]
    fps, frames = facts["fps"], facts["decoded_frames"]
    truth = out / "track" / "people"
    truth.mkdir(parents=True, exist_ok=True)
    with (truth / "ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, ["sequence", "frame", "track_id", "x", "y", "w", "h"])
        writer.writeheader()
        for row in people_rows(out):
            x1, y1, x2, y2 = row["box"]
            writer.writerow({"sequence": SEQUENCE, "frame": row["frame"] + 1, "track_id": row["track"] + 1,
                             "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    save(truth / "report.json", {"sequences": [SEQUENCE]})
    results = {}
    for name, root in MODELS.items():
        rows, threshold = model_rows(out, name)
        chosen = read(root / "journal.json")["decisions"].get("brackishmot_settings")
        settings = replace(settings_for(chosen, "botsort", fps), track_high_thresh=threshold,
                           new_track_thresh=threshold, track_low_thresh=threshold / 2)
        tracked = track_rows(rows, list(range(frames)), settings, fps)
        ids = {i: n for n, i in enumerate(sorted({r["id"] for r in tracked}), 1)}
        candidates = out / "track" / name
        write_mot([{"frame": r["frame"] + 1, "track_id": ids[r["id"]], "x": r["box"][0], "y": r["box"][1],
                    "w": r["box"][2] - r["box"][0], "h": r["box"][3] - r["box"][1], "score": r["confidence"]}
                   for r in tracked if r["box"][2] > r["box"][0] and r["box"][3] > r["box"][1]],
                  candidates / f"{SEQUENCE}.txt")
        report = candidates / "mot.json"
        if report.exists():
            report.unlink()
        mot = evaluate_mot(truth, candidates, report)
        results[name] = {"tracks": len(ids), "idf1": mot["identity"]["idf1"], "mota": mot["clear"]["mota"],
                         "id_switches": mot["clear"]["idsw"], "id_switches_per_people_track": mot["clear"]["idsw_per_gt_track"],
                         "people_tracks": mot["clear"]["gt_tracks"], "tracker": settings.__dict__}
        log(out, f"track/{name}", idf1=results[name]["idf1"], mota=results[name]["mota"])
    save(out / "track" / "track.json", results)
    return results


# ------------------------------------------------------------------ render --

COLOURS = {"matched": (80, 200, 80), "model_only": (0, 200, 255), "people_only": (255, 80, 200)}


def render_stage(out: Path, name: str = "round-2") -> dict:
    import cv2
    import numpy as np

    from scripts.sam3_botsort import by_frame, h264_profile, h264_writer, pair_frames

    facts = read(out / "inspect" / "inspect.json")["video"]
    predicted, threshold = model_rows(out, name)
    model_by, people_by = by_frame(predicted), by_frame(people_rows(out))
    size = (1280, 720)
    target = out / "render" / f"ds2-compare-{name}.mp4"
    legend = [("people and model", "matched"), (f"{name} only (>= {threshold})", "model_only"),
              ("people only", "people_only")]
    writer = h264_writer(target, facts["fps"], (size[0], size[1] + 50))
    capture, index, counts = cv2.VideoCapture(facts["path"]), 0, Counter()
    scale = size[0] / facts["width"]
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            view = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            pairs = pair_frames([index], model_by, people_by, IOU)
            drawn = ([(m["left"], "matched") for m in pairs["matches"]] + [(r, "model_only") for r in pairs["left_unique"]]
                     + [(r, "people_only") for r in pairs["right_unique"]])
            for row, kind in drawn:
                x1, y1, x2, y2 = (int(v * scale) for v in row["box"])
                cv2.rectangle(view, (x1, y1), (x2, y2), COLOURS[kind], 2)
                counts[kind] += 1
            canvas = np.zeros((size[1] + 50, size[0], 3), np.uint8)
            canvas[:size[1]] = view
            x = 12
            for label, kind in legend:
                cv2.rectangle(canvas, (x, size[1] + 14), (x + 26, size[1] + 36), COLOURS[kind], -1)
                cv2.putText(canvas, label, (x + 34, size[1] + 32), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                x += 60 + 11 * len(label)
            status = f"frame {index}"
            cv2.putText(canvas, status, (size[0] - 12 - 11 * len(status), size[1] + 32), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (255, 255, 255), 1)
            writer.write(canvas)
            index += 1
    finally:
        capture.release()
        writer.release()
    result = {"path": target.as_posix(), "frames": index, "encoding": h264_profile(target), "boxes_drawn": dict(counts)}
    log(out, f"render/{name}", **result)
    return result


# ------------------------------------------------------------------- audit --

def crop_tile(image, box, people_boxes, number: str, tile: int = 300):
    """The box with context (3x its size, at least 120 px), the box in yellow, people's
    boxes in thin magenta, scaled to a square tile with its number."""
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    side = max(120, 3 * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(max(0, min(width - side, cx - side / 2))), int(max(0, min(height - side, cy - side / 2)))
    right, bottom = int(min(width, left + side)), int(min(height, top + side))
    view = image[top:bottom, left:right].copy()
    k = tile / max(1, max(view.shape[:2]))
    view = cv2.resize(view, (max(1, int(view.shape[1] * k)), max(1, int(view.shape[0] * k))))
    for p in people_boxes:
        a = [int((p[0] - left) * k), int((p[1] - top) * k), int((p[2] - left) * k), int((p[3] - top) * k)]
        cv2.rectangle(view, (a[0], a[1]), (a[2], a[3]), (255, 80, 200), 1)
    a = [int((x1 - left) * k), int((y1 - top) * k), int((x2 - left) * k), int((y2 - top) * k)]
    pad = 4  # Drawn just outside the box, thin, so a faint fish inside stays visible
    cv2.rectangle(view, (a[0] - pad, a[1] - pad), (a[2] + pad, a[3] + pad), (0, 230, 255), 1)
    import numpy as np

    canvas = np.zeros((tile + 28, tile, 3), np.uint8)
    canvas[:view.shape[0], :view.shape[1]] = view
    cv2.putText(canvas, number, (6, tile + 21), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return canvas


def sheets(images: dict[int, object], items: list[dict], people_by: dict, target: Path, per_sheet: int = 20,
           columns: int = 4) -> list[str]:
    import cv2
    import numpy as np

    paths = []
    for start in range(0, len(items), per_sheet):
        chunk = items[start:start + per_sheet]
        tiles = [crop_tile(images[i["frame"]], i["box"], [p["box"] for p in people_by.get(i["frame"], [])],
                           i["number"]) for i in chunk]
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[r:r + columns]) for r in range(0, len(tiles), columns)])
        path = target.with_name(f"{target.stem}_{start // per_sheet + 1}.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(path.as_posix())
    return paths


def spread_sample(rows: list[dict], count: int, frames: int, seed: int = AUDIT_SEED) -> list[dict]:
    """A seeded sample spread over the video (thirds) and box sizes (short-side bins)."""
    strata = defaultdict(list)
    for row in rows:
        strata[(min(2, 3 * row["frame"] // frames), size_bin(row["box"]))].append(row)
    rng, keys = random.Random(seed), sorted(strata)
    for key in keys:
        rng.shuffle(strata[key])
    picked = []
    while len(picked) < min(count, len(rows)):
        for key in keys:
            if strata[key] and len(picked) < count:
                picked.append(strata[key].pop())
    return sorted(picked, key=lambda r: (r["frame"], r["box"][0]))


def audit_stage(out: Path) -> dict:
    """Numbered sheets of 60 model boxes with no human box (IoU 0.5) per student, and of the
    20 largest people's boxes each student missed. A person marks the audit boxes."""
    import cv2

    from scripts.sam3_botsort import by_frame

    facts = read(out / "inspect" / "inspect.json")["video"]
    frames, people = facts["decoded_frames"], people_rows(out)
    people_by = by_frame(people)
    picks, misses = {}, {}
    for name in MODELS:
        predicted, _ = model_rows(out, name)
        paired = match(range(frames), predicted, people)
        chosen = spread_sample(paired["left_unique"], AUDIT_PER_MODEL, frames)
        picks[name] = [r | {"number": f"{AUDIT_LETTERS[name]}{n}"} for n, r in enumerate(chosen, 1)]
        largest, tracks = [], set()  # The largest miss of each track, so one fish isn't shown 8 times
        for row in sorted(paired["right_unique"], key=lambda r: -short_side(r["box"])):
            if row["track"] not in tracks and len(largest) < 20:
                tracks.add(row["track"])
                largest.append(row)
        misses[name] = [r | {"number": f"M{AUDIT_LETTERS[name]}{n}", "confidence": None} for n, r in enumerate(largest, 1)]
    wanted = {r["frame"] for group in (*picks.values(), *misses.values()) for r in group}
    images, capture, index = {}, cv2.VideoCapture(facts["path"]), 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted:
            images[index] = image
        index += 1
    capture.release()
    result = {"instructions": "Mark each numbered yellow box: fish, not a fish, or can't tell. Thin magenta "
                              "boxes are people's boxes nearby.", "sheets": {}, "miss_sheets": {}}
    for name in MODELS:
        result["sheets"][name] = sheets(images, picks[name], people_by, out / "audit" / f"audit_{name}")
        result["miss_sheets"][name] = sheets(images, misses[name], people_by, out / "audit" / f"missed_{name}")
        with (out / "audit" / f"audit_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, ["number", "frame", "x1", "y1", "x2", "y2", "score", "verdict"])
            writer.writeheader()
            for r in picks[name]:
                writer.writerow({"number": r["number"], "frame": r["frame"],
                                 **dict(zip(("x1", "y1", "x2", "y2"), (round(v, 1) for v in r["box"]))),
                                 "score": round(r["confidence"], 3), "verdict": ""})
    save(out / "audit" / "audit.json", result)
    log(out, "audit", sheets=result["sheets"])
    return result


def audit_score_stage(out: Path) -> dict:
    """Precision two ways: against people's boxes, and with the audited share of
    unmatched boxes that are fish added back (Wilson 95% interval)."""
    score = read(out / "score" / "score.json")
    result = {}
    for name in MODELS:
        with (out / "audit" / f"audit_{name}.csv").open(newline="", encoding="utf-8") as stream:
            verdicts = Counter((r["verdict"] or "").strip().lower() for r in csv.DictReader(stream))
        fish, judged = verdicts["fish"], verdicts["fish"] + verdicts["not a fish"]
        base = score["models"][name]["iou_0.5"]
        unmatched = base["predicted"] - base["matched"]
        adjusted = lambda share: (base["matched"] + share * unmatched) / base["predicted"]
        interval = wilson(fish, judged)
        result[name] = {"verdicts": dict(verdicts), "fish_share_of_judged": fish / judged if judged else None,
                        "precision_against_people": base["precision"],
                        "precision_adjusted": adjusted(fish / judged) if judged else None,
                        "precision_adjusted_95": [adjusted(s) for s in interval] if interval else None}
    save(out / "audit" / "audit_score.json", result)
    log(out, "audit-score", result=result)
    return result


STAGES = {"inspect": inspect_stage, "detect": detect_stage, "score": score_stage, "track": track_stage,
          "render": render_stage, "audit": audit_stage, "audit-score": audit_score_stage}


def main() -> None:
    from scripts.sam3_botsort import OFFLINE

    os.environ.update(OFFLINE)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = ["inspect", "detect", "score", "track", "render", "audit"] if args.stage == "all" else [args.stage]
    for name in names:
        result = STAGES[name](args.out)
        print(f"== {name}")
        print(json.dumps(result, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
