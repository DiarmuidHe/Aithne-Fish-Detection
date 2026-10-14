"""Murky-water Part 3: FathomNet Megalodon beside production VIAME on dark ONC footage.

Stages: ``find`` (host; a read-only query for recordings of ONC live sessions), ``clips``
(host; identical lossless bounded clips), ``viame`` (VIAME worker image; live profile,
4 FPS), ``megalodon`` (host .venv; infer-megalodon) and ``compare`` (host). Megalodon has
one generic 'object' class, so its boxes are never called fish. No labels exist for this
footage: the comparison reports counts and agreement, not accuracy. ONC video is
CC BY-NC-SA 3.0. Nothing here changes production settings, jobs, uploads or CVAT.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath

from scripts.viame_zero_label_benchmark import digest, read, save

IN_CONTAINER = Path("/opt/noaa/viame").exists()
DATA = Path("/data") if IN_CONTAINER else Path("data")
ROOT = DATA / "outputs/murky-water-20261011/part3"
WEIGHTS = DATA / "models/megalodon/mbari-megalodon-yolov8x.pt"
# Bounded clips: the whole 30 s tempo-mini recording and the first 120 s of folger-pinnacle.
MAX_FRAMES = {"onc-tempo-mini": 120, "onc-folger-pinnacle": 480}
QUERY = ("SET default_transaction_read_only = on; "
         "SELECT s.source_key, s.id, v.id, v.storage_path, v.fps, v.width, v.height, v.duration_seconds "
         "FROM live_monitor_sessions s JOIN videos v ON v.source_session_id = s.id "
         "WHERE s.source_key LIKE 'onc-%' ORDER BY s.source_key;")
ACCEPT = .60


def local(path: str) -> Path:
    """A manifest path (POSIX, under data/ or /data/) on this side of the container."""
    parts = PurePosixPath(path).parts
    return DATA.joinpath(*parts[2:] if parts[0] == "/" else parts[1:])


def find() -> dict:
    """Read-only lookup of library recordings made by ONC live sessions."""
    rows = subprocess.run(["docker", "exec", "aithne-fish-detection-postgres-1", "psql", "-U", "fish",
                           "-d", "fish_monitor", "-At", "-F", "|", "-c", QUERY],
                          check=True, capture_output=True, text=True).stdout.splitlines()
    recordings = []
    for line in rows:
        if line == "SET":
            continue
        key, session, video, storage, fps, width, height, duration = line.split("|")
        path = local(storage)
        recordings.append({"source_key": key, "session_id": session, "video_id": video, "path": path.as_posix(),
                           "sha256": digest(path), "fps": float(fps), "width": int(width),
                           "height": int(height), "duration_seconds": float(duration)})
    manifest = {"query": QUERY, "recordings": recordings,
                "licence": "Ocean Networks Canada video: CC BY-NC-SA 3.0"}
    ROOT.mkdir(parents=True, exist_ok=True)
    save(ROOT / "manifest.json", manifest)
    return manifest


def clips() -> dict:
    """Lossless bounded clips both models read, with their frame count asserted."""
    import cv2

    manifest = read(ROOT / "manifest.json")
    for recording in manifest["recordings"]:
        source = local(recording["path"])
        if digest(source) != recording["sha256"]:
            raise RuntimeError(f"Recording changed: {source}")
        target = ROOT / "clips" / f"{recording['source_key']}.mkv"
        if "clip" in recording and target.exists() and digest(target) == recording["clip"]["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        capture = cv2.VideoCapture(str(source))
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"FFV1"), recording["fps"],
                                 (recording["width"], recording["height"]))
        written, brightness = 0, []
        try:
            while written < MAX_FRAMES[recording["source_key"]]:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
                written += 1
        finally:
            capture.release()
            writer.release()
        check, decoded = cv2.VideoCapture(str(target)), 0
        while check.read()[0]:
            decoded += 1
        check.release()
        if decoded != written:
            raise ValueError(f"{target}: decoded {decoded} of {written} frames")
        brightness.sort()
        recording["clip"] = {"path": target.as_posix(), "frames": written, "sha256": digest(target),
                             "codec": "FFV1 (lossless)", "seconds": written / recording["fps"],
                             "mean_grey_median_0_255": brightness[len(brightness) // 2]}
    save(ROOT / "manifest.json", manifest)
    return manifest


def viame() -> None:
    from scripts.murky_water_tracker import PRODUCTION_PIPE, VIAME_JOURNALS, Journal, run_pass

    journal = Journal(VIAME_JOURNALS[2])
    for recording in read(ROOT / "manifest.json")["recordings"]:
        clip = local(recording["clip"]["path"])
        output = ROOT / "viame" / recording["source_key"]
        run_pass(journal, f"part3/viame/{recording['source_key']}", "gpu",
                 [clip, output, "--fps", recording["fps"], "--pipeline", PRODUCTION_PIPE], output, timeout=900)


def megalodon() -> None:
    import os

    from scripts.external_fish import infer_megalodon
    from scripts.murky_water_detector import OFFLINE

    os.environ.update(OFFLINE)  # Ultralytics may not fetch anything (no analytics, no auto-install)
    for recording in read(ROOT / "manifest.json")["recordings"]:
        output = ROOT / "megalodon" / recording["source_key"]
        if not (output / "report.json").exists():
            infer_megalodon(WEIGHTS, local(recording["clip"]["path"]), output,
                            max_frames=recording["clip"]["frames"], stride=1, confidence=.25)


def viame_boxes(directory: Path, frames: int) -> tuple[set[int], dict, int]:
    """Analysed zero-based frames and tracked boxes by frame.

    VIAME ran at each clip's own frame rate, so every frame is analysed. The detection
    writer's zero-based ordinals (which count every frame it receives, with or without
    detections) are returned as evidence."""
    import csv

    from app.services.external_datasets import viame_tracks_to_mot

    ordinals = [int(row[2]) for row in csv.reader((directory / "computed_detections.csv").open(encoding="utf-8"))
                if row and not row[0].startswith("#")]
    analysed = set(range(frames))
    rows = viame_tracks_to_mot(directory / "viame_tracks.csv", frame_offset=-1)
    best = defaultdict(float)
    for row in rows:
        best[row["track_id"]] = max(best[row["track_id"]], row["score"])
    boxes = defaultdict(list)
    for row in rows:
        boxes[row["frame"] - 1].append({"frame": row["frame"] - 1, "score": row["score"],
                                        "box": (row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]),
                                        "accepted": best[row["track_id"]] >= ACCEPT})
    return analysed, boxes, max(ordinals, default=-1) + 1


def megalodon_boxes(path: Path, minimum: float) -> dict:
    import csv

    boxes = defaultdict(list)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if float(row["confidence"]) >= minimum:
            frame = int(row["frame"])
            boxes[frame].append({"frame": frame, "score": float(row["confidence"]),
                                 "box": tuple(float(row[k]) for k in ("tl_x", "tl_y", "br_x", "br_y"))})
    return boxes


def pair(frames, left: dict, right: dict) -> dict:
    """Greedy highest-IoU pairing per frame at IoU >= 0.5 (same rule as the benchmark)."""
    from app.services.detection_metrics import iou

    result = {"matched": 0, "matches": [], "left_unique": [], "right_unique": []}
    for frame in sorted(frames):
        a, b = left.get(frame, []), right.get(frame, [])
        edges = sorted(((iou(x["box"], y["box"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                        if iou(x["box"], y["box"]) >= .5), reverse=True)
        used_a, used_b = set(), set()
        for _, i, j in edges:
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                result["matches"].append({"left": a[i], "right": b[j]})
        result["left_unique"] += [x for i, x in enumerate(a) if i not in used_a]
        result["right_unique"] += [y for j, y in enumerate(b) if j not in used_b]
    result["matched"] = len(result["matches"])
    return result


def compare() -> dict:
    from scripts.murky_water_detector import contact_sheet

    manifest = read(ROOT / "manifest.json")
    result = {"licence": manifest["licence"], "clips": {},
              "note": "No labels: counts and agreement only. Megalodon's single class is a generic "
                      "'object' (fish, crabs, jellyfish, debris alike); VIAME's is fish. Unique boxes "
                      "are review suggestions, not errors."}
    sheets = []
    for recording in manifest["recordings"]:
        key = recording["source_key"]
        analysed, viame_all, received = viame_boxes(ROOT / "viame" / key, recording["clip"]["frames"])
        megalodon_report = read(ROOT / "megalodon" / key / "report.json")
        rows = {}
        for viame_label, keep in (("viame_tracked", lambda b: True), ("viame_accepted", lambda b: b["accepted"])):
            viame = {f: [b for b in v if keep(b)] for f, v in viame_all.items()}
            for minimum in (.25, .50):
                mega = megalodon_boxes(ROOT / "megalodon" / key / "candidates.csv", minimum)
                paired = pair(analysed, viame, mega)
                v_count = sum(len(viame.get(f, [])) for f in analysed)
                m_count = sum(len(mega.get(f, [])) for f in analysed)
                rows[f"{viame_label}__megalodon_{minimum:.2f}"] = {
                    "viame_boxes": v_count, "megalodon_boxes": m_count, "matched": paired["matched"],
                    "viame_only": len(paired["left_unique"]), "megalodon_only": len(paired["right_unique"]),
                    "agreement": paired["matched"] / max(1, v_count + m_count - paired["matched"]),
                    "frames_with_viame": sum(bool(viame.get(f)) for f in analysed),
                    "frames_with_megalodon": sum(bool(mega.get(f)) for f in analysed)}
                if viame_label == "viame_tracked" and minimum == .25:
                    sheets.append((local(recording["clip"]["path"]), paired))
        result["clips"][key] = {"frames": recording["clip"]["frames"], "analysed_by_viame": len(analysed),
                                "viame_writer_frames_at_least": received,
                                "mean_grey_median_0_255": recording["clip"]["mean_grey_median_0_255"],
                                "megalodon_input_size": megalodon_report["input_size"],
                                "comparisons": rows}
    contact_sheet(sheets, ROOT / "disagreements.jpg", per_clip=3, other="Megalodon")
    save(ROOT / "comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["find", "clips", "viame", "megalodon", "compare"])
    stage = parser.parse_args().stage
    ROOT.mkdir(parents=True, exist_ok=True)
    result = {"find": find, "clips": clips, "viame": viame, "megalodon": megalodon, "compare": compare}[stage]()
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
