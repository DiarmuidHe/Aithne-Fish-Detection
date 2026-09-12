"""Bounded, local-only VIAME experiment. No Settings, database or species imports.

Run from the repository root with --docker. A frozen copy executes in the existing
worker; no image build/pull, network request, camera capture or production job occurs.
Metrics are unlabeled proxies. This tool never promotes a production configuration.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager

from app.services.viame_parser import parse_viame_csv

ROOT = Path("data/outputs/viame-zero-label-benchmark")
VIAME = Path("/opt/noaa/viame")
PIPES = VIAME / "configs/pipelines"
MAX_PASSES = 10
MAX_SECONDS = 90 * 60
THRESHOLDS = (.45, .50, .55, .60, .65)
COMPOSE = ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml"]
WARNING = ("Unlabeled temporal proxies, not precision, recall, mAP, F1 or biological fish counts. "
           "Fusion shares DEIM with the baseline; agreement can reflect correlated errors.")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    """Atomic replacement: a killed writer cannot leave a half JSON journal."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows indexers/OneDrive can briefly hold the destination open.
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05 * (attempt + 1))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def lock(root):
    # OS advisory locks are released on process death, unlike stale PID files.
    import fcntl
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another benchmark owns this journal") from exc
        yield


def stats(values):
    values = sorted(values)
    def percentile(q):
        if not values:
            return None
        pos = (len(values) - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return {"n": len(values), "median": percentile(.5), "p90": percentile(.9),
            "p95": percentile(.95), "max": values[-1] if values else None}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def iou(a, b):
    intersection = area((max(a[0], b[0]), max(a[1], b[1]),
                         min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def observations(path, timestamps, offset=-1, timestamp_origin=0.):
    """Application CSV parser plus strict finite/geometry/time validation.

    CSV source frame IDs index decoded PTS, never sampled FPS. Repeated writer
    histories are deduplicated, while conflicting same-track/frame rows fail QA.
    """
    # The detection writer uses zero-based sampled ordinals plus HH:MM:SS PTS
    # in column 2. The track writer uses one-based source IDs. Applying the job
    # offset to both discards frame zero and mis-times every sampled detection.
    parsed = parse_viame_csv(Path(path), frame_number_offset=0)
    result, seen = [], {}
    issues = Counter(malformed=len(parsed.skipped_rows), invalid=0, duplicate_rows=0,
                     conflicting_rows=0, missing_timestamp=0)
    for detection in parsed.detections:
        box = (detection.bbox_left, detection.bbox_top,
               detection.bbox_right, detection.bbox_bottom)
        if (not all(math.isfinite(v) for v in (*box, detection.confidence))
                or area(box) <= 0 or not 0 <= detection.confidence <= 1):
            issues["invalid"] += 1
            continue
        timestamp_match = re.fullmatch(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", detection.source)
        if timestamp_match:
            hours, minutes, seconds = map(float, timestamp_match.groups())
            timestamp = hours * 3600 + minutes * 60 + seconds - timestamp_origin
            index = bisect.bisect_left(timestamps, timestamp)
            nearest = range(max(0, index - 1), min(len(timestamps), index + 1))
            frame = min(nearest, key=lambda i: abs(timestamps[i] - timestamp), default=-1)
            if frame < 0 or abs(timestamps[frame] - timestamp) > .002:
                issues["missing_timestamp"] += 1
                continue
        else:
            frame = detection.frame_number + offset
        if frame < 0 or frame >= len(timestamps) or not math.isfinite(timestamps[frame]):
            issues["missing_timestamp"] += 1
            continue
        row = {"id": detection.track_id, "frame": frame, "t": timestamps[frame],
               "box": list(box), "confidence": detection.confidence}
        key = (row["id"], frame)
        if key in seen:
            issues["duplicate_rows" if seen[key] == row else "conflicting_rows"] += 1
            continue
        seen[key] = row
        result.append(row)
    return sorted(result, key=lambda row: (row["t"], row["id"])), dict(issues)


def align(left, right, tolerance=.02, min_iou=.5):
    """Greedy highest-IoU one-to-one boxes on one-to-one nearest PTS frames.

    Frames farther than tolerance are unaligned, not counted as disagreements.
    Full unique box metadata is returned for inspection; it is never called false.
    """
    frames = []
    groups = []
    for rows in (left, right):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["t"]].append(row)
        groups.append(grouped)
        frames.append(sorted(grouped))
    candidates = []
    for t in frames[0]:
        index = bisect.bisect_left(frames[1], t)
        for other in frames[1][max(0, index - 1):index + 1]:
            if abs(t - other) <= tolerance:
                candidates.append((abs(t - other), t, other))
    used_left, used_right, matches, unique_left, unique_right = set(), set(), [], [], []
    for _, t, other in sorted(candidates):
        if t in used_left or other in used_right:
            continue
        used_left.add(t)
        used_right.add(other)
        a, b = groups[0][t], groups[1][other]
        edges = sorted(((-iou(x["box"], y["box"]), i, j)
                        for i, x in enumerate(a) for j, y in enumerate(b)
                        if iou(x["box"], y["box"]) >= min_iou))
        ai, bj = set(), set()
        for negative_iou, i, j in edges:
            if i not in ai and j not in bj:
                ai.add(i)
                bj.add(j)
                matches.append({"left": a[i], "right": b[j], "iou": -negative_iou})
        unique_left.extend(x for i, x in enumerate(a) if i not in ai)
        unique_right.extend(x for j, x in enumerate(b) if j not in bj)
    # Detection-free frames don't occur in CSVs. At shared sampling rates, boxes
    # on frames absent from the other CSV are valid disagreements as well.
    unaligned_left = [x for t in frames[0] if t not in used_left for x in groups[0][t]]
    unaligned_right = [x for t in frames[1] if t not in used_right for x in groups[1][t]]
    return {"matched": len(matches), "matches": matches, "left_unique": unique_left,
            "right_unique": unique_right, "left_unaligned": unaligned_left,
            "right_unaligned": unaligned_right,
            "agreement": ratio(len(matches), len(left) + len(right) - len(matches)),
            "timestamp_tolerance_seconds": tolerance, "box_iou_threshold": min_iou}


def metrics(detections, tracked, width, height, sampled_fps):
    tracks = defaultdict(list)
    for row in tracked:
        tracks[row["id"]].append(row)
    for rows in tracks.values():
        rows.sort(key=lambda row: row["t"])
    lengths = [len(rows) for rows in tracks.values()]
    sweep = {f"{cutoff:.2f}": sum(max(r["confidence"] for r in rows) >= cutoff
                                for rows in tracks.values()) for cutoff in THRESHOLDS}
    gaps, speeds, accelerations, changes = [], [], [], []
    edge_only, oob, tiny = 0, 0, 0
    for row in detections:
        x1, y1, x2, y2 = row["box"]
        oob += x1 < 0 or y1 < 0 or x2 > width or y2 > height
        tiny += min(x2 - x1, y2 - y1) < 4 or area(row["box"]) / (width * height) < .00001
    for rows in tracks.values():
        edge_only += all(r["box"][0] <= width * .01 or r["box"][1] <= height * .01
                         or r["box"][2] >= width * .99 or r["box"][3] >= height * .99
                         for r in rows)
        track_speed, track_acc, track_area, track_gap = [], [], [], []
        previous = None
        for a, b in zip(rows, rows[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0:
                continue
            ca, cb = center(a["box"]), center(b["box"])
            velocity = ((cb[0] - ca[0]) / width / dt, (cb[1] - ca[1]) / height / dt)
            track_speed.append(math.hypot(*velocity))
            track_gap.append(dt)
            track_area.append(abs(math.log(area(b["box"]) / area(a["box"]))) / dt)
            if previous:
                v, last_dt = previous
                track_acc.append(math.hypot(velocity[0] - v[0], velocity[1] - v[1])
                                 / ((dt + last_dt) / 2))
            previous = velocity, dt
        # Equal track weighting; each observation-heavy track contributes one median.
        for target, values in ((gaps, track_gap), (speeds, track_speed),
                               (accelerations, track_acc), (changes, track_area)):
            if values:
                target.append(statistics.median(values))
    frames = defaultdict(list)
    for row in tracked:
        frames[row["t"]].append(row)
    overlaps = defaultdict(list)
    for timestamp, rows in frames.items():
        for a, b in itertools.combinations(rows, 2):
            if a["id"] != b["id"] and iou(a["box"], b["box"]) >= .8:
                overlaps[tuple(sorted((a["id"], b["id"])))].append(timestamp)
    duplicates = []
    for pair, times in overlaps.items():
        streak = longest = 1
        for a, b in zip(times, times[1:]):
            streak = streak + 1 if b - a <= 1.5 / sampled_fps else 1
            longest = max(longest, streak)
        if longest >= 3:
            duplicates.append({"tracks": pair, "consecutive_frames": longest})
    fragments = []
    for aid, a in tracks.items():
        if len(a) < 2:
            continue  # No measured motion: cannot call a spatial pair fragmentation.
        last, prev = a[-1], a[-2]
        dt = last["t"] - prev["t"]
        if dt <= 0:
            continue
        ca, cp = center(last["box"]), center(prev["box"])
        for bid, b in tracks.items():
            gap = b[0]["t"] - last["t"]
            if aid == bid or not 0 < gap <= 1:
                continue
            dx, dy = (ca[0] - cp[0]) / dt * gap, (ca[1] - cp[1]) / dt * gap
            prediction = [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(last["box"])]
            if iou(prediction, b[0]["box"]) >= .3 and abs(
                    math.log(area(b[0]["box"]) / area(last["box"]))) <= math.log(2):
                fragments.append({"end": aid, "start": bid, "gap_seconds": gap})
    all_gaps = [b["t"] - a["t"] for rows in tracks.values() for a, b in zip(rows, rows[1:])]
    coverage = {}
    for minimum in (3, 5):
        persistent = [r for rows in tracks.values() if len(rows) >= minimum for r in rows]
        coverage[str(minimum)] = ratio(align(detections, persistent, min_iou=.95)["matched"],
                                       len(detections))
    return {"detections": len(detections), "tracks": len(tracks), "observations": len(tracked),
            "accepted_tracks": sweep, "track_length": stats(lengths),
            "length_fractions": {"1": ratio(lengths.count(1), len(lengths)),
                                 "2": ratio(lengths.count(2), len(lengths)),
                                 "3-4": ratio(sum(3 <= n <= 4 for n in lengths), len(lengths)),
                                 "5+": ratio(sum(n >= 5 for n in lengths), len(lengths))},
            "temporal_coverage": coverage, "gap_seconds": stats(all_gaps),
            "per_track_median_gap_seconds": stats(gaps),
            "center_velocity": stats(speeds), "center_acceleration": stats(accelerations),
            "absolute_log_area_change_per_second": stats(changes),
            "duplicate_pairs": duplicates, "duplicate_pair_rate": ratio(len(duplicates), len(tracks)),
            "fragmentation_pairs": fragments, "fragmentation_rate": ratio(len(fragments), len(tracks)),
            "out_of_bounds_boxes": oob, "tiny_boxes": tiny, "edge_only_tracks": edge_only}


def canonical(rows):
    """Track-label invariant partition signature, with coordinates and confidence."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append((row["t"], tuple(row["box"]), row["confidence"]))
    return sorted(sorted(group) for group in groups.values())


class Journal:
    def __init__(self, path, manifest_hash):
        self.path = Path(path)
        self.data = read(path) if self.path.exists() else {
            "schema_version": 1, "manifest_sha256": manifest_hash, "runs": {}}
        if self.data["manifest_sha256"] != manifest_hash:
            raise RuntimeError("Manifest changed; refusing to mix benchmark provenance")

    def used(self):
        rows = self.data["runs"].values()
        # Interrupted/unknown runs are charged their entire reserved timeout.
        return len(self.data["runs"]), sum(r.get("duration_seconds", r["reserved_seconds"])
                                          for r in rows)

    def reserve(self, name, config, timeout=600):
        if name in self.data["runs"]:
            if self.data["runs"][name]["config"] != config:
                raise RuntimeError("Run configuration changed on resume")
            return False
        count, seconds = self.used()
        remaining = math.floor(MAX_SECONDS - seconds)
        if count >= MAX_PASSES or remaining < 1:
            raise RuntimeError("Inference budget exhausted")
        self.data["runs"][name] = {"status": "reserved", "config": config,
                                    "reserved_seconds": min(timeout, remaining)}
        save(self.path, self.data)
        return True

    def finish(self, name, **values):
        self.data["runs"][name].update(values)
        save(self.path, self.data)


def clean_env():
    # Intentionally do not inherit credentials, database URLs, proxies or .env.
    keep = ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")
    return {**{key: os.environ[key] for key in keep if key in os.environ},
            "FISHIAL_ENABLED": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled", "PYTHONHASHSEED": "0"}


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True,
                          text=True, env=clean_env(), **kwargs).stdout


def probe(path):
    result = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,start_time:format=duration",
        "-of", "json", path]))
    stream = result["streams"][0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    return {"width": stream["width"], "height": stream["height"],
            "fps": numerator / denominator, "duration": float(result["format"]["duration"]),
            "average_frame_rate": stream["avg_frame_rate"],
            "pts_origin": float(stream.get("start_time", 0))}


def pts(path):
    result = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", path]))
    times = [float(frame["best_effort_timestamp_time"]) for frame in result["frames"]]
    times = [t - times[0] for t in times]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError("Non-monotonic source timestamps")
    return times


def image_summary(path):
    import cv2
    reader = cv2.VideoCapture(str(path))
    brightness, contrast, motion = [], [], []
    previous = None
    for second in range(math.floor(probe(path)["duration"])):
        reader.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = reader.read()
        if not ok:
            continue
        grey = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        brightness.append(float(grey.mean()))
        contrast.append(float(grey.std()))
        if previous is not None:
            motion.append(float(cv2.absdiff(previous, grey).mean()))
        previous = grey
    reader.release()
    return {"brightness_0_255": stats(brightness), "contrast_std_0_255": stats(contrast),
            "motion_mean_absolute_difference_1s": stats(motion)}


def inventory(root):
    import zipfile
    model_dir = PIPES / "models"
    files = sorted(PIPES.glob("*fish*.pipe")) + [PIPES / "common_default_tracker.pipe",
        PIPES / "common_default_input.pipe", PIPES / "common_default_input_with_downsampler.pipe",
        VIAME / "lib/python3.12/site-packages/viame/core/bytetrack_tracker.py"]
    models = sorted(p for p in model_dir.glob("*fish*") if p.is_file())
    inv = {"files": {str(p): digest(p) for p in files + models},
           "pipeline_text": {str(p): p.read_text() for p in files},
           "viame_release": None, "release_note": "No VIAME release marker found; file hashes identify build",
           "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"]).strip()}
    archive = model_dir / "fish_deim_v2_1024.zip"
    with zipfile.ZipFile(archive) as package:
        inv["model_archive_members"] = package.namelist()
        inv["model_spec"] = {n: package.read(n).decode() for n in package.namelist()
                             if n.endswith((".json", ".yaml"))}
        # Inspect actual ONNX protobuf, without creating an inference session.
        import onnx
        name = next(n for n in package.namelist() if n.endswith(".onnx"))
        data = package.read(name)
        model = onnx.load_model_from_string(data)
        inv["onnx_sha256"] = hashlib.sha256(data).hexdigest()
        inv["onnx_inputs"] = {v.name: [d.dim_value or d.dim_param
            for d in v.type.tensor_type.shape.dim] for v in model.graph.input}
    version = VIAME / "include/vital/version.h"
    inv["kwiver_version_header"] = version.read_text() if version.exists() else None
    save(root / "inventory.json", inv)
    return inv


def prepare(root):
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        for clip in manifest["clips"]:
            if digest(clip["path"]) != clip["sha256"] or digest(clip["source"]) != clip["source_sha256"]:
                raise RuntimeError("Benchmark media changed")
        if digest(root / "code/baseline.pipe") != manifest["baseline_sha256"]:
            raise RuntimeError("Frozen baseline changed")
        if digest(root / "inventory.json") != manifest["inventory_sha256"]:
            raise RuntimeError("Runtime inventory changed")
        for filename, sha in read(root / "inventory.json")["files"].items():
            if digest(filename) != sha:
                raise RuntimeError("Installed model or pipeline changed; cannot mix runs")
        return manifest
    inventory(root)
    sources, seen = [], set()
    for path in sorted(Path("/data/uploads").iterdir()):
        if path.suffix.lower() not in (".mov", ".mp4", ".mkv", ".avi"):
            continue
        sha = digest(path)
        if sha in seen:
            continue
        seen.add(sha)
        info = probe(path)
        sources.append({"path": str(path), "sha256": sha, **info})
    eligible = sorted((s for s in sources if s["duration"] >= 30),
                      key=lambda s: (-s["duration"], s["sha256"]))
    if not eligible:
        raise RuntimeError("No local source >=30 seconds; supply existing footage before inference")
    # Scene choice precedes inference. Three distinct long sources if available;
    # otherwise deterministic thirds of the longest, with an explicit limitation.
    selections = [(s, 0, min(60, s["duration"])) for s in eligible[:3]] if len(eligible) >= 3 else [
        (eligible[0], i * math.floor(eligible[0]["duration"] / 3),
         min(60, math.floor(eligible[0]["duration"] / 3))) for i in range(3)]
    clips = []
    for index, (source, start, duration) in enumerate(selections):
        path = root / f"clip-{index}.mp4"
        if not path.exists():
            command(["ffmpeg", "-nostdin", "-v", "error", "-n", "-ss", str(start),
                     "-i", source["path"], "-t", str(duration), "-an", "-c:v", "libx264",
                     "-crf", "18", "-preset", "fast", "-fps_mode", "passthrough", path])
        clips.append({"id": str(index), "source": source["path"], "source_sha256": source["sha256"],
                      "source_metadata": source, "start_seconds": start, "requested_duration": duration,
                      "path": str(path), "sha256": digest(path), **probe(path),
                      "timestamps": pts(path), "image_summary": image_summary(path)})
    live = root / "live-2fps.mp4"
    if not live.exists():
        command(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", clips[1]["path"],
                 "-vf", "fps=2,scale=1280:720:force_original_aspect_ratio=decrease",
                 "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", live])
    clips.append({**clips[1], "id": "live", "path": str(live), "sha256": digest(live),
                  **probe(live), "timestamps": pts(live), "image_summary": image_summary(live)})
    manifest = {"schema_version": 1, "sources": sources, "clips": clips,
                "selection": "Three long sources, or early/middle/late thirds of longest source; before inference",
                "limitation": "Single camera; thirds may be shorter than 30 seconds. Live uses the middle third.",
                "inventory_sha256": digest(root / "inventory.json"),
                "baseline_sha256": digest(root / "code/baseline.pipe"),
                "limits": {"passes": MAX_PASSES, "inference_wall_seconds": MAX_SECONDS,
                           "paid_api_calls": 0, "downloads": 0},
                "safe_configuration": read(root / "code/safe-config.json"),
                "plan": ["baseline-0", "baseline-1", "baseline-2", "repeat-1",
                         "fusion-1", "fusion-0", "fusion-2", "live-baseline", "live-retention"],
                "note": "Nine planned passes; reserve tenth for repeat of live retention if not dominated."}
    save(manifest_path, manifest)
    return manifest


def offline_history(root):
    """No geometry or timing guesses for older CSVs with unknown source provenance."""
    rows = []
    for path in sorted(Path("/data/jobs").rglob("viame_tracks.csv")):
        parsed = parse_viame_csv(path)
        valid = [t for t in parsed.tracks if math.isfinite(t.max_confidence)]
        rows.append({"csv": str(path), "sha256": digest(path), "tracks": len(valid),
                     "observations": len(parsed.detections), "malformed_rows": len(parsed.skipped_rows),
                     "accepted_tracks": {f"{c:.2f}": sum(t.max_confidence >= c for t in valid)
                                         for c in THRESHOLDS},
                     "note": "Historical output only; source timing/configuration not trusted; not ranked"})
    save(root / "offline-history.json", rows)


def run_config(name, root, manifest):
    live = name.startswith("live-")
    clip_id = "live" if live else name.rsplit("-", 1)[1]
    clip = next(c for c in manifest["clips"] if c["id"] == clip_id)
    fusion = name.startswith("fusion-")
    safe = manifest["safe_configuration"]
    fps = 2.0 if live else float(safe["VIAME_DOWNSAMPLE_FPS"])
    buffer = 6 if name in ("live-retention", "live-repeat") else int(safe["VIAME_TRACKER_BUFFER_FRAMES"])
    return {"clip": clip_id, "path": clip["path"], "sampled_fps": fps,
            "pipeline": str(PIPES / "tracker_default_fish_fusion.pipe") if fusion else str(root / "code/baseline.pipe"),
            "score": float(safe["VIAME_DETECTOR_SCORE_THRESHOLD"]),
            "high": float(safe["VIAME_TRACKER_HIGH_THRESHOLD"]),
            "low": float(safe["VIAME_TRACKER_LOW_THRESHOLD"]),
            "new": float(safe["VIAME_TRACKER_NEW_TRACK_THRESHOLD"]),
            "buffer": buffer, "retention_seconds": buffer / fps,
            "match_thresh": .98, "second_match_thresh": .98, "unconfirmed_match_thresh": .98,
            "std_weight_position": .26666666666666666, "std_weight_velocity": .17804493814764857,
            "frame_number_offset": int(safe["VIAME_FRAME_NUMBER_OFFSET"]),
            "renumber_frames": False, "fishial_enabled": False,
            "motion_resolution": "960x1728" if fusion else None,
            "onnx_resolution": [1024, 1024], "onnx_nms": .5,
            "fusion_note": "Stock fusion graph with current application confidence/buffer overrides" if fusion else None}


def execute_run(name, config, root, journal):
    if not journal.reserve(name, config):
        return
    directory = root / name
    directory.mkdir(exist_ok=False)
    settings = {"input:video_filename": config["path"], "input:video_reader:type": "vidl_ffmpeg",
                "downsampler:target_frame_rate": config["sampled_fps"],
                "downsampler:renumber_frames": "false", "detector:detector:onnx:score_thresh": config["score"],
                "track_writer:file_name": str(directory / "tracks.csv"),
                "detector_writer:file_name": str(directory / "detections.csv")}
    for key, value in {"high_thresh": config["high"], "low_thresh": config["low"],
                       "new_track_thresh": config["new"], "track_buffer": config["buffer"]}.items():
        settings["tracker:track_objects:bytetrack:" + key] = value
    args = ["bash", "-c", 'source "$1" && shift && exec kwiver runner "$@"',
            "benchmark", str(VIAME / "setup_viame.sh"), "-I", str(PIPES), config["pipeline"]]
    for key, value in settings.items():
        args.extend(["-s", f"{key}={value}"])
    timeout = journal.data["runs"][name]["reserved_seconds"]
    # Independent watchdog survives an abrupt death of this Python supervisor.
    # Keep a small margin for process cleanup within the aggregate wall budget.
    args = ["timeout", "--signal=KILL", str(max(.1, timeout - 2))] + args
    journal.finish(name, command=args, status="running", started_unix=time.time(),
                   runner_code_sha256=digest(Path(__file__)))
    started = time.monotonic()
    peak, first_output = 0, None
    process = None
    try:
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(args, cwd=directory, env=clean_env(), stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("Reserved inference timeout reached")
                output = directory / "detections.csv"
                if first_output is None and output.exists() and output.stat().st_size > 0:
                    first_output = time.monotonic() - started
                try:
                    memory = command(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=2)
                    peak = max(peak, sum(int(n) for n in memory.split()))
                except (subprocess.SubprocessError, ValueError):
                    pass
                time.sleep(.5)
            code = process.returncode
        hashes = {p.name: digest(p) for p in directory.glob("*.csv")}
        success = code == 0 and {"tracks.csv", "detections.csv"} <= hashes.keys()
        journal.finish(name, status="complete" if success else "failed", exit_status=code,
                       duration_seconds=time.monotonic() - started, csv_sha256=hashes,
                       peak_device_memory_mib=peak or None, first_csv_write_seconds=first_output,
                       startup_seconds=None,
                       startup_note="First CSV write is an upper bound including buffering, not isolated startup")
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        journal.finish(name, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       duration_seconds=time.monotonic() - started,
                       exit_status=process.returncode if process else None, error_type=type(exc).__name__)
        raise


def quality(m):
    return {"coverage_3": m["temporal_coverage"]["3"], "coverage_5": m["temporal_coverage"]["5"],
            "fragmentation": m["fragmentation_rate"], "duplicates": m["duplicate_pair_rate"],
            "velocity": m["center_velocity"]["median"], "acceleration": m["center_acceleration"]["median"],
            "log_area_change": m["absolute_log_area_change_per_second"]["median"]}


def compare_quality(base, candidate):
    a, b = quality(base), quality(candidate)
    return {key: {"baseline": a[key], "candidate": b[key],
                  "relative_change": ((b[key] - a[key]) / abs(a[key])) if a[key] not in (None, 0)
                  and b[key] is not None else None} for key in a}


def report(root, manifest, journal):
    runs, raw = {}, {}
    for name, entry in journal.data["runs"].items():
        if entry["status"] != "complete":
            runs[name] = entry
            continue
        clip = next(c for c in manifest["clips"] if c["id"] == entry["config"]["clip"])
        directory = root / name
        for filename, sha in entry["csv_sha256"].items():
            if digest(directory / filename) != sha:
                raise RuntimeError("Completed output modified; refusing to rescore")
        # MP4 trimming can retain a nonzero first PTS. The manifest's timestamp
        # array is clip-relative; CSV timestamp strings remain stream-absolute.
        origin = clip["pts_origin"] if "pts_origin" in clip else probe(clip["path"])["pts_origin"]
        det, di = observations(directory / "detections.csv", clip["timestamps"],
                               entry["config"]["frame_number_offset"], origin)
        track, ti = observations(directory / "tracks.csv", clip["timestamps"],
                                 entry["config"]["frame_number_offset"], origin)
        raw[name] = det, track
        m = metrics(det, track, clip["width"], clip["height"], entry["config"]["sampled_fps"])
        detector_parsed = parse_viame_csv(directory / "detections.csv")
        analyzed_lower_bound = max((d.frame_number + 1 for d in detector_parsed.detections), default=0)
        header = (directory / "tracks.csv").read_text(encoding="utf-8").splitlines()[:2]
        writer_time = re.search(r"exec_time:\s*(\d+)", " ".join(header))
        runs[name] = {**entry, "metrics": m, "csv_issues": {"detections": di, "tracks": ti},
                      "pts_origin_seconds": origin,
                      "analyzed_frames_lower_bound": analyzed_lower_bound,
                      "effective_analyzed_fps": analyzed_lower_bound / entry["duration_seconds"],
                      "effective_fps_note": "Lower bound: trailing frames with no detections are unobservable in CSV",
                      "writer_exec_seconds": int(writer_time.group(1)) if writer_time else None,
                      "writer_timing_note": "VIAME CSV header exec_time; not proven to exclude all startup",
                      "runtime_over_source_duration": entry["duration_seconds"] / clip["duration"]}
    pairs = {}
    for left, right in [("baseline-1", "repeat-1"), ("live-baseline", "live-retention"),
                        ("live-retention", "live-repeat")] + [(f"baseline-{i}", f"fusion-{i}") for i in range(3)]:
        if left not in raw or right not in raw:
            continue
        agreement = align(raw[left][0], raw[right][0])
        save(root / f"disagreements-{left}-{right}.json", agreement)
        pairs[f"{left}:{right}"] = {"detection_agreement": agreement["agreement"],
            "matched": agreement["matched"],
            "unique_left": len(agreement["left_unique"]) + len(agreement["left_unaligned"]),
            "unique_right": len(agreement["right_unique"]) + len(agreement["right_unaligned"]),
            "identical_track_partition": canonical(raw[left][1]) == canonical(raw[right][1]),
            "temporal_quality": compare_quality(runs[left]["metrics"], runs[right]["metrics"]),
            "runtime_ratio": runs[right]["duration_seconds"] / runs[left]["duration_seconds"]}
    aggregate = {}
    for profile in ("baseline", "fusion"):
        selected = [runs[f"{profile}-{i}"] for i in range(3) if f"{profile}-{i}" in raw]
        if selected:
            ms = [r["metrics"] for r in selected]
            aggregate[profile] = {"clips": len(ms), "detections": sum(m["detections"] for m in ms),
                "tracks": sum(m["tracks"] for m in ms), "observations": sum(m["observations"] for m in ms),
                "accepted_tracks": {f"{c:.2f}": sum(m["accepted_tracks"][f"{c:.2f}"] for m in ms) for c in THRESHOLDS},
                "macro_quality": {k: stats([quality(m)[k] for m in ms if quality(m)[k] is not None])
                                  for k in quality(ms[0])},
                "per_clip_distributions": {k: [m[k] for m in ms] for k in
                    ("track_length", "length_fractions", "gap_seconds", "center_velocity", "center_acceleration",
                     "absolute_log_area_change_per_second", "out_of_bounds_boxes", "tiny_boxes", "edge_only_tracks")},
                "runtime_seconds": sum(r["duration_seconds"] for r in selected)}
    count, seconds = journal.used()
    result = {"schema_version": 1, "warning": WARNING, "manifest_sha256": digest(root / "manifest.json"),
              "metric_code_sha256": digest(Path(__file__)),
              "budget": {"passes_used": count, "charged_inference_wall_seconds": seconds,
                         "pass_limit": MAX_PASSES, "seconds_limit": MAX_SECONDS,
                         "paid_api_calls": 0, "downloads": 0},
              "runs": runs, "aggregate": aggregate, "paired_comparisons": pairs,
              "ranking": "Pareto component table; no scalar score and no positive credit for detection counts",
              "promotion": {"decision": "retain_baseline", "automatic_promotion": False,
                            "reason": "All seven criteria require positive evidence; unknown is not a pass",
                            "criteria": promotion_criteria(runs, pairs),
                            "candidate_evidence": candidate_evidence(runs, pairs)},
              "not_tested": ["Association alternatives: valid IoU distance gates, but lower priority within ten passes",
                             "5 FPS live: omitted to prioritize retention and repeated baseline",
                             "Enhancement / new model / training: unsupported by this bounded evidence"],
              "limitations": [manifest["limitation"], "No trusted labels; proxies cannot prove accuracy",
                              "Live replay does not measure cross-segment joins or dropped segments",
                              "Device memory includes any concurrent GPU processes",
                              "Fixed startup not separately observable in installed logs; total time is conservative",
                              "First process may pay cold caches; compare repeat runtime before interpreting speed gains",
                              "One repeated baseline estimates only a lower bound on run-to-run noise"],
              "optional_labels": "Annotate all fish in ~30 stratified held-out frames (including empty frames); "
                                 "independently review boxes and use a fixed IoU rule for preliminary precision/recall. "
                                 "Identity/fragmentation accuracy also needs a few fully tracked short sequences."}
    save(root / "report.json", result)
    lines = ["# VIAME zero-label benchmark", "", WARNING, "",
             f"Decision: **retain production baseline**. {count}/{MAX_PASSES} passes; "
             f"{seconds:.3f}/{MAX_SECONDS} charged wall seconds; 0 paid API calls; 0 downloads.", "",
             "## Per-clip Pareto table", "",
             "Counts are context only. Trajectory statistics weight each track equally. Lower motion is not necessarily better.", "",
             "| Run | Detections | Tracks | Accepted .45/.50/.55/.60/.65 | Coverage ≥3 / ≥5 | Fragment rate | Duplicate rate | Acceleration | Seconds |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for name, run in runs.items():
        if "metrics" not in run:
            lines.append(f"| {name} ({run['status']}) | — | — | — | — | — | — | — | — |")
            continue
        m = run["metrics"]
        lines.append(f"| {name} | {m['detections']} | {m['tracks']} | "
            + "/".join(str(m["accepted_tracks"][f"{c:.2f}"]) for c in THRESHOLDS)
            + f" | {m['temporal_coverage']['3']} / {m['temporal_coverage']['5']} | {m['fragmentation_rate']} "
            + f"| {m['duplicate_pair_rate']} | {m['center_acceleration']['median']} | {run['duration_seconds']:.3f} |")
    lines += ["", "## Promotion criteria", "", "| Criterion | Evidence / outcome |", "|---|---|"]
    for criterion, value in result["promotion"]["criteria"].items():
        lines.append(f"| {criterion} | {value} |")
    for candidate, evidence in result["promotion"]["candidate_evidence"].items():
        lines += ["", f"### {candidate}", "", "| Criterion | Status |", "|---|---|"]
        lines += [f"| {key} | {value['status']} |" for key, value in evidence.items()]
    lines += ["", "Full component values, aggregate distributions, paired values and CSV diagnostics are in report.json.",
              "Raw unique/matched boxes and timestamps are retained in disagreements-*.json; unique does not mean false.",
              "", "## Limits and interpretation", ""] + [f"- {s}" for s in result["limitations"]]
    lines += ["", "No production parameters changed. No model-weight training was justified.", "",
              "## Deferred hypotheses", ""] + [f"- {s}" for s in result["not_tested"]]
    lines += ["", "Optional future truth set: " + result["optional_labels"], ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def promotion_criteria(runs, pairs):
    repeat = pairs.get("baseline-1:repeat-1", {})
    retention = pairs.get("live-baseline:live-retention", {})
    fusion = {str(i): pairs.get(f"baseline-{i}:fusion-{i}", {}) for i in range(3)}
    return {
        "1 Determinism": f"Baseline agreement={repeat.get('detection_agreement')}; identical track partition="
                         f"{repeat.get('identical_track_partition')}. Fusion not repeated; candidate tolerance unproven.",
        "2 Two independent proxies on every clip": "See paired_comparisons.temporal_quality. Live retention tested "
            "on one clip only; cannot pass every-clip criterion. Coverage 3 and 5 are one persistence family.",
        "3 No >5% regression": f"Per-proxy baseline/candidate/relative changes published for each pair; "
            f"live representative={retention.get('temporal_quality')}. Missing or zero denominators require review.",
        "4 Motion-fusion agreement": f"Baseline/fusion detection agreement by clip="
            f"{ {k: v.get('detection_agreement') for k, v in fusion.items()} }. "
            "No matched live-FPS fusion reference, so live candidate is unproven. Fusion cannot independently validate itself.",
        "5 Threshold stability": "Exact five-cutoff counts in every run and aggregate. Stability is contextual; "
            "extra counts do not earn promotion credit.",
        "6 Runtime / dropped segments": f"Fusion/baseline runtime ratios="
            f"{ {k: v.get('runtime_ratio') for k, v in fusion.items()} }; must be ≤1.25. "
            f"Live total runtime/duration={runs.get('live-retention', {}).get('runtime_over_source_duration')}; "
            "must be ≤0.8 after startup amortization. Startup isolated time and dropped segments unmeasured.",
        "7 Configurable / provenance / tests": "Experiment parameters and hashes persisted. No production candidate "
            "promotion or Settings changes; test results documented separately in README/work report."}


def candidate_evidence(runs, pairs):
    """Fail-closed evidence by criterion; no score or detection-count objective.

    Each coverage threshold belongs to the same persistence family. Smoothness
    uses acceleration, not low velocity (stationary background can be smooth).
    A positive change must exceed the observed repeated-baseline absolute noise.
    """
    repeat = pairs.get("baseline-1:repeat-1", {})
    noise = {k: abs(v["candidate"] - v["baseline"]) if v["candidate"] is not None
             and v["baseline"] is not None else None
             for k, v in repeat.get("temporal_quality", {}).items()}
    result = {}
    for candidate, comparisons in {
        "motion_fusion": [(f"baseline-{i}", f"fusion-{i}") for i in range(3)],
        "live_retention_3s": [("live-baseline", "live-retention")],
    }.items():
        clips = []
        for base_name, name in comparisons:
            if "metrics" not in runs.get(base_name, {}) or "metrics" not in runs.get(name, {}):
                continue
            base, new = runs[base_name], runs[name]
            components = compare_quality(base["metrics"], new["metrics"])
            improvements, regressions = set(), []
            for key, component in components.items():
                a, b = component["baseline"], component["candidate"]
                if a is None or b is None:
                    continue
                higher = key.startswith("coverage")
                better = b - a if higher else a - b
                family = "persistence" if higher else key
                if key != "velocity" and noise.get(key) is not None and better > max(noise[key], 1e-9):
                    improvements.add(family)
                if -better > .05 * abs(a) + 1e-9:
                    regressions.append(key)
            acceptance = {}
            for label, run in (("baseline", base), ("candidate", new)):
                counts = run["metrics"]["accepted_tracks"]
                values = [counts[f"{c:.2f}"] for c in THRESHOLDS]
                acceptance[label] = {"counts": counts, "largest_adjacent_relative_drop":
                    max((a - b) / max(a, 1) for a, b in zip(values, values[1:]))}
            clips.append({"baseline_run": base_name, "candidate_run": name,
                          "components": components, "independent_improved_families": sorted(improvements),
                          "regressions_over_5_percent": regressions, "acceptance": acceptance,
                          "runtime_ratio": new["duration_seconds"] / base["duration_seconds"],
                          "live_runtime_over_duration": new["runtime_over_source_duration"]})
        is_live = candidate.startswith("live")
        all_clips = len(clips) == 3
        repeat_pair = pairs.get("live-retention:live-repeat", {}) if is_live else {}
        result[candidate] = {
            "1_determinism": {"status": "pass_on_representative_only" if repeat_pair.get("identical_track_partition")
                and repeat_pair.get("detection_agreement") == 1 else "unproven",
                "baseline_noise_absolute": noise, "candidate_repeat": repeat_pair,
                "baseline_repeat_runtime_ratio": repeat.get("runtime_ratio")},
            "2_independent_temporal_improvements": {"status": "pass" if all_clips and all(
                len(c["independent_improved_families"]) >= 2 for c in clips) else "not_met",
                "required_clips": 3, "tested_clips": len(clips), "clip_values": clips},
            "3_no_regression_over_5_percent": {"status": "fail" if any(
                c["regressions_over_5_percent"] for c in clips) else ("pass" if all_clips else "unproven")},
            "4_independent_fusion_agreement": {"status": "unproven",
                "reason": "No same-FPS live fusion reference" if is_live else "Fusion cannot validate itself"},
            "5_nearby_threshold_stability": {"status": "pass" if all_clips and all(
                c["acceptance"]["candidate"]["largest_adjacent_relative_drop"] <=
                c["acceptance"]["baseline"]["largest_adjacent_relative_drop"] + .05 for c in clips) else "not_met",
                "rule": "Largest adjacent relative acceptance drop no more than baseline +5 percentage points"},
            "6_runtime": {"status": ("fail" if any(c["live_runtime_over_duration"] > .8 for c in clips)
                else "unproven_dropped_segments") if is_live else ("pass" if all_clips and all(
                    c["runtime_ratio"] <= 1.25 for c in clips) else "not_met"),
                "limit": .8 if is_live else 1.25},
            "7_production_configuration": {"status": "not_promoted",
                "reason": "Experiment settings/outputs provenanced; prerequisites for production change not met"},
        }
    return result


def docker_dispatch(report_only=False):
    root = ROOT.resolve()
    code = root / "code"
    code.mkdir(parents=True, exist_ok=True)
    script_name = f"benchmark-{digest(Path(__file__))[:16]}.py"
    snapshots = {script_name: Path(__file__), "baseline.pipe": Path("app/viame_pipelines/tracker_fish_no_motion.pipe")}
    for name, source in snapshots.items():
        target = code / name
        if not target.exists():
            shutil.copyfile(source, target)
        elif digest(source) != digest(target):
            raise RuntimeError(f"Frozen {name} differs; resume with existing frozen code or review before changing")
    safe_path = code / "safe-config.json"
    if not safe_path.exists():
        # Read only the explicit numerical allowlist from .env; never load Settings
        # or capture docker compose config, either of which contains credentials.
        safe = {"LIVE_FPS": "5", "MIN_FISH_CONFIDENCE": ".60", "VIAME_DOWNSAMPLE_FPS": "10",
                "VIAME_DETECTOR_SCORE_THRESHOLD": ".10", "VIAME_TRACKER_HIGH_THRESHOLD": ".45",
                "VIAME_TRACKER_LOW_THRESHOLD": ".10", "VIAME_TRACKER_NEW_TRACK_THRESHOLD": ".50",
                "VIAME_TRACKER_BUFFER_FRAMES": "30", "VIAME_FRAME_NUMBER_OFFSET": "-1"}
        if Path(".env").exists():
            for line in Path(".env").read_text(encoding="utf-8-sig").splitlines():
                key, _, value = line.partition("=")
                if key in safe and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.strip()):
                    safe[key] = value.strip()
        for key in safe:
            if key in os.environ and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", os.environ[key]):
                safe[key] = os.environ[key]
        save(safe_path, safe)
    container_root = "/data/outputs/viame-zero-label-benchmark"
    subprocess.run(COMPOSE + ["exec", "-T", "-e", "FISHIAL_ENABLED=false", "worker", "bash", "-c",
        'source /opt/noaa/viame/setup_viame.sh && exec python "$1" "$2"',
        "benchmark", container_root + "/code/" + script_name,
        "--report-only" if report_only else "--execute"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true", help="Run frozen harness in the existing GPU worker")
    parser.add_argument("--execute", action="store_true", help="Internal container entry point")
    parser.add_argument("--report-only", action="store_true", help="Rescore the existing journal, no inference")
    args = parser.parse_args()
    if args.docker:
        docker_dispatch(args.report_only)
        return
    if not (args.execute or args.report_only):
        parser.error("Use --docker from the repository root")
    root = Path("/data/outputs/viame-zero-label-benchmark")
    root.mkdir(parents=True, exist_ok=True)
    with lock(root):
        manifest = prepare(root)
        journal = Journal(root / "journal.json", digest(root / "manifest.json"))
        if args.report_only:
            report(root, manifest, journal)
            return
        if not (root / "offline-history.json").exists():
            offline_history(root)
        try:
            for name in manifest["plan"]:
                print(f"Benchmark {name}: resume or run", flush=True)
                execute_run(name, run_config(name, root, manifest), root, journal)
                result = report(root, manifest, journal)
                if journal.data["runs"][name]["status"] != "complete":
                    raise RuntimeError("Inference did not complete; stopping rather than spending further passes")
            # Spend the reserved pass on candidate repeat only if it shows a change
            # worth checking. Identical output has no independent proxy improvement.
            pair = result["paired_comparisons"].get("live-baseline:live-retention", {})
            if pair and not pair["identical_track_partition"]:
                execute_run("live-repeat", run_config("live-repeat", root, manifest), root, journal)
        finally:
            result = report(root, manifest, journal)
        print(json.dumps({"decision": "retain_baseline", "budget": result["budget"]}), flush=True)


if __name__ == "__main__":
    main()
