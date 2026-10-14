"""SAM 3 auto-labelling, BoT-SORT tracking and static-object suppression, on any upload.

PowerShell, from the repository root (host .venv; Docker only for the read-only VIAME query):

    $env:FISHIAL_PREPROCESS='none'; $env:YOLO_OFFLINE='true'
    .venv\\Scripts\\python.exe -m scripts.sam3_botsort run --video-id VIDEO_UUID

Stages (each takes ``--video-id`` or ``--video``; ``run`` does them in order):

- ``facts``: Part 0 facts that need no GPU (installed trackers, SAM 3 requirements, video).
- ``viame``: VIAME's stored result for the video, read-only from the database, checked
  against the job's own CSV; also its analysed frames and raw detections.
- ``sam3-probe``: Part 0 SAM 3 facts on this GPU (precision, input size, speed, memory,
  alignment overlay).
- ``sam3``: SAM 3 on the video for each prompt variant.
- ``static``: static-object suppression for every detector output present.
- ``labels``: reliable SAM 3 labels as an unreviewed VIAME-style candidate CSV.
- ``compare``: VIAME vs SAM 3 agreement and temporal proxies (no labels, no accuracy).
- ``render``: side-by-side H.264 video.
- ``track``: BoT-SORT and ByteTrack over the detections, after static suppression.
- ``brackishmot``: BoT-SORT replayed over the saved VIAME BrackishMOT detections and scored
  against real labels (Part 4.1; not per video).

Everything generated goes under ``data/outputs/sam3-botsort-<UTC date>/``. ``journal.json``
there records each step's inputs, settings, hashes, timings and status; a completed step
with unchanged inputs is never redone, and the SAM 3 GPU, training and VIAME budgets are
charged from it. Nothing here writes to the database, uploads, jobs, review decisions or
CVAT. Ultralytics (trackers, SAM 3 code) is AGPL-3.0.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from app.services.static_filter import StaticSettings, background_ncc, filter_static
from scripts.viame_zero_label_benchmark import digest, metrics, read, save

OUTPUTS = Path("data/outputs")
UPLOADS = Path("data/uploads")
JOBS = Path("data/jobs")
BUDGETS = {"sam3_gpu": 2 * 3600, "training": 2 * 3600, "viame": 30 * 60}
COMPOSE = ["docker", "compose"]
# Ultralytics stays offline: no analytics, no auto-install, no asset download.
OFFLINE = {"YOLO_OFFLINE": "true", "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
           "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9"}
PROMPTS = {"fish": "fish", "small-fish": "small fish"}
# Reliable-label rules, fixed after the Part 0 probe and before any comparison (see the
# results document): a SAM 3 track becomes a label only if it is long enough, confident
# enough and not static.
LABEL_RULES = {"min_track_frames": 5, "min_track_median_score": 0.5, "min_box_score": 0.25}
ACCEPT = .60  # Production MIN_FISH_CONFIDENCE: a VIAME track is shown when its best box reaches it.
PROXY_KEYS = ("fragmentation_rate", "duplicate_pair_rate")


def utc_today() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


# ------------------------------------------------------------------ journal --

class BudgetExceeded(RuntimeError):
    pass


class Journal:
    """Resumable record of every step. A step is skipped when it completed with the same
    inputs and settings; a step left "running" by a dead process is charged its elapsed
    time to its last heartbeat and redone."""

    def __init__(self, root: Path, budgets: dict | None = None):
        self.root = root
        self.path = root / "journal.json"
        self.touched: dict[str, set] = {"entries": set(), "decisions": set()}
        root.mkdir(parents=True, exist_ok=True)
        with self._lock():
            if self.path.exists():
                self.data = read(self.path)
            else:
                self.data = {"schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
                             "budgets_seconds": dict(budgets or BUDGETS), "entries": {}, "decisions": {}}
                save(self.path, self.data)

    @contextmanager
    def _lock(self, timeout: float = 60.0):
        """A lock directory (mkdir is atomic on Windows too) around each read-modify-write,
        so stages may run side by side; one left by a killed writer expires."""
        lock = self.root / ".journal.lock"
        deadline = time.monotonic() + timeout
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > timeout:
                    lock.rmdir()
                elif time.monotonic() > deadline:
                    raise RuntimeError(f"{lock} is held; is another stage stuck?")
                else:
                    time.sleep(.05)
        try:
            yield
        finally:
            lock.rmdir()

    def save(self) -> None:
        """Merge: re-read the file, overlay only what this process changed, write it back,
        and pick up other processes' entries (their time counts against the budgets)."""
        with self._lock():
            merged = read(self.path)
            for section, names in self.touched.items():
                for name in names:
                    merged[section][name] = self.data[section][name]
            save(self.path, merged)
            for section, names in self.touched.items():
                for name, value in merged[section].items():
                    if name not in names:
                        self.data[section][name] = value

    def charged(self, kind: str) -> float:
        return sum(attempt["duration_seconds"] for entry in self.data["entries"].values()
                   if entry.get("kind") == kind for attempt in entry.get("attempts", []))

    def remaining(self, kind: str) -> float | None:
        budget = self.data["budgets_seconds"].get(kind)
        return None if budget is None else budget - self.charged(kind)

    def run(self, name: str, kind: str, inputs: dict, settings: dict, step, estimate: float = 0.0):
        key = digest_json({"inputs": inputs, "settings": settings})
        entry = self.data["entries"].get(name)
        if entry and entry["status"] == "complete" and entry["key"] == key:
            return entry["result"]
        entry = entry or {"kind": kind, "attempts": []}
        if entry.get("status") == "running":  # A previous process died mid-step.
            entry["attempts"].append({"status": "interrupted", "started_unix": entry["started_unix"],
                                      "duration_seconds": max(0.0, entry["heartbeat_unix"] - entry["started_unix"])})
        remaining = self.remaining(kind)
        if remaining is not None and estimate > remaining:
            raise BudgetExceeded(f"{name}: needs about {estimate:.0f} s of {kind}, {remaining:.0f} s left")
        now = time.time()
        entry |= {"kind": kind, "status": "running", "key": key, "inputs": inputs, "settings": settings,
                  "started_unix": now, "heartbeat_unix": now}
        self.data["entries"][name] = entry
        self.touched["entries"].add(name)
        self.save()

        def heartbeat(**progress):
            entry["heartbeat_unix"] = time.time()
            entry["progress"] = progress
            if remaining is not None and time.time() - entry["started_unix"] > remaining:
                raise BudgetExceeded(f"{name}: {kind} budget used up")
            self.save()

        started = time.monotonic()
        try:
            result = step(heartbeat)
        except BaseException as error:
            entry["attempts"].append({"status": "failed", "started_unix": entry["started_unix"],
                                      "duration_seconds": time.monotonic() - started,
                                      "error": f"{type(error).__name__}: {error}"[-2000:]})
            entry["status"] = "failed"
            self.save()
            raise
        entry["attempts"].append({"status": "complete", "started_unix": entry["started_unix"],
                                  "duration_seconds": time.monotonic() - started})
        entry |= {"status": "complete", "result": result, "finished_unix": time.time()}
        self.save()
        return result

    def decide(self, name: str, value) -> None:
        self.data["decisions"][name] = value | {"decided_utc": datetime.now(UTC).isoformat()}
        self.touched["decisions"].add(name)
        self.save()


def digest_json(value) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


_FILE_HASHES: dict = {}


def file_id(path: Path) -> dict:
    """Path, size and SHA256 (hashed once per process for an unchanged file)."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_HASHES:
        _FILE_HASHES[key] = digest(path)
    return {"path": path.as_posix(), "bytes": stat.st_size, "sha256": _FILE_HASHES[key]}


def resolve_root(root: Path | None, key: str | None) -> Path:
    """An explicit root; else the newest experiment root that already holds this video
    (so a later day resumes instead of redoing work); else a new root dated today."""
    if root is not None:
        return root
    if key:
        existing = sorted(p.parent.parent for p in OUTPUTS.glob(f"sam3-botsort-*/videos/{key}/video.json"))
        if existing:
            return existing[-1]
    return OUTPUTS / f"sam3-botsort-{utc_today()}"


# ------------------------------------------------------------ video and DB --

def psql_copy(query: str) -> list[dict]:
    """Rows of a SELECT, read through the compose ``postgres`` service. The session is
    forced read-only (``default_transaction_read_only``), so no statement can write."""
    command = [*COMPOSE, "exec", "-T", "-e", "PGOPTIONS=-c default_transaction_read_only=on", "postgres",
               "psql", "-U", "fish", "-d", "fish_monitor", "-q", "-v", "ON_ERROR_STOP=1",
               "-c", f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER true)"]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    if completed.returncode:
        raise RuntimeError(f"Read-only database query failed (is Docker running?): {completed.stderr.strip()[-500:]}")
    return list(csv.DictReader(completed.stdout.splitlines()))


def local_path(storage_path: str) -> Path:
    """A container path such as /data/uploads/x.mp4 on this side."""
    parts = PurePosixPath(storage_path).parts
    if parts[:2] != ("/", "data"):
        raise ValueError(f"Unexpected storage path {storage_path}")
    return Path("data").joinpath(*parts[2:])


def probe_video(path: Path) -> dict:
    """Decoded frame count, rate, size and SHA256 of the file every model reads."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = 0
    while capture.grab():
        decoded += 1
    capture.release()
    return file_id(path) | {"fps": fps, "width": width, "height": height, "declared_frames": declared,
                            "decoded_frames": decoded, "seconds": decoded / fps if fps else None}


def video_lookup(video_id: str) -> dict:
    rows = psql_copy("SELECT id, original_filename, storage_path, fps, width, height, duration_seconds, "
                     f"content_sha256, processing_status FROM videos WHERE id = '{uuid.UUID(video_id)}'")
    if not rows:
        raise SystemExit(f"No video {video_id} in the database")
    return rows[0]


def video_record(args, journal_root: Path | None = None) -> tuple[Path, dict]:
    """Video facts for --video-id (read-only DB lookup) or --video (a path). Both together
    name the upload without the database (Docker down); the viame stage checks the record
    later."""
    if args.video_id and args.video:
        path, record, key = Path(args.video), {"id": str(uuid.UUID(args.video_id)), "not_consulted": True}, \
            str(uuid.UUID(args.video_id))
    elif args.video_id:
        record = video_lookup(args.video_id)
        path, key = local_path(record["storage_path"]), record["id"]
    else:
        path, record, key = Path(args.video), {}, Path(args.video).stem
    root = resolve_root(journal_root or args.root, key)
    directory = root / "videos" / key
    cached = directory / "video.json"
    if cached.exists():
        facts = read(cached)
        if facts["file"]["sha256"] != file_id(path)["sha256"]:
            raise RuntimeError(f"{path} changed since {cached} was written")
        return root, facts
    facts = {"key": key, "video_id": record.get("id"), "database": record, "file": probe_video(path),
             "probed_utc": datetime.now(UTC).isoformat()}
    if record.get("content_sha256") and record["content_sha256"] != facts["file"]["sha256"]:
        raise RuntimeError("The upload's SHA256 differs from the database record")
    directory.mkdir(parents=True, exist_ok=True)
    save(cached, facts)
    return root, facts


def video_dir(root: Path, facts: dict) -> Path:
    return root / "videos" / facts["key"]


def frame_time(frame: int, facts: dict) -> float:
    return frame / facts["file"]["fps"]


# ------------------------------------------------------------------ VIAME --

def _csv_rows(path: Path) -> list[list[str]]:
    return [r for r in csv.reader(path.open(encoding="utf-8")) if r and not r[0].startswith("#")]


def viame_analysed_frames(detections_csv: Path, fps: float) -> dict:
    """Source frames VIAME's downsampler passed, from the detection writer: column 3 is a
    zero-based ordinal over every frame it received, column 2 that frame's timestamp."""
    by_ordinal = {}
    for row in _csv_rows(detections_csv):
        hours, minutes, seconds = row[1].split(":")
        t = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        frame = t * fps
        if abs(frame - round(frame)) > 1e-3:
            raise ValueError(f"{detections_csv}: timestamp {row[1]} is not on a frame")
        if by_ordinal.setdefault(int(row[2]), round(frame)) != round(frame):
            raise ValueError(f"{detections_csv}: ordinal {row[2]} has two timestamps")
    ordinals = sorted(by_ordinal)
    missing = sorted(set(range(ordinals[-1] + 1)) - set(ordinals)) if ordinals else []
    frames = sorted(by_ordinal.values())
    # An ordinal with no detection leaves no row; place it between its neighbours.
    for ordinal in missing:
        before = max((o for o in ordinals if o < ordinal), default=None)
        after = min((o for o in ordinals if o > ordinal), default=None)
        if before is not None and after is not None:
            share = (ordinal - before) / (after - before)
            frames.append(round(by_ordinal[before] + share * (by_ordinal[after] - by_ordinal[before])))
    return {"frames": sorted(set(frames)), "ordinals": len(ordinals) + len(missing),
            "ordinals_without_detections": len(missing),
            "exact": not missing, "source": detections_csv.as_posix()}


def viame_raw_detections(detections_csv: Path, fps: float) -> list[dict]:
    rows = []
    for row in _csv_rows(detections_csv):
        hours, minutes, seconds = row[1].split(":")
        frame = round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * fps)
        rows.append({"id": None, "frame": frame, "t": frame / fps,
                     "box": [float(v) for v in row[3:7]], "confidence": float(row[7])})
    return rows


def viame_stage(args) -> dict:
    """VIAME's stored result: database rows (read-only), cross-checked against the job's
    CSV, plus its analysed frames, raw detections and run time. Never re-runs VIAME."""
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts) / "viame"
    fps = facts["file"]["fps"]

    def step(heartbeat):
        database_video = None
        if args.viame_source == "db":
            if not facts["video_id"]:
                raise SystemExit("--viame-source db needs --video-id")
            database_video = video_lookup(facts["video_id"])
            if local_path(database_video["storage_path"]).resolve() != Path(facts["file"]["path"]).resolve():
                raise RuntimeError(f"The database stores this video at {database_video['storage_path']}")
            if database_video["content_sha256"] not in ("", None, facts["file"]["sha256"]):
                raise RuntimeError("The upload's SHA256 differs from the database record")
            jobs = psql_copy("SELECT id, status, created_at, started_at, finished_at, output_csv_path, worker_mode, "
                             f"configuration_json FROM processing_jobs WHERE video_id = '{facts['video_id']}' "
                             "ORDER BY created_at")
            chosen = [j for j in jobs if (j["id"] == args.viame_job if args.viame_job else j["status"] == "completed")]
            if not chosen:
                raise SystemExit("No completed VIAME job for this video")
            job = chosen[-1]
            rows = psql_copy(
                "SELECT t.viame_track_id, t.review_state, d.frame_number, d.timestamp_seconds, d.x1, d.y1, "
                "d.x2, d.y2, d.confidence FROM fish_tracks t JOIN fish_detections d ON d.fish_track_id = t.id "
                f"WHERE t.video_id = '{facts['video_id']}' AND t.processing_job_id = '{job['id']}' "
                "ORDER BY d.frame_number, t.viame_track_id")
            tracks = [{"id": r["viame_track_id"], "frame": int(r["frame_number"]),
                       "t": int(r["frame_number"]) / fps, "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                       "confidence": float(r["confidence"])} for r in rows]
            review_states = sorted({r["review_state"] for r in rows})
            csv_path = local_path(job["output_csv_path"]) if job["output_csv_path"] else None
            offset = int(json.loads(job["configuration_json"] or "{}").get("frame_number_offset", -1))
        else:  # Docker down: the job's own CSV, recorded as such.
            job = {"id": args.viame_job}
            csv_path = next((JOBS / args.viame_job).glob("*/viame_tracks.csv"))
            offset, review_states = args.viame_offset, None
            tracks = None
        from app.services.viame_parser import parse_viame_csv

        parsed = parse_viame_csv(csv_path, strict=True, frame_number_offset=offset)
        from_csv = [{"id": d.track_id, "frame": d.frame_number, "t": d.frame_number / fps,
                     "box": [d.bbox_left, d.bbox_top, d.bbox_right, d.bbox_bottom], "confidence": d.confidence}
                    for d in parsed.detections]
        canonical = lambda rows: sorted((r["id"], r["frame"], *[round(v, 3) for v in r["box"]],
                                         round(r["confidence"], 5)) for r in rows)
        database_matches_csv = None if tracks is None else canonical(tracks) == canonical(from_csv)
        if tracks is None:
            tracks = from_csv
        header = csv_path.read_text(encoding="utf-8").splitlines()[:2]
        exec_time = next((int(m.group(1)) for line in header if (m := re.search(r"exec_time: (\d+)", line))), None)
        detections_csv = csv_path.parent / "computed_detections.csv"
        analysed = viame_analysed_frames(detections_csv, fps) if detections_csv.exists() else {
            "frames": sorted({r["frame"] for r in tracks}), "exact": False,
            "source": "track frames only (no detection CSV)"}
        if not {r["frame"] for r in tracks} <= set(analysed["frames"]):
            raise ValueError("A VIAME track box lies on a frame VIAME did not analyse")
        raw = viame_raw_detections(detections_csv, fps) if detections_csv.exists() else []
        directory.mkdir(parents=True, exist_ok=True)
        write_rows(tracks, directory / "tracks.jsonl")
        write_rows(raw, directory / "raw.jsonl")
        save(directory / "analysed.json", analysed)
        result = {"job_id": job["id"], "source": args.viame_source, "tracks": len({r["id"] for r in tracks}),
                  "boxes": len(tracks), "raw_detections": len(raw), "analysed_frames": len(analysed["frames"]),
                  "analysed_exact": analysed["exact"], "database_matches_job_csv": database_matches_csv,
                  "review_states": review_states, "job_csv": file_id(csv_path), "frame_number_offset": offset,
                  "viame_exec_seconds": exec_time, "database_video": database_video,
                  "job_times": {k: job.get(k) for k in ("created_at", "started_at", "finished_at")}}
        save(directory / "viame.json", result)
        return result

    inputs = {"video": facts["file"]["sha256"], "source": args.viame_source, "job": args.viame_job}
    return journal.run(f"{facts['key']}/viame/{args.viame_source}", "read", inputs, {}, step)


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ------------------------------------------------------------- detectors --

def detector_outputs(directory: Path, facts: dict) -> dict[str, dict]:
    """Every detector output present for a video: rows, analysed frames and frame rate."""
    outputs = {}
    viame = directory / "viame"
    if (viame / "tracks.jsonl").exists():
        analysed = read(viame / "analysed.json")["frames"]
        outputs["viame"] = {"rows": read_rows(viame / "tracks.jsonl"), "analysed": analysed,
                            "sampled_fps": len(analysed) / facts["file"]["seconds"]}
        if (viame / "raw.jsonl").exists():
            outputs["viame-raw"] = {"rows": read_rows(viame / "raw.jsonl"), "analysed": analysed,
                                    "sampled_fps": len(analysed) / facts["file"]["seconds"]}
    for frames_path in sorted((directory / "sam3").glob("*/frames.jsonl")):
        from scripts.sam3_autolabel import load_frames, observation_rows

        frames = load_frames(frames_path)
        tracked = frames_path.parent / "tracked.jsonl"  # Detector-only runs get BoT-SORT identities
        outputs[f"sam3-{frames_path.parent.name}"] = {
            "rows": read_rows(tracked) if tracked.exists() else observation_rows(frames, facts["file"]["fps"]),
            "analysed": [f["frame"] for f in frames], "sampled_fps": facts["file"]["fps"]}
    for labels_path in sorted((directory / "labels").glob("*/rows.jsonl")):  # Already static-filtered
        frames = outputs.get(f"sam3-{labels_path.parent.name}", {}).get("analysed")
        if frames is not None:
            outputs[f"labels-{labels_path.parent.name}"] = {"rows": read_rows(labels_path), "analysed": frames,
                                                            "sampled_fps": facts["file"]["fps"]}
    for tracks_path in sorted((directory / "student").glob("*/detections.jsonl")):
        rows = read_rows(tracks_path)
        analysed = read(tracks_path.parent / "analysed.json")["frames"]
        outputs[f"student-{tracks_path.parent.name}"] = {"rows": rows, "analysed": analysed,
                                                         "sampled_fps": facts["file"]["fps"]}
    return outputs


# ------------------------------------------------------------------ static --

def median_background(video: Path, frames: list[int], samples: int = 60, scale: float = .5):
    """Per-pixel median of evenly spaced frames, grey, at ``scale``: anything present for
    less than half the video drops out, so fish leave it and fixtures stay."""
    import cv2
    import numpy as np

    wanted = set(frames[::max(1, len(frames) // samples)][:samples])
    capture, stack, index = cv2.VideoCapture(str(video)), [], 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index in wanted:
            _, image = capture.retrieve()
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            stack.append(cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        index += 1
    capture.release()
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def background_scores(video: Path, rows: list[dict], background, scale: float = .5,
                      per_track: int = 8) -> dict:
    """Median NCC of up to ``per_track`` crops per track against the median background,
    read in one sequential pass over the video."""
    import cv2
    import numpy as np

    tracks = defaultdict(list)
    for row in rows:
        if row.get("id") is not None:
            tracks[row["id"]].append(row)
    wanted = defaultdict(list)
    for track_id, members in tracks.items():
        members.sort(key=lambda r: r["frame"])
        for row in members[::max(1, len(members) // per_track)][:per_track]:
            wanted[row["frame"]].append((track_id, row["box"]))
    scores = defaultdict(list)
    capture, index = cv2.VideoCapture(str(video)), 0
    last = max(wanted, default=-1)
    while index <= last and capture.grab():
        if index in wanted:
            _, image = capture.retrieve()
            grey = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), background.shape[::-1],
                              interpolation=cv2.INTER_AREA)
            for track_id, box in wanted[index]:
                x1, y1, x2, y2 = (round(v * scale) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(grey.shape[1], x2), min(grey.shape[0], y2)
                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    value = background_ncc(grey[y1:y2, x1:x2], background[y1:y2, x1:x2])
                    if value is not None:
                        scores[track_id].append(value)
        index += 1
    capture.release()
    return {track_id: float(np.median(values)) for track_id, values in scores.items()}


def static_stage(args) -> dict:
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts)
    video = Path(facts["file"]["path"])
    settings = StaticSettings()
    results = {}
    outputs = detector_outputs(directory, facts)
    background = None
    for name, output in outputs.items():
        if name.startswith("labels-"):  # Labels are made from static-filtered tracks
            continue

        def step(heartbeat, name=name, output=output):
            nonlocal background
            scores = None
            if any(r.get("id") is not None for r in output["rows"]):
                if background is None:
                    import cv2

                    background = median_background(video, list(range(facts["file"]["decoded_frames"])))
                    cv2.imwrite(str(directory / "static" / "median-background.png"), background)
                scores = background_scores(video, output["rows"], background)
            report = filter_static(output["rows"], analysed_frames=output["analysed"], fps=facts["file"]["fps"],
                                   settings=settings, background_scores=scores)
            target = directory / "static" / name
            write_rows(report["kept"], target / "kept.jsonl")
            write_rows(report["removed"], target / "removed.jsonl")
            summary = {k: report[k] for k in ("counts", "zones", "settings")} | {
                "removed_tracks": {k: v for k, v in report["tracks"].items() if v["removed"]},
                "background_flag_only": {k: v for k, v in report["tracks"].items()
                                         if "background_match" in v["flags"] and not v["removed"]}}
            save(target / "report.json", summary | {"tracks": report["tracks"]})
            static_sheet(video, report, target / "contact-sheet.jpg", name)
            return summary | {"tracks": None}

        (directory / "static").mkdir(parents=True, exist_ok=True)
        inputs = {"video": facts["file"]["sha256"], "rows": digest_json(output["rows"])}
        results[name] = journal.run(f"{facts['key']}/static/{name}", "cpu", inputs,
                                    {"static": settings.__dict__, "background": "median of 60 frames, grey, half size"},
                                    step)
    return {name: r["counts"] for name, r in results.items()}


def static_sheet(video: Path, report: dict, path: Path, name: str, tiles: int = 6) -> None:
    """Frames with the most removed boxes: removed red, kept green, static zones yellow."""
    import cv2
    import numpy as np

    removed = defaultdict(list)
    for row in report["removed"]:
        removed[row["frame"]].append(row)
    kept = defaultdict(list)
    for row in report["kept"]:
        kept[row["frame"]].append(row)
    # Spread the chosen frames over the video, preferring frames with many removals.
    ranked = sorted(removed, key=lambda f: (-len(removed[f]), f))
    chosen = []
    for frame in ranked:
        if all(abs(frame - c) > 60 for c in chosen):
            chosen.append(frame)
        if len(chosen) == tiles:
            break
    if len(chosen) < 2:  # Nothing (or almost nothing) removed: show kept boxes instead.
        frames = sorted(kept)
        chosen += [frames[i] for i in np.linspace(0, len(frames) - 1, tiles - len(chosen)).astype(int)] if frames else []
    capture, images = cv2.VideoCapture(str(video)), []
    for frame in sorted(set(chosen)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            continue
        for zone in report["zones"]:
            x1, y1, x2, y2 = map(int, zone["box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for colour, rows in (((0, 200, 0), kept[frame]), ((0, 0, 255), removed[frame])):
            for row in rows:
                x1, y1, x2, y2 = map(int, row["box"])
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(image, f"{name} frame {frame}: removed red, kept green, static zones yellow",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        images.append(cv2.resize(image, (960, 540)))
    capture.release()
    if images:
        if len(images) % 2:
            images.append(np.zeros_like(images[0]))
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.vstack([np.hstack(images[i:i + 2]) for i in range(0, len(images), 2)]))


def kept_rows(directory: Path, name: str, output: dict) -> list[dict]:
    """A detector's rows after static suppression (the static stage must have run)."""
    if name.startswith("labels-"):
        return output["rows"]
    path = directory / "static" / name / "kept.jsonl"
    if not path.exists():
        raise SystemExit(f"Run the static stage first ({path} is missing)")
    return read_rows(path)


# ---------------------------------------------------------------- compare --

def pair_frames(frames, left: dict, right: dict, threshold: float = .5) -> dict:
    """Greedy highest-IoU one-to-one pairs per frame on the given frames only."""
    from app.services.static_filter import box_iou

    result = {"matched": 0, "matches": [], "left_unique": [], "right_unique": []}
    for frame in sorted(frames):
        a, b = left.get(frame, []), right.get(frame, [])
        edges = sorted(((box_iou(x["box"], y["box"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                        if box_iou(x["box"], y["box"]) >= threshold), reverse=True)
        used_a, used_b = set(), set()
        for overlap, i, j in edges:
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                result["matches"].append({"left": a[i], "right": b[j], "iou": overlap})
        result["left_unique"] += [x for i, x in enumerate(a) if i not in used_a]
        result["right_unique"] += [y for j, y in enumerate(b) if j not in used_b]
    result["matched"] = len(result["matches"])
    return result


def by_frame(rows: list[dict]) -> dict[int, list[dict]]:
    frames = defaultdict(list)
    for row in rows:
        frames[row["frame"]].append(row)
    return frames


def proxies(rows: list[dict], facts: dict, sampled_fps: float) -> dict:
    """Unlabelled temporal proxies (the zero-label benchmark's definitions)."""
    m = metrics(rows, rows, facts["file"]["width"], facts["file"]["height"], sampled_fps)
    return {"tracks": m["tracks"], "boxes": m["observations"],
            "fragmentation": m["fragmentation_rate"], "duplicates": m["duplicate_pair_rate"],
            "single_frame_tracks": m["length_fractions"]["1"],
            "track_length_median": m["track_length"]["median"],
            "acceleration_median": m["center_acceleration"]["median"],
            "velocity_median": m["center_velocity"]["median"],
            "accepted_tracks": m["accepted_tracks"]}


def agreement(left: list[dict], right: list[dict], left_frames, right_frames) -> dict:
    """Boxes on frames both models analysed, at IoU >= 0.5. Frames only one model looked
    at are reported as unaligned, never as disagreement."""
    shared = set(left_frames) & set(right_frames)
    paired = pair_frames(shared, by_frame(left), by_frame(right))
    on_shared = lambda rows: sum(r["frame"] in shared for r in rows)
    left_n, right_n = on_shared(left), on_shared(right)
    return {"shared_frames": len(shared), "left_boxes": left_n, "right_boxes": right_n,
            "matched": paired["matched"], "left_only": len(paired["left_unique"]),
            "right_only": len(paired["right_unique"]),
            "left_unaligned": len(left) - left_n, "right_unaligned": len(right) - right_n,
            "agreement": paired["matched"] / max(1, left_n + right_n - paired["matched"]),
            "share_of_left_matched": paired["matched"] / max(1, left_n),
            "share_of_right_matched": paired["matched"] / max(1, right_n),
            "frame_shift_check": {str(s): pair_frames(shared, by_frame(left), {f + s: v for f, v in by_frame(right).items()
                                                                               if f + s in shared})["matched"]
                                  for s in (-2, -1, 0, 1, 2)}}, paired


def compare_stage(args) -> dict:
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts)
    outputs = detector_outputs(directory, facts)
    if "viame" not in outputs:
        raise SystemExit("No VIAME result for this video; compare needs one (run viame first)")
    names = [n for n in outputs if n.startswith(("sam3-", "labels-", "student-"))]

    def step(heartbeat):
        (directory / "compare").mkdir(parents=True, exist_ok=True)
        viame = kept_rows(directory, "viame", outputs["viame"])
        result = {"note": "No labels for this footage: agreement, disagreement and temporal proxies only, "
                          "never precision or recall. Unique boxes are review suggestions, not errors.",
                  "viame": {"proxies": proxies(viame, facts, outputs["viame"]["sampled_fps"])}, "models": {}}
        for name in names:
            rows = kept_rows(directory, name, outputs[name])
            summary, paired = agreement(viame, rows, outputs["viame"]["analysed"], outputs[name]["analysed"])
            accepted = accepted_ids(viame)
            accepted_rows = [r for r in viame if r["id"] in accepted]
            summary_accepted, _ = agreement(accepted_rows, rows, outputs["viame"]["analysed"], outputs[name]["analysed"])
            result["models"][name] = {"agreement_with_viame_tracked": summary,
                                      "agreement_with_viame_accepted": summary_accepted,
                                      "proxies": proxies(rows, facts, outputs[name]["sampled_fps"])}
            save(directory / "compare" / f"{name}-pairs.json",
                 {k: paired[k] for k in ("left_unique", "right_unique")} | {"matched": paired["matched"]})
        save(directory / "compare" / "comparison.json", result)
        return result

    inputs = {"rows": {n: digest_json(kept_rows(directory, n, outputs[n])) for n in ["viame", *names]}}
    return journal.run(f"{facts['key']}/compare", "cpu", inputs, {"iou": .5}, step)


def accepted_ids(rows: list[dict], threshold: float = ACCEPT) -> set:
    best = defaultdict(float)
    for row in rows:
        best[row["id"]] = max(best[row["id"]], row["confidence"])
    return {k for k, v in best.items() if v >= threshold}


# ----------------------------------------------------------------- render --

def h264_writer(path: Path, fps: float, size: tuple[int, int]):
    """A browser-playable H.264 MP4 writer: Windows Media Foundation on the host, else an
    ffmpeg on PATH. Refuses to fall back to MPEG-4 Part 2, which browsers do not play."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        writer = cv2.VideoWriter(str(path), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError("No H.264 encoder: run on Windows (Media Foundation) or encode with the worker "
                       "image's ffmpeg")


def h264_profile(path: Path) -> dict:
    """The avcC record of an MP4 (proof the stream is H.264, and which profile)."""
    data = path.read_bytes()
    at = data.find(b"avcC")
    if at < 0:
        return {"h264": False}
    profile, level = data[at + 5], data[at + 7]
    names = {66: "Baseline", 77: "Main", 100: "High"}
    return {"h264": True, "profile": names.get(profile, profile), "level": level / 10,
            "faststart": data.find(b"moov") < data.find(b"mdat")}


COLOURS = {"matched": (80, 220, 80), "viame_only": (255, 0, 255), "other_only": (0, 165, 255),
           "static": (150, 150, 150)}


def panel_title(name: str) -> str:
    kind, _, variant = name.partition("-")
    prompt = PROMPTS.get(variant, variant)
    return {"labels": f'SAM 3 labels, prompt "{prompt}" (unreviewed)',
            "sam3": f'SAM 3, prompt "{prompt}" (all tracked boxes)'}.get(kind, f"{name} (unreviewed)")


def render_stage(args) -> dict:
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts)
    outputs = detector_outputs(directory, facts)
    other = args.model or next((n for n in outputs if n.startswith("labels-")), None) or next(
        (n for n in outputs if n.startswith("sam3-")), None)
    if other is None:
        raise SystemExit("No SAM 3 output to render")
    with_viame = "viame" in outputs and not args.no_viame

    def step(heartbeat):
        import cv2
        import numpy as np

        right = by_frame(kept_rows(directory, other, outputs[other]))
        static_source = other.replace("labels-", "sam3-", 1)  # Labels come from these tracks
        right_static = by_frame(read_rows(directory / "static" / static_source / "removed.jsonl"))
        frames = sorted(set(outputs[other]["analysed"]))
        if with_viame:
            left = by_frame(kept_rows(directory, "viame", outputs["viame"]))
            left_static = by_frame(read_rows(directory / "static" / "viame" / "removed.jsonl"))
            frames = sorted(set(frames) & set(outputs["viame"]["analysed"]))
        rate = len(frames) / facts["file"]["seconds"]
        panel = (960, 540)
        width, height = panel[0] * (2 if with_viame else 1), panel[1] + 70
        target = directory / "render" / (f"side-by-side-viame-{other}.mp4" if with_viame else f"{other}.mp4")
        writer = h264_writer(target, rate, (width, height))
        wanted, capture, index = set(frames), cv2.VideoCapture(facts["file"]["path"]), 0
        scale = panel[0] / facts["file"]["width"]
        counts = defaultdict(int)
        try:
            while capture.grab():
                if index in wanted:
                    _, image = capture.retrieve()
                    canvas = np.zeros((height, width, 3), np.uint8)
                    pairs = pair_frames([index], left, right) if with_viame else {
                        "matches": [], "left_unique": [], "right_unique": right.get(index, [])}
                    panels = []
                    if with_viame:
                        panels.append(("VIAME (stored, production)", [(m["left"], "matched") for m in pairs["matches"]]
                                       + [(r, "viame_only") for r in pairs["left_unique"]], left_static.get(index, [])))
                    panels.append((panel_title(other), [(m["right"], "matched") for m in pairs["matches"]]
                                   + [(r, "other_only") for r in pairs["right_unique"]], right_static.get(index, [])))
                    for slot, (title, boxes, static) in enumerate(panels):
                        view = cv2.resize(image, panel, interpolation=cv2.INTER_AREA)
                        for row in static:
                            x1, y1, x2, y2 = (int(v * scale) for v in row["box"])
                            cv2.rectangle(view, (x1, y1), (x2, y2), COLOURS["static"], 1)
                        for row, kind in boxes:
                            x1, y1, x2, y2 = (int(v * scale) for v in row["box"])
                            cv2.rectangle(view, (x1, y1), (x2, y2), COLOURS[kind], 2)
                            counts[kind] += 1
                        cv2.putText(view, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                        canvas[:panel[1], slot * panel[0]:(slot + 1) * panel[0]] = view
                    legend = [("both models (IoU>=0.5)", "matched"), ("VIAME only", "viame_only"),
                              ("SAM 3 only" if other.startswith(("sam3", "labels")) else "model only", "other_only"),
                              ("static, suppressed", "static")]
                    x = 12
                    for label, kind in (legend if with_viame else legend[2:]):
                        cv2.rectangle(canvas, (x, panel[1] + 18), (x + 26, panel[1] + 40), COLOURS[kind], -1)
                        cv2.putText(canvas, label, (x + 34, panel[1] + 37), cv2.FONT_HERSHEY_SIMPLEX, .6,
                                    (255, 255, 255), 1)
                        x += 60 + 11 * len(label)
                    cv2.putText(canvas, f"frame {index}  t={index / facts['file']['fps']:.1f}s",
                                (width - 300, panel[1] + 37), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                    writer.write(canvas)
                    if len(counts) and index % 200 == 0:
                        heartbeat(frame=index)
                index += 1
        finally:
            capture.release()
            writer.release()
        return {"video": file_id(target), "frames": len(frames), "fps": rate, "encoding": h264_profile(target),
                "boxes_drawn": dict(counts), "with_viame": with_viame}

    inputs = {"video": facts["file"]["sha256"], "rows": digest_json(
        {n: digest_json(kept_rows(directory, n, outputs[n])) for n in (["viame"] if with_viame else []) + [other]})}
    return journal.run(f"{facts['key']}/render/{other}/{'viame' if with_viame else 'alone'}", "cpu", inputs, {}, step)


# ------------------------------------------------------------------ split --

def time_split(frames: int, fps: float, *, train: float = .75, test: float = .20, gap_seconds: float = 5.0,
               validation: float = 0.0) -> dict:
    """Contiguous time segments: train first, test last, at least ``gap_seconds`` apart so
    neighbouring frames cannot leak across. A validation share, if asked for, is cut from
    the end of the train segment with its own gap."""
    gap = math.ceil(gap_seconds * fps)
    test_start = frames - round(frames * test)
    train_end = min(round(frames * train), test_start - gap)
    if train_end <= 0 or test_start >= frames:
        raise ValueError("Video too short for this split")
    segments = {"test": [test_start, frames]}
    if validation:
        validation_start = train_end - round(frames * validation)
        if validation_start - gap <= 0:
            raise ValueError("Video too short for a validation segment")
        segments |= {"train": [0, validation_start - gap], "validation": [validation_start, train_end]}
    else:
        segments["train"] = [0, train_end]
    return {"segments": segments, "gap_frames": gap, "fps": fps, "frames": frames,
            "note": "Half-open [start, end) source-frame ranges. With one video this is a pipeline "
                    "smoke test, not a measure of generalisation."}


# --------------------------------------------------------------- tracking --

def tracker_candidates(fps: float, matches: tuple[float, ...] = (.8,)) -> dict:
    """The BrackishMOT grid: three threshold sets x three track lifetimes x score fusion
    on/off (x first-association match bounds), for BoT-SORT and Ultralytics' ByteTrack.
    Version 1 (fixed before any scoring) kept Ultralytics' match bound 0.8, i.e. IoU >= 0.2;
    version 2 adds 0.95 (IoU >= 0.05), closer to VIAME production's 0.98."""
    from app.services.box_tracking import TrackerSettings

    thresholds = {"viame": (.45, .50, .10), "mid": (.35, .40, .10), "ultralytics": (.25, .25, .10)}
    grid = {}
    for kind in ("botsort", "bytetrack"):
        for high, new, low in thresholds.values():
            for seconds in (1.5, 3.0, 7.5):
                for fuse in (True, False):
                    for match in matches:
                        settings = TrackerSettings(kind=kind, track_high_thresh=high, new_track_thresh=new,
                                                   track_low_thresh=low, track_buffer=max(1, round(seconds * fps)),
                                                   fuse_score=fuse, match_thresh=match)
                        grid[settings.tag()] = settings
    return grid


# Version 1 is the grid fixed before scoring, and its selection is the one used. Version 2
# was added after version 1 showed Ultralytics' strict default matching losing fish that
# move far between 4 FPS frames; it is selected on the tuning split only and reported
# beside v1 as an exploratory check, never used downstream.
GRID_VERSIONS = {"v1": (.8,), "v2": (.8, .95)}
SELECTED_VERSION = "v1"


def track_rows(rows: list[dict], frames: list[int], settings, fps: float, carry: list[list[int]] | None = None) -> list[dict]:
    """Track per-frame boxes with an Ultralytics tracker, feeding every analysed frame in
    order (empty ones too). ``carry`` optionally lists frame segments after which the
    tracker is reset (live mode's 2 s segments); None tracks straight through."""
    import numpy as np

    from app.services.box_tracking import BoxTracker

    grouped = by_frame(rows)
    tracker, out, offset = BoxTracker(settings), [], 0
    segments = carry or [frames]
    for number, segment in enumerate(segments):
        if carry is not None and number:
            tracker.reset()
            offset = 100000 * number
        for frame in segment:
            boxes = grouped.get(frame, [])
            for row in tracker.update(frame, np.array([b["box"] for b in boxes]).reshape(-1, 4),
                                      np.array([b["confidence"] for b in boxes])):
                out.append({"id": row["id"] + offset, "frame": row["frame"], "t": row["frame"] / fps,
                            "box": row["box"], "confidence": row["confidence"]})
    return out


def track_stage(args) -> dict:
    """Part 4.2: BoT-SORT and ByteTrack over each detector's boxes after static
    suppression, with the settings chosen on BrackishMOT. Proxies only."""
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts)
    outputs = detector_outputs(directory, facts)
    chosen = journal.data["decisions"].get("brackishmot_settings")
    results = {}
    for name in [n for n in outputs if n.startswith(("sam3-", "student-", "viame-raw"))]:
        base = kept_rows(directory, name, outputs[name])
        sampled = outputs[name]["sampled_fps"]
        for kind in ("botsort", "bytetrack"):
            settings = settings_for(chosen, kind, sampled)

            def step(heartbeat, base=base, settings=settings, name=name, sampled=sampled):
                started = time.perf_counter()
                tracked = track_rows(base, sorted(set(outputs[name]["analysed"])), settings, facts["file"]["fps"])
                seconds = time.perf_counter() - started
                write_rows(tracked, directory / "tracking" / name / settings.kind / "tracks.jsonl")
                return {"settings": settings.__dict__, "proxies": proxies(tracked, facts, sampled),
                        "tracker_seconds": seconds,
                        "tracker_ms_per_frame": 1000 * seconds / max(1, len(outputs[name]["analysed"]))}

            results[f"{name}/{kind}"] = journal.run(f"{facts['key']}/track/{name}/{kind}", "cpu",
                                                    {"rows": digest_json(base)}, settings.__dict__, step)
        if name.startswith("sam3-") and not (directory / "sam3" / name[5:] / "tracked.jsonl").exists():
            results[f"{name}/sam3-own"] = {"proxies": proxies(base, facts, sampled)}  # Video mode's own tracker
    if "viame" in outputs:
        results["viame/stored-bytetrack"] = {"proxies": proxies(kept_rows(directory, "viame", outputs["viame"]), facts,
                                                                outputs["viame"]["sampled_fps"])}
    save(directory / "tracking" / "comparison.json", results)
    return results


def settings_for(chosen: dict | None, kind: str, fps: float):
    """The BrackishMOT-selected settings for a tracker kind, with the track lifetime in
    seconds converted to this video's sampled frame rate; Ultralytics defaults otherwise."""
    from app.services.box_tracking import TrackerSettings

    if not chosen or kind not in chosen:
        return TrackerSettings(kind=kind)
    values = dict(chosen[kind])
    seconds = values.pop("buffer_seconds")
    return TrackerSettings(**values | {"kind": kind, "track_buffer": max(1, round(seconds * fps))})


# ------------------------------------------------------------ BrackishMOT --

BRACKISH = OUTPUTS / "murky-water-20261011" / "part1"
IMPORTED = Path("data/training/brackishmot-v1")
BRACKISH_IMAGES = Path("data/external/brackishmot")
BRACKISH_FPS = 15.0
VIAME_TAG = "h0.45-n0.50-l0.10-s0.10-b30"  # VIAME production ByteTrack in the murky-water replays


@functools.cache
def brackish_detections(sequence: str) -> dict[int, list[dict]]:
    """VIAME's saved production detections (score >= 0.10) by one-based MOT frame."""
    frames = defaultdict(list)
    for row in _csv_rows(BRACKISH / "detections" / "0.10" / f"{sequence.replace('/', '__')}.csv"):
        frame = int(row[1][:-4])
        frames[frame].append({"frame": frame, "box": [float(v) for v in row[3:7]], "confidence": float(row[7])})
    return frames


def brackish_track(sequence: str, positions: list[int], settings, segment_frames: int | None = None,
                   carry: bool = False, images: bool = False) -> tuple[list[dict], float]:
    """MOT rows for one sequence on its analysed frames, and the tracker's seconds.
    ``segment_frames`` splits the analysed frames into live-style segments; with
    ``carry`` the tracker keeps its state across them, otherwise it restarts each one."""
    import numpy as np

    from app.services.box_tracking import BoxTracker

    detections = brackish_detections(sequence)
    frames = [p + 1 for p in positions]
    tracker, rows, elapsed = BoxTracker(settings), [], 0.0
    offset = 0
    for index, frame in enumerate(frames):
        new_segment = segment_frames and index and (
            positions[index] // segment_frames != positions[index - 1] // segment_frames)
        if new_segment and not carry:
            tracker.reset()
            offset += 100000
        image = None
        if images:
            import cv2
            image = cv2.imread(str(BRACKISH_IMAGES / sequence / "img1" / f"{frame:06d}.jpg"))
        boxes = detections.get(frame, [])
        started = time.perf_counter()
        out = tracker.update(frame, np.array([b["box"] for b in boxes]).reshape(-1, 4),
                             np.array([b["confidence"] for b in boxes]), image=image)
        elapsed += time.perf_counter() - started
        for row in out:
            x1, y1, x2, y2 = row["box"]
            rows.append({"frame": row["frame"], "track_id": row["id"] + offset, "x": x1, "y": y1,
                         "w": x2 - x1, "h": y2 - y1, "score": row["confidence"]})
    return sorted(rows, key=lambda r: (r["frame"], r["track_id"])), elapsed


def brackish_evaluate(root: Path, label: str, profile: str, sequences: list[str], manifest: dict,
                      producer) -> dict:
    """Write MOT files for ``sequences`` with ``producer(sequence)`` and score them with
    evaluate_mot on analysed frames only. Cached by label."""
    from app.services.external_datasets import evaluate_mot, write_mot

    directory = root / "brackishmot" / "eval" / profile / label
    result = directory / "evaluation.json"
    if result.exists():
        return read(result)
    directory.mkdir(parents=True, exist_ok=True)
    analysed, seconds, frames = {}, 0.0, 0
    for sequence in sequences:
        positions = [i for i in manifest["analysed_positions"][profile] if i < manifest["sequences"][sequence]["frames"]]
        rows, elapsed = producer(sequence, positions)
        seconds += elapsed
        frames += len(positions)
        write_mot(rows, directory / f"{sequence}.txt")
        analysed[sequence] = [i + 1 for i in positions]
    (directory / "analysed_frames.json").write_text(json.dumps(analysed), encoding="utf-8")
    scratch = directory / "evaluation.tmp.json"
    scratch.unlink(missing_ok=True)  # Left by an interrupted run; evaluate_mot never overwrites.
    report = evaluate_mot(IMPORTED, directory, scratch)
    report["tracker_seconds"], report["tracker_frames"] = seconds, frames
    save(result, report)
    scratch.unlink()
    return report


def mot_summary(report: dict) -> dict:
    return {"idf1": report["identity"]["idf1"], "idp": report["identity"]["idp"], "idr": report["identity"]["idr"],
            "mota": report["clear"]["mota"], "tp": report["clear"]["tp"], "fp": report["clear"]["fp"],
            "fn": report["clear"]["fn"], "idsw": report["clear"]["idsw"],
            "idsw_per_gt_track": report["clear"]["idsw_per_gt_track"],
            "ms_per_frame": (1000 * report["tracker_seconds"] / report["tracker_frames"]
                             if report.get("tracker_frames") else None)}


def viame_reference(round_name: str, profile: str) -> dict:
    """VIAME's own ByteTrack at production settings, as scored in the murky-water work."""
    return read(BRACKISH / "eval" / round_name / profile / VIAME_TAG / "evaluation.json")


def rank(results: dict, base: dict) -> list[str]:
    """Murky-water rule: settings that keep MOTA >= VIAME production and FP <= 1.10x
    rank first, then by IDF1."""
    def key(tag):
        r = results[tag]
        compliant = r["mota"] >= base["mota"] and r["fp"] <= 1.10 * base["fp"]
        return (not compliant, -(r["idf1"] or 0), tag)
    return sorted(results, key=key)


def brackishmot_stage(args) -> dict:
    """Part 4.1: BoT-SORT (and Ultralytics ByteTrack as a control) over VIAME's saved
    BrackishMOT detections, tuned on the murky-water tuning split and scored on held-out
    sequences against real labels. No GPU."""
    root = args.root or resolve_root(None, None)
    journal = Journal(root)
    manifest = read(BRACKISH / "manifest.json")
    split = manifest["split"]
    profiles = {"batch-10fps": 10.0, "live-4fps": 4.0}
    outcome = {}
    for version, matches in GRID_VERSIONS.items():
        prefix = "brackishmot" if version == "v1" else f"brackishmot-{version}"
        result = {"match_thresh_values": matches}
        for profile, fps in profiles.items():
            grid = tracker_candidates(fps, matches)

            def evaluate(label, sequences, round_name, settings=None, grid=grid, profile=profile, **options):
                settings = settings or grid[label]
                producer = lambda sequence, positions: brackish_track(sequence, positions, settings, **options)
                return mot_summary(brackish_evaluate(root, f"{round_name}/{label}", profile, sequences, manifest,
                                                     producer))

            def step(heartbeat, grid=grid, profile=profile, evaluate=evaluate):
                base = {r: summary_of(viame_reference(r, profile)) for r in ("round-1", "round-2", "held-out")}
                one = {}
                for index, tag in enumerate(grid):
                    one[tag] = evaluate(tag, split["round_one"], "round-1")
                    heartbeat(round=1, done=index + 1, of=len(grid))
                found = {"viame_production": base, "round_1": one}
                for kind in ("botsort", "bytetrack"):
                    top = rank({t: v for t, v in one.items() if grid[t].kind == kind}, base["round-1"])[:3]
                    two = {t: evaluate(t, split["tuning"], "round-2") for t in top}
                    best = rank(two, base["round-2"])[0]
                    held = evaluate(best, split["held_out"], "held-out")
                    found[kind] = {"advanced": top, "round_2": two, "best": best, "settings": grid[best].__dict__,
                                   "held_out": held}
                    heartbeat(round=2, kind=kind)
                # Same thresholds and lifetime as VIAME production, for an algorithm-only comparison.
                from app.services.box_tracking import TrackerSettings
                for kind in ("botsort", "bytetrack"):
                    same = TrackerSettings(kind=kind, track_high_thresh=.45, new_track_thresh=.50,
                                           track_low_thresh=.10, track_buffer=30)
                    found[f"{kind}_at_viame_settings"] = {"settings": same.__dict__, "held_out": evaluate(
                        same.tag(), split["held_out"], "held-out", settings=same)}
                return found

            result[profile] = journal.run(f"{prefix}/{profile}", "cpu",
                                          {"manifest": file_id(BRACKISH / "manifest.json"),
                                           "truth": file_id(IMPORTED / "ground_truth.csv")},
                                          {"grid": sorted(grid)}, step)
            # The murky-water gate's per-sequence criterion, from the saved held-out scores.
            result[f"{profile}_per_sequence_vs_viame"] = {
                kind: sequence_gains(root, profile, f"held-out/{result[profile][kind]['best']}")
                for kind in ("botsort", "bytetrack")}
        result["segments"] = journal.run(f"{prefix}/segments", "cpu", {},
                                         {"segment_seconds": 2.0, "selected": result["live-4fps"]["botsort"]["best"]},
                                         lambda heartbeat, result=result: brackish_segments(root, manifest, result))
        outcome[version] = result
    outcome["gmc"] = journal.run("brackishmot/gmc", "cpu", {}, {"sequences": 3},
                                 lambda heartbeat: brackish_gmc(root, manifest, outcome["v1"]))
    final = outcome[SELECTED_VERSION]
    journal.decide("brackishmot_settings", {kind: settings_seconds(final["batch-10fps"][kind]["settings"], 10.0)
                                            for kind in ("botsort", "bytetrack")} | {"grid_version": SELECTED_VERSION})
    save(root / "brackishmot" / "summary.json", outcome)
    return outcome


def sequence_gains(root: Path, profile: str, label: str) -> dict:
    """On how many held-out sequences with fish IDF1 beats VIAME production's."""
    from scripts.murky_water_tracker import per_sequence_gain

    candidate = read(root / "brackishmot" / "eval" / profile / label / "evaluation.json")
    gain = per_sequence_gain(viame_reference("held-out", profile), candidate)
    worse = sum(r["gain"] < 0 for r in gain["rows"].values())
    return {k: gain[k] for k in ("sequences_with_fish", "sequences_with_gain", "majority")} | {
        "sequences_worse": worse}


def summary_of(report: dict) -> dict:
    return {"idf1": report["identity"]["idf1"], "idp": report["identity"]["idp"], "idr": report["identity"]["idr"],
            "mota": report["clear"]["mota"], "tp": report["clear"]["tp"], "fp": report["clear"]["fp"],
            "fn": report["clear"]["fn"], "idsw": report["clear"]["idsw"],
            "idsw_per_gt_track": report["clear"]["idsw_per_gt_track"]}


def settings_seconds(settings: dict, fps: float) -> dict:
    values = {k: v for k, v in settings.items() if k != "kind"}
    values["buffer_seconds"] = values.pop("track_buffer") / fps
    return values


def brackish_segments(root: Path, manifest: dict, outcome: dict) -> dict:
    """Live mode's 2 s segments at 4 FPS on held-out sequences: the tracker restarted per
    segment (as live mode does today) against one BoT-SORT whose state carries across."""
    profile, segment = "live-4fps", int(2.0 * BRACKISH_FPS)
    from app.services.box_tracking import TrackerSettings

    chosen = TrackerSettings(**outcome[profile]["botsort"]["settings"])
    viame_settings = TrackerSettings(kind="botsort", track_high_thresh=.45, new_track_thresh=.50,
                                     track_low_thresh=.10, track_buffer=30)
    results = {}
    for name, settings in (("botsort-selected", chosen), ("botsort-viame-settings", viame_settings)):
        for carry in (False, True):
            mode = "carry" if carry else "restart"
            producer = lambda sequence, positions, s=settings, c=carry: brackish_track(
                sequence, positions, s, segment_frames=segment, carry=c)
            report = brackish_evaluate(root, f"segments/{settings.tag()}/{mode}", profile,
                                       manifest["split"]["held_out"], manifest, producer)
            results[f"{name}/{mode}"] = {"settings": settings.tag()} | mot_summary(report) | {
                "precision": report["clear"]["tp"] / max(1, report["clear"]["tp"] + report["clear"]["fp"]),
                "recall": report["clear"]["tp"] / max(1, report["clear"]["tp"] + report["clear"]["fn"])}
    viame = read(BRACKISH / "journal.json")["decisions"]["segments/live-4fps"]["results"]
    return {"results": results, "viame_production_segments": viame,
            "note": "Restart rows offset IDs per segment, so their identity scores are not comparable; "
                    "compare boxes (TP/FP/FN). Carry rows keep IDs across segments."}


def brackish_gmc(root: Path, manifest: dict, outcome: dict) -> dict:
    """Camera-motion compensation cost on a fixed camera: BoT-SORT with sparse optical
    flow (reading every analysed image) against no compensation, on three held-out
    sequences with fish at 10 FPS."""
    from dataclasses import replace

    from app.services.box_tracking import TrackerSettings

    profile = "batch-10fps"
    sequences = [s for s in manifest["split"]["held_out"] if manifest["sequences"][s]["fish_boxes"]][:3]
    chosen = TrackerSettings(**outcome[profile]["botsort"]["settings"])
    results = {}
    for method in ("none", "sparseOptFlow"):
        settings = replace(chosen, gmc_method=method)
        producer = lambda sequence, positions, s=settings: brackish_track(sequence, positions, s, images=True)
        results[method] = mot_summary(brackish_evaluate(root, f"gmc/{method}", profile, sequences, manifest, producer))
    # How far the estimated camera motion moves a point, frame to frame (pixels).
    import cv2
    import numpy as np
    from ultralytics.trackers.utils.gmc import GMC

    shifts = []
    for sequence in sequences:
        gmc = GMC(method="sparseOptFlow")
        for position in [i for i in manifest["analysed_positions"][profile]
                         if i < manifest["sequences"][sequence]["frames"]]:
            warp = gmc.apply(cv2.imread(str(BRACKISH_IMAGES / sequence / "img1" / f"{position + 1:06d}.jpg")))
            centre = np.array([960.0, 540.0, 1.0])
            shifts.append(float(np.hypot(*(warp @ centre - centre[:2]))))
    shifts.sort()
    return {"sequences": sequences, "results": results,
            "estimated_camera_shift_px": {"median": shifts[len(shifts) // 2], "p95": shifts[int(.95 * (len(shifts) - 1))],
                                          "max": shifts[-1], "frames": len(shifts)},
            "note": "ms_per_frame is the tracker update alone, including GMC; image decoding is excluded."}


# ------------------------------------------------------------------ facts --

def facts_stage(args) -> dict:
    """Part 0 facts that need no GPU: the installed trackers and SAM 3 requirements."""
    import importlib.metadata

    from scripts.sam3_autolabel import SAM3_CHECKPOINT, missing_message, missing_requirements

    root = args.root or resolve_root(None, None)
    import ultralytics
    from ultralytics.utils import YAML

    cfg = Path(ultralytics.__file__).parent / "cfg" / "trackers"
    result = {"ultralytics": ultralytics.__version__, "ultralytics_licence": "AGPL-3.0",
              "botsort_yaml": YAML.load(cfg / "botsort.yaml"), "bytetrack_yaml": YAML.load(cfg / "bytetrack.yaml"),
              "lap": importlib.metadata.version("lap"),
              "track_buffer_unit": "tracker calls (ultralytics/trackers/byte_tracker.py: max_frames_lost = track_buffer)",
              "new_track_confirmation": "a track started after the first frame must match on the next call",
              "external_detections": "app/services/box_tracking.py Detections (conf, cls, xyxy, xywh arrays)",
              "reid": "with_reid False; model 'auto' reads YOLO backbone features only, so external boxes get none",
              "gmc": "sparseOptFlow by default; needs the frame image; off ('none') for a fixed camera",
              "sam3_loader": "ultralytics.models.sam SAM3VideoSemanticPredictor (build_sam3)",
              "sam3_missing_requirements": missing_requirements(),
              "sam3_checkpoint_present": SAM3_CHECKPOINT.is_file(),
              "sam3_checkpoint_instructions": missing_message(SAM3_CHECKPOINT.resolve())}
    save(root / "facts.json", result)
    return result


# --------------------------------------------------------------------- CLI --

def main() -> None:
    os.environ.update(OFFLINE)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["facts", "viame", "static", "compare", "render", "track", "brackishmot",
                                          "sam3-probe", "sam3", "variants", "labels", "run"])
    parser.add_argument("--video-id", help="Upload's video UUID (read-only database lookup)")
    parser.add_argument("--video", help="A video path (with --video-id: skip the lookup while Docker is down)")
    parser.add_argument("--root", type=Path, help="Experiment folder (default: the one already holding this "
                                                  "video, else data/outputs/sam3-botsort-<UTC date>)")
    parser.add_argument("--viame-source", choices=["db", "csv"], default="db")
    parser.add_argument("--viame-job", help="VIAME job id (default: the latest completed job)")
    parser.add_argument("--viame-offset", type=int, default=-1, help="Frame offset for --viame-source csv")
    parser.add_argument("--no-viame", action="store_true", help="Run without VIAME")
    parser.add_argument("--model", help="Detector output to render (default: the first SAM 3 variant)")
    parser.add_argument("--prompts", default=None, help="Comma-separated SAM 3 prompt variants")
    args = parser.parse_args()
    stages = {"facts": facts_stage, "viame": viame_stage, "static": static_stage, "compare": compare_stage,
              "render": render_stage, "track": track_stage, "brackishmot": brackishmot_stage}
    if args.stage in ("sam3-probe", "sam3", "variants", "labels", "run"):
        from scripts import sam3_stages
        stages |= sam3_stages.STAGES
    print(json.dumps(stages[args.stage](args), indent=2, default=str))


if __name__ == "__main__":
    main()
