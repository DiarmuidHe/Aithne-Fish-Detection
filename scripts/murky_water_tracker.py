"""Murky-water Part 1: tune VIAME's ByteTrack on BrackishMOT with the detector unchanged.

Run inside the VIAME worker image with this repository mounted read-only at /app:

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps -T
        --volume "${PWD}:/app:ro" -e PYTHONPATH=/app -e FISHIAL_ENABLED=false
        --entrypoint python3 worker /app/scripts/murky_water_tracker.py STAGE

Stages, in order: prepare, detect, validate, sweep, smartbay, report. Every VIAME pass
goes through scripts/viame_review_experiment.py. The detector runs once per frame;
tracker settings are swept by replaying its saved detections through the production
ByteTrack block, one VIAME run per sequence. journal.json records each pass's settings,
hashes, timing and status, so a resumed stage never repeats completed work. Nothing
here changes application settings, jobs, uploads, review decisions or CVAT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.services.external_datasets import (
    _iou,
    evaluate_mot,
    viame_tracks_to_mot,
    write_mot,
)
from scripts.viame_zero_label_benchmark import digest, metrics, observations, probe, read, save

ROOT = Path("/data/outputs/murky-water-20261011/part1")
SOURCE = Path("/data/external/brackishmot")
IMPORTED = Path("/data/training/brackishmot-v1")
SMARTBAY_MANIFEST = Path("/data/outputs/viame-zero-label-benchmark/manifest.json")
APP = Path("/app")
RUNNER = APP / "scripts/viame_review_experiment.py"
PRODUCTION_PIPE = APP / "app/viame_pipelines/tracker_fish_no_motion.pipe"
DETECT_PIPE = APP / "scripts/viame_pipelines/fish_detections_only.pipe"
REPLAY_PIPE = APP / "scripts/viame_pipelines/bytetrack_replay.pipe"
PROBE_PIPE = APP / "scripts/viame_pipelines/downsampler_probe.pipe"
VIAME = Path("/opt/noaa/viame")

NATIVE_FPS = 15.0
PROFILES = {"batch-10fps": 10.0, "live-4fps": 4.0}
# Production batch profile (app/config.py, docker-compose.gpu.yml, the .pipe block).
PRODUCTION = {"high": .45, "new_track": .50, "low": .10, "score": .10, "buffer": 30}
# One detector pass at a lower score keeps low_thresh .05 candidates possible; its
# rows filtered to >= .10 must reproduce production detections exactly (validate).
DETECTION_SCORE = .05
# Fixed before any scoring. (high, new_track, low); ByteTrack only starts tracks from
# detections >= high, so new_track >= high. The detector score threshold is min(.10, low).
THRESHOLD_SETS = {"A": (.45, .50, .10), "B": (.35, .40, .10),
                  "C": (.25, .30, .10), "D": (.35, .40, .05)}
BUFFER_SECONDS = (1.5, 3.0, 7.5)
GPU_BUDGET_SECONDS = 3 * 3600
# One VIAME GPU budget covers every part; each part writes only its own journal.
VIAME_JOURNALS = [ROOT / "journal.json", ROOT.parent / "part2" / "viame-journal.json",
                  ROOT.parent / "part3" / "viame-journal.json"]
CHUNK_SEQUENCES = 14
WORKERS = 8
ACCEPTANCE = ("0.55", "0.60", "0.65")
PROXIES = {"fragmentation": lambda m: m["fragmentation_rate"],
           "duplicates": lambda m: m["duplicate_pair_rate"],
           "singleton_tracks": lambda m: m["length_fractions"]["1"],
           "acceleration": lambda m: m["center_acceleration"]["median"]}


# ------------------------------------------------------------------ helpers --

def seq_key(sequence: str) -> str:
    return sequence.replace("/", "__")


def stable_rank(name: str) -> str:
    return hashlib.sha256(f"murky-water-v1:{name}".encode()).hexdigest()


def split_sequences(fish_boxes: dict[str, int]) -> dict[str, list[str]]:
    """Stable-hash halves within fish-bearing and fish-free strata.

    Alternating along the hash order gives each side half of each stratum; round one
    uses every other tuning sequence, again by hash order.
    """
    tuning, held_out, round_one = [], [], []
    for has_fish in (True, False):
        stratum = sorted((s for s, n in fish_boxes.items() if (n > 0) == has_fish), key=stable_rank)
        tuning += stratum[0::2]
        held_out += stratum[1::2]
        round_one += stratum[0::4]
    return {"tuning": sorted(tuning), "held_out": sorted(held_out), "round_one": sorted(round_one)}


def build_configs() -> list[dict]:
    return [{"name": f"{label}-{seconds:g}s", "high": high, "new_track": new, "low": low,
             "score": min(PRODUCTION["score"], low), "buffer_seconds": seconds}
            for label, (high, new, low) in THRESHOLD_SETS.items() for seconds in BUFFER_SECONDS]


def effective(config: dict | None, fps: float) -> dict:
    """Per-profile ByteTrack settings; ``None`` means the production defaults."""
    if config is None:
        return dict(PRODUCTION)
    return {"high": config["high"], "new_track": config["new_track"], "low": config["low"],
            "score": config["score"], "buffer": max(1, round(config["buffer_seconds"] * fps))}


def tag(settings: dict) -> str:
    return (f"h{settings['high']:.2f}-n{settings['new_track']:.2f}-l{settings['low']:.2f}"
            f"-s{settings['score']:.2f}-b{settings['buffer']}")


def sequence_to_video(images: list[Path], output: Path, fps: float) -> dict:
    """Lossless FFV1 video of an image sequence; asserts every frame decodes back."""
    import cv2

    if output.exists():
        raise FileExistsError(output)
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Cannot decode {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write FFV1")
    try:
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Cannot decode {path} at {width}x{height}")
            writer.write(frame)
    finally:
        writer.release()
    capture, decoded = cv2.VideoCapture(str(output)), 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != len(images):
        output.unlink(missing_ok=True)
        raise ValueError(f"Decoded {decoded} frames from {output}, expected {len(images)}")
    return {"path": str(output), "frames": decoded, "fps": fps, "codec": "FFV1 (lossless BGR)",
            "sha256": digest(output)}


def probe_selection(csv_path: Path) -> list[int]:
    """Zero-based list positions the downsampler passed, from the probe's writer."""
    rows = [r for r in csv.reader(csv_path.open(encoding="utf-8")) if r and not r[0].startswith("#")]
    return sorted(int(r[1][1:-4]) - 1 for r in rows)


def detection_rows(csv_path: Path) -> list[list[str]]:
    return [r for r in csv.reader(csv_path.open(encoding="utf-8")) if r and not r[0].startswith("#")]


def write_detections(rows: list[list[str]], path: Path, minimum: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if float(r[7]) >= minimum]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(kept)
    temporary.replace(path)
    return len(kept)


def relative_change(base, candidate):
    if base is None or candidate is None:
        return None
    return (candidate - base) / abs(base) if base else (0.0 if candidate == base else math.inf)


# ------------------------------------------------------------------ journal --

def _charged(passes: dict) -> float:
    total = 0.0
    for entry in passes.values():
        if entry.get("kind") != "gpu":
            continue
        if entry["status"] in ("complete", "failed", "interrupted"):
            total += entry["duration_seconds"]
        else:  # Still running (or died unseen): charge the whole reservation.
            total += entry["timeout_seconds"]
    return total


class Journal:
    def __init__(self, path: Path = VIAME_JOURNALS[0]):
        self.path = path
        self.lock = threading.Lock()
        self.data = read(path) if path.exists() else {"schema_version": 1, "passes": {},
                                                       "decisions": {}}

    def entry(self, name):
        return self.data["passes"].get(name)

    def update(self, name, **values):
        with self.lock:
            self.data["passes"].setdefault(name, {}).update(values)
            save(self.path, self.data)

    def decide(self, name, value):
        with self.lock:
            self.data["decisions"][name] = value
            save(self.path, self.data)

    def gpu_seconds(self):
        """GPU seconds charged across every part's VIAME journal."""
        others = sum(_charged(read(p)["passes"]) for p in VIAME_JOURNALS
                     if p != self.path and p.exists())
        return _charged(self.data["passes"]) + others


def run_pass(journal: Journal, name: str, kind: str, args: list, output: Path,
             timeout: int = 3600) -> dict:
    """One VIAME pass through viame_review_experiment.py; skipped once complete."""
    entry = journal.entry(name)
    if entry and entry["status"] == "complete":
        return entry
    if entry and entry["status"] == "running":
        # A previous process died mid-pass (for example a shutdown). Charge the time up
        # to its last output write, then redo the pass.
        files = [f for f in output.rglob("*") if f.is_file()] if output.exists() else []
        last = max((f.stat().st_mtime for f in files), default=entry["started_unix"])
        journal.update(name, status="interrupted", duration_seconds=max(0.0, last - entry["started_unix"]))
    if output.exists():  # A previous attempt died; keep its files for inspection.
        output.rename(output.with_name(f"{output.name}.attempt-{int(time.time())}"))
    if kind == "gpu" and journal.gpu_seconds() + timeout > GPU_BUDGET_SECONDS:
        raise RuntimeError(f"GPU budget would be exceeded by {name}")
    command = ["python3", str(RUNNER), "run", *map(str, args), "--timeout", str(timeout)]
    journal.update(name, kind=kind, status="running", command=command, timeout_seconds=timeout,
                   started_unix=time.time())
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 120)
    elapsed = time.monotonic() - started
    if completed.returncode:
        journal.update(name, status="failed", duration_seconds=elapsed,
                       error=completed.stderr[-2000:])
        raise RuntimeError(f"{name} failed: {completed.stderr[-500:]}")
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    journal.update(name, status="complete", duration_seconds=report["processing_seconds"],
                   wall_seconds=elapsed, outputs={p.name: digest(p) for p in output.glob("*.csv")},
                   kwiver_settings=report["kwiver_settings"])
    return journal.entry(name)


# ------------------------------------------------------------------ stages --

def prepare(journal: Journal) -> dict:
    """Inventory, frame-count checks, split and analysed frames, before any scoring."""
    from concurrent.futures import ProcessPoolExecutor

    path = ROOT / "manifest.json"
    if path.exists():
        return read(path)
    imported = read(IMPORTED / "report.json")
    truth = defaultdict(lambda: {"max_frame": 0, "ids": set()})
    with (IMPORTED / "ground_truth.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            entry = truth[row["sequence"]]
            entry["max_frame"] = max(entry["max_frame"], int(row["frame"]))
            entry["ids"].add(row["track_id"])
    sequences = {}
    pool = ProcessPoolExecutor(16)
    for sequence in sorted(imported["sequences"]):
        directory = SOURCE / sequence
        images = sorted((directory / "img1").glob("*.jpg"))
        numbers = [int(p.stem) for p in images]
        if numbers != list(range(1, len(images) + 1)):
            raise ValueError(f"{sequence}: image frames are not contiguous from 1")
        info = dict(line.split("=", 1) for line in (directory / "seqinfo.ini").read_text().splitlines()
                    if "=" in line)
        shapes = list(pool.map(_decoded_shape, map(str, images), chunksize=16))
        decoded = sum(shape is not None for shape in shapes)
        if decoded != len(images) or set(shapes) != {(1080, 1920)}:
            raise ValueError(f"{sequence}: decoded {decoded}/{len(images)} frames at {set(shapes)}")
        if truth[sequence]["max_frame"] > len(images):
            raise ValueError(f"{sequence}: ground truth beyond the last image")
        sequences[sequence] = {
            "frames": len(images), "decoded_frames": decoded, "seqinfo_length": int(info["seqLength"]),
            "seqinfo_matches_images": int(info["seqLength"]) == len(images),
            "frame_rate": float(info["frameRate"]), "fish_boxes": imported["sequences"][sequence]["fish_boxes"],
            "fish_identities": len(truth[sequence]["ids"]),
            "gt_sha256": imported["sequences"][sequence]["sha256"]}
        if sequences[sequence]["frame_rate"] != NATIVE_FPS:
            raise ValueError(f"{sequence}: unexpected frame rate")
    pool.shutdown()
    split = split_sequences({s: v["fish_boxes"] for s, v in sequences.items()})
    analysed = probe_downsampler(journal, max(v["frames"] for v in sequences.values()))
    inventory = {str(p): digest(p) for p in [
        PRODUCTION_PIPE, DETECT_PIPE, REPLAY_PIPE, PROBE_PIPE, RUNNER,
        VIAME / "configs/pipelines/common_fish_detector_box_only.pipe",
        VIAME / "configs/pipelines/common_default_input.pipe",
        VIAME / "configs/pipelines/common_no_load_input.pipe",
        VIAME / "configs/pipelines/common_default_input_with_downsampler.pipe",
        VIAME / "configs/pipelines/models/fish_deim_v2_1024.zip",
        VIAME / "lib/python3.12/site-packages/viame/core/bytetrack_tracker.py"]}
    manifest = {
        "schema_version": 1, "created_unix": time.time(),
        "source": {"archive_sha256": "e717dc1438aab65eaccb59b135b58f50ca7af09043ef7effbf9fd2825410c3ff",
                   "imported_report_sha256": digest(IMPORTED / "report.json"),
                   "ground_truth_sha256": digest(IMPORTED / "ground_truth.csv")},
        "sequences": sequences, "split": split, "analysed_positions": analysed,
        "profiles": PROFILES, "production": PRODUCTION, "detection_score": DETECTION_SCORE,
        "configs": build_configs(), "inventory": inventory,
        "protocol": {
            "round_one": "All 12 configurations per profile on round_one sequences (half of tuning)",
            "round_two": "Top 3 non-production configurations per profile on all tuning sequences",
            "held_out": "Best non-production configuration per profile vs production on held_out",
            "ranking": "Configurations with CLEAR MOTA >= production and CLEAR FP <= 1.10x production "
                       "rank first, then by IDF1; production is re-scored in every round as reference",
            "gate": "Held-out at both profiles: IDF1 +0.02 absolute; CLEAR MOTA not lower; CLEAR FP "
                    "<= 1.10x; IDSW per GT track not higher; IDF1 gain on a majority of held-out "
                    "sequences with fish; SmartBay proxies pooled over three clips regress <= 5%; "
                    "accepted-track largest adjacent drop over .55/.60/.65 <= production + 5 points; "
                    "SmartBay full-run seconds <= 1.10x production"}}
    save(path, manifest)
    return manifest


def _decoded_shape(path: str):
    import cv2
    image = cv2.imread(path)
    return None if image is None else image.shape[:2]


def probe_downsampler(journal: Journal, length: int) -> dict:
    """Ask VIAME's own downsampler which 15 FPS image-list entries each profile analyses."""
    directory = ROOT / "probe-input"
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"p{i:06d}.jpg" for i in range(1, length + 2)]
    for name in names:
        (directory / name).touch()
    (directory / "list.txt").write_text("".join(f"{directory / n}\n" for n in names))
    (directory / "probe.csv").write_text("".join(f"{i},{n},{i},0,0,10,10,0.5,-1,fish,0.5\n"
                                                 for i, n in enumerate(names)))
    selections = {}
    for profile, fps in PROFILES.items():
        output = ROOT / "probe" / profile
        run_pass(journal, f"probe/{profile}", "cpu",
                 [directory / "list.txt", output, "--image-list", "--frame-time", repr(1 / NATIVE_FPS),
                  "--fps", fps, "--pipeline", PROBE_PIPE, "--detections", directory / "probe.csv"], output)
        selections[profile] = probe_selection(output / "computed_detections.csv")
    return selections


def image_paths(sequence: str) -> list[Path]:
    return sorted((SOURCE / sequence / "img1").glob("*.jpg"))


def detect(journal: Journal, manifest: dict) -> None:
    """One detector-only pass per chunk of sequences at the lower detection score."""
    sequences = sorted(manifest["sequences"])
    chunks = [sequences[i:i + CHUNK_SEQUENCES] for i in range(0, len(sequences), CHUNK_SEQUENCES)]
    for index, chunk in enumerate(chunks):
        name, output = f"detect/chunk-{index:02d}", ROOT / "detect" / f"chunk-{index:02d}"
        listing = ROOT / "lists" / f"detect-chunk-{index:02d}.txt"
        entries = [(s, p) for s in chunk for p in image_paths(s)]
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("".join(f"{p}\n" for _, p in entries))
        run_pass(journal, name, "gpu", [listing, output, "--image-list", "--score", DETECTION_SCORE,
                                        "--pipeline", DETECT_PIPE], output, timeout=1800)
        split_detections(entries, output / "computed_detections.csv")
    # Nondeterminism: repeat the detector on the first three sequences of chunk 0.
    entries = [(s, p) for s in chunks[0][:3] for p in image_paths(s)]
    listing, output = ROOT / "lists" / "detect-repeat.txt", ROOT / "detect" / "repeat"
    listing.write_text("".join(f"{p}\n" for _, p in entries))
    run_pass(journal, "detect/repeat", "gpu", [listing, output, "--image-list", "--score",
                                              DETECTION_SCORE, "--pipeline", DETECT_PIPE], output,
             timeout=900)
    first = {(r[2], r[1], *r[3:8]) for r in detection_rows(ROOT / "detect/chunk-00/computed_detections.csv")
             if int(r[2]) < len(entries)}
    again = {(r[2], r[1], *r[3:8]) for r in detection_rows(output / "computed_detections.csv")}
    journal.decide("detector_repeat", {"frames": len(entries), "rows_first": len(first),
                                       "rows_repeat": len(again), "identical": first == again,
                                       "symmetric_difference": len(first ^ again)})


def split_detections(entries: list[tuple[str, Path]], csv_path: Path) -> None:
    """Per-sequence replay inputs; column 3 is the chunk list position, cross-checked."""
    by_sequence = defaultdict(list)
    for row in detection_rows(csv_path):
        sequence, image = entries[int(row[2])]
        if image.name != row[1]:
            raise ValueError(f"{csv_path}: row for {row[1]} at list position of {image}")
        by_sequence[sequence].append(row)
    for sequence in {s for s, _ in entries}:
        for minimum in (DETECTION_SCORE, PRODUCTION["score"]):
            target = ROOT / "detections" / f"{minimum:.2f}" / f"{seq_key(sequence)}.csv"
            if not target.exists():
                write_detections(by_sequence[sequence], target, minimum)


def analysed_list(manifest: dict, profile: str, sequence: str) -> tuple[Path, list[str]]:
    """The image list of frames VIAME's downsampler analyses for this profile."""
    positions = [i for i in manifest["analysed_positions"][profile]
                 if i < manifest["sequences"][sequence]["frames"]]
    images = image_paths(sequence)
    listing = ROOT / "lists" / profile / f"{seq_key(sequence)}.txt"
    if not listing.exists():
        listing.parent.mkdir(parents=True, exist_ok=True)
        temporary = listing.with_suffix(".tmp")
        temporary.write_text("".join(f"{images[i]}\n" for i in positions))
        temporary.replace(listing)
    return listing, [images[i].name for i in positions]


def replay(journal: Journal, manifest: dict, profile: str, settings: dict, sequence: str) -> Path:
    """Production ByteTrack over saved detections for one sequence; returns its MOT file."""
    listing, names = analysed_list(manifest, profile, sequence)
    detections = ROOT / "detections" / f"{settings['score']:.2f}" / f"{seq_key(sequence)}.csv"
    output = ROOT / "replays" / profile / tag(settings) / seq_key(sequence)
    mot = ROOT / "mot" / profile / tag(settings) / f"{sequence}.txt"
    run_pass(journal, f"replay/{profile}/{tag(settings)}/{sequence}", "cpu",
             [listing, output, "--image-list", "--pipeline", REPLAY_PIPE, "--detections", detections,
              "--high", settings["high"], "--low", settings["low"], "--new-track", settings["new_track"],
              "--buffer", settings["buffer"]], output, timeout=1800)
    if not mot.exists():
        write_mot(viame_tracks_to_mot(output / "viame_tracks.csv", input_names=names,
                                      frame_of={n: int(n[:-4]) for n in names}), mot)
    return mot


def evaluate(manifest: dict, round_name: str, profile: str, settings: dict,
             sequences: list[str]) -> dict:
    directory = ROOT / "eval" / round_name / profile / tag(settings)
    result = directory / "evaluation.json"
    if result.exists():
        return read(result)
    directory.mkdir(parents=True, exist_ok=True)
    analysed = {}
    for sequence in sequences:
        source = ROOT / "mot" / profile / tag(settings) / f"{sequence}.txt"
        target = directory / f"{sequence}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        analysed[sequence] = [i + 1 for i in manifest["analysed_positions"][profile]
                              if i < manifest["sequences"][sequence]["frames"]]
    (directory / "analysed_frames.json").write_text(json.dumps(analysed), encoding="utf-8")
    report = evaluate_mot(IMPORTED, directory, result)
    return report


def replay_all(journal: Journal, manifest: dict, jobs: list[tuple[str, dict, str]]) -> None:
    for profile, _, sequence in jobs:  # Shared list files exist before threads start.
        analysed_list(manifest, profile, sequence)
    with ThreadPoolExecutor(WORKERS) as pool:
        for future in [pool.submit(replay, journal, manifest, *job) for job in jobs]:
            future.result()


def summary(report: dict) -> dict:
    return {"idf1": report["identity"]["idf1"], "idp": report["identity"]["idp"],
            "idr": report["identity"]["idr"], "mota": report["clear"]["mota"],
            "tp": report["clear"]["tp"], "fp": report["clear"]["fp"], "fn": report["clear"]["fn"],
            "idsw": report["clear"]["idsw"], "idsw_per_gt_track": report["clear"]["idsw_per_gt_track"],
            "gt_tracks": report["clear"]["gt_tracks"],
            "greedy_diagnostic": {"mota": report["mota"], "id_switches": report["counts"]["id_switches"]}}


def ranked(results: dict, production: str) -> list[str]:
    base = results[production]

    def key(name):
        r = results[name]
        compliant = r["mota"] >= base["mota"] and r["fp"] <= 1.10 * base["fp"]
        return (not compliant, -(r["idf1"] or 0), name)
    return sorted((n for n in results if n != production), key=key)


def validate(journal: Journal, manifest: dict) -> None:
    """Prove the replay equals production, frame numbering, and the video path."""
    fish_tuning = [s for s in sorted(manifest["split"]["tuning"], key=stable_rank)
                   if manifest["sequences"][s]["fish_boxes"]][:3]
    findings = {"sequences": fish_tuning, "profiles": {}}
    for profile, fps in PROFILES.items():
        per_sequence = {}
        for sequence in fish_tuning:
            images = image_paths(sequence)
            listing = ROOT / "lists" / f"full-{seq_key(sequence)}.txt"
            listing.write_text("".join(f"{p}\n" for p in images))
            output = ROOT / "validate" / profile / seq_key(sequence)
            run_pass(journal, f"validate/{profile}/{sequence}", "gpu",
                     [listing, output, "--image-list", "--frame-time", repr(1 / NATIVE_FPS),
                      "--fps", fps, "--pipeline", PRODUCTION_PIPE], output, timeout=900)
            mot = replay(journal, manifest, profile, dict(PRODUCTION), sequence)
            positions = [i for i in manifest["analysed_positions"][profile] if i < len(images)]
            sampled = {images[i].name for i in positions}
            production_rows = detection_rows(output / "computed_detections.csv")
            saved = [r for r in detection_rows(ROOT / "detections" / f"{PRODUCTION['score']:.2f}"
                                               / f"{seq_key(sequence)}.csv") if r[1] in sampled]
            key = lambda r: (r[1], *r[3:8])
            tracks = lambda p: sorted((r[0], r[1], *r[3:]) for r in detection_rows(p))
            replay_dir = ROOT / "replays" / profile / tag(dict(PRODUCTION)) / seq_key(sequence)
            per_sequence[sequence] = {
                "production_frames_within_probe": {r[1] for r in production_rows} <= sampled,
                "detections_identical": sorted(map(key, production_rows)) == sorted(map(key, saved)),
                "detections": len(production_rows),
                "tracks_identical": tracks(output / "viame_tracks.csv") == tracks(replay_dir / "viame_tracks.csv"),
                "track_rows": len(detection_rows(output / "viame_tracks.csv")),
                "alignment": alignment(mot, sequence)}
            if profile == "batch-10fps":
                overlays(mot, sequence, ROOT / "validate" / "overlays")
        findings["profiles"][profile] = per_sequence
    # Replay determinism: the same detections through the same settings again.
    sequence, settings = fish_tuning[0], dict(PRODUCTION)
    listing, _ = analysed_list(manifest, "batch-10fps", sequence)
    detections = ROOT / "detections" / f"{settings['score']:.2f}" / f"{seq_key(sequence)}.csv"
    output = ROOT / "validate" / "replay-repeat" / seq_key(sequence)
    repeat = run_pass(journal, f"validate/replay-repeat/{sequence}", "cpu",
                      [listing, output, "--image-list", "--pipeline", REPLAY_PIPE, "--detections", detections,
                       "--high", settings["high"], "--low", settings["low"],
                       "--new-track", settings["new_track"], "--buffer", settings["buffer"]], output)
    first = journal.entry(f"replay/batch-10fps/{tag(settings)}/{sequence}")
    findings["replay_repeat"] = {"sequence": sequence, "identical_tracks_csv_rows": sorted(
        detection_rows(output / "viame_tracks.csv")) == sorted(detection_rows(
            ROOT / "replays" / "batch-10fps" / tag(settings) / seq_key(sequence) / "viame_tracks.csv")),
        "seconds": (first["duration_seconds"], repeat["duration_seconds"])}
    findings["video_path"] = video_path_check(journal, manifest, fish_tuning[0])
    journal.decide("validation", findings)
    save(ROOT / "validation.json", findings)


def load_truth(sequence: str) -> dict[int, list[dict]]:
    frames = defaultdict(list)
    with (IMPORTED / "ground_truth.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["sequence"] == sequence:
                frames[int(row["frame"])].append({k: float(row[k]) for k in "xywh"}
                                                 | {"track_id": int(row["track_id"])})
    return frames


def load_mot(path: Path) -> dict[int, list[dict]]:
    frames = defaultdict(list)
    for row in csv.reader(path.open(encoding="utf-8")):
        frames[int(row[0])].append({"track_id": int(row[1]), "x": float(row[2]), "y": float(row[3]),
                                    "w": float(row[4]), "h": float(row[5]), "score": float(row[6])})
    return frames


def alignment(mot: Path, sequence: str) -> dict:
    """Predicted boxes matching a GT fish (IoU >= .5) when frames are shifted by -2..+2."""
    truth, predicted = load_truth(sequence), load_mot(mot)
    return {str(shift): sum(any(_iou(p, g) >= .5 for g in truth.get(frame + shift, []))
                            for frame, boxes in predicted.items() for p in boxes)
            for shift in (-2, -1, 0, 1, 2)}


def overlays(mot: Path, sequence: str, directory: Path, count: int = 3) -> None:
    """Ground truth (green) and VIAME tracks (magenta) on the frames with most GT fish."""
    import cv2

    truth, predicted = load_truth(sequence), load_mot(mot)
    directory.mkdir(parents=True, exist_ok=True)
    frames = sorted(predicted, key=lambda f: (-len(truth.get(f, [])), f))[:count]
    for frame in frames:
        image = cv2.imread(str(SOURCE / sequence / "img1" / f"{frame:06d}.jpg"))
        for box in truth.get(frame, []):
            cv2.rectangle(image, (int(box["x"]), int(box["y"])),
                          (int(box["x"] + box["w"]), int(box["y"] + box["h"])), (0, 255, 0), 2)
        for box in predicted[frame]:
            cv2.rectangle(image, (int(box["x"]), int(box["y"])),
                          (int(box["x"] + box["w"]), int(box["y"] + box["h"])), (255, 0, 255), 2)
            cv2.putText(image, f"{box['track_id']} {box['score']:.2f}", (int(box["x"]), int(box["y"]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 255), 2)
        cv2.putText(image, f"{sequence} MOT frame {frame}: GT green, VIAME magenta", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(directory / f"{seq_key(sequence)}-{frame:06d}.jpg"), image)


def video_path_check(journal: Journal, manifest: dict, sequence: str) -> dict:
    """Production's video path (frame offset -1) must land on the same MOT frames."""
    video = ROOT / "video" / f"{seq_key(sequence)}.mkv"
    video.parent.mkdir(parents=True, exist_ok=True)
    info_path = video.with_suffix(".json")
    if not info_path.exists():
        if video.exists():
            video.unlink()
        save(info_path, sequence_to_video(image_paths(sequence), video, NATIVE_FPS))
    output = ROOT / "validate" / "video" / seq_key(sequence)
    run_pass(journal, f"validate/video/{sequence}", "gpu",
             [video, output, "--fps", PROFILES["batch-10fps"], "--pipeline", PRODUCTION_PIPE], output,
             timeout=900)
    from_video = load_mot_rows(viame_tracks_to_mot(output / "viame_tracks.csv", frame_offset=-1))
    image_list = load_mot(ROOT / "mot" / "batch-10fps" / tag(dict(PRODUCTION)) / f"{sequence}.txt")
    shared = set(from_video) & set(image_list)
    matched = sum(any(_iou(a, b) >= .9 for b in image_list[f]) for f in shared for a in from_video[f])
    return {"video": read(info_path), "video_frames": sorted(from_video)[:12],
            "image_list_frames": sorted(image_list)[:12],
            "same_frame_set": set(from_video) == set(image_list),
            "video_boxes": sum(map(len, from_video.values())),
            "video_boxes_matching_image_list_iou_0_9": matched,
            "alignment": alignment_rows(from_video, sequence)}


def load_mot_rows(rows: list[dict]) -> dict[int, list[dict]]:
    frames = defaultdict(list)
    for row in rows:
        frames[row["frame"]].append(row)
    return frames


def alignment_rows(predicted: dict, sequence: str) -> dict:
    truth = load_truth(sequence)
    return {str(shift): sum(any(_iou(p, g) >= .5 for g in truth.get(frame + shift, []))
                            for frame, boxes in predicted.items() for p in boxes)
            for shift in (-1, 0, 1)}


def sweep(journal: Journal, manifest: dict) -> None:
    configs = manifest["configs"]
    split = manifest["split"]
    for profile, fps in PROFILES.items():
        production = tag(dict(PRODUCTION))
        candidates = {tag(effective(c, fps)): effective(c, fps) for c in configs}
        candidates[production] = dict(PRODUCTION)
        names = {tag(effective(c, fps)): c["name"] for c in configs}
        # Round one: every configuration on half of the tuning sequences.
        replay_all(journal, manifest, [(profile, s, q) for s in candidates.values()
                                       for q in split["round_one"]])
        one = {t: summary(evaluate(manifest, "round-1", profile, s, split["round_one"]))
               for t, s in candidates.items()}
        top = ranked(one, production)[:3]
        journal.decide(f"round-1/{profile}", {"results": one, "advance": top, "names": names})
        # Round two: the top three and production on every tuning sequence.
        chosen = {t: candidates[t] for t in [production, *top]}
        replay_all(journal, manifest, [(profile, s, q) for s in chosen.values() for q in split["tuning"]])
        two = {t: summary(evaluate(manifest, "round-2", profile, s, split["tuning"]))
               for t, s in chosen.items()}
        best = ranked(two, production)[0]
        journal.decide(f"round-2/{profile}", {"results": two, "best": best})
        # Held-out: the single best against production.
        final = {t: candidates[t] for t in (production, best)}
        replay_all(journal, manifest, [(profile, s, q) for s in final.values() for q in split["held_out"]])
        held = {t: evaluate(manifest, "held-out", profile, s, split["held_out"]) for t, s in final.items()}
        journal.decide(f"held-out/{profile}", {"best": best, "settings": candidates[best],
                                               "results": {t: summary(r) for t, r in held.items()},
                                               "per_sequence": per_sequence_gain(held[production], held[best])})


def supplementary(journal: Journal, manifest: dict) -> None:
    """Outside the protocol: the highest-IDF1 round-two configuration on held-out, when
    the constraint-aware ranking passed it over. Reported as context, never promoted."""
    for profile in PROFILES:
        two = journal.data["decisions"][f"round-2/{profile}"]
        production = tag(dict(PRODUCTION))
        top = max((t for t in two["results"] if t != production), key=lambda t: two["results"][t]["idf1"])
        if top == two["best"]:
            journal.decide(f"supplementary/{profile}", {"config": top, "note": "Same as the protocol choice"})
            continue
        settings = next(effective(c, PROFILES[profile]) for c in manifest["configs"]
                        if tag(effective(c, PROFILES[profile])) == top)
        replay_all(journal, manifest, [(profile, settings, s) for s in manifest["split"]["held_out"]])
        base = evaluate(manifest, "held-out", profile, dict(PRODUCTION), manifest["split"]["held_out"])
        held = evaluate(manifest, "held-out", profile, settings, manifest["split"]["held_out"])
        journal.decide(f"supplementary/{profile}", {
            "config": top, "settings": settings, "results": {production: summary(base), top: summary(held)},
            "per_sequence": per_sequence_gain(base, held),
            "note": "Highest round-two IDF1, but outside the FP/MOTA constraints; not selected by the protocol"})


def segments(journal: Journal, manifest: dict, seconds: float = 2.0) -> None:
    """Supplementary, outside the gate: live mode runs VIAME on each 2 s camera segment,
    so the tracker restarts every 8 frames at 4 FPS and live_monitor joins segments.
    Each held-out sequence is replayed as independent 2 s segments and scored on boxes
    only (CLEAR TP/FP/FN); identities are not comparable across segments."""
    profile = "live-4fps"
    best = journal.data["decisions"][f"held-out/{profile}"]["settings"]
    results = {}
    for settings in (dict(PRODUCTION), best):
        jobs = [(settings, sequence) for sequence in manifest["split"]["held_out"]]
        with ThreadPoolExecutor(WORKERS) as pool:
            for future in [pool.submit(segment_replay, journal, manifest, *job, seconds) for job in jobs]:
                future.result()
        directory = ROOT / "eval" / "segments" / tag(settings)
        if not (directory / "evaluation.json").exists():
            directory.mkdir(parents=True, exist_ok=True)
            analysed = {}
            for sequence in manifest["split"]["held_out"]:
                target = directory / f"{sequence}.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / "mot-segments" / tag(settings) / f"{sequence}.txt", target)
                analysed[sequence] = [i + 1 for i in manifest["analysed_positions"][profile]
                                      if i < manifest["sequences"][sequence]["frames"]]
            (directory / "analysed_frames.json").write_text(json.dumps(analysed), encoding="utf-8")
            evaluate_mot(IMPORTED, directory, directory / "evaluation.json")
        report = read(directory / "evaluation.json")["clear"]
        results[tag(settings)] = {k: report[k] for k in ("tp", "fp", "fn")} | {
            "precision": report["tp"] / max(1, report["tp"] + report["fp"]),
            "recall": report["tp"] / max(1, report["tp"] + report["fn"])}
    journal.decide("segments/live-4fps", {"segment_seconds": seconds, "results": results,
                                          "note": "Supplementary; boxes only; not a gate criterion"})


def segment_replay(journal, manifest, settings, sequence, seconds) -> None:
    from itertools import groupby

    mot = ROOT / "mot-segments" / tag(settings) / f"{sequence}.txt"
    if mot.exists():
        return
    images = image_paths(sequence)
    positions = [i for i in manifest["analysed_positions"]["live-4fps"]
                 if i < manifest["sequences"][sequence]["frames"]]
    detections = ROOT / "detections" / f"{settings['score']:.2f}" / f"{seq_key(sequence)}.csv"
    rows = []
    for window, members in groupby(positions, key=lambda i: int(i // (seconds * NATIVE_FPS))):
        members = list(members)
        listing = ROOT / "lists" / "segments" / f"{seq_key(sequence)}-{window:03d}.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("".join(f"{images[i]}\n" for i in members))
        output = ROOT / "segments" / tag(settings) / seq_key(sequence) / f"{window:03d}"
        run_pass(journal, f"segment/{tag(settings)}/{sequence}/{window}", "cpu",
                 [listing, output, "--image-list", "--pipeline", REPLAY_PIPE, "--detections", detections,
                  "--high", settings["high"], "--low", settings["low"], "--new-track", settings["new_track"],
                  "--buffer", settings["buffer"]], output, timeout=900)
        names = [images[i].name for i in members]
        for row in viame_tracks_to_mot(output / "viame_tracks.csv", input_names=names,
                                       frame_of={n: int(n[:-4]) for n in names}):
            rows.append(row | {"track_id": row["track_id"] + 100000 * (window + 1)})
    write_mot(sorted(rows, key=lambda r: (r["frame"], r["track_id"])), mot)


def per_sequence_gain(base: dict, candidate: dict) -> dict:
    rows = {}
    for sequence, b in base["sequences"].items():
        c = candidate["sequences"][sequence]
        if b["gt_boxes"]:
            rows[sequence] = {"production_idf1": b["identity"]["idf1"],
                              "candidate_idf1": c["identity"]["idf1"],
                              "gain": (c["identity"]["idf1"] or 0) - (b["identity"]["idf1"] or 0)}
    gains = sum(r["gain"] > 0 for r in rows.values())
    return {"sequences_with_fish": len(rows), "sequences_with_gain": gains,
            "majority": gains > len(rows) / 2, "rows": rows}


def smartbay(journal: Journal, manifest: dict) -> None:
    """Full production-path passes on the fixed SmartBay clips: baseline vs best."""
    clips = [c for c in read(SMARTBAY_MANIFEST)["clips"] if c["id"] in ("0", "1", "2")]
    for clip in clips:
        if digest(clip["path"]) != clip["sha256"]:
            raise RuntimeError(f"SmartBay clip changed: {clip['path']}")
        # CSV timestamps are stream-absolute; the manifest's arrays are clip-relative.
        clip["pts_origin"] = clip.get("pts_origin", probe(clip["path"])["pts_origin"])
    save(ROOT / "smartbay-clips.json", {c["id"]: {"pts_origin": c["pts_origin"]} for c in clips})
    results = {}
    for profile, fps in PROFILES.items():
        best = journal.data["decisions"][f"held-out/{profile}"]["settings"]
        for label, settings in (("production", dict(PRODUCTION)), ("candidate", best)):
            for clip in clips:
                results[(profile, label, clip["id"])] = smartbay_pass(journal, profile, fps, settings, clip)
    repeat = smartbay_pass(journal, "batch-10fps", PROFILES["batch-10fps"], dict(PRODUCTION),
                           clips[1], repeat=True)
    summary_rows = {}
    for profile, fps in PROFILES.items():
        per_clip = {label: [results[(profile, label, c["id"])] for c in clips]
                    for label in ("production", "candidate")}
        summary_rows[profile] = compare_smartbay(per_clip, fps)
    first = results[("batch-10fps", "production", clips[1]["id"])]
    # File hashes always differ: VIAME stamps exec/export times in a header comment.
    summary_rows["repeat"] = {"clip": clips[1]["id"],
                              "identical_tracks": first["track_rows"] == repeat["track_rows"],
                              "proxies": {k: (first["proxies"][k], repeat["proxies"][k]) for k in PROXIES},
                              "seconds": (first["seconds"], repeat["seconds"])}
    journal.decide("smartbay", summary_rows)


def smartbay_pass(journal, profile, fps, settings, clip, repeat=False) -> dict:
    name = f"smartbay/{profile}/{tag(settings)}/clip-{clip['id']}" + ("/repeat" if repeat else "")
    output = ROOT / "smartbay" / profile / tag(settings) / (f"clip-{clip['id']}" + ("-repeat" if repeat else ""))
    entry = run_pass(journal, name, "gpu",
                     [clip["path"], output, "--fps", fps, "--score", settings["score"],
                      "--high", settings["high"], "--low", settings["low"],
                      "--new-track", settings["new_track"], "--buffer", settings["buffer"],
                      "--pipeline", PRODUCTION_PIPE], output, timeout=900)
    det, _ = observations(output / "computed_detections.csv", clip["timestamps"], -1, clip["pts_origin"])
    tracks, issues = observations(output / "viame_tracks.csv", clip["timestamps"], -1, clip["pts_origin"])
    m = metrics(det, tracks, clip["width"], clip["height"], fps)
    return {"metrics": m, "proxies": {k: f(m) for k, f in PROXIES.items()}, "csv_issues": issues,
            "accepted": {c: m["accepted_tracks"][c] for c in ACCEPTANCE},
            "seconds": entry["duration_seconds"], "tracks_sha256": entry["outputs"]["viame_tracks.csv"],
            "track_rows": sorted(detection_rows(output / "viame_tracks.csv"))}


def compare_smartbay(per_clip: dict, fps: float) -> dict:
    pooled = {}
    for label, runs in per_clip.items():
        tracks = sum(r["metrics"]["tracks"] for r in runs)
        singles = sum(round(r["metrics"]["length_fractions"]["1"] * r["metrics"]["tracks"])
                      for r in runs if r["metrics"]["tracks"])
        accelerations = [r["metrics"]["center_acceleration"]["median"] for r in runs
                         if r["metrics"]["center_acceleration"]["median"] is not None]
        accepted = {c: sum(r["accepted"][c] for r in runs) for c in ACCEPTANCE}
        values = [accepted[c] for c in ACCEPTANCE]
        pooled[label] = {
            "tracks": tracks, "detections": sum(r["metrics"]["detections"] for r in runs),
            "fragmentation": sum(len(r["metrics"]["fragmentation_pairs"]) for r in runs) / tracks if tracks else None,
            "duplicates": sum(len(r["metrics"]["duplicate_pairs"]) for r in runs) / tracks if tracks else None,
            "singleton_tracks": singles / tracks if tracks else None,
            "acceleration": sorted(accelerations)[len(accelerations) // 2] if accelerations else None,
            "accepted": accepted,
            "largest_adjacent_drop": max((a - b) / max(a, 1) for a, b in zip(values, values[1:])),
            "seconds": sum(r["seconds"] for r in runs)}
    base, cand = pooled["production"], pooled["candidate"]
    changes = {k: relative_change(base[k], cand[k]) for k in PROXIES}
    return {"pooled": pooled, "relative_change": changes,
            "proxy_regressions_over_5_percent": [k for k, v in changes.items() if v is None or v > .05],
            "acceptance_stable": cand["largest_adjacent_drop"] <= base["largest_adjacent_drop"] + .05,
            "runtime_ratio": cand["seconds"] / base["seconds"],
            "per_clip": {label: [{"proxies": r["proxies"], "accepted": r["accepted"],
                                  "tracks": r["metrics"]["tracks"], "seconds": r["seconds"]} for r in runs]
                         for label, runs in per_clip.items()}}


def gate(journal: Journal) -> dict:
    decisions = journal.data["decisions"]
    result = {}
    for profile in PROFILES:
        held = decisions[f"held-out/{profile}"]
        production, best = tag(dict(PRODUCTION)), held["best"]
        base, cand = held["results"][production], held["results"][best]
        smart = decisions.get("smartbay", {}).get(profile)
        criteria = {
            "idf1_gain_at_least_2_points": cand["idf1"] - base["idf1"] >= .02,
            "mota_not_lower": cand["mota"] >= base["mota"],
            "fp_rise_at_most_10_percent": cand["fp"] <= 1.10 * base["fp"],
            "idsw_per_gt_track_not_higher": cand["idsw_per_gt_track"] <= base["idsw_per_gt_track"],
            "gain_on_majority_of_sequences": held["per_sequence"]["majority"],
            "smartbay_proxies_within_5_percent": None if smart is None else not smart["proxy_regressions_over_5_percent"],
            "smartbay_acceptance_stable": None if smart is None else smart["acceptance_stable"],
            "runtime_within_1_1x": None if smart is None else smart["runtime_ratio"] <= 1.10}
        result[profile] = {"candidate": best, "criteria": criteria,
                           "passes": all(v is True for v in criteria.values()),
                           "idf1": (base["idf1"], cand["idf1"]), "mota": (base["mota"], cand["mota"]),
                           "fp": (base["fp"], cand["fp"]),
                           "idsw_per_gt_track": (base["idsw_per_gt_track"], cand["idsw_per_gt_track"])}
    result["promote"] = all(result[p]["passes"] for p in PROFILES)
    journal.decide("gate", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "detect", "validate", "sweep", "supplementary",
                                          "segments", "smartbay", "gate"])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    journal = Journal()
    manifest = prepare(journal)
    if args.stage == "detect":
        detect(journal, manifest)
    elif args.stage == "validate":
        validate(journal, manifest)
    elif args.stage == "sweep":
        sweep(journal, manifest)
    elif args.stage == "supplementary":
        supplementary(journal, manifest)
    elif args.stage == "segments":
        segments(journal, manifest)
    elif args.stage == "smartbay":
        smartbay(journal, manifest)
    elif args.stage == "gate":
        print(json.dumps(gate(journal), indent=2))
    print(json.dumps({"stage": args.stage, "gpu_seconds": journal.gpu_seconds()}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
