"""Part 4 of ``docs/smartbay-rolling-training-prompt.md``: label a round's windows with
SAM 3 on the EC2 labelling instance, resumably. ``run_labelling.sh`` starts the workers.

On the instance, from the repository root (after ``bootstrap_labelling.sh``):

    PYTHONPATH=.labelling-python-packages YOLO_OFFLINE=true YOLO_CONFIG_DIR=data/ultralytics \\
      /opt/pytorch/bin/python -m scripts.cloud_labelling.run_windows --worker 0/2

For each row of the round's ``selection/to_label.csv`` that is this worker's (rows
alternate between workers) and whose labels are not yet complete in the round journal:

1. download the source MP4 with the scan's User-Agent (retry with backoff), recording its
   size and SHA256;
2. cut the window by frame number (``trim``, ``libx264 -crf 14``), check its frame count,
   then delete the source, keeping its SHA256 in the window's record;
3. run stages ``sam3``, ``static`` and ``labels`` of ``scripts.sam3_botsort`` with
   ``--no-viame --prompts fish``. ``render`` is never run.

A clip that fails to download, is too short or fails its cut is replaced by the window's
reserve, as the selection planned. Everything stays under the round folder on the
persistent disk: ``windows/<id>.mp4`` and ``windows/<id>.json``, stage logs in ``logs/``,
and one JSON line per event in ``run_windows/worker-<k>.jsonl``. The journal makes a rerun
skip finished work. Labels are unreviewed SAM 3 suggestions, never ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from scripts.smartbay_archive_scan.workflow import AGENT

ROOT = Path("data/outputs/smartbay-rounds/round-1")
PROMPT = "fish"  # The recorded SAM 3 prompt choice (decision sam3_prompt_choice)
STAGES = ("sam3", "static", "labels")
RETRY_SECONDS = (5, 15, 45)
MIN_FREE_GB = 5
MAX_CONSECUTIVE_FAILURES = 3  # A broken GPU or budget stops the worker, not 200 failed windows


class DownloadFailed(RuntimeError):
    pass


class CutFailed(RuntimeError):
    pass


class WindowFailed(RuntimeError):
    pass


class StageFailed(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_for(path: Path, worker: int, workers: int) -> list[dict]:
    """This worker's rows of ``to_label.csv``: every ``workers``-th, from row ``worker``."""
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for i, row in enumerate(csv.DictReader(stream)) if i % workers == worker]


def labelled(root: Path, window_id: str) -> bool:
    """Whether the labels stage of ``window_id`` is complete in the round journal."""
    journal = root / "journal.json"
    if not journal.exists():
        return False
    entry = read_json(journal)["entries"].get(f"{window_id}/labels/{PROMPT}")
    return bool(entry and entry["status"] == "complete")


def download(url: str, path: Path, *, agent: str = AGENT, retries=RETRY_SECONDS,
             opener=urllib.request.urlopen, sleep=time.sleep) -> dict:
    """Stream ``url`` to ``path`` (through a ``.part`` file), hashing as it goes; retry with
    the given back-off, then raise ``DownloadFailed``."""
    part = path.with_name(path.name + ".part")
    for attempt, wait in enumerate((*retries, None), start=1):
        digest, size, started = hashlib.sha256(), 0, time.monotonic()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": agent})
            with opener(request, timeout=60) as response, part.open("wb") as stream:
                for block in iter(lambda: response.read(1 << 20), b""):
                    digest.update(block)
                    stream.write(block)
                    size += len(block)
            part.replace(path)
            return {"url": url, "bytes": size, "sha256": digest.hexdigest(), "attempts": attempt,
                    "seconds": round(time.monotonic() - started, 1)}
        except (OSError, urllib.error.URLError) as error:
            part.unlink(missing_ok=True)
            if wait is None:
                raise DownloadFailed(f"{url}: {error}") from error
            sleep(wait)


def frame_count(path: Path, *, run=subprocess.run) -> int:
    """Decoded video frames, counted by ffprobe."""
    out = run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
               "stream=nb_read_frames", "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def clip_frames(path: Path, *, run=subprocess.run) -> int:
    """A source clip's frame count from its container, without decoding it (the cut's own
    frame count is decoded and checked)."""
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
               "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def cut_command(source: Path, target: Path, start: int, count: int) -> list[str]:
    # veryfast: the same CRF 14 quality with a fraction of the CPU, which the two SAM 3
    # workers share (the default preset slowed both by about 15% on 4 vCPUs).
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-vf",
            f"trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(target)]


def cut(source: Path, target: Path, start: int, count: int, *, run=subprocess.run, frames=frame_count) -> int:
    """Cut frames ``start``..``start + count - 1`` of ``source`` into ``target``; refuse a
    result with any other frame count."""
    part = target.with_name(target.stem + ".part.mp4")
    try:
        run(cut_command(source, part, start, count), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        part.unlink(missing_ok=True)
        raise CutFailed(f"ffmpeg failed: {(error.stderr or '').strip()[-300:]}") from error
    got = frames(part)
    if got != count:
        part.unlink(missing_ok=True)
        raise CutFailed(f"cut has {got} frames, expected {count}")
    part.replace(target)
    return got


def choices(row: dict) -> list[tuple[str, str, str, int]]:
    """The window, then its reserve: (which, window ID, source URL, start frame)."""
    options = [("window", row["window_id"], row["source_mp4_url"], int(row["start_frame"]))]
    if row.get("reserve_window_id"):
        options.append(("reserve", row["reserve_window_id"], row["reserve_url"], int(row["reserve_start_frame"])))
    return options


def prepare(row: dict, windows: Path, *, fetch=download, trim=cut, frames=clip_frames) -> dict:
    """The window's cut video and its record, from its own clip or else its reserve. A
    window cut earlier with an intact file is reused."""
    count, failures = int(row["frame_count"]), []
    windows.mkdir(parents=True, exist_ok=True)
    for _, window_id, _, _ in choices(row):
        target, record_path = windows / f"{window_id}.mp4", windows / f"{window_id}.json"
        if record_path.exists() and target.exists():
            record = read_json(record_path)
            if record["cut"]["sha256"] == sha256(target):
                return record
    for which, window_id, url, start in choices(row):
        target, record_path = windows / f"{window_id}.mp4", windows / f"{window_id}.json"
        source = windows / Path(urlparse(url).path).name
        try:
            fetched = fetch(url, source)
            available = frames(source)
            if available < start + count:
                raise CutFailed(f"clip has {available} frames; the window needs {start} + {count}")
            started = time.monotonic()
            got = trim(source, target, start, count)
            seconds = round(time.monotonic() - started, 1)
        except (DownloadFailed, CutFailed, subprocess.CalledProcessError, ValueError) as error:
            failures.append({"which": which, "window_id": window_id, "error": f"{type(error).__name__}: {error}"})
            continue  # A failed clip's source stays for inspection; the next option is tried
        source.unlink()
        record = {"window_id": row["window_id"], "labelled_as": window_id, "reserve_used": which == "reserve",
                  "round": row.get("round"), "day": row["day"], "role": row["role"], "evidence": row["evidence"],
                  "source": fetched | {"frames": available, "deleted_after_cut": True}, "start_frame": start,
                  "frame_count": got, "cut": {"bytes": target.stat().st_size, "sha256": sha256(target),
                                              "seconds": seconds,
                                              "command": cut_command(Path(source.name), Path(target.name), start, count)},
                  "failures": failures, "attribution": row["attribution"], "cut_utc": utc_now()}
        save_json(record_path, record)
        return record
    raise WindowFailed(json.dumps(failures))


def run_stage(stage: str, video: Path, root: Path, logs: Path) -> float:
    """One ``scripts.sam3_botsort`` stage on one window; its output goes to a log file."""
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{video.stem}.{stage}.log"
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.run([sys.executable, "-m", "scripts.sam3_botsort", stage, "--video", str(video),
                               "--root", str(root), "--no-viame", "--prompts", PROMPT],
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        raise StageFailed(f"{stage} exited {code}: {' | '.join(tail)[-500:]}")
    return round(time.monotonic() - started, 1)


def run(rows: list[dict], root: Path, events: Path, *, prepare_window=prepare, stage=run_stage, done=labelled,
        free_gb=lambda path: shutil.disk_usage(path).free / 2**30) -> dict:
    """Label ``rows`` in order: skip those already labelled, stop when the disk runs low or
    after ``MAX_CONSECUTIVE_FAILURES`` failed windows in a row, and log every event."""
    windows, logs, summary, consecutive = root / "windows", root / "logs", Counter(), 0
    events.parent.mkdir(parents=True, exist_ok=True)

    def event(kind: str, **fields) -> None:
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": utc_now(), "event": kind} | fields) + "\n")

    event("start", rows=len(rows))
    for row in rows:
        if any(done(root, window_id) for _, window_id, _, _ in choices(row)):
            summary["already_labelled"] += 1
            continue
        if free_gb(root) < MIN_FREE_GB:
            event("stop", reason=f"under {MIN_FREE_GB} GB free")
            break
        try:
            record = prepare_window(row, windows)
            video = windows / f"{record['labelled_as']}.mp4"
            seconds = {name: stage(name, video, root, logs) for name in STAGES}
        except (WindowFailed, StageFailed) as error:
            consecutive += 1
            summary["failed"] += 1
            event("failed", window_id=row["window_id"], error=str(error)[-1000:])
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                event("stop", reason=f"{consecutive} failed windows in a row")
                break
            continue
        consecutive = 0
        summary["labelled"] += 1
        summary["reserve_used"] += record["reserve_used"]
        event("labelled", window_id=row["window_id"], labelled_as=record["labelled_as"], stage_seconds=seconds)
    event("end", **summary)
    return dict(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="Round folder holding journal.json")
    parser.add_argument("--to-label", type=Path, help="Default <root>/selection/to_label.csv")
    parser.add_argument("--worker", default="0/1", help="k/n: take every n-th row from row k")
    parser.add_argument("--limit", type=int, help="Stop after this many of this worker's rows")
    args = parser.parse_args()
    worker, workers = map(int, args.worker.split("/"))
    if not 0 <= worker < workers:
        parser.error("--worker must be k/n with 0 <= k < n")
    rows = rows_for(args.to_label or args.root / "selection" / "to_label.csv", worker, workers)[:args.limit]
    summary = run(rows, args.root, args.root / "run_windows" / f"worker-{worker}.jsonl")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
