"""Copy the EC2 archive scan into the local ``data/archive_scan`` on a schedule, so losing
the instance costs at most one interval of results.

PowerShell, from the repository root (a pull now, then every 30 minutes until Ctrl+C):

    .venv\\Scripts\\python.exe -m scripts.smartbay_archive_pull --host ubuntu@<public IP> --every 30

Add ``--once`` for a single pull. The instance's public IP changes after a stop/start, and
SSH only works while the security group allows this PC's current public IP.

Each pull copies only files that are new or changed on the instance:

- no local file is ever deleted;
- a local file that differs and is newer than the instance's copy is a conflict, and the
  whole pull is skipped and reported;
- while the scanner runs, the day it is writing is left for a later pull;
- every copied file must match the instance's SHA-256, every local target must be
  unchanged since planning, and a year ledger may only gain completed days, all checked
  before any local file changes.

A repeatable ``--exclude <glob>`` leaves out files whose path (relative to ``--remote``) or
name matches: the instance neither hashes nor sends them. A labelling round needs it for
files rewritten while it runs (``journal.json``, live logs) and for the cut windows:

    .venv\\Scripts\\python.exe -m scripts.smartbay_archive_pull --host ubuntu@<IP> `
      --remote /home/ubuntu/Aithne-Fish-Detection/data/outputs/smartbay-rounds/round-1 `
      --local data/outputs/smartbay-rounds/round-1/cloud --exclude journal.json --exclude "*.mp4" --every 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path

LOCAL = Path("data/archive_scan")
REMOTE = "/home/ubuntu/Aithne-Fish-Detection/data/archive_scan"
KEY = Path.home() / ".ssh" / "fish-sam3-key.pem"
NEVER_COPY = (".cloud-scan.lock",)
NEVER_COPY_SUFFIXES = (".json.tmp",)
MTIME_SLACK = 2.0  # seconds; filesystems round mtimes differently

# Runs on the instance with its system python3: hashes the scan folder except the files the
# globs after the folder exclude (the rule of ``excluded``), and reports whether a scanner
# process is alive and which day each year ledger is in the middle of.
REMOTE_MANIFEST = r"""
import fnmatch, hashlib, json, os, sys
root, patterns = sys.argv[1], sys.argv[2:]
files, excluded = {}, 0
for dirpath, _dirs, names in os.walk(root):
    for name in names:
        full = os.path.join(dirpath, name)
        rel = os.path.relpath(full, root).replace(os.sep, "/")
        if any(fnmatch.fnmatchcase(rel, p) or fnmatch.fnmatchcase(name, p) for p in patterns):
            excluded += 1
            continue
        digest = hashlib.sha256()
        try:
            stat = os.stat(full)
            with open(full, "rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
        except FileNotFoundError:
            continue
        files[rel] = [stat.st_size, stat.st_mtime, digest.hexdigest()]
running = False
for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
    if pid.isdigit() and int(pid) != os.getpid():
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                running = running or b"smartbay_archive_scan.workflow" in stream.read()
        except OSError:
            pass
in_progress = []
for name in sorted(os.listdir(root)):
    ledger = os.path.join(root, name, "year_state.json")
    if os.path.isfile(ledger):
        with open(ledger, encoding="utf-8") as stream:
            state = json.load(stream)
        if state.get("current_day"):
            in_progress.append(f"{state['camera']}_{state['current_day']}/")
print(json.dumps({"files": files, "scanner_running": running, "in_progress": in_progress, "excluded": excluded}))
"""


def excluded(rel: str, patterns: tuple[str, ...]) -> bool:
    """Whether a relative path, or its file name, matches any ``--exclude`` glob."""
    return any(fnmatchcase(rel, p) or fnmatchcase(rel.rsplit("/", 1)[-1], p) for p in patterns)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_manifest(root: Path, cache: dict | None = None) -> dict[str, list]:
    """``{relative path: [size, mtime, sha256]}``; ``cache`` skips rehashing unchanged files."""
    cache = {} if cache is None else cache
    files = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        key = (rel, stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = sha256(path)
        files[rel] = [stat.st_size, stat.st_mtime, cache[key]]
    return files


def plan(local: dict, remote: dict, skip_prefixes: tuple[str, ...] = (),
         exclude: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """The remote files to copy, and the conflicts that forbid copying anything."""
    fetch, conflicts = [], []
    for rel, (_size, mtime, digest) in sorted(remote.items()):
        if rel.split("/")[-1] in NEVER_COPY or rel.endswith(NEVER_COPY_SUFFIXES) or rel.startswith(skip_prefixes) \
                or excluded(rel, exclude):
            continue
        if rel not in local:
            fetch.append(rel)
        elif local[rel][2] != digest:
            if local[rel][1] > mtime + MTIME_SLACK:
                conflicts.append(f"{rel}: the local copy differs and is newer than the instance's")
            else:
                fetch.append(rel)
    return fetch, conflicts


def check_and_apply(staging: Path, fetch: list[str], remote: dict, local: dict, root: Path) -> list[str]:
    """Verify every staged file and local target, then copy them all; on any problem copy
    nothing and return the problems."""
    problems = []
    for rel in fetch:
        source, target = staging / rel, root / rel
        if not source.is_file() or sha256(source) != remote[rel][2]:
            problems.append(f"{rel}: the downloaded copy does not match the instance's SHA-256")
            continue
        if rel in local:
            if not target.is_file() or sha256(target) != local[rel][2]:
                problems.append(f"{rel}: the local file changed during the pull")
        elif target.exists():
            problems.append(f"{rel}: a local file appeared during the pull")
        if rel.endswith("year_state.json") and target.is_file():
            kept = json.loads(target.read_text(encoding="utf-8"))["completed_days"]
            new = json.loads(source.read_text(encoding="utf-8"))["completed_days"]
            if lost := sorted(set(kept) - set(new)):
                problems.append(f"{rel}: the instance's ledger lacks locally completed days {lost}")
    if problems:
        return problems
    for rel in fetch:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging / rel, root / rel)
    return []


def ssh_command(host: str, key: Path, remote_command: str) -> list[str]:
    return ["ssh", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", host, remote_command]


def pull_once(host: str, key: Path, remote_root: str, root: Path, cache: dict, exclude: tuple[str, ...] = ()) -> bool:
    """One pull; prints a single status line (plus any problems) and returns success."""
    arguments = " ".join(shlex.quote(a) for a in (remote_root, *exclude))
    listing = subprocess.run(ssh_command(host, key, f"python3 -I - {arguments}"), input=REMOTE_MANIFEST,
                             capture_output=True, text=True, timeout=1800)
    if listing.returncode != 0:
        print(f"{utc_now()} could not read the instance: {listing.stderr.strip()[-300:]}", flush=True)
        return False
    report = json.loads(listing.stdout)
    skip = tuple(report["in_progress"]) if report["scanner_running"] else ()
    root.mkdir(parents=True, exist_ok=True)
    local = local_manifest(root, cache)
    fetch, conflicts = plan(local, report["files"], skip, exclude)
    if conflicts:
        print(f"{utc_now()} pull skipped: {len(conflicts)} conflict(s)", *conflicts[:20], sep="\n  ", flush=True)
        return False
    if fetch:
        with tempfile.TemporaryDirectory(prefix="archive-pull-") as temporary:
            staging = Path(temporary)
            archive = staging / "pull.tgz"
            with archive.open("wb") as stream:
                copied = subprocess.run(ssh_command(host, key, f"tar -czf - -C {remote_root} -T -"),
                                        input="".join(f"{rel}\n" for rel in fetch).encode(), stdout=stream,
                                        stderr=subprocess.PIPE, timeout=3600)
            if copied.returncode != 0:
                print(f"{utc_now()} transfer failed: {copied.stderr.decode(errors='replace').strip()[-300:]}",
                      flush=True)
                return False
            with tarfile.open(archive) as bundle:
                bundle.extractall(staging / "files", filter="data")
            problems = check_and_apply(staging / "files", fetch, report["files"], local, root)
        if problems:
            print(f"{utc_now()} pull not applied:", *problems[:20], sep="\n  ", flush=True)
            return False
    size = sum(report["files"][rel][0] for rel in fetch)
    status = []
    for summary in sorted(root.glob("*/year_summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        target = data.get("selected_days", data["discovered_days"])
        status.append(f"{summary.parent.name} {data['completed_days']}/{target} days "
                      f"(last {data['last_completed_day']})")
    waiting = f"; left for later: {', '.join(skip)}" if skip else ""
    left_out = [f"{report['excluded']} excluded"] if exclude else []
    print(utc_now(), "; ".join([f"pulled {len(fetch)} files ({size / 1e6:.1f} MB)", *status, *left_out]) + waiting,
          flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="user@public-ip of the scan instance")
    parser.add_argument("--key", type=Path, default=KEY)
    parser.add_argument("--remote", default=REMOTE, help="Scan folder on the instance")
    parser.add_argument("--local", type=Path, default=LOCAL, help="Scan folder on this PC")
    parser.add_argument("--every", type=float, default=30, help="Minutes between pulls")
    parser.add_argument("--once", action="store_true", help="Pull once and exit (status 1 on failure)")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                        help="Leave out files whose relative path or name matches (repeatable)")
    args = parser.parse_args()
    cache: dict = {}
    while True:
        try:
            ok = pull_once(args.host, args.key, args.remote, args.local, cache, tuple(args.exclude))
        except Exception as error:  # a scheduled backup keeps trying; one bad pull changes nothing
            print(f"{utc_now()} pull failed: {type(error).__name__}: {error}", flush=True)
            ok = False
        if args.once:
            sys.exit(0 if ok else 1)
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
