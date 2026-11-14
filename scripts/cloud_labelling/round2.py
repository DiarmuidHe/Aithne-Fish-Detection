"""Round 2 across several instances: label the new windows on N labellers launched from one image,
merge their labels with round 1's on the PC, then train on one instance. Runbook:
``docs/smartbay-round-2.md``. From the repository root on the PC:

    .venv\\Scripts\\python.exe -m scripts.cloud_labelling.round2 STAGE

- ``labelling-bundle``: one bundle every labeller unpacks: the code, round 2's selection and a
  journal seeded with round 1's tracker and prompt decisions. Each labeller then runs its own shard
  (``run_labelling.sh`` with ``AITHNE_SHARD=i/n``), so one ``to_label.csv`` is split without overlap.
- ``merge``: round 1's labels and every labeller's copied-back folder into ``round-2/cloud``, the
  folder training reads. A file two sources hold with different bytes, or a journal step two
  labellers both ran, stops the merge. ``to_label.csv`` then lists every window of the round (round
  0 and 1's reused, round 2's new), and the round checks run on it without contact sheets, as the
  videos stay on the instances.
- ``training-bundle``: the code, the merged round and round 1's journal (for its frozen validation
  days), for the training instance, which restores every window from the public archive.
- ``round1-bundle``: round 1's labels alone, for when one instance labels everything and then
  merges and trains by itself (``run_round2.sh after-label``), with no PC step in between.
- ``split`` (on that instance, while it labels): hand the end of its workers' lists to new
  instances (``split/instance-<i>.csv``, run with ``AITHNE_TO_LABEL``), marked in its journal so its
  workers skip them. ``relay_shards.sh`` copies the new instances' results to it, and
  ``after-label`` with ``AITHNE_EXTRA_SHARDS`` merges them, dropping the marks.

Labels are SAM 3's unreviewed suggestions, never ground truth. Nothing here uses the network.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from scripts.cloud_labelling import package_base_image, package_hard_negatives
from scripts.cloud_labelling.round_checks import check

ROUND1 = Path("data/outputs/smartbay-rounds/round-1/cloud")  # Round 1 as copied back to the PC
ROUND2 = Path("data/outputs/smartbay-rounds/round-2")
INSTANCE_ROUND1 = "data/outputs/smartbay-rounds/round-1"
INSTANCE_ROUND2 = "data/outputs/smartbay-rounds/round-2"
SELECTION_FILES = ("to_label.csv", "manifest.csv", "day_roles.json", "summary.json")
# Round 1's settings every labeller must use, copied from its journal before any window is labelled;
# its exclusion rules join them at the merge, so round 2's windows are judged by the same rules.
COPIED_DECISIONS = ("brackishmot_settings", "sam3_prompt_choice", "sam3_choice_rule", "labelling_processes")
MERGE_DECISIONS = (*COPIED_DECISIONS, "auto_exclusions")
# Per labeller: two SAM 3 workers each charge their own time, about 221 s of sam3_gpu a window in
# round 1 (46,818 s for 212). The cap allows 300 s a window of the instance's share: room for a slow
# run, and a stop for a runaway one.
SAM3_SECONDS_PER_WINDOW = 300
LABELLER_BUDGETS = {"sam3_gpu": 30_000, "training": 0, "viame": 0}  # 4 labellers, about 61 windows each


def labeller_budgets(windows: int, shards: int) -> dict:
    return LABELLER_BUDGETS | {"sam3_gpu": -(-windows // shards) * SAM3_SECONDS_PER_WINDOW}
# Training on about 2.3 times round 1's data: 15 epochs of about 21 minutes, the probe epoch and evaluation.
TRAINING_BUDGET_SECONDS = 6 * 3600
# What a labelled window carries forward; the contact sheets and background images stay behind.
WINDOW_FILES = ("video.json", "labels/fish/rows.jsonl", "labels/fish/report.json", "sam3/fish/frames.jsonl",
                "sam3/fish/tracked.jsonl", "sam3/fish/report.json", "static/sam3-fish/removed.jsonl",
                "static/sam3-fish/kept.jsonl", "static/sam3-fish/report.json")
# What training, evaluation and the student comparison read of each window
TRAINING_WINDOW_FILES = ("video.json", "labels/fish/rows.jsonl", "sam3/fish/frames.jsonl",
                         "static/sam3-fish/removed.jsonl", "static/sam3-fish/report.json")


def sha256_of(path: Path) -> str:
    return package_hard_negatives.sha256_of(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8", newline="\n")


# --------------------------------------------------------------- labelling --

def seed_journal(round1_journal: dict, selection: Path, shards: int = 4) -> dict:
    """A labeller's starting journal: round 1's labelling decisions, and a SAM 3 budget for its
    share of the windows when ``shards`` instances split them."""
    with (selection / "to_label.csv").open(newline="", encoding="utf-8") as stream:
        budgets = labeller_budgets(sum(1 for _ in csv.DictReader(stream)), shards)
    decisions = {name: round1_journal["decisions"][name] for name in COPIED_DECISIONS}
    now = datetime.now(UTC).isoformat()
    decisions["journal_seed"] = {"copied": list(COPIED_DECISIONS), "from": f"{INSTANCE_ROUND1}/journal.json",
                                 "to_label_sha256": sha256_of(selection / "to_label.csv"), "shards": shards,
                                 "budgets_seconds": budgets, "decided_utc": now}
    return {"schema_version": 1, "created_utc": now, "budgets_seconds": budgets, "entries": {},
            "decisions": decisions}


def labelling_bundle(selection: Path, output: Path, round1: Path = ROUND1, shards: int = 4) -> dict:
    """The code, the selection and the seeded journal, at the instance's round 2 paths."""
    missing = [name for name in SELECTION_FILES if not (selection / name).is_file()]
    if missing:
        raise SystemExit(f"{selection} lacks {missing}: run the round 2 selection first")
    journal = output.with_name("journal-seed.json")
    seed = seed_journal(read_json(round1 / "journal.json"), selection, shards)
    write_json(journal, seed)
    files = package_base_image.contents(Path(".")) | {
        f"{INSTANCE_ROUND2}/selection/{name}": selection / name for name in SELECTION_FILES} | {
        f"{INSTANCE_ROUND2}/journal.json": journal}
    with (selection / "to_label.csv").open(newline="", encoding="utf-8") as stream:
        windows = sum(1 for _ in csv.DictReader(stream))
    return package_hard_negatives.build(Path("."), Path("."), output, files=files) | {
        "windows_to_label": windows, "shards": shards, "budgets_seconds": seed["budgets_seconds"]}


# ----------------------------------------------------------- mid-run split --

PLACEHOLDER = "placeholder"  # A journal entry marking a window another instance labels


def worker_position(rows: list[dict], events: Path) -> int:
    """Where one worker is in its list ``rows`` (in order): the row after the last one it labelled
    or gave up on, by its event log."""
    index, position = {row["window_id"]: i for i, row in enumerate(rows)}, 0
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event"] in ("labelled", "failed") and event["window_id"] in index:
                position = max(position, index[event["window_id"]] + 1)
    return position


def split_remaining(root: Path, instances: int, *, workers: int = 2, head_start: int = 1,
                    dry_run: bool = False) -> dict:
    """Hand the end of a running labeller's lists to ``instances`` new ones (round 2, 2026-11-14).

    The running labeller's ``workers`` keep their order (``run_windows.rows_for``): each keeps the
    row it is on and its share of what is left, plus ``head_start`` rows while the new instances
    start. Every later row goes, in ``to_label.csv`` order, round robin to the new instances, as
    ``split/instance-<i>.csv`` (i from 1, for ``AITHNE_TO_LABEL``), and is marked labelled in the
    running journal by a placeholder entry: its workers skip it, and ``merge`` drops it."""
    from scripts.cloud_labelling.run_windows import PROMPT, rows_for
    from scripts.sam3_botsort import Journal

    out = root / "split"
    if (out / "split.json").exists():
        raise SystemExit(f"{out / 'split.json'} exists: {root} was split already")
    to_label = root / "selection" / "to_label.csv"
    with to_label.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        order = {row["window_id"]: i for i, row in enumerate(reader)}
        fields = reader.fieldnames
    entries = read_json(root / "journal.json")["entries"]

    def done(row: dict) -> bool:
        return any(entries.get(f"{w}/labels/{PROMPT}", {}).get("status") == "complete"
                   for w in (row["window_id"], row.get("reserve_window_id")) if w)

    lists = [rows_for(to_label, k, workers) for k in range(workers)]
    positions = [worker_position(rows, root / "run_windows" / f"worker-{k}.jsonl") for k, rows in enumerate(lists)]
    remaining = sum(len(rows) - p for rows, p in zip(lists, positions, strict=True))
    keep = -(-remaining // ((instances + 1) * workers)) + head_start
    kept, handed = [], []
    for rows, p in zip(lists, positions, strict=True):
        kept += [row["window_id"] for row in rows[p:p + keep]]
        handed += [row for row in rows[p + keep:] if not done(row)]
    handed.sort(key=lambda row: order[row["window_id"]])
    assigned = {i: handed[i - 1::instances] for i in range(1, instances + 1)}
    now = datetime.now(UTC).isoformat()
    summary = {"decided_utc": now, "workers": workers, "positions": positions, "remaining": remaining,
               "keep_per_worker": keep, "kept": kept, "kept_count": len(kept),
               "instances": {str(i): [row["window_id"] for row in rows] for i, rows in assigned.items()},
               "counts": {str(i): len(rows) for i, rows in assigned.items()}}
    if dry_run:
        return summary
    out.mkdir(parents=True, exist_ok=True)
    for i, rows in assigned.items():
        with (out / f"instance-{i}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    journal = Journal(root)  # Its lock and merge-on-save keep the running workers' entries
    for i, rows in assigned.items():
        for row in rows:
            name = f"{row['window_id']}/labels/{PROMPT}"
            journal.data["entries"][name] = {"kind": PLACEHOLDER, "status": "complete", "result": {}, "attempts": [],
                                             PLACEHOLDER: {"instance": i, "decided_utc": now}}
            journal.touched["entries"].add(name)
    journal.save()
    write_json(out / "split.json", summary)
    return summary


# ------------------------------------------------------------------- merge --

class MergeConflict(RuntimeError):
    pass


def copy_file(source: Path, target: Path, origin: dict[Path, str], name: str) -> None:
    """Copy unless the target already holds the same bytes; a difference is a conflict."""
    if target.exists():
        if not filecmp.cmp(source, target, shallow=False):
            raise MergeConflict(f"{target} differs between {origin.get(target, 'the merged round')} and {name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    origin[target] = name


def merge(selection: Path, shards: list[Path], out: Path, round1: Path = ROUND1) -> dict:
    """Round 1's labelled windows and every shard's, with their journal steps, into ``out``."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; merge into a new folder")
    origin: dict[Path, str] = {}
    entries, step_source, sources = {}, {}, {}
    for name, root in [("round-1", round1), *((f"shard-{i}", s) for i, s in enumerate(shards))]:
        journal = read_json(root / "journal.json")
        # A window's own steps only: round 1's student, restore and probe steps stay behind, as do a
        # split's placeholders (the instance the window was handed to holds its real steps)
        steps = {k: v for k, v in journal["entries"].items() if k.startswith("SmartBayCam") and PLACEHOLDER not in v}
        for key, value in steps.items():
            if key in entries and entries[key] != value:
                raise MergeConflict(f"journal step {key} is in both {step_source[key]} and {name}")
            entries[key], step_source[key] = value, name
        windows = sorted(p.name for p in (root / "videos").iterdir() if p.is_dir()) if (root / "videos").exists() else []
        for window in windows:
            for relative in WINDOW_FILES:
                source = root / "videos" / window / relative
                if source.exists():
                    copy_file(source, out / "videos" / window / relative, origin, name)
        for record in sorted((root / "windows").glob("*.json")) if (root / "windows").exists() else []:
            copy_file(record, out / "windows" / record.name, origin, name)
        for events in sorted((root / "run_windows").glob("worker-*.jsonl")) if name != "round-1" else []:
            copy_file(events, out / "run_windows" / f"{name}-{events.name}", origin, name)
        sources[name] = {"root": root.as_posix(), "windows": len(windows), "journal_steps": len(steps)}
    first = read_json(round1 / "journal.json")
    decisions = {k: first["decisions"][k] for k in MERGE_DECISIONS if k in first["decisions"]}
    decisions["round_2_merge"] = {"sources": sources, "decided_utc": datetime.now(UTC).isoformat()}
    write_json(out / "journal.json", {"schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
                                      "budgets_seconds": {"sam3_gpu": 0, "training": TRAINING_BUDGET_SECONDS,
                                                          "viame": 0},
                                      "entries": entries, "decisions": decisions})
    target = out / "selection"
    for name in SELECTION_FILES:
        copy_file(selection / name, target / name, origin, "selection")
    # Training reads every window of the round, not just those this round labelled
    (target / "to_label.csv").replace(target / "labelled_in_round_2.csv")
    shutil.copyfile(target / "manifest.csv", target / "to_label.csv")
    report = check(out, target / "to_label.csv", sheets=False)
    unlabelled = [w["window_id"] for w in report["per_window"] if w["labelled_as"] is None]
    summary = {"out": out.as_posix(), "sources": decisions["round_2_merge"]["sources"], "windows": report["windows"],
               "labelled": report["labelled"], "kept": report["kept"], "excluded": report["excluded"],
               "exclusion_reasons": report["exclusion_reasons"], "unlabelled": unlabelled[:20],
               "unlabelled_count": len(unlabelled)}
    write_json(out / "merge_summary.json", summary)
    return summary


# ---------------------------------------------------------------- training --

def training_files(out: Path, round1: Path = ROUND1) -> dict[str, Path]:
    """The code, the merged round at the instance's round 2 path and round 1's journal."""
    files = package_base_image.contents(Path("."))
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out).as_posix()
        if not path.is_file() or relative.startswith(("run_windows/", "logs/", "upload/")):
            continue
        if relative.startswith("videos/"):
            if relative.split("/", 2)[2] in TRAINING_WINDOW_FILES:
                files[f"{INSTANCE_ROUND2}/{relative}"] = path
        elif path.suffix in (".json", ".jsonl", ".csv", ".sha256"):
            files[f"{INSTANCE_ROUND2}/{relative}"] = path
    files[f"{INSTANCE_ROUND1}/journal.json"] = round1 / "journal.json"
    for needed in ("journal.json", "checks/quality_report.json", "selection/to_label.csv", "selection/day_roles.json"):
        if f"{INSTANCE_ROUND2}/{needed}" not in files:
            raise SystemExit(f"{out} lacks {needed}: run merge first")
    return files


def training_bundle(out: Path, output: Path, round1: Path = ROUND1) -> dict:
    return package_hard_negatives.build(Path("."), Path("."), output, files=training_files(out, round1))


def round1_files(round1: Path = ROUND1) -> dict[str, Path]:
    """Round 1's labels at the instance's round 1 path, for a merge on the instance itself: its
    journal, each window's files that training reads and the window records."""
    files = {f"{INSTANCE_ROUND1}/journal.json": round1 / "journal.json"}
    for path in sorted((round1 / "videos").rglob("*")):
        relative = path.relative_to(round1).as_posix()
        if path.is_file() and relative.split("/", 2)[2] in TRAINING_WINDOW_FILES:
            files[f"{INSTANCE_ROUND1}/{relative}"] = path
    for path in sorted((round1 / "windows").iterdir()):
        if path.suffix in (".json", ".sha256"):
            files[f"{INSTANCE_ROUND1}/windows/{path.name}"] = path
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["labelling-bundle", "merge", "training-bundle", "round1-bundle", "split"])
    parser.add_argument("--root", type=Path, default=ROUND2, help="split: the running labeller's round folder")
    parser.add_argument("--instances", type=int, default=3, help="split: how many new instances take windows")
    parser.add_argument("--head-start", type=int, default=1,
                        help="split: extra rows each running worker keeps while the new instances start")
    parser.add_argument("--dry-run", action="store_true", help="split: show the split, change nothing")
    parser.add_argument("--output", type=Path, help="labelling-bundle: the bundle's path")
    parser.add_argument("--selection", type=Path, default=ROUND2 / "selection")
    parser.add_argument("--shard", type=Path, action="append", default=[],
                        help="merge: a labeller's round 2 folder as copied back (repeatable, in shard order)")
    parser.add_argument("--out", type=Path, default=ROUND2 / "cloud", help="merge, training-bundle: the merged round")
    parser.add_argument("--round1", type=Path, default=ROUND1)
    parser.add_argument("--shards", type=int, default=4, help="labelling-bundle: how many instances split the windows")
    args = parser.parse_args()
    upload = ROUND2 / "upload"
    if args.stage == "labelling-bundle":
        result = labelling_bundle(args.selection, args.output or upload / "aithne-round2-labelling.tgz", args.round1,
                                  args.shards)
    elif args.stage == "split":
        result = split_remaining(args.root, args.instances, head_start=args.head_start, dry_run=args.dry_run)
    elif args.stage == "merge":
        if not args.shard:
            parser.error("merge needs every labeller's folder as --shard")
        result = merge(args.selection, args.shard, args.out, args.round1)
    elif args.stage == "round1-bundle":
        result = package_hard_negatives.build(Path("."), Path("."), upload / "aithne-round1-labels.tgz",
                                              files=round1_files(args.round1))
    else:
        result = training_bundle(args.out, upload / "aithne-round2-training.tgz", args.round1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
