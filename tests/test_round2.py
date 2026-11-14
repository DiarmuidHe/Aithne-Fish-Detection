"""Round 2 across instances: the labelling bundle, the merge of round 1 and the labellers' folders,
and the training bundle, on tiny synthetic folders (no network, GPU or video)."""
import csv
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.cloud_labelling import round2
from scripts.cloud_labelling.round_checks import RULES

FIELDS = ["window_id", "round", "day", "role", "source_mp4_url", "clip_time", "start_frame", "frame_count", "evidence",
          "light", "hour_bin", "reserve_window_id", "reserve_url", "reserve_start_frame"]
WINDOWS = {"SmartBayCam1_2025-07-01_1200_f0100": ("0", "2025-07-01"),
           "SmartBayCam1_2025-10-10_1200_f0100": ("2", "2025-10-10"),
           "SmartBayCam2_2025-10-12_1200_f0100": ("2", "2025-10-12@SmartBayCam2")}
OLD, NEW1, NEW2 = WINDOWS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def labelled(root, window, boxes=3):
    """A window as a labeller leaves it: its records, its outputs and its three journal steps."""
    directory = root / "videos" / window
    write_json(directory / "video.json", {"key": window, "file": {"path": f"x/{window}.mp4", "sha256": "v"}})
    rows = "".join(json.dumps({"id": f"t{i}", "frame": i, "box": [0, 0, 10, 10], "confidence": .9}) + "\n"
                   for i in range(boxes))
    (directory / "labels" / "fish").mkdir(parents=True)
    (directory / "labels" / "fish" / "rows.jsonl").write_text(rows)
    write_json(directory / "static" / "sam3-fish" / "report.json", {"counts": {"boxes": 10, "removed_boxes": 1}})
    (directory / "static" / "sam3-fish" / "contact-sheet.jpg").write_bytes(b"jpeg")  # stays behind
    write_json(root / "windows" / f"{window}.json", {"source": {"sha256": "s"}})
    return {f"{window}/{stage}": {"status": "complete", "result": {}} for stage in ("sam3/fish", "static/sam3-fish",
                                                                                 "labels/fish")}


@pytest.fixture
def rounds(tmp_path):
    round1 = tmp_path / "round-1"
    entries = labelled(round1, OLD) | {"round/student/train": {"status": "complete"}}
    decisions = {name: {"copied": name} for name in round2.COPIED_DECISIONS} | {
        "auto_exclusions": {"rules": RULES}, "student_model": {"checkpoint": {}}}
    write_json(round1 / "journal.json", {"budgets_seconds": {}, "entries": entries, "decisions": decisions})
    selection = tmp_path / "selection"
    selection.mkdir()
    rows = [{"window_id": w, "round": r, "day": d, "role": "train", "source_mp4_url": f"https://x/{w}.mp4",
             "clip_time": "", "start_frame": "100", "frame_count": "450", "evidence": "strong", "light": "mid",
             "hour_bin": "12-18", "reserve_window_id": "", "reserve_url": "", "reserve_start_frame": ""}
            for w, (r, d) in WINDOWS.items()]
    for name, chosen in (("manifest.csv", rows), ("to_label.csv", rows[1:])):
        with (selection / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(chosen)
    write_json(selection / "day_roles.json", {"roles": {d: "train" for _, d in WINDOWS.values()}})
    write_json(selection / "summary.json", {})
    shards = []
    for i, window in enumerate((NEW1, NEW2)):
        shard = tmp_path / f"shard-{i}"
        seed = round2.seed_journal(json.loads((round1 / "journal.json").read_text()), selection)
        write_json(shard / "journal.json", seed | {"entries": labelled(shard, window)})
        (shard / "run_windows").mkdir()
        (shard / "run_windows" / f"worker-{2 * i}.jsonl").write_text("{}\n")
        shards.append(shard)
    return round1, selection, shards, tmp_path / "round-2" / "cloud"


def test_a_labeller_starts_from_round_ones_decisions_and_its_own_budget(rounds):
    round1, selection, _, _ = rounds
    seed = round2.seed_journal(json.loads((round1 / "journal.json").read_text()), selection)
    assert set(seed["decisions"]) == {*round2.COPIED_DECISIONS, "journal_seed"}
    assert seed["entries"] == {} and seed["budgets_seconds"]["sam3_gpu"] == 1 * round2.SAM3_SECONDS_PER_WINDOW
    alone = round2.seed_journal(json.loads((round1 / "journal.json").read_text()), selection, shards=1)
    assert alone["budgets_seconds"]["sam3_gpu"] == 2 * round2.SAM3_SECONDS_PER_WINDOW  # Both windows, one instance
    assert round2.labeller_budgets(244, 1)["sam3_gpu"] == 73_200 and round2.labeller_budgets(244, 4)["sam3_gpu"] == 18_300
    assert seed["decisions"]["journal_seed"]["to_label_sha256"] == round2.sha256_of(selection / "to_label.csv")


def test_the_merge_joins_round_one_and_every_labeller_and_lists_every_window(rounds):
    round1, selection, shards, out = rounds
    summary = round2.merge(selection, shards, out, round1)
    assert summary["windows"] == 3 and summary["labelled"] == 3 and summary["kept"] == 3
    assert summary["unlabelled_count"] == 0
    journal = json.loads((out / "journal.json").read_text())
    assert len(journal["entries"]) == 9 and "round/student/train" not in journal["entries"]
    assert "student_model" not in journal["decisions"] and journal["decisions"]["auto_exclusions"]["rules"] == RULES
    assert journal["budgets_seconds"]["training"] == round2.TRAINING_BUDGET_SECONDS
    with (out / "selection" / "to_label.csv").open(newline="", encoding="utf-8") as stream:
        assert [r["window_id"] for r in csv.DictReader(stream)] == list(WINDOWS)
    assert (out / "selection" / "labelled_in_round_2.csv").exists()
    assert not (out / "videos" / OLD / "static" / "sam3-fish" / "contact-sheet.jpg").exists()
    assert {p.name for p in (out / "run_windows").iterdir()} == {"shard-0-worker-0.jsonl", "shard-1-worker-2.jsonl"}


def test_the_merge_refuses_two_labellers_with_different_results_for_one_window(rounds):
    round1, selection, shards, out = rounds
    labels = shards[1] / "videos" / NEW1 / "labels" / "fish" / "rows.jsonl"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}\n")
    with pytest.raises(round2.MergeConflict, match="differs"):
        round2.merge(selection, shards, out, round1)


def test_the_merge_refuses_a_journal_step_two_labellers_both_ran(rounds):
    round1, selection, shards, out = rounds
    journal = json.loads((shards[1] / "journal.json").read_text())
    journal["entries"][f"{NEW1}/sam3/fish"] = {"status": "failed"}
    (shards[1] / "journal.json").write_text(json.dumps(journal))
    with pytest.raises(round2.MergeConflict, match="journal step"):
        round2.merge(selection, shards, out, round1)


def test_the_merge_reports_a_window_no_labeller_finished(rounds):
    round1, selection, shards, out = rounds
    summary = round2.merge(selection, shards[:1], out, round1)
    assert summary["unlabelled"] == [NEW2] and summary["labelled"] == 2


def test_the_training_bundle_carries_what_training_reads_at_the_instances_paths(rounds, monkeypatch):
    round1, selection, shards, out = rounds
    round2.merge(selection, shards, out, round1)
    monkeypatch.setattr(round2.package_base_image, "contents", lambda repo: {})
    files = round2.training_files(out, round1)
    names = set(files)
    root = round2.INSTANCE_ROUND2
    assert f"{root}/videos/{NEW2}/labels/fish/rows.jsonl" in names and f"{root}/windows/{OLD}.json" in names
    assert f"{root}/checks/quality_report.json" in names and f"{round2.INSTANCE_ROUND1}/journal.json" in names
    assert not any(n.endswith(".jpg") or "/run_windows/" in n for n in names)
    assert files[f"{round2.INSTANCE_ROUND1}/journal.json"] == round1 / "journal.json"


def test_the_labelling_bundle_holds_code_selection_and_seed_only(rounds, tmp_path):
    round1, selection, _, _ = rounds
    output = tmp_path / "up" / "labelling.tgz"
    summary = round2.labelling_bundle(selection, output, round1)
    with tarfile.open(output) as archive:
        names = set(archive.getnames())
        seed = json.loads(archive.extractfile(f"{round2.INSTANCE_ROUND2}/journal.json").read())
    assert f"{round2.INSTANCE_ROUND2}/selection/to_label.csv" in names and "scripts/cloud_labelling/round2.py" in names
    assert not any(n.endswith((".pt", ".pth", ".mp4", ".env")) for n in names)
    assert seed["entries"] == {} and summary["windows_to_label"] == 2


def test_shards_split_one_to_label_without_overlap_or_gaps():
    """run_labelling.sh's arithmetic: instance i of n runs workers i*2 and i*2+1 of 2n."""
    script = Path("scripts/cloud_labelling/run_labelling.sh").read_text(encoding="utf-8")
    assert 'global=$((SHARD_INDEX * WORKERS + k))' in script and '--worker "$global/$((SHARDS * WORKERS))"' in script
    from scripts.cloud_labelling.run_windows import rows_for
    rows = Path("rows.csv")
    try:
        rows.write_text("window_id\n" + "".join(f"w{i}\n" for i in range(23)))
        taken = [r["window_id"] for i in range(4) for k in range(2) for r in rows_for(rows, i * 2 + k, 8)]
    finally:
        rows.unlink()
    assert sorted(taken) == sorted(f"w{i}" for i in range(23))


def test_the_dataset_reads_a_window_restore_recut_at_the_sha256_restore_recorded(tmp_path, monkeypatch):
    """A re-cut can differ in bytes (the encoder) while its source matched: restore records it."""
    pytest.importorskip("cv2")
    from test_hard_negatives import make_round, make_video

    from scripts import student_detector as sd

    monkeypatch.chdir(tmp_path)
    root = Path("round-1")
    windows = make_round(root)
    test_window = next(w for w in windows if not w["excluded"] and w["role"] == "test")
    video = Path(test_window["video"])
    make_video(video, seed=99)  # Same frame count, different bytes
    with pytest.raises(RuntimeError, match="changed since it was labelled"):
        sd.round_data(root)
    (root / "restored.json").write_text(json.dumps({"windows": {test_window["window_id"]: {"sha256": sd.sha256(video)}}}))
    (root / "journal.json").write_text(json.dumps(json.loads((root / "journal.json").read_text()) | {"entries": {}}))
    result = sd.round_data(root)
    assert result["counts"]["test"]["windows"] == 1


def split_round(root, rows=30, finished=((0, 4), (1, 5))):
    """A running labeller's round: ``rows`` windows to label, two workers, and each worker's last
    finished row (an index into to_label.csv), labelled in the journal and in its event log."""
    ids = [f"SmartBayCam1_2025-10-{10 + i // 3:02d}_1200_f{i:04d}" for i in range(rows)]
    (root / "selection").mkdir(parents=True)
    with (root / "selection" / "to_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({"window_id": w, "day": "2025-10-10", "role": "train"} for w in ids)
    entries = {}
    for worker, last in finished:
        events = [{"event": "start"}]
        for i in range(worker, last + 1, 2):
            entries[f"{ids[i]}/labels/fish"] = {"kind": "sam3_gpu", "status": "complete", "result": {}}
            events.append({"event": "labelled", "window_id": ids[i]})
        (root / "run_windows").mkdir(exist_ok=True)
        (root / "run_windows" / f"worker-{worker}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    write_json(root / "journal.json", {"budgets_seconds": {}, "entries": entries, "decisions": {}})
    return ids


def test_a_split_hands_the_end_of_each_running_workers_list_to_the_new_instances(tmp_path):
    pytest.importorskip("cv2")
    from scripts.cloud_labelling.run_windows import labelled
    from scripts.sam3_botsort import Journal

    root = tmp_path / "round-2"
    ids = split_round(root)
    preview = round2.split_remaining(root, 3, head_start=1, dry_run=True)
    assert not (root / "split").exists() and preview["positions"] == [3, 3]  # Rows 6 and 7 are in progress
    summary = round2.split_remaining(root, 3, head_start=1)
    assert summary == preview | {"decided_utc": summary["decided_utc"]}
    # 24 rows left; each worker keeps ceil(24 / 8) + 1 = 4, from the row it is on
    assert summary["keep_per_worker"] == 4 and summary["kept"] == [ids[i] for i in (6, 8, 10, 12, 7, 9, 11, 13)]
    handed = [w for i in "123" for w in summary["instances"][i]]
    assert sorted(handed) == ids[14:] and summary["counts"] == {"1": 6, "2": 5, "3": 5}
    assert summary["instances"]["1"] == ids[14::3]  # Round robin in to_label order: a mix of days
    with (root / "split" / "instance-2.csv").open(newline="", encoding="utf-8") as stream:
        assert [r["window_id"] for r in csv.DictReader(stream)] == ids[15::3]
    # The running workers skip every handed window, and keep their own
    assert all(labelled(root, w) for w in handed) and not any(labelled(root, w) for w in summary["kept"])
    # A running worker's later save keeps the placeholders
    worker = Journal(root)
    worker.data["entries"][f"{ids[6]}/labels/fish"] = {"kind": "sam3_gpu", "status": "complete", "result": {}}
    worker.touched["entries"].add(f"{ids[6]}/labels/fish")
    worker.save()
    assert all(labelled(root, w) for w in handed) and worker.charged("sam3_gpu") == 0
    with pytest.raises(SystemExit, match="split already"):
        round2.split_remaining(root, 3)


def test_the_merge_drops_a_split_placeholder_for_the_instance_that_labelled_the_window(rounds):
    round1, selection, shards, out = rounds
    journal = json.loads((shards[0] / "journal.json").read_text())
    journal["entries"][f"{NEW2}/labels/fish"] = {"kind": "placeholder", "status": "complete", "result": {},
                                                 "placeholder": {"instance": 1}}
    (shards[0] / "journal.json").write_text(json.dumps(journal))
    summary = round2.merge(selection, shards, out, round1)
    merged = json.loads((out / "journal.json").read_text())["entries"]
    assert summary["labelled"] == 3 and "placeholder" not in merged[f"{NEW2}/labels/fish"]


def test_an_instance_given_a_list_labels_that_list():
    script = Path("scripts/cloud_labelling/run_labelling.sh").read_text(encoding="utf-8")
    assert '${TO_LABEL:+--to-label "$TO_LABEL"}' in script and 'TO_LABEL="${AITHNE_TO_LABEL:-}"' in script
    after = Path("scripts/cloud_labelling/run_round2.sh").read_text(encoding="utf-8")
    assert '${EXTRA[@]+"${EXTRA[@]}"}' in after and '"$shard/.complete"' in after


@pytest.mark.parametrize("script", ["run_round2.sh", "run_scan_days.sh", "run_labelling.sh", "relay_shards.sh"])
def test_the_instance_scripts_parse_and_have_unix_line_endings(script):
    import shutil

    path = Path("scripts/cloud_labelling") / script
    assert b"\r" not in path.read_bytes()
    if not shutil.which("bash"):
        pytest.skip("no bash here")
    bash = subprocess.run(["bash", "-n", path.as_posix()], capture_output=True, text=True, check=False)
    assert bash.returncode == 0, bash.stderr
