import hashlib
import json
import subprocess
import sys

from scripts import smartbay_archive_pull as pull

ROUND_EXCLUDES = ("journal.json", "*.mp4", "*.part", "logs/*", "run_windows/*")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def entry(text, mtime=100.0):
    return [len(text), mtime, sha(text)]


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_plan_copies_new_and_changed_files_but_not_identical_lock_temp_or_live_day():
    local = {"a.json": entry("same"), "b.json": entry("old", 50.0)}
    remote = {"a.json": entry("same"), "b.json": entry("new", 90.0), "Cam_2025-09-01/summary.json": entry("x"),
              ".cloud-scan.lock": entry(""), "Cam_2025/year_state.json.tmp": entry("t"),
              "Cam_2025-09-02/state.json": entry("partial")}
    fetch, conflicts = pull.plan(local, remote, ("Cam_2025-09-02/",))
    assert fetch == ["Cam_2025-09-01/summary.json", "b.json"]
    assert conflicts == []


def test_plan_reports_a_newer_differing_local_file_as_a_conflict():
    fetch, conflicts = pull.plan({"log": entry("local edit", 500.0)}, {"log": entry("instance", 100.0)})
    assert conflicts and "newer" in conflicts[0]


def test_apply_copies_verified_files_and_keeps_local_only_files(tmp_path):
    root, staging = tmp_path / "local", tmp_path / "staging"
    write(root, "kept/thumb.jpg", "stale intermediate")
    write(root, "log", "line 1\n")
    write(staging, "log", "line 1\nline 2\n")
    write(staging, "day/summary.json", "{}")
    local = pull.local_manifest(root)
    remote = {"log": entry("line 1\nline 2\n"), "day/summary.json": entry("{}")}
    assert pull.check_and_apply(staging, ["day/summary.json", "log"], remote, local, root) == []
    assert (root / "log").read_text(encoding="utf-8") == "line 1\nline 2\n"
    assert (root / "day/summary.json").is_file()
    assert (root / "kept/thumb.jpg").is_file()


def test_apply_changes_nothing_when_any_downloaded_file_fails_its_hash(tmp_path):
    root, staging = tmp_path / "local", tmp_path / "staging"
    write(staging, "good.json", "good")
    write(staging, "bad.json", "corrupted")
    remote = {"good.json": entry("good"), "bad.json": entry("expected")}
    problems = pull.check_and_apply(staging, ["bad.json", "good.json"], remote, {}, root)
    assert problems and "bad.json" in problems[0]
    assert not (root / "good.json").exists()


def test_apply_refuses_a_year_ledger_that_would_drop_completed_days(tmp_path):
    root, staging = tmp_path / "local", tmp_path / "staging"
    rel = "Cam_2025/year_state.json"
    older = json.dumps({"completed_days": {"2025-08-20": {}}})
    newer = json.dumps({"completed_days": {"2025-08-20": {}, "2025-08-21": {}}})
    write(root, rel, newer)
    write(staging, rel, older)
    problems = pull.check_and_apply(staging, [rel], {rel: entry(older)}, pull.local_manifest(root), root)
    assert problems and "2025-08-21" in problems[0]
    assert (root / rel).read_text(encoding="utf-8") == newer


def test_exclude_globs_match_a_relative_path_or_a_file_name():
    assert pull.excluded("journal.json", ROUND_EXCLUDES)
    assert pull.excluded("windows/SmartBayCam1_2025-06-22_0225_f0946.mp4", ROUND_EXCLUDES)
    assert pull.excluded("windows/SmartBayCam1_2025-06-22_0225.mp4.part", ROUND_EXCLUDES)
    assert pull.excluded("logs/w.sam3.log", ROUND_EXCLUDES) and pull.excluded("run_windows/worker-0.jsonl", ROUND_EXCLUDES)
    assert not pull.excluded("windows/SmartBayCam1_2025-06-22_0225.mp4.sha256", ROUND_EXCLUDES)
    assert not pull.excluded("videos/w/labels/fish/candidates.csv", ROUND_EXCLUDES)
    assert not pull.excluded("videos/w/sam3/fish/frames.jsonl", ROUND_EXCLUDES)
    assert not pull.excluded("journal.json", ())


def test_plan_leaves_excluded_files_out():
    remote = {"journal.json": entry("busy"), "windows/w.mp4": entry("video"), "windows/w.json": entry("{}"),
              "videos/w/labels/fish/report.json": entry("{}"), "logs/w.sam3.log": entry("live")}
    fetch, conflicts = pull.plan({}, remote, exclude=ROUND_EXCLUDES)
    assert fetch == ["videos/w/labels/fish/report.json", "windows/w.json"] and conflicts == []
    assert len(pull.plan({}, remote)[0]) == len(remote)  # no --exclude: unchanged behaviour


def test_the_instance_neither_hashes_nor_lists_excluded_files(tmp_path):
    for rel, text in {"journal.json": "{}", "windows/w.mp4": "video", "windows/w.json": "{}",
                      "logs/w.sam3.log": "live", "videos/w/sam3/fish/report.json": "{}"}.items():
        write(tmp_path, rel, text)
    listing = subprocess.run([sys.executable, "-I", "-", str(tmp_path), *ROUND_EXCLUDES], input=pull.REMOTE_MANIFEST,
                             capture_output=True, text=True, check=True)
    report = json.loads(listing.stdout)
    assert sorted(report["files"]) == ["videos/w/sam3/fish/report.json", "windows/w.json"]
    assert report["excluded"] == 3
    assert report["files"]["windows/w.json"][2] == sha("{}")
