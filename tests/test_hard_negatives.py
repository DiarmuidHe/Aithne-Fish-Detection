"""Hard-negative mining end to end on a tiny synthetic round: real (tiny) videos, fake RF-DETR,
no network, GPU or checkpoint."""
import csv
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import hard_negatives as hn
from scripts import student_detector as sd
from scripts.cloud_labelling import package_hard_negatives as package

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

FRAMES, WIDTH, HEIGHT, FPS = 40, 96, 64, 15.0
STUCK = [40.0, 10.0, 70.0, 40.0]  # The debris: same place, same look, in every window that has it
DAYS = {"train": "2025-07-01", "validation": "2025-07-02", "test": "2025-07-03", "excluded": "2025-07-04",
        "excluded_test": "2025-07-05"}


def make_video(path, stuck=True, seed=0):
    """A textured, noisy scene; ``stuck`` draws the same blotchy object in every frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 255, (30, 30), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(FRAMES):
        image = rng.integers(60, 90, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        if stuck:
            image[10:40, 40:70] = np.repeat(texture[:, :, None], 3, axis=2)
        writer.write(image)
    writer.release()


def rows(frames, box, track="sam3-1"):
    return [{"id": track, "frame": f, "t": f / FPS, "box": box, "confidence": .9} for f in frames]


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


def make_round(root):
    """Four windows of a labelled round. SAM 3 labels a fish on every window, and nothing on the
    stuck object, so the student's boxes on it are the hard negatives to find."""
    to_label, per_window = [], []
    for name, day in DAYS.items():
        key = f"SmartBayCam1_{day}_1200_f0100"
        role = "test" if name.endswith("test") else "train"
        excluded = name.startswith("excluded")
        video = root / "windows" / f"{key}.mp4"
        make_video(video, seed=len(to_label))
        to_label.append({"window_id": key, "round": "1", "day": day, "role": role, "evidence": "strong",
                         "source_mp4_url": f"https://example.test/{key}.mp4", "start_frame": "100",
                         "frame_count": str(FRAMES), "reserve_window_id": "", "reserve_url": "", "reserve_start_frame": ""})
        per_window.append({"window_id": key, "labelled_as": key, "excluded": excluded})
        directory = root / "videos" / key
        directory.mkdir(parents=True)
        (directory / "video.json").write_text(json.dumps({"key": key, "file": {
            "path": video.as_posix(), "sha256": sd.sha256(video), "fps": FPS, "decoded_frames": FRAMES,
            "width": WIDTH, "height": HEIGHT}}))
        fish = rows(range(FRAMES), [2.0, 44.0, 22.0, 60.0])
        write_jsonl(directory / "labels" / "fish" / "rows.jsonl", fish)
        write_jsonl(directory / "sam3" / "fish" / "frames.jsonl",
                    [{"frame": f, "boxes": [r["box"] for r in fish if r["frame"] == f], "scores": [.9]} for f in range(FRAMES)])
        write_jsonl(directory / "static" / "sam3-fish" / "removed.jsonl", [])
        (root / "windows" / f"{key}.json").write_text(json.dumps({
            "start_frame": 100, "source": {"url": f"https://example.test/{key}.mp4", "sha256": "source-" + key},
            "cut": {"sha256": sd.sha256(video), "command": ["ffmpeg", "-y", "-i", "source.mp4", "-vf", "trim", "cut.mp4"]}}))
    (root / "checks").mkdir()
    (root / "checks" / "quality_report.json").write_text(json.dumps({"per_window": per_window}))
    (root / "selection").mkdir()
    with (root / "selection" / "to_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(to_label[0]))
        writer.writeheader()
        writer.writerows(to_label)
    roles = {day: ("test" if name.endswith("test") else "train") for name, day in DAYS.items()}
    (root / "selection" / "day_roles.json").write_text(json.dumps({"roles": roles}))
    validation = {DAYS["validation"]}
    windows = sd.round_windows(root)
    coco = sd.round_coco(windows, {w["window_id"]: fish_for(root, w) for w in windows}, validation)
    (root / "student").mkdir()
    (root / "student" / "round_dataset.json").write_text(json.dumps({"counts": sd.coco_counts(coco), "windows": []}))
    (root / "student" / "round_eval.json").write_text(json.dumps({
        "threshold": .5, "agreement": {"all": {"agreement": .7}}, "empty_water_proxy": {}}))
    entries = {"round/student/train": {"status": "complete", "result": {"epochs_run": 2}}}
    for window in windows:  # Round 1 only ran its student on validation and test windows
        if not window["excluded"] and window["day"] in (DAYS["validation"], DAYS["test"]):
            entries[f"round/student/infer/{window['window_id']}"] = {"status": "complete", "inputs": {"model": "model-sha"}}
            write_jsonl(root / "student" / "rfdetr-nano" / window["window_id"] / "frames.jsonl", student_frames())
    (root / "journal.json").write_text(json.dumps({
        "schema_version": 1, "budgets_seconds": {}, "entries": entries, "decisions": {
            "validation_days": {"days": sorted(validation)}, "student_threshold": {"threshold": .5},
            "student_model": {"checkpoint": {"path": "model.pth", "sha256": "model-sha"}, "trained_on": "round round-1"}}}))
    return windows


def fish_for(root, window):
    return [json.loads(line) for line in
            (root / "videos" / window["window_id"] / "labels" / "fish" / "rows.jsonl").read_text().splitlines()]


def student_frames(boxes=(STUCK,), frames=FRAMES):
    return [{"frame": f, "boxes": [list(b) for b in boxes], "ids": [], "scores": [.8] * len(boxes)} for f in range(frames)]


@pytest.fixture
def round_(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # The round's records hold paths relative to the repository root
    make_round(Path("round-1"))
    monkeypatch.setattr(hn, "load_student", lambda checkpoint: "model")
    monkeypatch.setattr(hn, "detect_video", lambda model, video, heartbeat=None: (student_frames(), {"fps": 30}))
    return Path("round-1"), Path("out")


def mine_all(round_):
    root, out = round_
    return hn.mine(root, out)


# ------------------------------------------------------------------ the rule --

def test_a_zone_sam3_labelled_is_never_a_hard_negative_however_still_it_is():
    assert hn.verdict({"labels": .5, "raw": .5, "removed": 0}, .99) == "sam3_labelled"


def test_a_zone_sam3_only_fired_on_weakly_is_still_a_hard_negative_when_it_stays_put():
    # The real debris on the local clip: raw SAM 3 boxes on 39% of its frames, never a label.
    assert hn.verdict({"labels": 0, "raw": .39, "removed": .11}, .98) == "hard_negative"


def test_a_zone_whose_crops_change_is_not_a_hard_negative_and_unchecked_is_not_either():
    assert hn.verdict({"labels": 0, "raw": 0, "removed": 0}, .7) == "moving_content"
    assert hn.verdict({"labels": 0, "raw": 0, "removed": 0}, None) == "not_checked"


def test_sam3_counts_as_found_for_a_larger_box_around_the_same_object():
    boxes = np.array([[0.0, 0.0, 200.0, 200.0], [500.0, 500.0, 520.0, 520.0]])
    assert hn._overlaps([40.0, 40.0, 60.0, 60.0], boxes).tolist() == [True, False]


def test_stillness_separates_one_object_from_different_things_passing(tmp_path):
    still, changing = tmp_path / "still.mp4", tmp_path / "changing.mp4"
    make_video(still)
    rng = np.random.default_rng(1)
    writer = cv2.VideoWriter(str(changing), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(FRAMES):
        writer.write(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    frames = list(range(FRAMES))
    assert hn.zone_stillness(still, STUCK, frames) >= hn.RULE["min_stillness"]
    assert hn.zone_stillness(changing, STUCK, frames) < hn.RULE["min_stillness"]
    assert hn.zone_stillness(still, [40.0, 10.0, 42.0, 12.0], frames) is None  # Too small to judge


def test_mine_window_finds_the_stuck_object_and_splits_the_boxes_sam3_has_nothing_near(tmp_path):
    video = tmp_path / "w.mp4"
    make_video(video)
    window = {"window_id": "w", "frames": FRAMES, "fps": FPS}
    labels = rows(range(FRAMES), [2.0, 44.0, 22.0, 60.0])
    raw = [{"frame": r["frame"], "box": r["box"]} for r in labels]
    frames = student_frames([STUCK, [2.0, 44.0, 22.0, 60.0]])
    frames[3]["boxes"].append([80.0, 50.0, 95.0, 60.0])  # A lone box: no SAM 3 box near, on no zone
    frames[3]["scores"].append(.9)
    result = hn.mine_window(window, frames, labels, raw, [], .5, video)
    verdicts = {tuple(z["box"]): z["verdict"] for z in result["zones"]}
    assert verdicts[tuple(STUCK)] == "hard_negative" and verdicts[(2.0, 44.0, 22.0, 60.0)] == "sam3_labelled"
    assert result["unsupported_boxes"] == FRAMES + 1
    assert result["unsupported_by_zone"] == {"hard_negative": FRAMES, "no_zone": 1}
    assert result["unsupported_sample"] == [{"frame": 3, "box": [80.0, 50.0, 95.0, 60.0], "confidence": .9}]
    assert next(z for z in result["zones"] if z["verdict"] == "hard_negative")["clean_frames"] == list(range(FRAMES))


def test_a_box_that_never_returns_to_one_place_makes_no_zone():
    window = {"window_id": "w", "frames": FRAMES, "fps": FPS}
    frames = [{"frame": f, "boxes": [[float(2 * f), 5.0, float(2 * f + 10), 15.0]], "ids": [], "scores": [.9]}
              for f in range(FRAMES)]
    assert hn.mine_window(window, frames, [], [], [], .5)["zones"] == []


def test_a_box_present_only_briefly_makes_no_zone_as_a_resting_fish_would():
    window = {"window_id": "w", "frames": 400, "fps": FPS}
    frames = student_frames(frames=400)
    for frame in frames[60:]:
        frame["boxes"], frame["scores"] = [], []  # 15% of the frames, but under half of its time
    assert hn.mine_window(window, frames, [], [], [], .5)["zones"] == []


# ----------------------------------------------------------- windows and use --

def test_only_a_training_day_window_can_feed_a_dataset():
    validation = {DAYS["validation"]}
    w = lambda day, role="train", excluded=False: {"day": day, "role": role, "excluded": excluded}
    assert hn.use_of(w(DAYS["train"]), validation) == "train"
    assert hn.use_of(w(DAYS["validation"]), validation) == "report"
    assert hn.use_of(w(DAYS["test"], "test"), validation) == "report"
    assert hn.use_of(w(DAYS["excluded"], excluded=True), validation) == "report_excluded"
    assert hn.use_of(w(DAYS["validation"], excluded=True), validation) is None
    assert hn.use_of(w(DAYS["test"], "test", excluded=True), validation) is None
    assert hn.use_of(w("2025-09-01", "expert"), validation) is None


def _window(tmp_path):
    return {"window_id": "w", "video": (tmp_path / "w.mp4").as_posix(), "video_sha256": "old", "frames": FRAMES}


def _record():
    return {"source": {"url": "https://example.test/c.mp4", "sha256": "good"},
            "cut": {"command": ["ffmpeg", "-y", "-i", "c.mp4", "-vf", "trim", "w.mp4"]}}


def _fetch(sha):
    def fetch(url, path):
        path.write_bytes(b"source")
        return {"url": url, "sha256": sha}
    return fetch


def _cut(command, cwd, **kwargs):
    assert command[3] == "w.source.mp4" and command[-1] == "w.part.mp4"  # The record's command, on private names
    (cwd / command[-1]).write_bytes(b"cut")
    return subprocess.CompletedProcess(command, 0)


def test_restore_runs_the_records_cut_and_accepts_different_bytes_with_the_same_frames(tmp_path):
    result = hn.restore_window(_window(tmp_path), {}, _record(), fetch=_fetch("good"), run=_cut, frames=lambda p: FRAMES)
    assert result["byte_identical"] is False and result["source_verified"] is True
    assert result["sha256"] == sd.sha256(tmp_path / "w.mp4")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.mp4"]  # No source or partial cut is left


def test_restore_refuses_a_source_that_changed_since_it_was_labelled(tmp_path):
    with pytest.raises(hn.RestoreFailed, match="changed"):
        hn.restore_window(_window(tmp_path), {}, _record(), fetch=_fetch("other"), run=_cut, frames=lambda p: FRAMES)
    assert not (tmp_path / "w.mp4").exists() and not list(tmp_path.iterdir())


def test_restore_refuses_a_cut_with_the_wrong_frame_count(tmp_path):
    with pytest.raises(hn.RestoreFailed, match="frames"):
        hn.restore_window(_window(tmp_path), {}, _record(), fetch=_fetch("good"), run=_cut, frames=lambda p: FRAMES - 1)
    assert not list(tmp_path.iterdir())


def test_restore_reports_a_failed_cut_as_a_failure_of_that_window(tmp_path):
    def broken(command, cwd, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="boom")

    with pytest.raises(hn.RestoreFailed, match="w:"):
        hn.restore_window(_window(tmp_path), {}, _record(), fetch=_fetch("good"), run=broken, frames=lambda p: FRAMES)


def test_restore_keeps_windows_in_place_and_cuts_the_missing_one(round_, monkeypatch):
    root, out = round_
    missing = next(p for p in (root / "windows").glob("*.mp4") if DAYS["train"] in p.name)
    original = missing.read_bytes()
    missing.unlink()

    def recut(window, row, record, **kwargs):
        Path(window["video"]).write_bytes(original)
        return {"status": "cut", "sha256": window["video_sha256"], "byte_identical": True, "source_verified": True}

    monkeypatch.setattr(hn, "restore_window", recut)
    result = hn.restore(root, out)
    assert result["windows"] == 4 and result["byte_identical"] == 4  # Not the excluded test-day window
    state = json.loads((out / "restored.json").read_text())["windows"]
    assert state[missing.stem]["status"] == "cut" and sum(r["status"] == "present" for r in state.values()) == 3


def test_a_failed_window_stops_the_stage_after_the_rest_are_done(round_, monkeypatch):
    root, out = round_
    next(p for p in (root / "windows").glob("*.mp4") if DAYS["train"] in p.name).unlink()

    def fail(window, row, record, **kwargs):
        raise hn.RestoreFailed("gone")

    monkeypatch.setattr(hn, "restore_window", fail)
    with pytest.raises(hn.RestoreFailed, match="1 windows failed"):
        hn.restore(root, out)
    assert len(json.loads((out / "restored.json").read_text())["windows"]) == 3


# ---------------------------------------------------------------------- mine --

def test_mine_runs_the_student_only_where_the_round_has_not_and_reports_every_window(round_, monkeypatch):
    root, out = round_
    seen = []

    def detect(model, video, heartbeat=None):
        seen.append(video.name)
        return student_frames(), {"fps": 30}

    monkeypatch.setattr(hn, "detect_video", detect)
    result = hn.mine(root, out)
    assert len(seen) == 2 and all(DAYS["train"] in n or DAYS["excluded"] in n for n in seen)
    assert result["by_use"]["train"]["zones"] == {"hard_negative": 1}
    assert result["by_use"]["report"]["zones"] == {"hard_negative": 2}
    assert result["by_use"]["report_excluded"]["zones"] == {"hard_negative": 1}
    assert not any(DAYS["excluded_test"] in p.name for p in (out / "mined" / "zones").iterdir())
    assert (out / "review" / "hard_negative_1.jpg").exists()
    table = list(csv.DictReader((out / "hard_negatives.csv").open(encoding="utf-8")))
    assert {r["use"] for r in table} == {"train", "report", "report_excluded"}
    assert all(r["verdict"] == "hard_negative" for r in table)


def test_mine_reuses_finished_work_on_a_rerun(round_, monkeypatch):
    root, out = round_
    hn.mine(root, out)
    monkeypatch.setattr(hn, "detect_video", lambda *a, **k: pytest.fail("inference was repeated"))
    hn.mine(root, out)


def test_existing_only_needs_no_student_and_tolerates_missing_videos(round_, monkeypatch):
    root, out = round_
    for video in (root / "windows").glob("*.mp4"):
        video.unlink()
    monkeypatch.setattr(hn, "load_student", lambda checkpoint: pytest.fail("loaded the student"))
    result = hn.mine(root, out, existing_only=True)
    assert result["by_use"] == {"report": result["by_use"]["report"]}
    assert result["by_use"]["report"]["zones"] == {"not_checked": 2}
    assert result["review_sheets"] == []


# ---------------------------------------------------------------------- data --

def _images(out, split):
    return json.loads((out / "dataset" / split / "_annotations.coco.json").read_text())


def test_data_waits_for_a_persons_approval(round_):
    root, out = round_
    mine_all(round_)
    with pytest.raises(SystemExit, match="review"):
        hn.build(root, out)
    assert not (out / "dataset").exists()


def test_data_adds_hard_negatives_to_training_and_leaves_the_other_splits_as_they_were(round_):
    root, out = round_
    mine_all(round_)
    result = hn.build(root, out, approval="looks right")
    base = json.loads((root / "student" / "round_dataset.json").read_text())["counts"]
    assert result["base_counts"] == base and result["counts"]["validation"] == base["validation"]
    assert result["counts"]["test"] == base["test"]
    train = _images(out, "train")
    marked = [i for i in train["images"] if i.get("hard_negative_boxes") == [STUCK]]
    grid = [i for i in marked if i["frame"] % sd.ROUND_STRIDE == 0]
    added = [i for i in marked if i["frame"] % sd.ROUND_STRIDE]
    assert len(grid) == 8 and len(added) == 32  # All 40 frames of the window, as 32 are under the per-zone limit
    assert len(train["images"]) == base["train"]["images"] + 32
    assert all((out / "dataset" / "train" / i["file_name"]).exists() for i in train["images"])
    assert not any("hard_negative_boxes" in i for s in ("valid", "test") for i in _images(out, s)["images"])
    # No box on the object itself, and the fish on those frames keep their labels
    on_added = [a for a in train["annotations"] if a["image_id"] in {i["id"] for i in added}]
    assert len(on_added) == 32 and all(a["bbox"] == [2.0, 44.0, 20.0, 16.0] for a in on_added)
    source = next(i for i in added if i["frame"] == 1)
    assert source["source_frame"] == 101
    decided = json.loads((out / "journal.json").read_text())["decisions"]["hard_negatives_approved"]
    assert decided["words"] == "looks right"


def test_a_rejected_zone_stays_out_and_a_typo_stops_the_build(round_):
    root, out = round_
    mine_all(round_)
    zone = f"SmartBayCam1_{DAYS['train']}_1200_f0100#0"
    (out / "review" / "rejected_zones.txt").write_text("nonsense#9\n")
    with pytest.raises(SystemExit, match="unknown"):
        hn.build(root, out, approval="yes")
    (out / "review" / "rejected_zones.txt").write_text(zone + "\n")
    with pytest.raises(SystemExit, match="No hard negatives"):  # It was the only one
        hn.build(root, out, approval="yes")
    assert not (out / "dataset").exists()


def test_excluded_windows_feed_training_only_when_asked_and_never_from_held_out_days(round_):
    root, out = round_
    mine_all(round_)
    zones, _ = hn.chosen_zones(out, use_excluded=False)
    assert [z["window_id"] for z in zones] == [f"SmartBayCam1_{DAYS['train']}_1200_f0100"]
    result = hn.build(root, out, approval="yes", use_excluded=True)
    assert result["hard_negatives"]["zones"] == 2
    days = {i["day"] for i in _images(out, "train")["images"]}
    assert days == {DAYS["train"], DAYS["excluded"]}
    excluded = [i for i in _images(out, "train")["images"] if i["day"] == DAYS["excluded"]]
    assert len(excluded) == 40  # Its grid frames too: that window has no images of its own
    assert result["counts"]["validation"]["images"] == FRAMES // sd.ROUND_STRIDE


def test_the_limit_spreads_a_zones_new_frames_over_its_time():
    window = {"window_id": "w", "frames": 450, "width": WIDTH, "height": HEIGHT, "start_frame": 0, "day": DAYS["train"],
              "evidence": "strong", "round": "1", "excluded": False, "role": "train"}
    coco = sd.round_coco([window], {}, set())
    zone = {"zone_id": "w#0", "window_id": "w", "box": STUCK, "clean_frames": list(range(450))}
    added = hn.add_hard_negatives(coco, {"w": window}, {}, [zone], limit=45)
    assert added["images_added"] == 45 and added["images_marked"] == 90
    frames = [i["frame"] for i in coco["train"]["images"][90:]]
    assert max(frames) - min(frames) > 400 and len(set(frames)) == 45


def test_the_build_refuses_a_base_dataset_that_differs_from_the_rounds(round_):
    root, out = round_
    mine_all(round_)
    report = root / "student" / "round_dataset.json"
    data = json.loads(report.read_text())
    data["counts"]["train"]["images"] += 1
    report.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="differs"):
        hn.build(root, out, approval="yes")


# ----------------------------------------------------------------- train, eval --

def test_training_repeats_round_ones_epochs_and_evaluation_compares_the_two_students(round_, monkeypatch):
    root, out = round_
    mine_all(round_)
    hn.build(root, out, approval="yes")
    asked = {}

    def fake_train(dataset, output, epochs, seconds, heartbeat):
        asked.update(epochs=epochs, dataset=dataset)
        return {"seconds": 1.0, "epochs_run": epochs, "best": {"path": "new.pth", "sha256": "new-sha"}, "files": []}

    monkeypatch.setattr(hn, "_train", fake_train)
    monkeypatch.setattr(hn, "verify_rfdetr", lambda: {"sha256": "coco"})
    hn.train(root, out)
    assert asked["epochs"] == 2 and asked["dataset"].name == "dataset"
    decisions = json.loads((out / "journal.json").read_text())["decisions"]
    assert decisions["student_model"]["trained_on"] == "round out"

    monkeypatch.setattr(sd, "load_student", lambda checkpoint: "model")

    def detect(model, video, heartbeat=None):  # The new student no longer fires on the object
        return student_frames(boxes=[[2.0, 44.0, 22.0, 60.0]]), {
            "total_ms": {"p50": 30, "p95": 35}, "fps": 30, "gpu": "fake", "keeps_up_with_live": True}

    monkeypatch.setattr(sd, "detect_video", detect)
    result = hn.evaluate(root, out)
    assert (out / "comparison.json").exists()
    static = result["static_zones_mined_from_each_student"]["test"]
    assert static["before"]["hard_negative_zones"] == 1 and static["after"]["hard_negative_zones"] == 0
    assert static["before"]["no_sam3_box_near_on_hard_negative_zones"] == FRAMES
    assert static["after"]["no_sam3_box_near"] == 0
    assert result["agreement"]["after"]["all"]["agreement"] == 1.0


# ----------------------------------------------------------------- the bundle --

def make_repo(repo):
    """A repository and a round's results, with the things that must never travel beside them."""
    for name in (*package.CODE, *package.MODELS, "scripts/cloud_labelling/run_x.sh", "scripts/cloud_labelling/run_x.py",
                 "scripts/smartbay_archive_scan/core.py", ".env", "scripts/cloud_labelling/operator-policy.json"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(b"content of " + name.encode() + b"\n")
    results = repo / "results"
    student = results / "models" / "m" / "train" / "checkpoint_best_total.pth"
    student.parent.mkdir(parents=True)
    student.write_bytes(b"student")
    (student.parent / "last.ckpt").write_bytes(b"too large to carry")
    path = f"{package.INSTANCE_ROUND}/models/m/train/checkpoint_best_total.pth"
    for name in ("checks/quality_report.json", "selection/day_roles.json", "selection/to_label.csv",
                 "student/round_dataset.json", "student/round_eval.json", "student/rfdetr-nano/w/frames.jsonl",
                 "windows/w.json", "videos/w/video.json", "videos/w/labels/fish/rows.jsonl",
                 "videos/w/sam3/fish/frames.jsonl", "videos/w/static/sam3-fish/removed.jsonl",
                 "windows/w.mp4", "videos/w/static/sam3-fish/contact-sheet.jpg"):
        (results / name).parent.mkdir(parents=True, exist_ok=True)
        (results / name).write_bytes(b"{}")
    (results / "journal.json").write_text(json.dumps({"decisions": {"student_model": {"checkpoint": {
        "path": path, "sha256": package.sha256_of(student)}}}}))
    return results


def test_the_bundle_holds_the_allowlist_only_at_the_instances_paths_with_checksums(tmp_path):
    results = make_repo(tmp_path)
    output = tmp_path / "up" / "b.tgz"
    summary = package.build(tmp_path, results, output)
    with tarfile.open(output) as archive:
        names = set(archive.getnames())
        members = {m.name: m for m in archive.getmembers()}
        sums = archive.extractfile("SHA256SUMS").read().decode()
        script = archive.extractfile("scripts/hard_negatives.py").read()
    assert f"{package.INSTANCE_ROUND}/videos/w/labels/fish/rows.jsonl" in names
    assert f"{package.INSTANCE_ROUND}/models/m/train/checkpoint_best_total.pth" in names
    assert "data/models/rfdetr/rf-detr-nano.pth" in names and "data/models/sam3/LICENSE" in names
    assert not any(n.endswith((".env", ".mp4", ".jpg", ".ckpt", ".pem", ".json.tmp", "operator-policy.json", "sam3.pt"))
                   for n in names)
    assert not any(n.startswith("results/") for n in names)
    assert members["scripts/cloud_labelling/run_x.sh"].mode == 0o755 and members["scripts/hard_negatives.py"].mode == 0o644
    assert summary["files"] == len(names) - 1 == len(sums.splitlines())
    assert script == b"content of scripts/hard_negatives.py\n"
    assert not output.with_name("b.tgz.tmp").exists()
    assert output.with_name("b.tgz.sha256").read_text().split()[0] == package.sha256_of(output)


def test_the_bundle_refuses_a_student_the_journal_did_not_record(tmp_path):
    results = make_repo(tmp_path)
    (results / "models" / "m" / "train" / "checkpoint_best_total.pth").write_bytes(b"another student")
    with pytest.raises(SystemExit, match="recorded"):
        package.contents(tmp_path, results)


def test_the_bundle_refuses_a_shell_script_that_would_not_run_on_the_instance(tmp_path):
    results = make_repo(tmp_path)
    (tmp_path / "scripts" / "cloud_labelling" / "run_x.sh").write_bytes(b"#!/bin/bash\r\necho\r\n")
    with pytest.raises(SystemExit, match="line endings"):
        package.contents(tmp_path, results)


def test_the_bundle_refuses_a_round_with_files_missing(tmp_path):
    results = make_repo(tmp_path)
    (results / "videos" / "w" / "labels" / "fish" / "rows.jsonl").unlink()
    with pytest.raises(SystemExit, match="rows.jsonl"):
        package.contents(tmp_path, results)


def test_every_module_the_stages_import_is_in_the_bundle():
    """Nothing the instance imports on start-up may be left out: a new import that isn't listed
    would only fail after the instance is running and billing. Only module-level imports count:
    sam3_botsort imports murky_water_tracker inside a report stage that these stages never call."""
    import ast

    known = {package.Path(name) for name in package.CODE} | {
        p for folder in package.CODE_FOLDERS for p in package.Path(folder).rglob("*.py")}
    seen, queue = set(), [package.Path("scripts/hard_negatives.py")]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            for name in names:
                if name.split(".")[0] in ("app", "scripts"):
                    target = package.Path(name.replace(".", "/") + ".py")
                    package_init = package.Path(name.replace(".", "/")) / "__init__.py"
                    found = target if target.exists() else package_init if package_init.exists() else None
                    if found:
                        queue.append(found)
    assert seen <= known, sorted(map(str, seen - known))
