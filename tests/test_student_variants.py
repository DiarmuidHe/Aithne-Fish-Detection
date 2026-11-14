"""The student comparison end to end on the tiny synthetic round of ``test_hard_negatives``: real
(tiny) videos, fake RF-DETR, no network, GPU or checkpoint."""
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_hard_negatives import DAYS, FPS, FRAMES, STUCK, make_video, student_frames, write_jsonl

from scripts import student_detector as sd
from scripts import student_variants as sv
from scripts.cloud_labelling import package_hard_negatives as package
from scripts.cloud_labelling import package_student_variants as variants_package
from scripts.sam3_autolabel import UnverifiedCheckpoint
from scripts.sam3_botsort import BudgetExceeded

FISH = [2.0, 44.0, 22.0, 60.0]
TEST_WINDOW = f"SmartBayCam1_{DAYS['test']}_1200_f0100"


@pytest.fixture
def round_(tmp_path, monkeypatch):
    from test_hard_negatives import make_round

    monkeypatch.chdir(tmp_path)  # The round's records hold paths relative to the repository root
    root = Path("round-1")
    make_round(root)
    (root / "student" / "round_eval.json").write_text(json.dumps(round_eval_record(.7)))
    return root, Path("variants")


def round_eval_record(agreement):
    pooled = {"agreement": agreement, "share_of_sam3_matched": agreement}
    return {"threshold": .5, "agreement": {"all": pooled, **{s: pooled for s in sd.STRATA}},
            "empty_water_proxy": {"none": {"student_boxes_per_frame": .3}, "murky": {"student_boxes_per_frame": .9}},
            "speed_1080p": {"fps_median": 33.0, "gpu": "A10G"}}


def fake_rfdetr(monkeypatch, asked=None):
    """Training that records what it was asked; a "model" that is its size's name; a Nano that
    keeps boxing the stuck object and bigger sizes that box only the fish."""
    def fake_train(dataset, output, epochs, seconds, heartbeat, **kwargs):
        if asked is not None:
            asked.append({"dataset": dataset, "output": output, "epochs": epochs} | kwargs)
        return {"seconds": 1.0, "epochs_run": epochs, "stopped_by_max_time": False,
                "best": {"path": f"{output}/best.pth", "sha256": f"sha-{kwargs['variant']}"}, "files": []}

    def detect(model, video, heartbeat=None):
        boxes = [STUCK, FISH] if model == "nano" else [FISH]
        return student_frames(boxes=boxes), {"total_ms": {"p50": 30, "p95": 35}, "fps": 30.0, "gpu": "fake",
                                             "keeps_up_with_live": True}

    monkeypatch.setattr(sv, "_train", fake_train)
    monkeypatch.setattr(sv, "verify_rfdetr", lambda path=None, variant="nano": {"sha256": f"coco-{variant}"})
    monkeypatch.setattr(sd, "load_student", lambda checkpoint: checkpoint.get("variant", "nano"))
    monkeypatch.setattr(sd, "detect_video", detect)


# ------------------------------------------------------------------- data --

def test_the_dataset_is_round_ones_rebuilt_unchanged(round_):
    root, out = round_
    result = sv.build(root, out)
    base = json.loads((root / "student" / "round_dataset.json").read_text())
    assert result["counts"] == base["counts"]
    train = json.loads((out / "dataset" / "train" / "_annotations.coco.json").read_text())
    assert all((out / "dataset" / "train" / image["file_name"]).exists() for image in train["images"])
    windows = json.loads((out / "student" / "round_dataset.json").read_text())["windows"]
    assert {w["window_id"]: w["split"] for w in windows}[TEST_WINDOW] == "test"


def test_the_dataset_is_refused_when_it_differs_from_round_ones(round_):
    root, out = round_
    report = root / "student" / "round_dataset.json"
    data = json.loads(report.read_text())
    data["counts"]["train"]["boxes"] += 1
    report.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="differs"):
        sv.build(root, out)


# ------------------------------------------------------------ train, eval --

def test_each_arm_trains_in_its_own_folder_for_round_ones_epochs(round_, monkeypatch):
    root, out = round_
    sv.build(root, out)
    asked = []
    fake_rfdetr(monkeypatch, asked)
    sv.train("medium", root, out)
    sv.train("nano-gentle", root, out)
    medium, gentle = asked
    assert medium["epochs"] == gentle["epochs"] == 2 and medium["dataset"] == gentle["dataset"]
    assert medium["variant"] == "medium" and medium["train_settings"] == {}
    assert gentle["variant"] == "nano" and gentle["train_settings"] == {"lr_encoder": sv.GENTLE}
    decision = json.loads((out / "medium" / "journal.json").read_text())["decisions"]["student_model"]
    assert decision["checkpoint"]["variant"] == "medium" and decision["trained_on"].startswith("round")
    assert decision["backbone"] == "DINOv2 ViT-S" and decision["coco_checkpoint"] == {"sha256": "coco-medium"}
    assert (out / "medium" / "student" / "round_dataset.json").exists()
    assert not (out / "nano-gentle" / "models" / "rfdetr-medium-medium").exists()


def test_an_arm_that_cannot_finish_in_its_budget_stops_after_its_first_epoch(round_, monkeypatch):
    root, out = round_
    sv.build(root, out)
    fake_rfdetr(monkeypatch)
    epochs = []

    def first_epoch_only(dataset, output, epochs_asked, seconds, heartbeat, **kwargs):
        heartbeat(epoch=1, of=epochs_asked)
        epochs.append(2)
        return {}

    monkeypatch.setattr(sv, "_train", first_epoch_only)
    clock = itertools.count(step=100)  # Every reading 100 s on: the first epoch "took" 100 s
    monkeypatch.setattr(sv, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setitem(sv.ARMS["medium"], "budget_hours", 300 / 3600)  # 180 s once the margin is held back
    with pytest.raises(BudgetExceeded, match="first epoch"):
        sv.train("medium", root, out)
    assert not epochs


def test_compare_puts_every_student_beside_round_one_on_the_same_test_days(round_, monkeypatch):
    root, out = round_
    sv.build(root, out)
    fake_rfdetr(monkeypatch)
    sv.train("medium", root, out)
    evaluation = sv.evaluate("medium", root, out)
    assert evaluation["agreement"]["all"]["agreement"] == 1.0
    assert (out / "medium" / "student" / "rfdetr-medium" / TEST_WINDOW / "frames.jsonl").exists()
    result = sv.compare(root, out)
    students = result["students"]
    assert set(students) == {"round-1", "medium"}
    assert students["round-1"]["no_sam3_box_near"] == FRAMES  # Round 1's student boxes the stuck object
    assert students["medium"]["no_sam3_box_near"] == 0
    assert students["round-1"]["hard_negative_zones"] == 1 and students["medium"]["hard_negative_zones"] == 0
    assert students["medium"]["decoder_layers"] == 4 and students["round-1"]["variant"] == "nano"
    assert (out / "comparison.json").exists()


# ------------------------------------------------------------------ local --

def make_clip(folder, key):
    """A SAM 3 output folder for a clip from a camera no student trained on."""
    video = folder / f"{key}.mp4"
    make_video(video, seed=7)
    directory = folder / "videos" / key
    directory.mkdir(parents=True)
    (directory / "video.json").write_text(json.dumps({"key": key, "file": {
        "path": video.as_posix(), "sha256": sd.sha256(video), "fps": FPS, "decoded_frames": FRAMES,
        "width": 96, "height": 64}}))
    fish = [{"id": "sam3-1", "frame": f, "t": f / FPS, "box": FISH, "confidence": .9} for f in range(FRAMES)]
    write_jsonl(directory / "labels" / "fish" / "rows.jsonl", fish)
    write_jsonl(directory / "sam3" / "fish" / "frames.jsonl",
                [{"frame": f, "boxes": [FISH], "scores": [.9]} for f in range(FRAMES)])
    return directory


def test_local_scores_every_student_on_an_unseen_camera_with_its_speed(round_, monkeypatch):
    root, out = round_
    sv.build(root, out)
    fake_rfdetr(monkeypatch)
    sv.train("medium", root, out)
    assert "medium" not in sv.students(root, out, local=True)  # No threshold until evaluated on validation days
    sv.evaluate("medium", root, out)
    gate = make_clip(Path("clips"), "gate-upload")
    result = sv.local([gate], root, out)
    clip = result["videos"]["gate-upload"]
    assert clip["camera_in_training"] is False
    assert clip["students"]["round-1"]["no_sam3_box_near"] == FRAMES
    assert clip["students"]["medium"]["no_sam3_box_near"] == 0
    assert clip["students"]["medium"]["keeps_up_with_live"] and clip["students"]["medium"]["fps_this_pc"] == 30.0
    assert (out / "local_comparison.json").exists()

    other = make_clip(Path("clips"), "SmartBayCam1_2025-08-01_0000")
    summary = sd.local_eval(out / "medium", [other])
    assert set(summary["videos"]) == {"gate-upload", "SmartBayCam1_2025-08-01_0000"}  # earlier clips stay


# ---------------------------------------------------------------- weights --

def test_weights_are_fetched_once_and_kept_only_when_rfdetrs_md5_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "RFDETR_CHECKPOINT", tmp_path / "models" / "rf-detr-nano.pth")
    monkeypatch.setitem(sd.VARIANTS["medium"], "md5", hashlib.md5(b"medium").hexdigest())
    fetched = []

    def fetch(url, path, content=b"medium"):
        fetched.append(url)
        path.write_bytes(content)
        return {"seconds": 1.0}

    result = sv.weights(["medium"], tmp_path / "out", fetch=fetch)
    assert result["medium"]["fetched"] and fetched == [sd.VARIANTS["medium"]["url"]]
    assert sv.weights(["medium"], tmp_path / "out", fetch=fetch)["medium"]["fetched"] is False
    assert len(fetched) == 1

    (tmp_path / "models" / "rf-detr-medium.pth").unlink()
    with pytest.raises(UnverifiedCheckpoint):
        sv.weights(["medium"], tmp_path / "out", fetch=lambda url, path: fetch(url, path, b"tampered"))
    assert not (tmp_path / "models" / "rf-detr-medium.pth").exists()


def test_the_guard_covers_setup_and_every_arms_budget_and_evaluation():
    assert sv.guard_minutes(list(sv.DEFAULT_ARMS)) == 75 + 15 + (180 + 20) + (210 + 25)
    assert sv.guard_minutes(["vitb-gentle"]) > 9 * 60


def test_every_arms_size_exists_and_fits_its_backbone():
    for arm, spec in sv.ARMS.items():
        assert spec["variant"] in sd.VARIANTS, arm
    assert sd.VARIANTS["large-vitb"]["backbone"] == "DINOv2 ViT-B"
    assert {sd.VARIANTS[v]["backbone"] for v in ("nano", "small", "medium")} == {"DINOv2 ViT-S"}


# ---------------------------------------------------------------- the bundle --

def test_the_bundle_carries_the_variants_script_and_everything_it_imports():
    """The same check as the hard-negatives bundle: an import missing from the bundle would only
    fail once the instance is running and billing."""
    import ast

    known = {Path(name) for name in (*package.CODE, *variants_package.EXTRA)} | {
        p for folder in package.CODE_FOLDERS for p in Path(folder).rglob("*.py")}
    seen, queue = set(), [Path("scripts/student_variants.py")]
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
                    target = Path(name.replace(".", "/") + ".py")
                    package_init = Path(name.replace(".", "/")) / "__init__.py"
                    found = target if target.exists() else package_init if package_init.exists() else None
                    if found:
                        queue.append(found)
    assert seen <= known, sorted(map(str, seen - known))
