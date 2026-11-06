"""Parts 5-7 end to end on a tiny synthetic round: real (tiny) videos, fake RF-DETR."""
import csv
import json

import pytest

from scripts import student_detector as sd
from scripts.cloud_labelling import round_checks as rc

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

DAYS = {"train": "2025-07-01", "validation": "2025-07-02", "test": "2025-07-03", "excluded": "2025-07-04"}
FRAMES, WIDTH, HEIGHT = 20, 64, 36


def make_round(root):
    rows, entries = [], {}
    for name, day in DAYS.items():
        key = f"SmartBayCam1_{day}_1200_f0100"
        role, evidence = ("test", "none") if name == "test" else ("train", "strong")
        rows.append({"window_id": key, "round": "1", "day": day, "role": role, "evidence": evidence, "light": "mid",
                     "hour_bin": "12-18", "source_mp4_url": "https://example.test/x.mp4", "start_frame": "100",
                     "frame_count": str(FRAMES), "reserve_window_id": "", "reserve_url": "", "reserve_start_frame": ""})
        video = root / "windows" / f"{key}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
        for i in range(FRAMES):
            writer.write(np.full((HEIGHT, WIDTH, 3), 10 * i, np.uint8))
        writer.release()
        directory = root / "videos" / key
        (directory / "labels" / "fish").mkdir(parents=True)
        (directory / "static" / "sam3-fish").mkdir(parents=True)
        (directory / "video.json").write_text(json.dumps({"key": key, "file": {
            "path": str(video), "sha256": sd.sha256(video), "fps": 15.0, "decoded_frames": FRAMES,
            "width": WIDTH, "height": HEIGHT}}))
        labels = [{"id": "sam3-1", "frame": f, "box": [5, 5, 25, 20], "confidence": .9} for f in range(FRAMES)]
        (directory / "labels" / "fish" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
        removed = 15 if name == "excluded" else 0
        (directory / "static" / "sam3-fish" / "report.json").write_text(
            json.dumps({"counts": {"boxes": 20, "removed_boxes": removed}}))
        entries |= {f"{key}/{stage}": {"status": "complete", "kind": "cpu", "attempts": []} for stage in rc.STAGE_ENTRIES}
    (root / "selection").mkdir()
    with (root / "selection" / "to_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    roles = {day: ("test" if name == "test" else "train") for name, day in DAYS.items()} | {"2025-07-05": "expert"}
    (root / "selection" / "day_roles.json").write_text(json.dumps({"roles": roles}))
    (root / "journal.json").write_text(json.dumps({
        "schema_version": 1, "created_utc": "x", "budgets_seconds": {"training": 10800}, "entries": entries,
        "decisions": {"validation_days": {"days": [DAYS["validation"]], "considered": sorted(roles)}}}))


def test_checks_dataset_training_and_evaluation_fit_together(tmp_path, monkeypatch):
    root = tmp_path / "round-1"
    make_round(root)
    report = rc.check(root, root / "selection" / "to_label.csv", sheets=False)
    assert report["kept"] == 3 and report["exclusion_reasons"] == {"max_static_removed_share": 1}

    data = sd.round_data(root)
    assert {k: (v["images"], v["boxes"]) for k, v in data["counts"].items()} == \
        {"train": (4, 4), "validation": (4, 4), "test": (4, 4)}
    for split, folder in sd.SPLIT_DIRS.items():
        coco = json.loads((root / "dataset" / folder / "_annotations.coco.json").read_text())
        assert sorted(p.name for p in (root / "dataset" / folder).glob("*.jpg")) == \
            sorted(i["file_name"] for i in coco["images"])
        assert {i["day"] for i in coco["images"]} == {DAYS[split]}
    assert not list((root / "dataset").rglob(f"*{DAYS['excluded']}*"))

    calls = []

    def fake_train(dataset, output, epochs, max_seconds, heartbeat):
        calls.append((output.name, epochs))
        output.mkdir(parents=True, exist_ok=True)
        best = output / "checkpoint_best_total.pth"
        best.write_bytes(b"weights")
        return {"seconds": 100.0, "epochs_asked": epochs, "epochs_run": epochs, "best": sd.file_id(best)}

    monkeypatch.setattr(sd, "_train", fake_train)
    monkeypatch.setattr(sd, "verify_rfdetr", lambda: {"sha256": "coco-checkpoint"})
    trained = sd.round_train(root)
    assert calls == [("probe", 1), ("train", 50)] and trained["epochs_run"] == 50
    journal = json.loads((root / "journal.json").read_text())
    assert journal["decisions"]["student_model"]["trained_on"] == "round round-1"

    def fake_detect(model, video, **kwargs):
        boxes = [] if DAYS["test"] in str(video) else [[5, 5, 25, 20]]  # silent on the test (none) window
        frames = [{"frame": f, "boxes": boxes, "ids": [], "scores": [.7] * len(boxes)} for f in range(FRAMES)]
        speed = {"gpu": "fake", "total_ms": {"p50": 20.0, "p95": 30.0}, "fps": 50.0, "keeps_up_with_live": True}
        return frames, speed

    monkeypatch.setattr(sd, "load_student", lambda checkpoint: object())
    monkeypatch.setattr(sd, "detect_video", fake_detect)
    result = sd.round_eval(root)
    journal = json.loads((root / "journal.json").read_text())
    assert journal["decisions"]["student_threshold"]["threshold"] == 0.7  # the highest threshold keeping the boxes
    assert result["agreement"]["none"] == {"windows": 1, "student_boxes": 0, "sam3_label_boxes": 20, "matched": 0,
                                           "agreement": 0.0, "share_of_student_matched": 0.0,
                                           "share_of_sam3_matched": 0.0}
    assert result["empty_water_proxy"]["none"]["student_boxes_per_frame"] == 0.0
    assert result["empty_water_proxy"]["none"]["sam3_label_boxes_per_frame"] == 1.0
    assert result["test_days"] == [DAYS["test"]] and result["speed_1080p"]["windows"] == 2
    assert (root / "student" / "round_eval.json").exists()
