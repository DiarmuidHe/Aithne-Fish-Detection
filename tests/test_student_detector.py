"""Part 3 student detector: no network, no GPU, no checkpoint, no rfdetr import."""
import pytest

from app.services.external_datasets import sha256
from scripts import student_detector
from scripts.sam3_autolabel import UnverifiedCheckpoint
from scripts.sam3_botsort import BudgetExceeded, Journal, save, time_split, write_rows
from scripts.student_detector import (
    choose_threshold,
    coco_split,
    load_student,
    local_checkpoint,
    local_eval,
    raw_rows,
    segment_of,
    size_epochs,
    student_tracker,
    training_overlap,
    verify_rfdetr,
)

FPS = 10.0


def _split():
    return time_split(200, FPS, train=.6, test=.2, gap_seconds=1.0, validation=.1)


def test_split_segments_and_gaps():
    split = _split()
    assert split["segments"] == {"test": [160, 200], "train": [0, 90], "validation": [100, 120]}
    assert segment_of(0, split) == "train" and segment_of(95, split) is None
    assert segment_of(119, split) == "validation" and segment_of(130, split) is None
    assert segment_of(199, split) == "test"


def test_coco_split_keeps_negatives_clips_boxes_and_drops_slivers():
    rows = [{"frame": 5, "box": [-10, 20, 50, 60]},       # clipped at the left edge
            {"frame": 5, "box": [100, 100, 101, 140]},    # 1 px wide: dropped
            {"frame": 95, "box": [10, 10, 40, 40]},       # a gap frame: never written
            {"frame": 170, "box": [1900, 1000, 1950, 1100]}]  # clipped at the far corner
    coco = coco_split(rows, _split(), 1920, 1080)
    assert len(coco["train"]["images"]) == 90 and len(coco["validation"]["images"]) == 20
    assert coco["train"]["annotations"] == [{"id": 1, "image_id": 5, "category_id": 1, "bbox": [0.0, 20.0, 50.0, 40.0],
                                             "area": 2000.0, "iscrowd": 0}]
    assert coco["test"]["annotations"][0]["bbox"] == [1900.0, 1000.0, 20.0, 80.0]
    assert all(image["id"] != 95 for data in coco.values() for image in data["images"])
    assert coco["test"]["images"][0] == {"id": 160, "file_name": "000160.jpg", "width": 1920, "height": 1080}


def test_epochs_are_sized_from_the_probe_and_refused_when_too_few_fit():
    assert size_epochs(60, 7200 - 60) == 50  # capped
    assert size_epochs(300, 6000) == 18  # (6000 - 600) / 300
    with pytest.raises(BudgetExceeded):
        size_epochs(600, 6000)


def _frame(frame, boxes):
    return {"frame": frame, "boxes": [b for b, _ in boxes], "scores": [s for _, s in boxes], "ids": []}


def test_threshold_is_chosen_on_validation_frames_only():
    fish = [0, 0, 40, 40]
    labels = [{"id": "l", "frame": f, "t": f / FPS, "box": fish, "confidence": .9} for f in range(10, 20)]
    # Validation [10, 20): the fish at .45 plus a .30 false box. Outside it, a frame full of .9 noise.
    frames = [_frame(f, [(fish, .45), ([500, 500, 540, 540], .30)]) for f in range(10, 20)]
    frames.append(_frame(25, [([900, 900, 940, 940], .9)] * 3))
    chosen = choose_threshold(frames, labels, [10, 20], FPS, thresholds=[.1, .2, .3, .35, .4, .5])
    assert chosen["sweep"]["0.1"]["student_boxes"] == 20 and chosen["sweep"]["0.5"]["matched"] == 0
    assert chosen["threshold"] == .4  # .35 and .4 tie at f1 1.0: ties go higher


def test_raw_rows_apply_the_threshold():
    rows = raw_rows([_frame(3, [([0, 0, 5, 5], .2), ([1, 1, 6, 6], .6)])], FPS, .5)
    assert rows == [{"id": None, "frame": 3, "t": .3, "box": [1, 1, 6, 6], "confidence": .6}]


def test_tracker_gates_move_to_the_threshold(tmp_path):
    journal = Journal(tmp_path)
    journal.decide("brackishmot_settings", {"botsort": {"track_high_thresh": .45, "new_track_thresh": .5,
                                                        "track_low_thresh": .1, "buffer_seconds": 7.5,
                                                        "match_thresh": .8, "fuse_score": False}})
    settings = student_tracker(journal, .3, FPS)
    assert (settings.track_high_thresh, settings.new_track_thresh, settings.track_low_thresh) == (.3, .3, .15)
    assert settings.track_buffer == 75 and settings.kind == "botsort" and not settings.fuse_score


def test_rfdetr_checkpoint_is_refused_before_any_load(tmp_path, monkeypatch):
    with pytest.raises(UnverifiedCheckpoint, match="storage.googleapis.com/rfdetr"):
        verify_rfdetr(tmp_path / "missing.pth")
    fake = tmp_path / "rf-detr-nano.pth"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(UnverifiedCheckpoint, match="bytes"):
        verify_rfdetr(fake)
    monkeypatch.setattr(student_detector, "RFDETR_BYTES", fake.stat().st_size)
    with pytest.raises(UnverifiedCheckpoint, match="pinned"):
        verify_rfdetr(fake)


def test_changed_student_is_refused(tmp_path):
    path = tmp_path / "checkpoint_best_total.pth"
    path.write_bytes(b"weights")
    with pytest.raises(UnverifiedCheckpoint, match="changed"):
        load_student({"path": str(path), "sha256": "0" * 64})


def test_local_checkpoint_moves_the_instance_path_under_root(tmp_path):
    root = tmp_path / "round-1" / "cloud"
    pulled = root / "models" / "rfdetr-nano-round-1" / "train" / "checkpoint_best_total.pth"
    pulled.parent.mkdir(parents=True)
    pulled.write_bytes(b"weights")
    recorded = {"path": "data/outputs/smartbay-rounds/round-1/models/rfdetr-nano-round-1/train/checkpoint_best_total.pth",
                "sha256": "abc"}
    assert local_checkpoint(root, recorded) == {"path": str(pulled), "sha256": "abc"}


def test_training_overlap_reports_day_role_and_same_clip(tmp_path):
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection" / "day_roles.json").write_text('{"roles": {"2025-08-01": "train"}}')
    (tmp_path / "selection" / "to_label.csv").write_text(
        "window_id,source_mp4_url\nw1,https://x/SmartBayCam1/2025/08/01/SmartBayCam1_2025-08-01_1414.mp4\n")
    model = {"trained_on": "round round-1"}
    assert training_overlap(tmp_path, model, "SmartBayCam1_2025-08-01_0000") == {
        "trained_on_this_video": False, "camera_in_round": True, "day_role": "train", "same_clip_in_round": False}
    assert training_overlap(tmp_path, model, "SmartBayCam1_2025-08-01_1414_f0862")["same_clip_in_round"]
    assert training_overlap(tmp_path, model, "85ea19d4") == {"trained_on_this_video": False}


def test_local_eval_scores_against_sam3_labels_and_reuses_inference(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"frames")
    folder = tmp_path / "videos" / "clip"
    (folder / "labels" / "fish").mkdir(parents=True)
    (folder / "student").mkdir()
    save(folder / "video.json", {"key": "clip", "file": {"path": str(video), "sha256": sha256(video), "fps": FPS,
                                                         "decoded_frames": 4}})
    save(folder / "student" / "split.json", {"segments": {"test": [2, 4]}})
    box = [0, 0, 10, 10]
    write_rows([{"frame": f, "box": box} for f in range(4)], folder / "labels" / "fish" / "rows.jsonl")
    root = tmp_path / "student-root"
    root.mkdir()
    save(root / "journal.json", {"decisions": {"student_model": {"trained_on": "other", "checkpoint": {
        "path": str(tmp_path / "w.pth"), "sha256": "w"}}, "student_threshold": {"threshold": .5}}})
    calls = []
    frames = [{"frame": 0, "boxes": [box], "scores": [.9]}, {"frame": 1, "boxes": [box], "scores": [.3]},
              {"frame": 2, "boxes": [box, [50, 50, 60, 60]], "scores": [.9, .8]}, {"frame": 3, "boxes": [], "scores": []}]
    monkeypatch.setattr(student_detector, "load_student", lambda checkpoint: "model")
    monkeypatch.setattr(student_detector, "detect_video", lambda model, path: calls.append(path) or (frames, {"fps": 9}))
    result = local_eval(root, [folder])["videos"]["clip"]
    assert result["agreement"]["all"].items() >= {"student_boxes": 3, "sam3_label_boxes": 4, "matched": 2,
                                                  "agreement": .4, "frames_with_a_student_box": .5}.items()
    assert result["agreement"]["test_segment"].items() >= {"student_boxes": 2, "sam3_label_boxes": 2,
                                                           "matched": 1}.items()
    assert result["speed"] == {"fps": 9} and len(calls) == 1
    local_eval(root, [folder])
    assert len(calls) == 1  # Same model and video: the frames are reused
