"""SAM 3 / BoT-SORT / static-suppression pipeline: no network, no GPU, no checkpoint."""
import json

import pytest

from app.services.static_filter import StaticSettings, filter_static, static_zones
from app.services.viame_parser import parse_viame_csv
from scripts import sam3_autolabel
from scripts.sam3_autolabel import (
    UnverifiedCheckpoint,
    build_predictor,
    observation_rows,
    reliable_labels,
    verify_checkpoint,
    write_candidates,
)
from scripts.sam3_botsort import (
    BudgetExceeded,
    Journal,
    agreement,
    time_split,
    viame_analysed_frames,
)

FPS = 10.0


def _track(track_id, frames, box, step=(0.0, 0.0), confidence=.8):
    """One observation per frame, the box moving by ``step`` pixels per frame."""
    rows = []
    for index, frame in enumerate(frames):
        dx, dy = step[0] * index, step[1] * index
        rows.append({"id": track_id, "frame": frame, "t": frame / FPS, "confidence": confidence,
                     "box": [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]})
    return rows


# ------------------------------------------------------------- static filter --

def test_still_box_is_removed_and_swimming_fish_kept():
    frames = list(range(400))  # 40 s at 10 FPS
    sign = _track("sign", frames, (900, 800, 1060, 980))
    fish = _track("fish", frames[:60], (100, 100, 180, 140), step=(8, 2))
    report = filter_static(sign + fish, analysed_frames=frames, fps=FPS)
    assert report["tracks"]["sign"]["removed"] and "still_track" in report["tracks"]["sign"]["flags"]
    assert not report["tracks"]["fish"]["removed"]
    assert {r["id"] for r in report["kept"]} == {"fish"}
    assert report["counts"]["removed_boxes"] == len(sign)


def test_fish_that_rests_then_moves_is_kept():
    frames = list(range(400))
    rest = _track("fish", frames[:300], (500, 500, 580, 540))  # 30 s at rest
    swim = _track("fish", frames[300:], (500, 500, 580, 540), step=(6, 0))  # then swims off
    report = filter_static(rest + swim, analysed_frames=frames, fps=FPS)
    decision = report["tracks"]["fish"]
    assert decision["excursions"] > StaticSettings().max_excursions
    assert not decision["removed"] and report["counts"]["removed_boxes"] == 0


def test_brief_rest_is_far_below_every_duration_threshold():
    frames = list(range(400))
    resting = _track("fish", frames[100:150], (500, 500, 580, 540))  # 5 s, perfectly still
    report = filter_static(resting, analysed_frames=frames, fps=FPS)
    assert not report["tracks"]["fish"]["removed"]


def test_recurring_static_zone_is_found_and_its_short_tracks_removed():
    frames = list(range(600))
    # A post detected in short bursts all through the video: no burst is long enough to be
    # a still track, but together they are a static zone.
    bursts = [r for start in range(0, 600, 40)
              for r in _track(f"post-{start}", frames[start:start + 25], (1000, 900, 1030, 1080))]
    passer = _track("fish", frames[200:240], (700, 920, 780, 960), step=(8, 0))  # swims through it
    zones = static_zones(bursts + passer, frames, FPS)
    assert len(zones) == 1 and zones[0]["time_bin_share"] == 1.0
    report = filter_static(bursts + passer, analysed_frames=frames, fps=FPS)
    removed = {r["id"] for r in report["removed"]}
    assert removed == {r["id"] for r in bursts}
    assert all("static_zone" in report["tracks"][i]["flags"] for i in removed)
    assert not report["tracks"]["fish"]["removed"]


def test_untracked_boxes_in_a_static_zone_are_removed():
    frames = list(range(300))
    raw = [{"id": None, "frame": f, "t": f / FPS, "confidence": .15, "box": [300, 300, 360, 380]} for f in frames]
    raw += [{"id": None, "frame": 7, "t": .7, "confidence": .9, "box": [1200, 200, 1300, 260]}]
    report = filter_static(raw, analysed_frames=frames, fps=FPS)
    assert report["counts"]["removed_untracked_boxes"] == 300
    assert [r["box"] for r in report["kept"]] == [[1200, 200, 1300, 260]]


def test_background_match_removes_fixtures_but_not_a_resting_fish():
    frames = list(range(100))
    lamp = _track("lamp", frames[:60], (10, 10, 60, 60))  # 6 s: too short for a still track
    fish = _track("fish", frames[:60], (500, 500, 580, 540))  # Resting 6 s, not in the background
    blink = _track("blink", frames[:20], (900, 10, 950, 60))  # Matches, but under 3 s
    scores = {"lamp": .97, "fish": .41, "blink": .99}
    report = filter_static(lamp + fish + blink, analysed_frames=frames, fps=FPS, background_scores=scores)
    assert report["tracks"]["lamp"]["flags"] == ["background_match"] and report["tracks"]["lamp"]["removed"]
    assert not report["tracks"]["fish"]["removed"] and not report["tracks"]["blink"]["removed"]
    flag_only = StaticSettings(background_removes=False)
    report = filter_static(lamp, analysed_frames=frames, fps=FPS, settings=flag_only, background_scores=scores)
    assert report["tracks"]["lamp"]["flags"] == ["background_match"] and not report["tracks"]["lamp"]["removed"]


# ------------------------------------------------------------- SAM 3 labels --

def _mock_sam3_frames():
    """What infer_video yields: zero-based source frames, boxes, SAM 3 object ids."""
    frames = []
    for frame in range(12):
        boxes, ids, scores, trackers = [], [], [], []
        if frame >= 2:  # Object 4 appears on frame 2 and stays.
            boxes.append([100.0 + frame, 200.0, 180.0 + frame, 240.0]); ids.append(4)
            scores.append(.82); trackers.append(.9 if frame != 6 else .1)
        if frame in (5, 6):  # Object 9 is seen twice only.
            boxes.append([900.0, 500.0, 950.0, 530.0]); ids.append(9); scores.append(.95); trackers.append(.95)
        if frame >= 3:  # Object 11 is long but never confident.
            boxes.append([1500.0, 100.0, 1560.0, 150.0]); ids.append(11); scores.append(.3); trackers.append(.4)
        frames.append({"frame": frame, "boxes": boxes, "ids": ids, "scores": scores, "tracker_scores": trackers})
    return frames


def test_mocked_sam3_output_becomes_candidate_rows_on_source_frames(tmp_path):
    rows = observation_rows(_mock_sam3_frames(), fps=12.5)
    kept, report = reliable_labels(rows, min_track_frames=5, min_track_median_score=.5, min_box_score=.25)
    assert report["tracks"] == {"short": 1, "low_score": 1, "kept": 1}
    assert report["low_score_boxes_dropped"] == 1  # Frame 6's tracker score 0.1
    count = write_candidates(kept, "upload.mp4", tmp_path / "candidates.csv")
    parsed = parse_viame_csv(tmp_path / "candidates.csv", strict=True)  # As cvat-suggestions reads it
    assert count == len(parsed.detections) == 9
    assert [d.frame_number for d in parsed.detections] == [2, 3, 4, 5, 7, 8, 9, 10, 11]
    first = parsed.detections[0]
    assert first.track_id == "sam3-4" and first.class_name == "fish" and first.source == "upload.mp4"
    assert (first.bbox_left, first.bbox_right) == (102.0, 182.0)
    assert len(parsed.tracks) == 1


def test_sha256_check_refuses_an_unverified_checkpoint_before_loading(tmp_path, monkeypatch):
    fake = tmp_path / "sam3.pt"
    fake.write_bytes(b"not the pinned checkpoint")
    loaded = []
    monkeypatch.setattr(sam3_autolabel, "_import_predictor",
                        lambda *a: loaded.append("import") or (None, None))
    import torch
    monkeypatch.setattr(torch, "load", lambda *a, **k: loaded.append("torch.load"))
    with pytest.raises(UnverifiedCheckpoint, match="refusing to load"):
        build_predictor(fake)
    with pytest.raises(UnverifiedCheckpoint, match="SHA256"):
        verify_checkpoint(fake, expected_bytes=None)
    with pytest.raises(UnverifiedCheckpoint, match="huggingface.co/facebook/sam3"):
        verify_checkpoint(tmp_path / "missing.pt")
    assert loaded == []
    digest = verify_checkpoint(fake, expected_sha256=sam3_autolabel.sha256(fake), expected_bytes=None)["sha256"]
    loader = sam3_autolabel._verified_loader(fake.resolve())
    with pytest.raises(UnverifiedCheckpoint, match="only"):
        loader(str(tmp_path / "other.pt"))
    assert loaded == [] and len(digest) == 64


# ------------------------------------------------------------------ tracker --

def _tracker(**values):
    pytest.importorskip("ultralytics")
    from app.services.box_tracking import BoxTracker, TrackerSettings
    return BoxTracker(TrackerSettings(**values))


@pytest.mark.parametrize("kind", ["botsort", "bytetrack"])
def test_tracker_keeps_an_id_through_a_one_frame_gap(kind, monkeypatch):
    import numpy as np

    monkeypatch.setenv("YOLO_OFFLINE", "true")
    tracker, rows = _tracker(kind=kind), []
    for frame in range(8):
        if frame == 4:  # The fish is missed on one frame.
            rows += tracker.update(frame, np.empty((0, 4)), np.empty(0))
            continue
        boxes = np.array([[100 + 6 * frame, 100, 160 + 6 * frame, 140], [900, 500, 960, 560]])
        rows += tracker.update(frame, boxes, np.array([.9, .7]))
    fish = [r for r in rows if r["det_index"] == 0]
    assert [r["frame"] for r in fish] == [0, 1, 2, 3, 5, 6, 7]
    assert len({r["id"] for r in fish}) == 1
    assert fish[0]["box"] == [100.0, 100.0, 160.0, 140.0]  # The detector's box, not a smoothed one


def test_tracker_reports_a_late_tracks_first_box_once_confirmed(monkeypatch):
    import numpy as np

    monkeypatch.setenv("YOLO_OFFLINE", "true")
    tracker, rows = _tracker(), []
    for frame in range(6):
        boxes = [[100, 100, 160, 140]] + ([[600 + 4 * frame, 300, 660 + 4 * frame, 340]] if frame >= 2 else [])
        rows += tracker.update(frame, np.array(boxes, dtype=float), np.array([.9] * len(boxes)))
    late = sorted(r["frame"] for r in rows if r["box"][0] >= 600)
    assert late == [2, 3, 4, 5]  # Frame 2 is held until frame 3 confirms the track


# -------------------------------------------------------------------- split --

def test_time_split_has_its_gap():
    split = time_split(1440, 12.5)  # The SmartBay test video
    train, test = split["segments"]["train"], split["segments"]["test"]
    assert train[0] == 0 and test[1] == 1440 and test[0] == 1440 - 288
    assert test[0] - train[1] >= split["gap_frames"] == 63  # >= 5 s at 12.5 FPS
    split = time_split(1440, 12.5, validation=.1)
    segments = split["segments"]
    assert segments["validation"][0] - segments["train"][1] >= 63
    assert segments["test"][0] - segments["validation"][1] >= 63
    with pytest.raises(ValueError):
        time_split(60, 12.5)


# ------------------------------------------------------------------ journal --

def test_journal_skips_completed_work_and_charges_interrupted_runs(tmp_path):
    calls = []
    journal = Journal(tmp_path, budgets={"sam3_gpu": 100})
    step = lambda heartbeat: calls.append(1) or {"boxes": 5}
    assert journal.run("video/sam3/fish", "sam3_gpu", {"video": "abc"}, {"prompt": "fish"}, step) == {"boxes": 5}
    resumed = Journal(tmp_path)
    assert resumed.run("video/sam3/fish", "sam3_gpu", {"video": "abc"}, {"prompt": "fish"}, step) == {"boxes": 5}
    assert calls == [1]  # Completed with the same inputs: not redone
    resumed.run("video/sam3/fish", "sam3_gpu", {"video": "def"}, {"prompt": "fish"}, step)
    assert calls == [1, 1]  # Different inputs: redone

    data = json.loads((tmp_path / "journal.json").read_text())  # A process died mid-step
    data["entries"]["video/sam3/small"] = {"kind": "sam3_gpu", "status": "running", "key": "x", "attempts": [],
                                           "started_unix": 1000.0, "heartbeat_unix": 1040.0}
    (tmp_path / "journal.json").write_text(json.dumps(data))
    journal = Journal(tmp_path)
    journal.run("video/sam3/small", "sam3_gpu", {}, {}, lambda heartbeat: {"ok": True})
    attempts = journal.data["entries"]["video/sam3/small"]["attempts"]
    assert [a["status"] for a in attempts] == ["interrupted", "complete"]
    assert attempts[0]["duration_seconds"] == 40.0
    with pytest.raises(BudgetExceeded):
        journal.run("video/sam3/more", "sam3_gpu", {}, {}, step, estimate=90)
    assert calls == [1, 1]


def test_journal_merges_entries_written_by_another_process(tmp_path):
    first, second = Journal(tmp_path), Journal(tmp_path)
    first.run("a", "cpu", {}, {}, lambda heartbeat: 1)
    second.run("b", "cpu", {}, {}, lambda heartbeat: 2)
    on_disk = json.loads((tmp_path / "journal.json").read_text())["entries"]
    assert set(on_disk) == {"a", "b"}


# ---------------------------------------------------------- VIAME and pairing --

def test_viame_analysed_frames_come_from_detection_timestamps(tmp_path):
    lines = ["# header", "0,00:00:00.000000,0,1,1,5,5,0.5,0,fish,0.5", "1,00:00:00.160000,1,1,1,5,5,0.5,0,fish,0.5",
             "2,00:00:00.320000,3,1,1,5,5,0.5,0,fish,0.5"]  # Ordinal 2 had no detection
    path = tmp_path / "computed_detections.csv"
    path.write_text("\n".join(lines) + "\n")
    analysed = viame_analysed_frames(path, 12.5)
    assert analysed["frames"] == [0, 2, 3, 4] and not analysed["exact"]
    assert analysed["ordinals_without_detections"] == 1


def test_agreement_counts_only_frames_both_models_analysed():
    viame = [{"id": "1", "frame": 0, "box": [0, 0, 10, 10]}, {"id": "1", "frame": 2, "box": [0, 0, 10, 10]}]
    sam3 = [{"id": "s", "frame": 0, "box": [1, 0, 11, 10]}, {"id": "s", "frame": 1, "box": [0, 0, 10, 10]},
            {"id": "t", "frame": 2, "box": [50, 50, 60, 60]}]
    summary, _ = agreement(viame, sam3, left_frames=[0, 2], right_frames=[0, 1, 2])
    assert summary["shared_frames"] == 2 and summary["matched"] == 1
    assert (summary["left_only"], summary["right_only"]) == (1, 1)
    assert summary["right_unaligned"] == 1  # Frame 1: VIAME never looked, so not a disagreement


def test_detector_only_frames_get_identities_from_the_tracker(monkeypatch):
    """SAM 3's detector gives boxes without ids; BoT-SORT supplies them before labelling."""
    pytest.importorskip("ultralytics")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    from app.services.box_tracking import TrackerSettings
    from scripts.sam3_botsort import track_rows

    frames = [{"frame": f, "boxes": [[100.0 + 5 * f, 100.0, 160.0 + 5 * f, 140.0]], "ids": [], "scores": [.9],
               "tracker_scores": [None]} for f in range(8)]
    rows = observation_rows(frames, fps=12.5)
    assert {r["id"] for r in rows} == {None}
    tracked = track_rows(rows, list(range(8)), TrackerSettings(), 12.5)
    assert [r["frame"] for r in tracked] == list(range(8)) and len({r["id"] for r in tracked}) == 1
    kept, _ = reliable_labels(tracked, min_track_frames=5, min_track_median_score=.5, min_box_score=.25)
    assert len(kept) == 8
