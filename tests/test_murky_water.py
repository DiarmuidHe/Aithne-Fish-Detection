import csv
import itertools
import json
import random
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.external_datasets import (
    evaluate_mot,
    extract_brackish_mot,
    import_brackish_mot,
    min_cost_assignment,
    sha256,
    viame_tracks_to_mot,
)
from scripts.murky_water_tracker import probe_selection, sequence_to_video, split_sequences
from scripts.viame_review_experiment import kwiver_settings, pipeline_processes

HEADER = "# 1: Detection or Track-id, 2: Video or Image Identifier, 3: Unique Frame Identifier\n"


def _box(track_id, x):
    return {"track_id": track_id, "x": x, "y": 0, "w": 10, "h": 10}


def _scored(tmp_path, gt_rows, pred_rows, analysed=None):
    """gt_rows/pred_rows: (frame, id, x) on one sequence; boxes 10x10 at (x, 0)."""
    root = tmp_path / "mot" / "seq" / "gt"
    root.mkdir(parents=True)
    (root / "gt.txt").write_text("".join(f"{f},{i},{x},0,10,10,1,1,1\n" for f, i, x in gt_rows))
    imported = tmp_path / "imported"
    import_brackish_mot(tmp_path / "mot", imported)
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "seq.txt").write_text("".join(f"{f},{i},{x},0,10,10,0.9\n" for f, i, x in pred_rows))
    if analysed is not None:
        (candidates / "analysed_frames.json").write_text(json.dumps({"seq": analysed}))
    return evaluate_mot(imported, candidates, tmp_path / "evaluation.json")


def test_assignment_matches_brute_force_on_rectangular_matrices():
    rng = random.Random(7)
    for _ in range(200):
        rows, columns = rng.randint(1, 5), rng.randint(1, 5)
        cost = [[rng.choice([rng.random(), float(rng.randint(-3, 3))]) for _ in range(columns)]
                for _ in range(rows)]
        size = min(rows, columns)
        best = min(sum(cost[i][j] for i, j in zip(r, c))
                   for r in itertools.combinations(range(rows), size)
                   for c in itertools.permutations(range(columns), size))
        pairs = min_cost_assignment(cost)
        assert len(pairs) == size and len({i for i, _ in pairs}) == len({j for _, j in pairs}) == size
        assert sum(cost[i][j] for i, j in pairs) == pytest.approx(best)


def test_idf1_and_clear_on_an_identity_swap(tmp_path):
    gt = [(f, 1, 0) for f in range(1, 5)] + [(f, 2, 100) for f in range(1, 5)]
    # Prediction 10 follows fish 1 then fish 2; prediction 20 does the opposite.
    pred = [(f, 10, 0 if f <= 2 else 100) for f in range(1, 5)] + [
        (f, 20, 100 if f <= 2 else 0) for f in range(1, 5)]
    report = _scored(tmp_path, gt, pred)
    assert report["identity"] | {} == {"idtp": 4, "idfp": 4, "idfn": 4, "idf1": .5, "idp": .5, "idr": .5}
    assert report["clear"]["idsw"] == 2 and report["clear"]["mota"] == pytest.approx(6 / 8)
    assert report["clear"]["gt_tracks"] == 2 and report["clear"]["idsw_per_gt_track"] == 1
    assert report["counts"]["id_switches"] == 2  # Greedy diagnostic agrees here.


def test_idf1_and_clear_on_a_fragmented_track(tmp_path):
    gt = [(f, 1, 0) for f in range(1, 7)]
    pred = [(f, 5, 0) for f in range(1, 4)] + [(f, 6, 0) for f in range(4, 7)]
    report = _scored(tmp_path, gt, pred)
    assert (report["identity"]["idtp"], report["identity"]["idf1"]) == (3, .5)
    assert report["clear"]["idsw"] == 1 and report["clear"]["mota"] == pytest.approx(5 / 6)


def test_clear_prefers_continuing_a_pair_over_a_better_overlap(tmp_path):
    # Frame 2: prediction 7 (continuing) overlaps less than prediction 8; no switch.
    gt = [(1, 1, 0), (2, 1, 0)]
    pred = [(1, 7, 0), (2, 7, 2), (2, 8, 0)]
    report = _scored(tmp_path, gt, pred)
    assert report["clear"]["idsw"] == 0 and report["clear"]["fp"] == 1
    assert report["identity"]["idtp"] == 2


def test_only_analysed_frames_are_scored(tmp_path):
    gt = [(f, 1, 0) for f in range(1, 7)]
    report = _scored(tmp_path, gt, [(f, 3, 0) for f in (1, 3, 5)], analysed=[1, 3, 5])
    assert report["clear"]["fn"] == 0 and report["identity"]["idf1"] == 1
    assert report["sequences"]["seq"]["gt_boxes"] == 3
    assert report["analysed_frames"].startswith("only analysed_frames.json (3 frames)")


def test_predictions_on_unanalysed_frames_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unanalysed frame 2"):
        _scored(tmp_path, [(1, 1, 0)], [(1, 3, 0), (2, 3, 0)], analysed=[1, 3])


def _viame(path, rows):
    path.write_text(HEADER + "".join(
        f"{tid},{name},{frame},{x},5,{x + 20},25,0.7,0,fish,0.7\n" for tid, name, frame, x in rows))
    return path


def test_viame_image_list_csv_to_mot_at_native_and_downsampled_fps(tmp_path):
    native = [f"{i:06d}.jpg" for i in range(1, 6)]
    csv_path = _viame(tmp_path / "native.csv", [(1, "000001.jpg", 1, 0), (1, "000002.jpg", 2, 3)])
    rows = viame_tracks_to_mot(csv_path, input_names=native, frame_of={n: int(n[:6]) for n in native})
    assert [(r["frame"], r["track_id"], r["x"], r["w"], r["h"]) for r in rows] == [
        (1, 1, 0, 20, 20), (2, 1, 3, 20, 20)]
    # 10 of 15 FPS: VIAME numbers rows by list position; MOT keeps the source frame.
    sampled = ["000001.jpg", "000002.jpg", "000004.jpg", "000005.jpg"]
    csv_path = _viame(tmp_path / "sampled.csv", [(2, "000004.jpg", 3, 0), (2, "000005.jpg", 4, 1)])
    rows = viame_tracks_to_mot(csv_path, input_names=sampled, frame_of={n: int(n[:6]) for n in sampled})
    assert [r["frame"] for r in rows] == [4, 5]


def test_viame_image_list_rejects_a_misaligned_row(tmp_path):
    names = ["000001.jpg", "000003.jpg"]
    csv_path = _viame(tmp_path / "bad.csv", [(1, "000003.jpg", 1, 0)])
    with pytest.raises(ValueError, match="not 000003.jpg"):
        viame_tracks_to_mot(csv_path, input_names=names, frame_of={n: int(n[:6]) for n in names})


def test_production_video_csv_offset_gives_one_based_mot_frames(tmp_path):
    # Production video CSVs number source frames from one; offset -1 is zero-based.
    csv_path = _viame(tmp_path / "video.csv", [(4, "input_list.txt", 1, 0), (4, "input_list.txt", 4, 2)])
    assert [r["frame"] for r in viame_tracks_to_mot(csv_path, frame_offset=-1)] == [1, 4]
    with pytest.raises(ValueError, match="before the first frame"):
        viame_tracks_to_mot(csv_path, frame_offset=-2)


def test_sequence_to_video_asserts_frame_count(tmp_path):
    import cv2
    import numpy as np

    images = []
    for index in range(4):
        path = tmp_path / f"{index + 1:06d}.png"
        cv2.imwrite(str(path), np.full((24, 32, 3), index * 40, dtype=np.uint8))
        images.append(path)
    info = sequence_to_video(images, tmp_path / "seq.mkv", 15)
    assert info["frames"] == 4 and Path(info["path"]).is_file()
    (tmp_path / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot decode"):
        sequence_to_video([*images, tmp_path / "broken.png"], tmp_path / "broken.mkv", 15)


def test_split_is_stable_and_stratified():
    boxes = {f"train/s{i:02d}": (i % 3) * 10 for i in range(40)}
    first, second = split_sequences(boxes), split_sequences(dict(reversed(list(boxes.items()))))
    assert first == second
    assert not set(first["tuning"]) & set(first["held_out"])
    assert set(first["round_one"]) <= set(first["tuning"])
    fish = {s for s, n in boxes.items() if n}
    assert abs(len(fish & set(first["tuning"])) - len(fish & set(first["held_out"]))) <= 1


def test_probe_positions_are_zero_based(tmp_path):
    path = tmp_path / "passed.csv"
    path.write_text(HEADER + "0,p000001.jpg,0,0,0,10,10,0.5,0,fish,0.5\n"
                             "1,p000002.jpg,1,0,0,10,10,0.5,0,fish,0.5\n"
                             "2,p000004.jpg,2,0,0,10,10,0.5,0,fish,0.5\n")
    assert probe_selection(path) == [0, 1, 3]


def _tracker_block(path):
    text = Path(path).read_text(encoding="utf-8")
    return re.search(r"process tracker\n.*?endblock", text, re.S).group(0)


def test_replay_tracker_block_matches_production():
    assert _tracker_block("scripts/viame_pipelines/bytetrack_replay.pipe") == _tracker_block(
        "app/viame_pipelines/tracker_fish_no_motion.pipe")


def test_runner_sets_only_keys_for_declared_processes(tmp_path):
    (tmp_path / "inner.pipe").write_text("process detection_reader\n  :: detected_object_input\n")
    pipe = tmp_path / "outer.pipe"
    pipe.write_text("include inner.pipe\nprocess tracker\n  :: track_objects\nprocess track_writer\n")
    processes = pipeline_processes(pipe)
    assert processes == {"detection_reader", "tracker", "track_writer"}
    args = SimpleNamespace(video=tmp_path / "list.txt", image_list=True, frame_time=1 / 15, fps=10,
                           score=.1, detections=tmp_path / "d.csv", high=.35, low=.1, new_track=.4,
                           buffer=12, match=None, second_match=None, unconfirmed_match=None)
    settings = kwiver_settings(args, processes)
    assert settings["input:video_reader:type"] == "image_list"
    assert settings["tracker:track_objects:bytetrack:track_buffer"] == "12"
    assert not any(k.startswith(("downsampler:", "detector:")) for k in settings)
    assert "tracker:track_objects:bytetrack:match_thresh" not in settings
    with pytest.raises(ValueError, match="--detections"):
        kwiver_settings(SimpleNamespace(**{**vars(args), "detections": None}), processes)


def test_extract_brackish_mot_keeps_only_real_sequences(tmp_path):
    archive = tmp_path / "mot.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("BrackishMOT/train/seq-1/gt/gt.txt", "1,1,0,0,5,5,1,1,1\n")
        bundle.writestr("BrackishMOT/train/seq-1/img1/000001.jpg", b"x")
        bundle.writestr("brackishMOTSynth/brackishMOTSynth/s/gt/gt.txt", "1,1,0,0,5,5,1,1,1\n")
    out = tmp_path / "out"
    report = extract_brackish_mot(archive, out, sha256(archive))
    assert report["files"] == 2 and (out / "train/seq-1/gt/gt.txt").is_file()
    assert not any("Synth" in str(p) for p in out.rglob("*"))
    with pytest.raises(ValueError, match="SHA256"):
        extract_brackish_mot(archive, tmp_path / "other", "0" * 64)


def test_extract_brackish_mot_rejects_traversal(tmp_path):
    archive = tmp_path / "mot.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("BrackishMOT/train/../../evil/gt.txt", "x")
    with pytest.raises(ValueError, match="Unexpected"):
        extract_brackish_mot(archive, tmp_path / "out", sha256(archive))
    assert not (tmp_path / "out").exists()


def test_import_excludes_rows_without_identity(tmp_path):
    gt = tmp_path / "mot" / "seq" / "gt"
    gt.mkdir(parents=True)
    (gt / "gt.txt").write_text("1,-1,0,0,5,5,1,5,1\n1,2,9,9,5,5,1,5,1\n")
    report = import_brackish_mot(tmp_path / "mot", tmp_path / "imported")
    assert report["sequences"]["seq"]["fish_boxes"] == 1
    assert len(report["sequences"]["seq"]["excluded_unidentified_rows"]) == 1
    rows = list(csv.DictReader((tmp_path / "imported/ground_truth.csv").open()))
    assert [r["track_id"] for r in rows] == ["2"]


def _summary(idf1, mota, fp, idsw_per_track):
    return {"idf1": idf1, "mota": mota, "fp": fp, "idsw_per_gt_track": idsw_per_track}


def test_ranking_puts_constraint_violators_after_compliant_configs():
    from scripts.murky_water_tracker import ranked

    results = {"prod": _summary(.40, .30, 100, .5),
               "big-idf1-but-fp": _summary(.60, .35, 111, .4),   # FP +11%: violates
               "worse-mota": _summary(.55, .29, 90, .4),         # MOTA falls: violates
               "modest": _summary(.45, .31, 105, .5),
               "better": _summary(.50, .32, 100, .5)}
    assert ranked(results, "prod") == ["better", "modest", "big-idf1-but-fp", "worse-mota"]


def test_gate_requires_every_criterion_at_both_profiles(tmp_path):
    from scripts.murky_water_tracker import PRODUCTION, Journal, gate, tag

    journal = Journal(tmp_path / "journal.json")
    production = tag(dict(PRODUCTION))
    smart = {"proxy_regressions_over_5_percent": [], "acceptance_stable": True, "runtime_ratio": 1.02}
    for profile, idf1 in (("batch-10fps", .43), ("live-4fps", .41)):
        journal.decide(f"held-out/{profile}", {
            "best": "cand", "results": {production: _summary(.40, .30, 100, .5),
                                        "cand": _summary(idf1, .31, 108, .5)},
            "per_sequence": {"majority": True}})
    journal.decide("smartbay", {"batch-10fps": smart, "live-4fps": smart})
    result = gate(journal)
    assert result["batch-10fps"]["passes"] and not result["live-4fps"]["passes"]
    assert result["live-4fps"]["criteria"]["idf1_gain_at_least_2_points"] is False
    assert result["promote"] is False


def test_gate_treats_missing_smartbay_evidence_as_not_passed(tmp_path):
    from scripts.murky_water_tracker import PRODUCTION, Journal, gate, tag

    journal = Journal(tmp_path / "journal.json")
    for profile in ("batch-10fps", "live-4fps"):
        journal.decide(f"held-out/{profile}", {
            "best": "cand", "results": {tag(dict(PRODUCTION)): _summary(.40, .30, 100, .5),
                                        "cand": _summary(.50, .31, 100, .4)},
            "per_sequence": {"majority": True}})
    result = gate(journal)
    assert result["batch-10fps"]["criteria"]["runtime_within_1_1x"] is None
    assert not result["batch-10fps"]["passes"] and not result["promote"]
