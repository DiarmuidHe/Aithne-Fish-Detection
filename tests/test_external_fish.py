import csv
import json
import uuid
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services.external_datasets import (
    _official_yolo_lines,
    evaluate_mot,
    import_brackish,
    import_brackish_mot,
    make_brackish_manifest,
    parse_mot,
)
from app.services.viame_parser import parse_viame_csv
from scripts.external_fish import _fishial_boxes


def _manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label", "sequence", "frame", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _class_map(path):
    path.write_text(json.dumps({"0": "crab", "1": "fish", "2": "jellyfish",
                                "3": "shrimp", "4": "small_fish", "5": "starfish"}))


def test_brackish_maps_only_fish_and_records_sequence(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    Image.new("RGB", (16, 16)).save(root / "f.jpg")
    (root / "f.txt").write_text("1 .5 .5 .25 .25\n4 .3 .3 .2 .2\n0 .7 .7 .1 .1\n")
    manifest = tmp_path / "frames.csv"
    classes = tmp_path / "classes.json"
    _class_map(classes)
    _manifest(manifest, [{"image": "f.jpg", "label": "f.txt", "sequence": "recording-1",
                          "frame": 12, "split": "train"}])
    out = tmp_path / "converted"
    report = import_brackish(root, manifest, classes, out)
    assert report["fish_boxes"] == 2
    assert len(next((out / "labels/train").glob("*.txt")).read_text().splitlines()) == 2
    metadata = json.loads((out / "metadata.jsonl").read_text().strip())
    assert (metadata["source_sequence"], metadata["source_frame"]) == ("recording-1", 12)
    assert metadata["source_image_sha256"] and metadata["source_label_sha256"]


def test_brackish_refuses_sequence_split_leakage_before_writing(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (16, 16)).save(root / f"{name}.jpg")
        (root / f"{name}.txt").write_text("1 .5 .5 .25 .25\n")
    manifest = tmp_path / "frames.csv"
    classes = tmp_path / "classes.json"
    _class_map(classes)
    _manifest(manifest, [{"image": f"{name}.jpg", "label": f"{name}.txt",
                          "sequence": "same-recording", "frame": frame, "split": split}
                         for frame, (name, split) in enumerate((("a", "train"), ("b", "test")))])
    out = tmp_path / "converted"
    with pytest.raises(ValueError, match="Split leakage"):
        import_brackish(root, manifest, classes, out)
    assert not out.exists()


def test_brackish_uses_supplied_class_ids(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    Image.new("RGB", (16, 16)).save(root / "f.jpg")
    (root / "f.txt").write_text("0 .5 .5 .25 .25\n2 .3 .3 .2 .2\n")
    manifest = tmp_path / "frames.csv"
    _manifest(manifest, [{"image": "f.jpg", "label": "f.txt", "sequence": "rec",
                          "frame": 1, "split": "train"}])
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps({"0": "fish", "1": "small_fish", "2": "crab",
                                   "3": "shrimp", "4": "jellyfish", "5": "starfish"}))
    out = tmp_path / "converted"
    report = import_brackish(root, manifest, classes, out)
    assert report["fish_boxes"] == 1
    assert report["class_mapping"]["0"]["target"] == "fish"


def test_brackish_rejects_duplicate_image_across_distinct_sequences(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (16, 16)).save(root / f"{name}.jpg")
        (root / f"{name}.txt").write_text("1 .5 .5 .2 .2\n")
    manifest = tmp_path / "frames.csv"
    _manifest(manifest, [{"image": "a.jpg", "label": "a.txt", "sequence": "rec-1",
                          "frame": 1, "split": "train"},
                         {"image": "b.jpg", "label": "b.txt", "sequence": "rec-2",
                          "frame": 1, "split": "test"}])
    classes = tmp_path / "classes.json"
    _class_map(classes)
    with pytest.raises(ValueError, match="identical image"):
        import_brackish(root, manifest, classes, tmp_path / "converted")


def test_manifest_assigns_complete_sequence_one_split(tmp_path):
    root = tmp_path / "source"
    for folder in ("train", "valid"):
        (root / folder / "images").mkdir(parents=True)
        (root / folder / "labels").mkdir()
    for folder, frame in (("train", 1), ("valid", 2)):
        (root / folder / "images" / f"recA_{frame:04}.jpg").write_bytes(b"image")
        (root / folder / "labels" / f"recA_{frame:04}.txt").write_text("")
    manifest = tmp_path / "manifest.csv"
    make_brackish_manifest(root, manifest, r"(?P<sequence>recA)_(?P<frame>\d+)")
    rows = list(csv.DictReader(manifest.open()))
    assert len({row["split"] for row in rows}) == 1
    assert {int(row["frame"]) for row in rows} == {1, 2}


def test_official_archive_pixel_boxes_are_normalized_and_clipped():
    classes = {1: "fish", 2: "small_fish", 3: "crab", 4: "shrimp",
               5: "jellyfish", 6: "starfish"}
    lines, audit = _official_yolo_lines(b"1 480 270 96 54\n2 6 516 13 17\n3 20 20 10 10\n",
                                        "source.txt", classes)
    assert lines[0] == "0 0.50000000 0.50000000 0.10000000 0.10000000"
    assert len(lines) == 2
    assert audit[1]["clipped_to_frame"] is True
    assert audit[2]["target_class"] is None


def test_mot_preserves_ids_frames_and_sequence_isolation(tmp_path):
    root = tmp_path / "mot"
    for name in ("s1", "s2"):
        gt = root / name / "gt"
        gt.mkdir(parents=True)
        (gt / "gt.txt").write_text("1,7,10,20,30,40,1,1,1\n2,7,11,20,30,40,1,5,1\n")
    out = tmp_path / "imported"
    report = import_brackish_mot(root, out)
    assert set(report["sequences"]) == {"s1", "s2"}
    rows = list(csv.DictReader((out / "ground_truth.csv").open()))
    assert len(rows) == 4
    assert {(r["sequence"], r["frame"], r["track_id"]) for r in rows} == {
        (s, f, "7") for s in ("s1", "s2") for f in ("1", "2")}
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    for name in ("s1", "s2"):
        (candidates / f"{name}.txt").write_text("1,3,10,20,30,40,0.9\n2,3,11,20,30,40,0.9\n")
    result = evaluate_mot(out, candidates, tmp_path / "metrics.json")
    assert result["counts"] == {"tp": 4, "fp": 0, "fn": 0, "id_switches": 0}


def test_mot_rejects_invalid_track_id(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,0,10,20,30,40,1,1,1\n")
    with pytest.raises(ValueError, match="invalid MOT"):
        parse_mot(path)


def test_fishial_candidate_csv_is_cvat_parseable(tmp_path):
    path = tmp_path / "candidates.csv"
    path.write_text("track_id,video_id,frame,tl_x,tl_y,br_x,br_y,confidence,target_length,class,score\n"
                    "fishial-0-0,clip.mp4,0,1,2,10,20,0.8,-1,fish,0.8\n")
    parsed = parse_viame_csv(path, strict=True)
    assert len(parsed.tracks) == 1
    assert parsed.tracks[0].observations[0].frame_number == 0


def test_fishial_export_postprocessing_filters_and_suppresses_overlap():
    import numpy as np
    # Center x/y, width/height, score columns from the pinned [5,8400] export.
    prediction = np.array([[50, 51, 200], [50, 50, 200], [20, 20, 30],
                           [20, 20, 30], [.9, .8, .1]], dtype=float)
    boxes = _fishial_boxes(prediction, .25)
    assert len(boxes) == 1
    assert boxes[0][:4] == [40, 40, 60, 60]


def test_cvat_suggestions_use_separate_candidate_task(tmp_path, monkeypatch):
    import scripts.import_model_suggestions as module

    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("track_id,video_id,frame,tl_x,tl_y,br_x,br_y,confidence,target_length,class,score\n"
                        "fishial-0-0,clip.mp4,0,1,2,10,20,0.8,-1,fish,0.8\n")
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, kind, identity):
            return SimpleNamespace(width=100, height=100)

    class Gateway:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_candidate_task(self, video, name):
            calls.append(("candidate", name))
            return 42

        def has_annotations(self, task_id):
            return False

        def import_tracks(self, task_id, parsed, width, height):
            calls.append(("import", len(parsed.tracks)))
            return 1

    monkeypatch.setattr(module, "SessionLocal", Session)
    monkeypatch.setattr(module, "get_settings", lambda: object())
    monkeypatch.setattr(module, "gateway_for", lambda settings: Gateway())
    result = module.import_suggestions(uuid.uuid4(), "fishial-v12", csv_path)
    assert calls == [("candidate", "fishial-v12"), ("import", 1)]
    assert result["status"] == "unreviewed_suggestions"
