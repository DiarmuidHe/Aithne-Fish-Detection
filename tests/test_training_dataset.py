from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.training_dataset import build_dataset, inventory_jobs, preview_manifest


def _video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def test_dataset_alignment_splits_and_review_gate(tmp_path: Path):
    sources = []
    for name, split, reviewed in (("a", "train", False), ("b", "val", True), ("c", "test", True)):
        _video(tmp_path / f"{name}.mp4")
        (tmp_path / f"{name}.csv").write_text(
            "1,input_list.txt,1,8,4,24,20,0.9,-1,cod,0.9\n"
            "1,input_list.txt,1,8,4,24,20,0.9,-1,cod,0.9\n"
            "2,input_list.txt,2,10,4,20,20,0.3,-1,cod,0.9\n"
            "3,input_list.txt,3,80,4,90,20,0.8,-1,cod,0.9\n", encoding="utf-8")
        sources.append({"video": f"{name}.mp4", "annotations": f"{name}.csv",
                        "frame_offset": -1, "group": name, "split": split, "reviewed": reviewed})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"class_mapping": {"cod": "fish"}, "sources": sources}))
    report = build_dataset(manifest, tmp_path / "out")
    assert report["frames"] == {"train": 0, "val": 2, "test": 2}
    assert report["evaluation_ready"]
    assert list((tmp_path / "out" / "previews" / "val").glob("*.jpg"))
    labels = list((tmp_path / "out" / "labels" / "val").glob("*.txt"))
    assert any("0 0.25000000 0.25000000 0.25000000 0.33333333" in p.read_text() for p in labels)
    issues = json.loads((tmp_path / "out" / "review_issues.json").read_text())
    assert {x["reason"] for x in issues} >= {"duplicate", "low confidence", "box outside frame or inverted"}
    with pytest.raises(FileExistsError):
        build_dataset(manifest, tmp_path / "out")


def test_pseudo_training_and_group_leakage_guard(tmp_path: Path):
    _video(tmp_path / "a.mp4")
    (tmp_path / "a.csv").write_text("1,a.mp4,0,0,0,10,10,.9,-1,fish,.9\n")
    data = {"class_mapping": {"fish": "fish"}, "sources": [
        {"video": "a.mp4", "annotations": "a.csv", "group": "site-a", "split": "train"}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    assert build_dataset(manifest, tmp_path / "out", include_pseudo=True)["frames"]["train"] == 1
    _video(tmp_path / "b.mp4")
    (tmp_path / "b.csv").write_text("1,b.mp4,0,0,0,10,10,.9,-1,fish,.9\n")
    data["sources"].append({"video": "b.mp4", "annotations": "b.csv", "group": "site-a",
                            "split": "val", "reviewed": True})
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="spans multiple splits"):
        build_dataset(manifest, tmp_path / "bad")


def test_inventory_pairs_job_input_list_with_upload(tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    _video(uploads / "a.mp4")
    run = tmp_path / "jobs" / "job1" / "run1"
    run.mkdir(parents=True)
    (run / "input_list.txt").write_text("/data/uploads/a.mp4\n")
    (run / "viame_tracks.csv").write_text(
        "# exported_by: write_object_track_set_viame_csv\n"
        "1,input_list.txt,1,0,0,10,10,.9,-1,cod,.9\n")
    result = inventory_jobs(tmp_path / "jobs", uploads, tmp_path / "inventory.json")
    assert result["paired"] == 1
    draft = json.loads((tmp_path / "inventory.json").read_text())
    assert draft["class_mapping"] == {"cod": "cod"}
    assert draft["sources"][0]["frame_offset"] == -1
    assert draft["sources"][0]["reviewed"] is False


def test_duplicate_video_runs_cannot_overwrite_each_other(tmp_path: Path):
    _video(tmp_path / "a.mp4")
    (tmp_path / "a.csv").write_text("1,a.mp4,0,0,0,10,10,.9,-1,fish,.9\n")
    manifest = tmp_path / "manifest.json"
    source = {"video": "a.mp4", "annotations": "a.csv", "reviewed": True}
    manifest.write_text(json.dumps({"class_mapping": {"fish": "fish"},
                                    "sources": [source, source]}))
    with pytest.raises(ValueError, match="multiple active annotation files"):
        build_dataset(manifest, tmp_path / "out")


def test_uncertain_pseudo_frame_is_excluded_whole(tmp_path: Path):
    _video(tmp_path / "a.mp4")
    (tmp_path / "a.csv").write_text(
        "1,a.mp4,0,0,0,10,10,.9,-1,fish,.9\n"
        "2,a.mp4,0,20,20,30,30,.2,-1,fish,.9\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"class_mapping": {"fish": "fish"},
                                    "sources": [{"video": "a.mp4", "annotations": "a.csv",
                                                 "split": "train", "reviewed": False}]}))
    report = build_dataset(manifest, tmp_path / "out", include_pseudo=True)
    assert report["frames"]["train"] == 0
    preview = preview_manifest(manifest, tmp_path / "previews", limit=1)
    assert preview["previews"] == 1
    assert len(list((tmp_path / "previews").glob("*.jpg"))) == 1
