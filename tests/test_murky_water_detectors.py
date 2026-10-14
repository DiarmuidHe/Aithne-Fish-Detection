import json
from pathlib import Path

import pytest

import scripts.external_fish as external_fish
from app.services.external_datasets import leakage_guard, sha256
from app.services.viame_parser import parse_viame_csv
from scripts.fish_model import combined_data_yaml
from scripts.murk_augmentation import (
    TEST_RANGES,
    TRAIN_RANGES,
    augment_train,
    degrade,
    sample_params,
    synthetic_test,
)


def _dataset(root: Path) -> Path:
    import cv2
    import numpy as np

    rows = []
    for split, count in (("train", 3), ("val", 1), ("test", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            stem = f"{split}{index}"
            image = np.random.default_rng(index).integers(0, 255, (54, 96, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), image)
            (root / "labels" / split / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            rows.append({"split": split, "source_sequence": f"rec-{split}", "source_frame": index + 1,
                         "output_image": f"images/{split}/{stem}.jpg",
                         "output_label": f"labels/{split}/{stem}.txt", "reviewed": True,
                         "reviewed_for_smartbay": False, "unverified_empty": False})
    (root / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "classes.json").write_text(json.dumps({"0": "fish"}))
    (root / "report.json").write_text("{}")
    return root


def test_degradation_is_seeded_per_image():
    import numpy as np

    image = np.random.default_rng(0).integers(0, 255, (54, 96, 3), dtype=np.uint8)
    first, again = sample_params(7, "frame-a", TRAIN_RANGES), sample_params(7, "frame-a", TRAIN_RANGES)
    assert first == again and first != sample_params(8, "frame-a", TRAIN_RANGES)
    assert np.array_equal(degrade(image, first), degrade(image, again))
    for name, (low, high) in TRAIN_RANGES.items():
        assert low <= first[name] <= high


def test_test_ranges_are_disjoint_from_training():
    for name, (low, high) in TEST_RANGES.items():
        if name == "transmission_gradient":
            continue
        train_low, train_high = TRAIN_RANGES[name]
        assert high <= train_low or low >= train_high, name


def test_augment_train_copies_only_train_and_is_reproducible(tmp_path):
    dataset = _dataset(tmp_path / "base")
    report = augment_train(dataset, tmp_path / "murk-a", seed=3)
    augment_train(dataset, tmp_path / "murk-b", seed=3)
    assert report["copies"] == 3 and report["splits_augmented"] == ["train"]
    assert sorted(p.parent.name for p in (tmp_path / "murk-a" / "images").rglob("*.jpg")) == ["train"] * 3
    rows = [json.loads(line) for line in (tmp_path / "murk-a" / "metadata.jsonl").read_text().splitlines()]
    assert {r["split"] for r in rows} == {"train"} and all(r["synthetic"] for r in rows)
    assert {r["source_sequence"] for r in rows} == {"rec-train"}
    for a in (tmp_path / "murk-a" / "images" / "train").glob("*.jpg"):
        assert sha256(a) == sha256(tmp_path / "murk-b" / "images" / "train" / a.name)
    assert (tmp_path / "murk-a" / "contact-sheet.jpg").is_file()


def test_synthetic_test_set_is_separate_and_labelled(tmp_path):
    dataset = _dataset(tmp_path / "base")
    report = synthetic_test(dataset, tmp_path / "dark", seed=11)
    assert report["frames"] == {"train": 0, "val": 0, "test": 2}
    assert not (tmp_path / "dark" / "images" / "train").exists()
    rows = [json.loads(line) for line in (tmp_path / "dark" / "metadata.jsonl").read_text().splitlines()]
    assert all(r["purpose"] == "synthetic robustness evaluation only" for r in rows)


def test_extra_training_folder_must_be_train_only(tmp_path):
    dataset = _dataset(tmp_path / "base")
    augment_train(dataset, tmp_path / "murk", seed=1)
    data = combined_data_yaml(dataset.resolve(), [(tmp_path / "murk").resolve()], tmp_path / "data.yaml")
    text = data.read_text()
    assert "murk/images/train" in text and "test: images/test" in text
    half = combined_data_yaml(dataset.resolve(), [(tmp_path / "murk").resolve()], tmp_path / "half.yaml", .67)
    listing = (tmp_path / "half.murk.txt").read_text().splitlines()
    assert len(listing) == 2 and (tmp_path / "half.murk.txt").as_posix() in half.read_text()
    rows = (tmp_path / "murk" / "metadata.jsonl").read_text().splitlines()
    leaked = json.loads(rows[0]) | {"split": "test"}
    (tmp_path / "murk" / "metadata.jsonl").write_text("\n".join([json.dumps(leaked), *rows[1:]]))
    with pytest.raises(SystemExit, match="only train rows"):
        combined_data_yaml(dataset.resolve(), [(tmp_path / "murk").resolve()], tmp_path / "x.yaml")


def test_validation_workers_cap_only_the_validation_loader(monkeypatch):
    import types

    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect import DetectionTrainer

    from scripts.fish_model import _validation_workers_trainer

    seen = []

    def fake_loader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        seen.append((mode, self.args.workers))  # Ultralytics doubles this for validation

    monkeypatch.setattr(DetectionTrainer, "get_dataloader", fake_loader)
    trainer = object.__new__(_validation_workers_trainer(2))
    trainer.args = types.SimpleNamespace(workers=4)
    trainer.get_dataloader("train-images", mode="train")
    trainer.get_dataloader("val-images", mode="val")
    assert seen == [("train", 4), ("val", 1)] and trainer.args.workers == 4


def test_leakage_guard_excludes_training_overlap_and_undetermined():
    overlap = {"s1": {"status": "overlap", "recording": "rec-train"},
               "s2": {"status": "overlap", "recording": "rec-test"},
               "s3": {"status": "none", "recording": None},
               "s4": {"status": "undetermined", "recording": None}}
    result = leakage_guard(overlap, {"rec-train", "rec-val"}, ["s1", "s2", "s3", "s4", "s5"])
    assert result == {"scorable": ["s2", "s3"], "excluded_training_overlap": {"s1": "rec-train"},
                      "undetermined": ["s4", "s5"]}


def test_fetch_megalodon_verifies_hash_and_cleans_up(tmp_path, monkeypatch):
    payload = b"not really a checkpoint"

    def fake_download(url, target):
        assert url == external_fish.MEGALODON_URL and external_fish.MEGALODON_COMMIT in url
        Path(target).write_bytes(payload)

    monkeypatch.setattr(external_fish.urllib.request, "urlretrieve", fake_download)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        external_fish.fetch_megalodon(tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    import hashlib
    monkeypatch.setattr(external_fish, "MEGALODON_SHA256", hashlib.sha256(payload).hexdigest())
    report = external_fish.fetch_megalodon(tmp_path / "good")
    provenance = json.loads((tmp_path / "good" / "provenance.json").read_text())
    assert report["commit"] == provenance["commit"] and provenance["class_mapping"] == {"0": "object"}
    assert "CC BY 4.0" in provenance["license"]


def test_infer_megalodon_refuses_unverified_weights_before_loading(tmp_path):
    weights = tmp_path / "megalodon.pt"
    weights.write_bytes(b"pickle that must never be loaded")
    with pytest.raises(ValueError, match="refusing to load"):
        external_fish.infer_megalodon(weights, tmp_path / "video.mp4", tmp_path / "out")
    assert not (tmp_path / "out").exists()


class _Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def test_megalodon_output_becomes_object_candidate_rows(tmp_path):
    import cv2
    import numpy as np

    video = tmp_path / "clip.mkv"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 4, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    boxes = type("Boxes", (), {"xyxy": _Tensor([[-5.0, 2.0, 30.0, 20.0], [10.0, 10.0, 10.0, 12.0]]),
                               "conf": _Tensor([.8, .4])})()
    rows = external_fish._megalodon_rows(boxes)
    assert rows == [(-5.0, 2.0, 30.0, 20.0, .8), (10.0, 10.0, 10.0, 12.0, .4)]
    frames, written = external_fish._candidate_csv(video, tmp_path / "out", lambda frame: rows,
                                                    max_frames=2, stride=1, prefix="megalodon",
                                                    label="object")
    assert (frames, written) == (2, 2)  # The zero-width box is dropped on both frames.
    parsed = parse_viame_csv(tmp_path / "out" / "candidates.csv", strict=True)
    assert {d.class_name for d in parsed.detections} == {"object"}
    assert [(d.track_id, d.frame_number, d.bbox_left) for d in parsed.detections] == [
        ("megalodon-0-0", 0, 0.0), ("megalodon-1-0", 1, 0.0)]


def test_temporal_alignment_finds_a_subclip_and_rejects_unrelated_footage():
    import numpy as np

    from app.services.external_datasets import best_alignment, classify_alignment

    rng = np.random.default_rng(5)
    background = rng.uniform(0, 255, (36, 64)).astype("float32")
    # A static scene: only small moving patches and noise change between frames.
    recording = np.stack([background + rng.normal(0, 3, (36, 64)).astype("float32") for _ in range(60)])
    other = np.stack([background + rng.normal(0, 3, (36, 64)).astype("float32") for _ in range(60)])
    clip = [frame + rng.normal(0, .5, frame.shape).astype("float32") for frame in recording[20:45]]
    found = classify_alignment(best_alignment(clip, {"rec-a": recording, "rec-b": other}))
    assert (found["status"], found["recording"], found["offset"]) == ("overlap", "rec-a", 20)
    unrelated = [background + rng.normal(0, 3, (36, 64)).astype("float32") for _ in range(25)]
    assert classify_alignment(best_alignment(unrelated, {"rec-a": recording}))["status"] != "overlap"


def test_analysed_video_frames_are_inferred_from_the_downsampler_phase():
    from scripts.murky_water_detector import infer_analysed

    probe = [i for i in range(30) if i % 3 != 2]  # VIAME's 10-of-15 FPS image-list pattern
    shifted = {i for i in range(30) if i % 3 != 0}  # The same pattern one frame later
    observed = shifted - {4, 10}  # Two analysed frames happened to have no detections
    result = infer_analysed(observed, probe, 30)
    assert result["frames"] == sorted(shifted) and result["shift"] == 1
    assert result["unobserved"] == 2 and not result["ambiguous"]
    # The video path occasionally breaks the pattern; a full set at the right rate is accepted.
    irregular = {i for i in range(30) if i % 3 != 1} - {12} | {13}
    assert infer_analysed(irregular, probe, 30)["frames"] == sorted(irregular)
    with pytest.raises(ValueError, match="no downsampler phase"):
        infer_analysed(set(range(30)), probe, 30)


def test_dill_stand_in_resolves_only_allow_listed_builtin_types():
    import importlib.util
    import sys

    if importlib.util.find_spec("dill") is not None:
        pytest.skip("real dill installed")
    with external_fish._dill_types_only():
        from dill._dill import _load_type
        assert _load_type("set") is set
        with pytest.raises(ValueError, match="allow-list"):
            _load_type("FunctionType")
    assert "dill" not in sys.modules
