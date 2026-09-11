"""Unpaid capture and a restart-safe six-image screening experiment."""
import json
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from app.db.models import utc_now
from app.services.fishial import FishialClient, FishialError
from scripts import fishial_capture, fishial_replay
from scripts.fishial_replay_journal import ReplayJournal


def crops(directory):
    directory.mkdir()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(200, dtype=np.uint8)
    frame[:, :, 1] = 80
    frame[:, :, 2] = 20
    for name in ("crop-1.jpg", "crop-2.jpg"):
        cv2.imwrite(str(directory / name), frame)
    (directory / "manifest.json").write_text(json.dumps({
        "source": "smartbay-cam3", "region": "north_east_atlantic", "preprocess": "none",
        "crops": [{"file": name, "expected_box": [.1, .1, .9, .9]}
                  for name in ("crop-1.jpg", "crop-2.jpg")]}))


@pytest.mark.parametrize("failure", [None, 401, 500, "timeout"])
def test_comparison_applies_modes_and_never_exceeds_six_requests_after_restart(
        test_settings, tmp_path, monkeypatch, failure):
    directory = tmp_path / "experiment"
    crops(directory)
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    monkeypatch.setattr(fishial_replay, "get_settings", lambda: test_settings)
    requests = []

    def handle(request):
        if request.url.path.endswith("auth"):
            return httpx.Response(200, json={"access_token": "token"})
        requests.append(request.content)
        if failure == "timeout":
            raise httpx.ReadTimeout("hidden", request=request)
        if failure:
            return httpx.Response(failure)
        return httpx.Response(200, json={"ok": True, "objects": [
            {"bbox": [20, 10, 180, 90], "species": [{"id": "saithe", "certainty": .8}]},
            {"bbox": [0, 0, 15, 15], "species": [{"id": "gar", "certainty": .99}]}],
            "definitions": {"saithe": {"scientificName": "Pollachius virens"},
                            "gar": {"scientificName": "Lepisosteus oculatus"}}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        monkeypatch.setattr(fishial_replay, "FishialClient", lambda settings: FishialClient(settings, http))
        args = ["--replay", str(directory), "--compare-preprocess", "--max-calls", "6"]
        assert fishial_replay.main(args) == 0
        count = len(requests)
        assert 0 < count <= 6
        assert fishial_replay.main(args) == 0
        assert len(requests) == count
    report = json.loads((directory / "report.json").read_text())
    assert report["calls_reserved"] == count
    assert "Keep fishial_preprocess=none" in report["recommendation"]
    if failure is None:
        assert count == 6
        assert len(set(requests)) == 3  # Two identical fixtures, three different modes.
        assert report["usable_by_mode"] == {"none": 2, "clahe": 2, "white_balance": 2}
        assert all(r["species"] == [["Pollachius virens", .8]] for r in report["results"].values())
    assert test_settings.fishial_preprocess == "none"


def test_journal_preserves_uncertain_reservations_and_refuses_budget_changes(tmp_path):
    path = tmp_path / "journal.sqlite"
    spec = {"max_calls": 1, "max_retries": 2}
    first = ReplayJournal(path, spec)
    assert first.claim("none/a.jpg")
    first.reserve("none/a.jpg", False)
    first.close()  # Simulated crash after reservation, before recording the answer.
    resumed = ReplayJournal(path, spec)
    assert not resumed.claim("none/a.jpg")
    assert resumed.spent == 1
    with pytest.raises(FishialError, match="budget exhausted"):
        resumed.reserve("none/b.jpg", False)
    assert resumed.results()["none/a.jpg"]["reason"] == "interrupted; not replayed"
    resumed.close()
    with pytest.raises(SystemExit, match="changed"):
        ReplayJournal(path, {**spec, "max_calls": 6})


@pytest.mark.parametrize("ceiling", ["0", "-1"])
def test_nonpositive_replay_budget_rejected_before_capture(ceiling, tmp_path):
    with pytest.raises(SystemExit) as exc:
        fishial_replay.main(["--replay", str(tmp_path), "--capture-source", "smartbay-cam3",
                            "--max-calls", ceiling])
    assert exc.value.code == 2


@pytest.mark.parametrize("modes", ["none,none", "none,unknown", "", "none,", "funie_gan,none"])
def test_explicit_modes_reject_duplicates_and_unknown(modes):
    with pytest.raises(SystemExit):
        fishial_replay.parse_modes(modes)


def test_pairs_hold_back_retry_budget_and_freeze_model_provenance(test_settings, tmp_path, monkeypatch):
    from app.services.fish_enhancement import PROVENANCE, EnhancedCrop, enhance_crop
    directory = tmp_path / "pairs"
    crops(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["crops"][0]["ground_truth_scientific_name"] = "Pollachius virens"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    test_settings.fishial_funie_model_path = "fake.pth"
    test_settings.fishial_funie_model_sha256 = "0" * 64
    monkeypatch.setattr(fishial_replay, "get_settings", lambda: test_settings)
    monkeypatch.setattr(fishial_replay, "initialize_enhancement", lambda *args: None)
    selected, requests = [], []
    def fake(image, settings):
        selected.append(settings.fishial_preprocess)
        if settings.fishial_preprocess == "none":
            return enhance_crop(image, settings)
        result = enhance_crop(image, settings.model_copy(update={"fishial_preprocess": "both"}))
        return EnhancedCrop(result.image, {**result.metadata, "mode": "funie_gan",
            "model_id": PROVENANCE["model_id"], "model_sha256": "0" * 64})
    monkeypatch.setattr(fishial_replay, "enhance_crop", fake)
    def handle(request):
        if request.url.path.endswith("auth"):
            return httpx.Response(200, json={"access_token": "token"})
        requests.append(request.content)
        return httpx.Response(500) if len(requests) == 1 else httpx.Response(200, json={
            "ok": True, "objects": [{"species": [{"id": "fish", "certainty": .8}]}],
            "definitions": {"fish": {"scientificName": "Pollachius virens"}}})
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        monkeypatch.setattr(fishial_replay, "FishialClient", lambda settings: FishialClient(settings, http))
        args = ["--replay", str(directory), "--preprocess-modes", "none,funie_gan", "--max-calls", "2"]
        assert fishial_replay.main(args) == 0
        assert fishial_replay.main(args) == 0
        assert selected == ["none", "funie_gan"] and len(requests) == 2
        report = json.loads((directory / "report.json").read_text())
        assert report["labelled_by_mode"] == {"none": {"abstained": 1}, "funie_gan": {"correct": 1}}
        result = report["results"]["funie_gan/crop-1.jpg"]
        assert result["preprocessing"]["model_sha256"] == "0" * 64
        assert (directory / "variants/funie_gan/crop-1.jpg").read_bytes() == requests[1]
        test_settings.fishial_funie_model_sha256 = "1" * 64
        with pytest.raises(SystemExit, match="changed"):
            fishial_replay.main(args)
        assert len(requests) == 2


def test_capture_reuses_ranking_and_retains_clean_targets_without_fishial(
        test_settings, tmp_path, monkeypatch):
    monkeypatch.setattr(FishialClient, "__init__", lambda *a, **k: pytest.fail("Capture constructed a paid client"))
    test_settings.viame_mock = False
    # Capture can stage with Fishial disabled and no credentials in production settings.
    frame = np.full((300, 500, 3), [80, 90, 60], dtype=np.uint8)
    segment = SimpleNamespace(path=tmp_path / "segment.mp4", started_at=utc_now())
    segment.path.touch()

    class Reader:
        index = 0

        def read(self):
            self.index += 1
            return (True, frame.copy()) if self.index <= 12 else (False, None)

        def release(self):
            pass

    class Capture:
        closed = False

        def next_segment(self):
            return segment

        def close(self):
            self.closed = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Reader())
    observations = [SimpleNamespace(frame_number=number, track_id=str(left),
        bbox_left=left, bbox_top=50, bbox_right=left + 100, bbox_bottom=150,
        confidence=.9, class_name="fish") for number in range(12) for left in (40, 280)]
    directory = tmp_path / "capture-test"
    manifest = fishial_capture.collect(directory, "smartbay-cam3", test_settings,
        resolver=lambda *args: "https://camera.invalid/", capture_factory=lambda *args: capture,
        detector=lambda *args: observations)
    assert capture.closed
    assert manifest["capture_api_calls"] == 0
    assert len(manifest["crops"]) == 2
    assert len({entry["track"] for entry in manifest["crops"]}) == 2
    for entry in manifest["crops"]:
        saved = cv2.imread(str(directory / entry["file"]))
        assert saved.shape[0] < frame.shape[0] and saved.shape[1] < frame.shape[1]
        assert np.max(np.ptp(saved, axis=(0, 1))) == 0  # No annotation drawn on the uniform fish crop.
        assert len(entry["expected_box"]) == 4
    # A rerun reads the completed capture without touching the camera.
    assert fishial_capture.collect(directory, "smartbay-cam3", test_settings,
        resolver=lambda *args: pytest.fail("Captured twice")) == manifest
