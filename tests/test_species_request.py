"""Operator-requested identification: selection, spend, state and both footage kinds."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.db.models import (
    FishDetection,
    FishTrack,
    LiveFishDetection,
    LiveFishTrack,
    LiveMonitorSession,
    Video,
    VideoProcessingStatus,
    utc_now,
)
from app.services.fishial import FishialPrediction
from app.services.live_monitor import live_path
from app.services.species_request import (
    IdentificationBusyError,
    LiveRecordingFrameReader,
    VideoFrameReader,
    choose_frames,
    claim,
    footage_for_live_track,
    footage_for_track,
    identification_payload,
    identify,
    shortlist,
)


@pytest.fixture
def enabled_settings(test_settings):
    test_settings.fishial_enabled = True
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    return test_settings


def write_video(path: Path, frames: int = 40, size=(640, 360)) -> None:
    """Frames whose fish patch gets sharper over time, so quality has a clear order."""

    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    try:
        for number in range(frames):
            frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
            # A textured, colourful patch where every detection box sits. More
            # texture later in the clip, so the ranker has something to prefer.
            step = max(2, 20 - number // 2)
            patch = frame[100:260, 200:440]
            patch[::step, :] = (20, 200, 240)
            patch[:, ::step] = (240, 40, 20)
            writer.write(frame)
    finally:
        writer.release()


def stored_video(db, settings, tmp_path, frames=40) -> Video:
    path = tmp_path / f"{uuid.uuid4()}.mp4"
    write_video(path, frames)
    video = Video(original_filename="source.mp4", storage_path=str(path), fps=10.0,
                  width=640, height=360, model_name="m", pipeline_name="p",
                  confidence_threshold=0.5,
                  processing_status=VideoProcessingStatus.COMPLETED.value)
    db.add(video)
    db.commit()
    return video


def stored_track(db, video, frames=range(0, 30, 3), confidence=0.9) -> FishTrack:
    numbers = list(frames)
    # Unique per call: one video may hold several tracks, and the label is unique.
    track = FishTrack(video_id=video.id, viame_track_id=uuid.uuid4().hex[:8],
                      first_frame=numbers[0],
                      last_frame=numbers[-1], detection_count=len(numbers),
                      mean_confidence=confidence, max_confidence=confidence)
    db.add(track)
    db.flush()
    for number in numbers:
        db.add(FishDetection(fish_track_id=track.id, frame_number=number,
                             timestamp_seconds=number / 10, x1=200, y1=100, x2=440, y2=260,
                             confidence=confidence))
    db.commit()
    return track


class StubClient:
    """One image call per identify(), like the adapters the worker injects."""

    def __init__(self, results):
        self.results, self.images, self.boxes = list(results), [], []

    def identify(self, image, expected_box=None):
        self.images.append(image)
        self.boxes.append(expected_box)
        value = self.results[min(len(self.images) - 1, len(self.results) - 1)]
        if isinstance(value, Exception):
            raise value
        name, score = value
        return FishialPrediction([(name, score)], {"stored": True}, 0, 1.0, 1)


# --- Selection ---------------------------------------------------------------


def test_shortlist_offers_every_window_before_repeating_one(enabled_settings):
    from app.services.species_request import Observation

    # Ten frames at 10 fps is one second, which a 0.6s separation splits into two
    # windows. Both windows must be offered before either is offered twice.
    observations = [Observation(number, (200, 100, 440, 260), 0.9) for number in range(10)]
    chosen = shortlist(observations, 10.0, enabled_settings, wanted=5)
    assert sorted(window for window, _ in chosen[:2]) == [0, 1]
    # Frames beyond the independent ones still follow, so a request for five frames
    # of a briefly-seen fish is not silently cut to two.
    assert len(chosen) > 2


def test_shortlist_rejects_frames_below_the_safety_floors(enabled_settings):
    from app.services.species_request import Observation

    enabled_settings.fishial_min_crop_pixels = 40
    observations = [
        Observation(0, (0, 0, 10, 10), 0.9),        # too small
        Observation(20, (200, 100, 440, 260), 0.1),  # below the confidence floor
        Observation(40, (200, 100, 440, 260), 0.9),  # usable
    ]
    chosen = shortlist(observations, 10.0, enabled_settings, wanted=5)
    assert [observation.frame_number for _, observation in chosen] == [40]


def test_shortlist_ranks_by_crop_size_then_returns_a_wider_list_than_asked(enabled_settings):
    from app.services.species_request import Observation

    enabled_settings.fishial_min_frame_separation_seconds = 0.1
    observations = [Observation(number * 2, (0, 0, 60 + number * 40, 60 + number * 40), 0.9)
                    for number in range(6)]
    chosen = shortlist(observations, 10.0, enabled_settings, wanted=1)
    # Biggest crop first, and three candidates decoded for every frame bought.
    assert chosen[0][1].frame_number == 10
    assert len(chosen) == 3


def test_choose_frames_reads_the_video_and_keeps_the_best_measured_crops(
        db_session_factory, enabled_settings, tmp_path):
    import cv2

    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        footage = footage_for_track(db, track, enabled_settings)
        chosen = choose_frames(footage, enabled_settings, 3, cv2)

    assert len(chosen) == 3
    # Ranked best first, each from a different separation window, each a real JPEG
    # of the crop rather than the whole frame.
    assert [entry.score for entry in chosen] == sorted((e.score for e in chosen), reverse=True)
    assert len({entry.window for entry in chosen}) == 3
    for entry in chosen:
        assert entry.image[:2] == b"\xff\xd8"
        assert 0 <= entry.expected_box[0] < entry.expected_box[2] <= 1
        assert entry.quality["short_side"] == pytest.approx(160.0)


def test_choose_frames_fills_the_request_after_covering_every_moment(
        db_session_factory, enabled_settings, tmp_path):
    import cv2

    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        # Eight frames at 10 fps: 0.8s of footage, so two 0.6s windows.
        track = stored_track(db, video, frames=range(8))
        footage = footage_for_track(db, track, enabled_settings)
        chosen = choose_frames(footage, enabled_settings, 6, cv2)

    # A fish seen for two moments still gets the six frames that were paid for -
    # weaker evidence than six moments, but far better evidence than two frames.
    assert len(chosen) == 6
    # Both moments are covered before either repeats, so the strongest independent
    # evidence is always at the front.
    assert {entry.window for entry in chosen[:2]} == {0, 1}


def test_video_frame_reader_returns_the_frames_it_was_asked_for(enabled_settings, tmp_path):
    import cv2

    path = tmp_path / "source.mp4"
    write_video(path, frames=30)
    reader = VideoFrameReader(path, cv2)
    try:
        got = [number for number, _ in reader.read([2, 11, 25])]
    finally:
        reader.close()
    assert got == [2, 11, 25]


# --- Claiming ----------------------------------------------------------------


def test_claim_refuses_while_another_request_holds_the_fish(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        claim(db, track, 4, enabled_settings)
        assert track.fishial_state == "submitted"
        assert track.fishial_votes["request"]["frames"] == 4
        with pytest.raises(IdentificationBusyError):
            claim(db, track, 4, enabled_settings)


def test_claim_takes_over_a_request_whose_process_died(
        db_session_factory, enabled_settings, tmp_path):
    from datetime import timedelta

    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        claim(db, track, 4, enabled_settings)
        track.fishial_requested_at = utc_now() - timedelta(
            seconds=enabled_settings.fishial_request_stale_seconds + 1)
        db.commit()
        claim(db, track, 2, enabled_settings)
        assert track.fishial_votes["request"]["frames"] == 2


def test_claim_clears_a_previous_answer_so_votes_are_never_mixed(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        track.fishial_state = "identified"
        track.fishial_species = "Gadus morhua"
        track.fishial_species_confidence = 0.9
        track.fishial_votes_json = json.dumps({"frames": [{"species": "Gadus morhua",
                                                           "voted": True, "score": 0.9,
                                                           "attempts": [{"retry": False}]}]})
        db.commit()
        claim(db, track, 3, enabled_settings)
        assert track.fishial_state == "submitted"
        assert track.fishial_species is None
        assert track.fishial_votes["frames"] == []


# --- Identifying -------------------------------------------------------------


def test_identify_a_library_track_reaches_consensus_and_stores_the_name(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        claim(db, track, 3, enabled_settings)
        client = StubClient([("Gadus morhua", 0.9)])
        identify(db, track, enabled_settings, client=client)

        assert track.fishial_state == "identified"
        assert track.fishial_species == "Gadus morhua"
        assert track.fishial_species_confidence == pytest.approx(0.9)
        assert track.fishial_frames_used == 3
        # Two agreeing frames settle it under the request's own consensus rule, so
        # the third is chosen but never bought.
        assert len(client.images) == 2
        assert all(image[:2] == b"\xff\xd8" for image in client.images)
        assert all(box is not None for box in client.boxes)
        payload = identification_payload(track)
        assert payload["frames_requested"] == 3
        assert payload["frames_selected"] == 3
        assert payload["frames_submitted"] == 2
        assert payload["windows"] == 3
        assert payload["tally"] == {"Gadus morhua": 2}


def test_identify_records_disagreement_as_review_rather_than_a_name(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        claim(db, track, 3, enabled_settings)
        identify(db, track, enabled_settings,
                 client=StubClient([("Gadus morhua", 0.9), ("Salmo salar", 0.9),
                                    ("Molva molva", 0.9)]))

        assert track.fishial_state == "review_required"
        assert track.fishial_species is None
        payload = identification_payload(track)
        # Under the request's relaxed rule two of three frames could still agree, so
        # the third is worth buying; only then is the disagreement settled.
        assert payload["frames_submitted"] == 3
        assert {c["species"] for c in payload["diagnostics"]["candidates"]} == {
            "Gadus morhua", "Salmo salar", "Molva molva"}


def test_identify_stops_paying_once_the_answer_can_no_longer_change(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        claim(db, track, 5, enabled_settings)
        client = StubClient([("Gadus morhua", 0.9)])
        identify(db, track, enabled_settings, client=client)

        payload = identification_payload(track)
        assert track.fishial_state == "identified"
        # Five frames were selected, but once three have agreed the remaining two
        # cannot change the answer, so they are not bought.
        assert payload["frames_selected"] == 5
        assert len(client.images) == 3
        assert payload["calls_saved"] == 2
        assert payload["stopped_early"] == "decided"


def test_identify_flags_a_name_that_is_implausible_here_rather_than_hiding_it(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        session = LiveMonitorSession(source_url="https://camera.example/",
                                     source_key="smartbay-cam3",
                                     species_id_region="north_east_atlantic")
        db.add(session)
        db.commit()
        video = stored_video(db, enabled_settings, tmp_path)
        video.source_session_id = session.id
        db.commit()
        track = stored_track(db, video)
        claim(db, track, 3, enabled_settings)
        identify(db, track, enabled_settings,
                 client=StubClient([("Sparisoma aurofrenatum", 0.95)]))

        # A Caribbean parrotfish on a Galway camera is reported WITH the warning, so
        # the operator can judge it. The automatic pass still drops such a name.
        assert track.fishial_state == "identified"
        assert track.fishial_species == "Sparisoma aurofrenatum"
        payload = identification_payload(track)
        assert payload["implausible_for_region"] == "Sparisoma aurofrenatum"


def test_identify_reports_footage_that_is_gone_without_claiming_a_species(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, enabled_settings, tmp_path)
        track = stored_track(db, video)
        claim(db, track, 3, enabled_settings)
        Path(video.storage_path).unlink()
        identify(db, track, enabled_settings, client=StubClient([("Gadus morhua", 0.9)]))

        assert track.fishial_state == "error"
        assert track.fishial_species is None
        assert "no longer available" in track.fishial_votes["reason"]


# --- Live footage -------------------------------------------------------------


def live_session_with_recording(db, settings, tmp_path, chunks=3, frames_per_chunk=10):
    """A running session whose retained chunks hold the frames its tracker analyzed."""

    session = LiveMonitorSession(source_url="https://camera.example/", status="running")
    db.add(session)
    db.commit()
    directory = live_path(settings, str(session.id), "recording")
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for index in range(chunks):
        write_video(directory / f"{index:08d}.mp4", frames_per_chunk)
        parts.append({"start_frame": index * frames_per_chunk, "frames": frames_per_chunk,
                      "width": 640, "height": 360, "scale": 1.0,
                      "offset_x": 0, "offset_y": 0})
    (directory / "manifest.json").write_text(json.dumps(
        {"session_id": str(session.id), "fps": settings.live_fps, "width": 640,
         "height": 360, "truncated": False, "parts": parts}), encoding="utf-8")
    return session


def live_track_with_detections(db, session, numbers) -> LiveFishTrack:
    track = LiveFishTrack(session_id=session.id, status="active", first_seen_at=utc_now(),
                          last_seen_at=utc_now(), x1=200, y1=100, x2=440, y2=260,
                          detection_count=len(numbers), max_confidence=0.9,
                          mean_confidence=0.9)
    db.add(track)
    db.flush()
    for number in numbers:
        db.add(LiveFishDetection(track_id=track.id, observed_at=utc_now(), frame_number=number,
                                 confidence=0.9, x1=200, y1=100, x2=440, y2=260))
    db.commit()
    return track


def test_live_recording_reader_maps_session_frames_onto_the_right_chunk(
        db_session_factory, enabled_settings, tmp_path):
    import cv2

    with db_session_factory() as db:
        session = live_session_with_recording(db, enabled_settings, tmp_path)
    directory = live_path(enabled_settings, str(session.id), "recording")
    reader = LiveRecordingFrameReader(directory, cv2)
    try:
        assert reader.locate(0) == 0
        assert reader.locate(15) == 1
        assert reader.locate(29) == 2
        assert reader.locate(30) is None  # past the retained footage
        assert [number for number, _ in reader.read([3, 15, 28])] == [3, 15, 28]
    finally:
        reader.close()


def test_identify_a_fish_in_a_running_session_from_retained_footage(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        session = live_session_with_recording(db, enabled_settings, tmp_path)
        # Automatic identification is off for this session; an operator may still ask.
        assert session.species_id_enabled is False
        track = live_track_with_detections(db, session, range(0, 30, 4))
        claim(db, track, 3, enabled_settings)
        client = StubClient([("Gadus morhua", 0.92)])
        identify(db, track, enabled_settings, client=client)

        assert track.fishial_state == "identified"
        assert track.fishial_species == "Gadus morhua"
        # Three frames chosen, two bought: the second agreeing answer settles it.
        assert len(client.images) == 2
        db.refresh(session)
        # Operator spend is counted apart from the automatic budget it must not eat.
        assert session.species_id_manual_api_calls == 2
        assert session.species_id_api_calls == 0


def test_live_footage_skips_observations_no_retained_chunk_holds(
        db_session_factory, enabled_settings, tmp_path):
    with db_session_factory() as db:
        session = live_session_with_recording(db, enabled_settings, tmp_path, chunks=2)
        track = live_track_with_detections(db, session, [0, 5, 19, 40, 99])
        footage = footage_for_live_track(db, track, enabled_settings)

    # 40 and 99 were analyzed after the recording stopped retaining chunks.
    assert [observation.frame_number for observation in footage.observations] == [0, 5, 19]


# --- API ----------------------------------------------------------------------


@pytest.fixture
def scheduled(monkeypatch):
    """Capture the background task instead of letting it reach the provider."""

    import app.api.live
    import app.api.tracks

    tasks = []
    monkeypatch.setattr(app.api.tracks, "run_request",
                        lambda *args, **kwargs: tasks.append(args))
    monkeypatch.setattr(app.api.live, "run_request",
                        lambda *args, **kwargs: tasks.append(args))
    return tasks


def test_identify_endpoint_claims_the_fish_and_reports_the_limits(
        client, db_session_factory, test_settings, tmp_path, scheduled):
    test_settings.fishial_enabled = True
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video)
        track_id = str(track.id)

    status = client.get("/system/status").json()["species_identification"]
    assert status == {"available": True, "max_frames": 12, "default_frames": 5}

    response = client.post(f"/tracks/{track_id}/identify", json={"frames": 40})
    assert response.status_code == 202
    # Clamped to the deployment ceiling rather than refused.
    assert response.json()["frames_requested"] == 12
    assert response.json()["state"] == "submitted"
    # The work is handed to a background task, so the operator is not made to wait
    # on footage decoding and a classifier round trip.
    assert [args[1:3] for args in scheduled] == [(FishTrack, uuid.UUID(track_id))]

    with db_session_factory() as db:
        assert db.get(FishTrack, uuid.UUID(track_id)).fishial_state == "submitted"
    assert client.post(f"/tracks/{track_id}/identify").status_code == 409
    assert client.get(f"/tracks/{track_id}/identification").json()["state"] == "submitted"


def test_identify_endpoint_refuses_when_fishial_is_not_configured(
        client, db_session_factory, test_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video)
        track_id = str(track.id)

    assert client.get("/system/status").json()["species_identification"]["available"] is False
    response = client.post(f"/tracks/{track_id}/identify")
    assert response.status_code == 409
    assert "not configured" in response.json()["detail"]


def test_live_identify_endpoint_claims_one_chosen_fish(
        client, db_session_factory, test_settings, tmp_path, scheduled):
    test_settings.fishial_enabled = True
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    with db_session_factory() as db:
        session = live_session_with_recording(db, test_settings, tmp_path)
        track = live_track_with_detections(db, session, range(0, 30, 4))
        track_id = str(track.id)

    response = client.post(f"/live/tracks/{track_id}/identify", json={"frames": 3})
    assert response.status_code == 202
    assert response.json()["identification"]["frames_requested"] == 3
    assert response.json()["fishial_state"] == "submitted"
    assert [args[1:3] for args in scheduled] == [(LiveFishTrack, uuid.UUID(track_id))]
    assert client.post(f"/live/tracks/{uuid.uuid4()}/identify").status_code == 404
