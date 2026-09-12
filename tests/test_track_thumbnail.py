"""Per-track preview crops, and the relaxations that keep requested crops askable."""

from __future__ import annotations

import uuid

import pytest
from pydantic import SecretStr
from test_species_request import stored_track, stored_video

from app.db.models import FishTrack, Video
from app.services.species_request import request_settings, shortlist
from app.services.track_thumbnail import (
    THUMBNAIL_SIZE,
    best_detection,
    existing_thumbnail,
    generate_video_thumbnails,
    thumbnail_path,
)


def test_best_detection_prefers_the_biggest_clearest_sighting(db_session_factory, test_settings,
                                                              tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video, frames=[0, 5, 10])
        detections = sorted(track.detections, key=lambda d: d.frame_number)
        # Frame 5 shows the fish twice as wide as the others at the same confidence.
        detections[1].x2 = detections[1].x1 + 400
        detections[1].y2 = detections[1].y1 + 400
        db.commit()
        assert best_detection(detections, test_settings).frame_number == 5


def test_best_detection_ignores_a_degenerate_box(test_settings):
    class Box:
        def __init__(self, frame_number, x1, y1, x2, y2, confidence=0.9):
            self.frame_number = frame_number
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
            self.confidence = confidence

    assert best_detection([Box(0, 10, 10, 10, 10)], test_settings) is None
    assert best_detection([], test_settings) is None


def test_thumbnails_are_generated_for_every_track_in_one_pass(db_session_factory, test_settings,
                                                              tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        tracks = [stored_track(db, video, frames=range(start, start + 9))
                  for start in (0, 12, 24)]
        found = generate_video_thumbnails(db, video, test_settings)

        assert set(found) == {track.id for track in tracks}
        for track in tracks:
            path = thumbnail_path(video.id, track.id, test_settings)
            assert path.is_file()
            assert path.read_bytes()[:2] == b"\xff\xd8"
            assert existing_thumbnail(track, test_settings) == path


def test_thumbnail_is_a_square_crop_of_the_fish_not_the_whole_frame(
        db_session_factory, test_settings, tmp_path):
    import cv2

    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video)
        generate_video_thumbnails(db, video, test_settings)
        image = cv2.imread(str(thumbnail_path(video.id, track.id, test_settings)))

    # A fixed square keeps the table column even whatever shape the fish was.
    assert image.shape[:2] == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    # The source frame is flat grey outside the fish patch; a thumbnail that had
    # captured the whole frame would be almost entirely that one colour.
    assert image.std() > 5


def test_thumbnails_are_reused_until_a_refresh_is_asked_for(db_session_factory, test_settings,
                                                            tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video)
        generate_video_thumbnails(db, video, test_settings)
        path = thumbnail_path(video.id, track.id, test_settings)
        path.write_bytes(b"\xff\xd8stale")

        generate_video_thumbnails(db, video, test_settings)
        assert path.read_bytes() == b"\xff\xd8stale"

        generate_video_thumbnails(db, video, test_settings, refresh=True)
        assert path.read_bytes() != b"\xff\xd8stale"


def test_missing_footage_leaves_a_blank_cell_rather_than_failing(db_session_factory,
                                                                 test_settings, tmp_path):
    from pathlib import Path

    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        stored_track(db, video)
        Path(video.storage_path).unlink()
        # A table that will not load is a far worse outcome than a missing preview.
        assert generate_video_thumbnails(db, video, test_settings) == {}


def test_thumbnail_endpoints_generate_then_serve(client, db_session_factory, test_settings,
                                                 tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video)
        video_id, track_id = str(video.id), str(track.id)

    # Nothing is generated until the video's own endpoint is asked.
    assert client.get(f"/tracks/{track_id}/thumbnail").status_code == 404

    rows = client.get(f"/videos/{video_id}/thumbnails").json()
    assert rows == [{"track_id": track_id, "url": f"/tracks/{track_id}/thumbnail"}]

    response = client.get(f"/tracks/{track_id}/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # A track id is replaced when processing reruns, so its crop never changes.
    assert "immutable" in response.headers["cache-control"]
    assert response.content[:2] == b"\xff\xd8"

    assert client.get(f"/videos/{uuid.uuid4()}/thumbnails").status_code == 404
    assert client.get(f"/tracks/{uuid.uuid4()}/thumbnail").status_code == 404


# --- The relaxations ----------------------------------------------------------


@pytest.fixture
def enabled_settings(test_settings):
    test_settings.fishial_enabled = True
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    return test_settings


def test_request_settings_relaxes_the_floors_the_automatic_pass_enforces(enabled_settings):
    relaxed = request_settings(enabled_settings)

    # A request is one operator authorising one spend, so it is not gated by floors
    # calibrated to protect an unattended budget.
    assert relaxed.fishial_min_frame_confidence < enabled_settings.fishial_min_frame_confidence
    assert relaxed.fishial_min_crop_pixels < enabled_settings.fishial_min_crop_pixels
    assert relaxed.fishial_min_votes < enabled_settings.fishial_min_votes
    assert relaxed.fishial_vote_ratio <= enabled_settings.fishial_vote_ratio
    assert relaxed.fishial_min_species_score < enabled_settings.fishial_min_species_score
    assert relaxed.fishial_region_filter_enabled is False
    # Nothing else moves: the automatic pass's own settings are untouched.
    assert relaxed.fishial_quality_weights == enabled_settings.fishial_quality_weights
    assert relaxed.fishial_min_frame_separation_seconds == (
        enabled_settings.fishial_min_frame_separation_seconds)
    assert enabled_settings.fishial_min_votes == 3


def test_a_small_faint_fish_the_automatic_pass_rejects_is_still_askable(enabled_settings):
    from app.services.species_request import Observation

    # 30 px across at 0.45 confidence: below both automatic floors, and exactly the
    # kind of fish an operator points at and asks "what is that?".
    observations = [Observation(number * 8, (100, 100, 130, 130), 0.45) for number in range(4)]
    assert shortlist(observations, 10.0, enabled_settings, wanted=3) == []
    assert shortlist(observations, 10.0, request_settings(enabled_settings), wanted=3)


def test_an_edge_cropped_fish_is_still_askable(enabled_settings):
    import cv2
    import numpy as np

    from app.services.species_quality import is_clear_frame

    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    # Flush against the left edge: truncated, but often still nameable.
    box = (0.0, 100.0, 60.0, 260.0)
    assert not is_clear_frame(cv2, frame, box, 0.9, enabled_settings)
    assert is_clear_frame(cv2, frame, box, 0.9, request_settings(enabled_settings))


def test_video_and_track_rows_are_unaffected_by_thumbnail_generation(
        db_session_factory, test_settings, tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video)
        before = (track.review_state, track.species, track.fishial_state)
        generate_video_thumbnails(db, video, test_settings)
        db.refresh(track)
        db.refresh(video)
        # Previews are derived media: they read stored detections and change nothing.
        assert (track.review_state, track.species, track.fishial_state) == before
        assert db.get(Video, video.id).processing_status == "completed"
        assert db.get(FishTrack, track.id) is not None


def test_thumbnails_do_not_write_outside_the_clip_cache(db_session_factory, test_settings,
                                                        tmp_path):
    with db_session_factory() as db:
        video = stored_video(db, test_settings, tmp_path)
        track = stored_track(db, video)
        path = thumbnail_path(video.id, track.id, test_settings)
    assert path.is_relative_to(test_settings.output_root.resolve())
    assert str(video.id) in path.parts
