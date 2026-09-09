from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from app.config import Settings
from app.db.models import FishDetection, FishTrack, Video
from app.services.track_clip import (
    NoDetectionsForClipError,
    _build_plan,
    clip_paths,
    generate_track_clip,
    video_clip_directory,
)
from app.workers.processing_worker import process_job


def _write_test_video(path: Path, frame_count: int = 40) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360))
    assert writer.isOpened()
    try:
        for frame_number in range(frame_count):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = 40 + frame_number % 30
            frame[:, :, 1] = 60
            frame[:, :, 2] = 90
            writer.write(frame)
    finally:
        writer.release()


def _completed_video(client, db_session_factory, test_settings, tmp_path: Path) -> UUID:
    source_path = tmp_path / "source.mp4"
    _write_test_video(source_path)
    with source_path.open("rb") as source_file:
        response = client.post("/videos", files={"file": ("tiny.mp4", source_file, "video/mp4")})
    assert response.status_code == 201
    video_id = UUID(response.json()["id"])

    job_id = UUID(client.post(f"/videos/{video_id}/process").json()["id"])
    with db_session_factory() as db:
        video = db.get(Video, video_id)
        video.fps = 10.0
        db.commit()
    process_job(job_id=job_id, settings=test_settings, session_factory=db_session_factory)
    return video_id


class _StubDetection:
    def __init__(self, frame_number, x1, y1, x2, y2, confidence=0.9):
        self.frame_number = frame_number
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence = confidence


class _StubTrack:
    id = UUID("11111111-2222-3333-4444-555555555555")
    viame_track_id = "7"
    species = "fish"
    detection_count = 2
    max_confidence = 0.91


def test_clip_plan_pads_crops_and_follows_the_fish_between_reported_frames():
    settings = Settings(clip_padding_seconds=0.5, clip_zoom_margin=2.0, clip_min_crop_pixels=64)
    detections = [
        _StubDetection(10, 100.0, 100.0, 140.0, 130.0),
        _StubDetection(14, 200.0, 160.0, 240.0, 190.0),
    ]

    plan = _build_plan(_StubTrack(), detections, 640, 360, 10.0, settings)

    # 0.5 s of padding at 10 fps on both sides of the reported detections.
    assert (plan.start_frame, plan.end_frame) == (5, 19)
    # Widest box is 40 px, doubled by the zoom margin, floored by the minimum crop.
    assert (plan.crop_width, plan.crop_height) == (80, 64)
    assert plan.width % 2 == 0 and plan.height % 2 == 0
    assert plan.width / plan.crop_width == pytest.approx(plan.height / plan.crop_height)

    # Frame 12 sits halfway between the two detections, so the crop is centred there.
    left, top = plan.origins[12]
    assert left + plan.crop_width / 2 == pytest.approx(170.0, abs=1.0)
    assert top + plan.crop_height / 2 == pytest.approx(145.0, abs=1.0)
    # Padding frames hold the first and last known position.
    assert plan.origins[5] == plan.origins[10]
    assert plan.origins[19] == plan.origins[14]
    assert set(plan.boxes) == {10, 14}


def test_clip_plan_keeps_the_crop_inside_the_frame_at_the_edges():
    settings = Settings(clip_padding_seconds=0.0, clip_zoom_margin=3.0, clip_min_crop_pixels=64)
    detections = [_StubDetection(4, 600.0, 10.0, 636.0, 40.0)]

    plan = _build_plan(_StubTrack(), detections, 640, 360, 10.0, settings)

    left, top = plan.origins[4]
    assert left >= 0 and top >= 0
    assert left + plan.crop_width <= 640
    assert top + plan.crop_height <= 360


def test_fish_clips_endpoint_renders_one_clip_per_accepted_track(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)

    response = client.post(f"/videos/{video_id}/fish-clips")

    assert response.status_code == 200
    clips = response.json()
    assert [clip["viame_track_id"] for clip in clips] == ["1", "3"]
    for clip in clips:
        assert clip["cached"] is False
        assert clip["media_type"] == "video/mp4"
        assert clip["size_bytes"] > 0
        assert clip["frame_count"] > 0
        assert clip["width"] % 2 == 0 and clip["height"] % 2 == 0
        assert clip["duration_seconds"] > 0
        assert clip["url"] == f"/tracks/{clip['track_id']}/clip"

    # Track 1 runs over frames 0-2 with 0.6 s of padding at 10 fps.
    first = clips[0]
    assert first["start_seconds"] == pytest.approx(0.0)
    assert first["end_seconds"] == pytest.approx(0.9, abs=0.15)
    assert first["detection_count"] == 3

    directory = video_clip_directory(video_id, test_settings)
    written = sorted(path.suffix for path in directory.iterdir())
    assert written == [".json", ".json", ".mp4", ".mp4"]
    assert not any(path.name.endswith(".tmp.mp4") for path in directory.iterdir())


def test_generated_clip_is_served_inline_and_as_a_download(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)
    clip = client.post(f"/videos/{video_id}/fish-clips").json()[0]
    track_id = clip["track_id"]

    inline = client.get(f"/tracks/{track_id}/clip")
    assert inline.status_code == 200
    assert inline.headers["content-type"] == "video/mp4"
    assert inline.headers["content-disposition"].startswith("inline")
    assert len(inline.content) == clip["size_bytes"]

    download = client.get(f"/tracks/{track_id}/clip?download=true")
    assert download.status_code == 200
    assert download.headers["content-disposition"].startswith("attachment")
    assert "fish-1-" in download.headers["content-disposition"]


def test_clips_are_reused_until_a_refresh_is_requested(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)
    first = client.post(f"/videos/{video_id}/fish-clips").json()
    track_id = first[0]["track_id"]
    clip_path, _ = clip_paths(video_id, UUID(track_id), test_settings)
    first_written = clip_path.stat().st_mtime_ns

    cached = client.post(f"/videos/{video_id}/fish-clips").json()
    assert all(clip["cached"] is True for clip in cached)
    assert clip_path.stat().st_mtime_ns == first_written

    listed = client.get(f"/videos/{video_id}/fish-clips").json()
    assert [clip["track_id"] for clip in listed] == [clip["track_id"] for clip in first]

    refreshed = client.post(f"/tracks/{track_id}/clip?refresh=true").json()
    assert refreshed["cached"] is False
    assert clip_path.is_file()


def test_low_confidence_tracks_are_excluded_unless_requested(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)

    accepted = client.post(f"/videos/{video_id}/fish-clips").json()
    everything = client.post(f"/videos/{video_id}/fish-clips?accepted_only=false").json()

    assert [clip["viame_track_id"] for clip in accepted] == ["1", "3"]
    assert [clip["viame_track_id"] for clip in everything] == ["1", "2", "3"]
    # A single low-confidence track can still be inspected on its own.
    low = next(clip for clip in everything if clip["viame_track_id"] == "2")
    assert client.get(f"/tracks/{low['track_id']}/clip").status_code == 200


def test_clip_listing_and_download_before_generation_return_nothing(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)
    track_id = client.get(f"/videos/{video_id}/track-summaries").json()[0]["id"]

    assert client.get(f"/videos/{video_id}/fish-clips").json() == []
    missing = client.get(f"/tracks/{track_id}/clip")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Fish clip has not been generated"


def test_clip_requests_reject_unprocessed_videos_and_unknown_tracks(client, db_session_factory):
    upload = client.post("/videos", files={"file": ("raw.mp4", b"not a video", "video/mp4")})
    video_id = upload.json()["id"]

    assert client.post(f"/videos/{video_id}/fish-clips").status_code == 409
    assert client.post(f"/videos/{video_id}/fish-clips").json()["detail"] == (
        "Video processing is not completed"
    )
    unknown = client.post("/tracks/11111111-2222-3333-4444-555555555555/clip")
    assert unknown.status_code == 404


def test_missing_source_video_is_reported_without_leaking_paths(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)
    with db_session_factory() as db:
        video = db.get(Video, video_id)
        Path(video.storage_path).unlink()

    response = client.post(f"/videos/{video_id}/fish-clips")

    assert response.status_code == 409
    assert response.json()["detail"] == "Source video is not available"


def test_reprocessing_discards_clips_rendered_from_replaced_tracks(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)
    client.post(f"/videos/{video_id}/fish-clips")
    directory = video_clip_directory(video_id, test_settings)
    assert list(directory.glob("*.mp4"))

    job_id = UUID(client.post(f"/videos/{video_id}/process").json()["id"])
    process_job(job_id=job_id, settings=test_settings, session_factory=db_session_factory)

    assert not list(directory.glob("*.mp4"))
    assert client.get(f"/videos/{video_id}/fish-clips").json() == []


def test_clip_metadata_sidecar_survives_a_restart_and_matches_the_file(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)
    clip = client.post(f"/videos/{video_id}/fish-clips").json()[0]
    clip_path, metadata_path = clip_paths(video_id, UUID(clip["track_id"]), test_settings)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["viame_track_id"] == clip["viame_track_id"]
    assert payload["frame_count"] == clip["frame_count"]
    assert clip_path.stat().st_size == clip["size_bytes"]
    clip_path.relative_to(test_settings.output_root.resolve())


def test_a_track_without_detections_reports_a_clear_error(
    client, db_session_factory, test_settings, tmp_path: Path
):
    video_id = _completed_video(client, db_session_factory, test_settings, tmp_path)
    with db_session_factory() as db:
        video = db.get(Video, video_id)
        track = db.query(FishTrack).filter(FishTrack.video_id == video_id).first()
        track.detections.clear()
        db.commit()
        with pytest.raises(NoDetectionsForClipError):
            generate_track_clip(db=db, video=video, track=track, settings=test_settings)


def _write_moving_fish_video(path: Path, positions: dict[int, tuple[int, int]], frames: int = 30):
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360))
    assert writer.isOpened()
    try:
        for frame_number in range(frames):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            position = positions.get(frame_number)
            if position is not None:
                left, top = position
                frame[top : top + 30, left : left + 40] = (255, 255, 255)
            writer.write(frame)
    finally:
        writer.release()


def test_rendered_clip_keeps_the_fish_near_the_centre_of_every_frame(
    db_session_factory, test_settings, tmp_path: Path
):
    import cv2
    import numpy as np

    positions = {frame: (60 + frame * 20, 40 + frame * 8) for frame in range(4, 12)}
    source_path = tmp_path / "moving-fish.mp4"
    _write_moving_fish_video(source_path, positions)

    with db_session_factory() as db:
        video = Video(
            original_filename="moving-fish.mp4",
            storage_path=str(source_path),
            processing_status="completed",
            fps=10.0,
            model_name="test",
            pipeline_name="test",
            confidence_threshold=0.6,
        )
        db.add(video)
        db.flush()
        track = FishTrack(
            video_id=video.id,
            viame_track_id="9",
            first_frame=min(positions),
            last_frame=max(positions),
            first_timestamp_seconds=0.4,
            last_timestamp_seconds=1.1,
            detection_count=len(positions),
            mean_confidence=0.9,
            max_confidence=0.95,
        )
        db.add(track)
        db.flush()
        for frame_number, (left, top) in positions.items():
            db.add(
                FishDetection(
                    fish_track_id=track.id,
                    frame_number=frame_number,
                    timestamp_seconds=frame_number / 10.0,
                    x1=float(left),
                    y1=float(top),
                    x2=float(left + 40),
                    y2=float(top + 30),
                    confidence=0.9,
                )
            )
        db.commit()

        clip = generate_track_clip(db=db, video=video, track=track, settings=test_settings)

    capture = cv2.VideoCapture(str(clip.path))
    try:
        offsets = []
        for index in range(clip.frame_count):
            success, frame = capture.read()
            if not success:
                break
            # Ignore the caption strip drawn along the top of every clip.
            body = frame[frame.shape[0] // 4 :, :]
            bright = np.argwhere(body.max(axis=2) > 200)
            if not len(bright):
                continue
            centre_y, centre_x = bright.mean(axis=0)
            offsets.append(
                (
                    abs(centre_x - clip.width / 2) / clip.width,
                    abs(centre_y + frame.shape[0] // 4 - clip.height / 2) / clip.height,
                )
            )
    finally:
        capture.release()

    assert len(offsets) >= len(positions)
    # The crop follows the fish, so it never drifts far from the middle of the clip.
    assert max(offset_x for offset_x, _ in offsets) < 0.2
    assert max(offset_y for _, offset_y in offsets) < 0.25
