from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.db.models import Video
from app.services.video_annotator import _load_annotations_by_frame
from app.workers.processing_worker import process_job


def _upload_fake_video(client, filename: str = "underwater.mp4"):
    return client.post(
        "/videos",
        files={"file": (filename, b"fake mp4 bytes", "video/mp4")},
    )


def _write_test_video(path: Path, frame_count: int = 35) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (640, 360),
    )
    assert writer.isOpened()
    try:
        for frame_number in range(frame_count):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = 30 + frame_number % 40
            frame[:, :, 1] = 55
            frame[:, :, 2] = 80
            writer.write(frame)
    finally:
        writer.release()


def _upload_generated_video(client, tmp_path: Path):
    video_path = tmp_path / "source.mp4"
    _write_test_video(video_path)
    with video_path.open("rb") as video_file:
        return client.post(
            "/videos",
            files={"file": ("tiny.mp4", video_file, "video/mp4")},
        )


def _process_with_mock_worker(client, db_session_factory, test_settings, video_id: UUID) -> None:
    job_response = client.post(f"/videos/{video_id}/process")
    job_id = UUID(job_response.json()["id"])

    with db_session_factory() as db:
        video = db.get(Video, video_id)
        video.fps = 10.0
        db.commit()

    process_job(job_id=job_id, settings=test_settings, session_factory=db_session_factory)


def test_annotation_endpoint_rejects_unprocessed_video(client):
    video_response = _upload_fake_video(client)
    video_id = video_response.json()["id"]

    response = client.post(f"/videos/{video_id}/annotate")

    assert response.status_code == 409
    assert response.json()["detail"] == "Video processing is not completed"


def test_annotation_uses_only_database_tracks_above_video_threshold(
    client,
    db_session_factory,
    test_settings,
):
    video_response = _upload_fake_video(client)
    video_id = UUID(video_response.json()["id"])
    _process_with_mock_worker(client, db_session_factory, test_settings, video_id)

    with db_session_factory() as db:
        video = db.get(Video, video_id)
        annotations_by_frame = _load_annotations_by_frame(db, video)

    assert set(annotations_by_frame) == {0, 1, 2, 30, 31}
    assert {item.track_label for items in annotations_by_frame.values() for item in items} == {
        "1",
        "3",
    }


def test_annotation_endpoint_generates_video_under_output_root(
    client,
    db_session_factory,
    test_settings,
    tmp_path: Path,
):
    video_response = _upload_generated_video(client, tmp_path)
    video_id = UUID(video_response.json()["id"])
    _process_with_mock_worker(client, db_session_factory, test_settings, video_id)

    response = client.post(f"/videos/{video_id}/annotate")

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_id"] == str(video_id)
    assert payload["media_type"] == "video/mp4"
    assert payload["filename"].endswith(".mp4")
    assert UUID(Path(payload["filename"]).stem)
    assert payload["size_bytes"] > 0
    assert payload["frame_count"] == 35
    assert payload["width"] == 640
    assert payload["height"] == 360

    with db_session_factory() as db:
        video = db.get(Video, video_id)
        output_path = Path(video.annotated_video_path).resolve()

    output_path.relative_to(test_settings.output_root.resolve())
    assert output_path.is_file()
    assert output_path.name == payload["filename"]

    get_response = client.get(f"/videos/{video_id}/annotated-video")

    assert get_response.status_code == 200
    assert get_response.headers["content-type"] == "video/mp4"
    assert get_response.headers["content-disposition"].startswith("inline")
    assert len(get_response.content) == payload["size_bytes"]

    download_response = client.get(f"/videos/{video_id}/annotated-video?download=true")
    assert download_response.status_code == 200
    assert download_response.headers["content-disposition"].startswith("attachment")


def test_get_annotated_video_before_generation_returns_404(
    client,
    db_session_factory,
    test_settings,
):
    video_response = _upload_fake_video(client)
    video_id = UUID(video_response.json()["id"])
    _process_with_mock_worker(client, db_session_factory, test_settings, video_id)

    response = client.get(f"/videos/{video_id}/annotated-video")

    assert response.status_code == 404
    assert response.json()["detail"] == "Annotated video has not been generated"


def test_get_annotated_video_missing_file_returns_404(
    client,
    db_session_factory,
    test_settings,
):
    video_response = _upload_fake_video(client)
    video_id = UUID(video_response.json()["id"])
    _process_with_mock_worker(client, db_session_factory, test_settings, video_id)

    with db_session_factory() as db:
        video = db.get(Video, video_id)
        video.annotated_video_path = str(test_settings.output_root / "missing.mp4")
        video.annotated_at = datetime.now(timezone.utc)
        db.commit()

    response = client.get(f"/videos/{video_id}/annotated-video")

    assert response.status_code == 404
    assert response.json()["detail"] == "Annotated video file is not available"
