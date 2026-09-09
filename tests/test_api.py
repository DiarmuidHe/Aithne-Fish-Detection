from __future__ import annotations

from uuid import UUID

import pytest

from app.db.models import Video
from app.workers.processing_worker import process_job


def _upload_video(client, filename: str = "underwater.mp4"):
    return client.post(
        "/videos",
        files={"file": (filename, b"fake mp4 bytes", "video/mp4")},
    )


def test_api_video_creation(client):
    response = _upload_video(client)

    assert response.status_code == 201
    payload = response.json()
    assert payload["original_filename"] == "underwater.mp4"
    assert payload["storage_path"].endswith(".mp4")
    assert payload["processing_status"] == "uploaded"


def test_api_rejects_unsupported_video_extension(client):
    response = _upload_video(client, filename="notes.txt")

    assert response.status_code == 400


def test_job_creation_returns_queued_job(client):
    video_response = _upload_video(client)
    video_id = video_response.json()["id"]

    response = client.post(f"/videos/{video_id}/process")

    assert response.status_code == 202
    payload = response.json()
    assert payload["video_id"] == video_id
    assert payload["status"] == "queued"

    job_response = client.get(f"/jobs/{payload['id']}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "queued"


def test_mock_worker_populates_summary_and_tracks(client, db_session_factory, test_settings):
    video_response = _upload_video(client)
    video_id = UUID(video_response.json()["id"])
    job_response = client.post(f"/videos/{video_id}/process")
    job_id = UUID(job_response.json()["id"])

    with db_session_factory() as db:
        video = db.get(Video, video_id)
        video.fps = 30.0
        db.commit()

    process_job(job_id=job_id, settings=test_settings, session_factory=db_session_factory)

    summary_response = client.get(f"/videos/{video_id}/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["status"] == "completed"
    assert summary["fish_tracks"] == 2
    assert summary["total_detections"] == 5
    assert summary["first_fish_timestamp_seconds"] == pytest.approx(0.0)
    assert summary["last_fish_timestamp_seconds"] == pytest.approx(31 / 30)

    tracks_response = client.get(f"/videos/{video_id}/tracks")
    assert tracks_response.status_code == 200
    tracks = tracks_response.json()
    assert len(tracks) == 2
    assert tracks[0]["detections"]

