from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import SecretStr

from app.db.models import CvatSync, JobStatus, ProcessingJob, Video
from app.services.cvat_sync import queue_cvat_video, sync_cvat_video


def _enable_cvat(settings):
    settings.cvat_dev_mode = True
    settings.cvat_access_token = SecretStr("test-token")


def test_upload_only_queues_cvat_when_requested(client, db_session_factory, test_settings,
                                                monkeypatch):
    _enable_cvat(test_settings)
    called = []
    monkeypatch.setattr(
        "app.api.videos.sync_cvat_video",
        lambda video_id, settings: called.append(video_id),
    )

    ordinary = client.post("/videos", files={"file": ("ordinary.mp4", b"video", "video/mp4")})
    mirrored = client.post("/videos", files={"file": ("mirror.mp4", b"video", "video/mp4")},
                           data={"sync_to_cvat": "true"})
    assert ordinary.status_code == mirrored.status_code == 201
    mirrored_id = uuid.UUID(mirrored.json()["id"])
    assert called == [mirrored_id]
    with db_session_factory() as db:
        assert db.get(CvatSync, uuid.UUID(ordinary.json()["id"])) is None
        assert db.get(CvatSync, mirrored_id).status == "queued"
    assert client.get(f"/cvat/videos/{mirrored_id}").json()["status"] == "queued"


def test_cvat_imports_first_completed_job_without_overwriting_human_work(
    client, db_session_factory, test_settings,
):
    _enable_cvat(test_settings)
    response = client.post("/videos", files={"file": ("fish.mp4", b"video", "video/mp4")})
    video_id = uuid.UUID(response.json()["id"])
    with db_session_factory() as db:
        video = db.get(Video, video_id)
        video.width, video.height = 1920, 1080
        queue_cvat_video(db, video_id)
        db.add(ProcessingJob(
            video_id=video_id,
            status=JobStatus.COMPLETED.value,
            finished_at=datetime.now(UTC),
            output_csv_path=str(Path("tests/fixtures/sample_viame_output.csv").resolve()),
        ))
        db.commit()

    class Gateway:
        def __init__(self):
            self.created = 0
            self.imported = 0
            self.annotated = False

        def create_task(self, video):
            self.created += 1
            return 42

        def has_annotations(self, task_id):
            return self.annotated

        def import_tracks(self, task_id, parsed, width, height):
            self.imported += 1
            assert task_id == 42 and width == 1920 and height == 1080
            assert parsed.tracks
            return len(parsed.tracks)

    gateway = Gateway()

    @contextmanager
    def gateway_factory(_settings):
        yield gateway

    sync_cvat_video(video_id, test_settings, db_session_factory, gateway_factory)
    sync_cvat_video(video_id, test_settings, db_session_factory, gateway_factory)
    with db_session_factory() as db:
        link = db.get(CvatSync, video_id)
        assert link.status == "review"
        assert link.task_id == 42
        assert link.imported_job_id is not None
    assert gateway.created == gateway.imported == 1

    with db_session_factory() as db:
        link = db.get(CvatSync, video_id)
        link.imported_job_id = None
        db.commit()
    gateway.annotated = True
    sync_cvat_video(video_id, test_settings, db_session_factory, gateway_factory)
    with db_session_factory() as db:
        assert db.get(CvatSync, video_id).status == "needs_review"
    assert gateway.imported == 1
