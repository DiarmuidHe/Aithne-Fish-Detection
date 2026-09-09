from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import JobStatus, ProcessingJob, Video, VideoProcessingStatus
from app.services.viame_runner import VIAMERunnerError
from app.workers.processing_worker import process_job, record_worker_heartbeat


def _upload_video(client, filename: str = "dashboard-sample.mp4"):
    return client.post(
        "/videos",
        files={"file": (filename, b"fake video content", "video/mp4")},
    )


class FailingRunner:
    def run(self, **_kwargs):
        raise VIAMERunnerError("VIAME model initialization failed")


def test_dashboard_and_static_assets_are_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Upload an underwater video" in response.text
    assert "Show low-confidence tracks" in response.text
    assert "Generate annotated video" in response.text

    css = client.get("/static/dashboard.css")
    javascript = client.get("/static/dashboard.js")
    assert css.status_code == 200
    assert "@media (max-width: 820px)" in css.text
    assert javascript.status_code == 200
    assert 'apiRequest("/system/status")' in javascript.text
    assert 'includes(video.processing_status)' in javascript.text


def test_dashboard_offers_cropped_clips_of_individual_fish(client):
    page = client.get("/").text
    assert "Generate fish clips" in page
    assert "Individual fish clips" in page
    assert "Show cropped clip of this fish" in page

    product = client.get("/static/product.js")
    assert product.status_code == 200
    assert "fish-clips" in product.text
    assert "clip-card" in product.text
    assert ".clip-grid" in client.get("/static/dashboard.css").text


def test_video_list_exposes_latest_job_and_safe_configuration(client):
    upload = _upload_video(client)
    video_id = upload.json()["id"]
    queued = client.post(f"/videos/{video_id}/process")

    assert queued.status_code == 202
    assert "stdout_log_path" not in queued.json()
    assert queued.json()["configuration"]["worker_mode"] == "mock"
    assert queued.json()["configuration"]["confidence_threshold"] == 0.60
    assert len(queued.json()["configuration"]["run_command_sha256"]) == 64

    listed = client.get("/videos").json()
    assert listed[0]["latest_job"]["id"] == queued.json()["id"]
    assert listed[0]["latest_job"]["status"] == "queued"
    assert "/" not in listed[0]["storage_path"]
    assert "\\" not in listed[0]["storage_path"]
    assert len(listed[0]["content_sha256"]) == 64


def test_duplicate_process_requests_return_the_same_active_job(client, db_session_factory):
    video_id = _upload_video(client).json()["id"]

    first = client.post(f"/videos/{video_id}/process")
    second = client.post(f"/videos/{video_id}/process")

    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    with db_session_factory() as db:
        active_jobs = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == UUID(video_id),
                ProcessingJob.status.in_(
                    [JobStatus.QUEUED.value, JobStatus.PROCESSING.value]
                ),
            )
            .all()
        )
    assert len(active_jobs) == 1


def test_database_invariant_rejects_two_active_jobs(client, db_session_factory):
    video_id = UUID(_upload_video(client).json()["id"])
    client.post(f"/videos/{video_id}/process")

    with db_session_factory() as db:
        db.add(
            ProcessingJob(
                video_id=video_id,
                status=JobStatus.QUEUED.value,
                worker_mode="mock",
                configuration_json="{}",
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


def test_failure_is_visible_and_failed_video_can_be_retried(
    client,
    db_session_factory,
    test_settings,
):
    video_id = UUID(_upload_video(client).json()["id"])
    first_job = client.post(f"/videos/{video_id}/process").json()

    process_job(
        UUID(first_job["id"]),
        settings=test_settings,
        runner=FailingRunner(),
        session_factory=db_session_factory,
    )

    failed_job = client.get(f"/jobs/{first_job['id']}")
    assert failed_job.status_code == 200
    assert failed_job.json()["status"] == "failed"
    assert failed_job.json()["error_message"] == "VIAME model initialization failed"
    listed_video = client.get("/videos").json()[0]
    assert listed_video["processing_status"] == "failed"
    assert listed_video["latest_job"]["error_message"] == "VIAME model initialization failed"

    with db_session_factory() as db:
        job = db.get(ProcessingJob, UUID(first_job["id"]))
        job.error_message = (
            "VIAME exited with code 1; stderr log: /private/jobs/job-stderr.log"
        )
        db.commit()
    sanitized_job = client.get(f"/jobs/{first_job['id']}").json()
    assert sanitized_job["error_message"] == (
        "VIAME exited with code 1. See worker logs."
    )
    assert "/private/" not in sanitized_job["error_message"]

    retry = client.post(f"/videos/{video_id}/process")
    assert retry.status_code == 202
    assert retry.json()["status"] == "queued"
    assert retry.json()["id"] != first_job["id"]


def test_stale_processing_job_is_failed_before_retry(client, db_session_factory):
    video_id = UUID(_upload_video(client).json()["id"])
    first_job_id = UUID(client.post(f"/videos/{video_id}/process").json()["id"])
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    with db_session_factory() as db:
        job = db.get(ProcessingJob, first_job_id)
        job.status = JobStatus.PROCESSING.value
        job.started_at = stale_time
        job.heartbeat_at = stale_time
        job.video.processing_status = VideoProcessingStatus.PROCESSING.value
        db.commit()

    retry = client.post(f"/videos/{video_id}/process")

    assert retry.status_code == 202
    assert UUID(retry.json()["id"]) != first_job_id
    with db_session_factory() as db:
        stale_job = db.get(ProcessingJob, first_job_id)
        assert stale_job.status == JobStatus.FAILED.value
        assert "Worker stopped" in stale_job.error_message


def test_worker_and_database_readiness_are_reported(client, db_session_factory):
    before = client.get("/system/status")
    assert before.status_code == 200
    assert before.json()["database"]["available"] is True
    assert before.json()["worker"]["available"] is False
    assert before.json()["processing_mode"] == "mock"
    assert before.json()["compose_command"] == "docker compose up -d --build"

    record_worker_heartbeat(
        worker_id="test-worker",
        mode="mock",
        current_job_id=None,
        session_factory=db_session_factory,
    )
    after = client.get("/system/status")
    assert after.json()["ready"] is True
    assert after.json()["worker"]["available"] is True
    assert after.json()["worker"]["active_workers"] == 1


def test_track_summary_toggle_and_completed_run_configuration(
    client,
    db_session_factory,
    test_settings,
):
    video_id = UUID(_upload_video(client).json()["id"])
    job_id = UUID(client.post(f"/videos/{video_id}/process").json()["id"])
    with db_session_factory() as db:
        video = db.get(Video, video_id)
        video.fps = 30.0
        db.commit()

    process_job(job_id, settings=test_settings, session_factory=db_session_factory)

    summary = client.get(f"/videos/{video_id}/summary").json()
    assert summary["fish_tracks"] == 2
    assert summary["total_detections"] == 5
    assert summary["confidence_threshold"] == 0.60
    assert summary["pipeline_name"] == str(test_settings.viame_tracker_pipeline)

    accepted = client.get(f"/videos/{video_id}/track-summaries").json()
    all_tracks = client.get(
        f"/videos/{video_id}/track-summaries?accepted_only=false"
    ).json()
    assert len(accepted) == 2
    assert all(track["accepted"] for track in accepted)
    assert len(all_tracks) == 3
    assert any(not track["accepted"] for track in all_tracks)
