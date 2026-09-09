from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.config import Settings, get_settings
from app.db.database import SessionLocal, init_db
from app.db.models import (
    FishDetection,
    FishTrack,
    JobStatus,
    ProcessingJob,
    VideoProcessingStatus,
    WorkerHeartbeat,
)
from app.services.fish_counter import accepted_tracks
from app.services.processing_config import settings_for_processing_job
from app.services.track_clip import remove_video_clips
from app.services.viame_parser import parse_viame_csv
from app.services.viame_runner import VIAMERunner, build_viame_runner

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def claim_next_job(
    session_factory: sessionmaker = SessionLocal,
    worker_id: str | None = None,
    worker_mode: str | None = None,
) -> uuid.UUID | None:
    with session_factory() as db:
        statement = (
            select(ProcessingJob)
            .where(ProcessingJob.status == JobStatus.QUEUED.value)
            .order_by(ProcessingJob.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if worker_mode is not None:
            statement = statement.where(ProcessingJob.worker_mode == worker_mode)
        job = db.scalars(statement).first()
        if job is None:
            return None

        job.status = JobStatus.PROCESSING.value
        job.started_at = utc_now()
        job.heartbeat_at = job.started_at
        job.worker_id = worker_id
        job.error_message = None
        if job.video is not None:
            job.video.processing_status = VideoProcessingStatus.PROCESSING.value
        db.commit()
        return job.id


def process_job(
    job_id: uuid.UUID,
    settings: Settings | None = None,
    runner: VIAMERunner | None = None,
    session_factory: sessionmaker = SessionLocal,
    worker_id: str | None = None,
) -> None:
    settings = settings or get_settings()

    with session_factory() as db:
        job = _get_job_for_update(db, job_id)
        video = job.video
        if job.worker_mode != settings.worker_mode:
            job.status = JobStatus.FAILED.value
            job.finished_at = utc_now()
            job.heartbeat_at = job.finished_at
            job.error_message = (
                f"Job requires a {job.worker_mode} worker, but this worker is {settings.worker_mode}"
            )
            video.processing_status = VideoProcessingStatus.FAILED.value
            db.commit()
            return
        run_settings = settings_for_processing_job(settings, job.configuration)
        job.status = JobStatus.PROCESSING.value
        job.started_at = job.started_at or utc_now()
        job.heartbeat_at = utc_now()
        job.worker_id = worker_id or job.worker_id
        job.error_message = None
        video.processing_status = VideoProcessingStatus.PROCESSING.value
        db.commit()

        video_id = video.id
        video_path = Path(video.storage_path)
        fps = video.fps

    runner = runner or build_viame_runner(run_settings)
    logger.info("processing job started job_id=%s video_id=%s", job_id, video_id)
    try:
        result = runner.run(
            video_path=video_path,
            job_id=job_id,
            pipeline_path=run_settings.viame_tracker_pipeline,
        )
        parsed = parse_viame_csv(
            result.output_csv_path,
            fps=fps,
            frame_number_offset=run_settings.viame_frame_number_offset,
        )
        accepted = accepted_tracks(parsed.tracks, run_settings.min_fish_confidence)

        with session_factory() as db:
            job = _get_job_for_update(db, job_id)
            if job.status != JobStatus.PROCESSING.value:
                logger.warning(
                    "discarding results for inactive job job_id=%s status=%s",
                    job_id,
                    job.status,
                )
                return
            video = job.video
            _replace_video_results(db, video.id, run_settings)
            for parsed_track in parsed.tracks:
                track = FishTrack(
                    video_id=video.id,
                    processing_job_id=job.id,
                    viame_track_id=parsed_track.viame_track_id,
                    first_frame=parsed_track.first_frame,
                    last_frame=parsed_track.last_frame,
                    first_timestamp_seconds=parsed_track.first_timestamp_seconds,
                    last_timestamp_seconds=parsed_track.last_timestamp_seconds,
                    detection_count=parsed_track.detection_count,
                    mean_confidence=parsed_track.mean_confidence,
                    max_confidence=parsed_track.max_confidence,
                    species=parsed_track.species,
                    species_confidence=parsed_track.species_confidence,
                )
                db.add(track)
                db.flush()
                for observation in parsed_track.observations:
                    db.add(
                        FishDetection(
                            fish_track_id=track.id,
                            frame_number=observation.frame_number,
                            timestamp_seconds=observation.timestamp_seconds,
                            x1=observation.x1,
                            y1=observation.y1,
                            x2=observation.x2,
                            y2=observation.y2,
                            confidence=observation.confidence,
                            class_name=observation.class_name,
                            class_confidence=observation.class_confidence,
                        )
                    )

            video.processing_status = VideoProcessingStatus.COMPLETED.value
            video.viame_version = result.viame_version or run_settings.viame_version
            video.model_name = run_settings.model_name
            video.model_version = run_settings.model_version
            video.pipeline_name = str(result.pipeline_path)
            video.confidence_threshold = run_settings.min_fish_confidence
            job.status = JobStatus.COMPLETED.value
            job.finished_at = utc_now()
            job.heartbeat_at = job.finished_at
            job.stdout_log_path = str(result.stdout_log_path)
            job.stderr_log_path = str(result.stderr_log_path)
            job.output_csv_path = str(result.output_csv_path)
            db.commit()

        logger.info(
            "processing job completed job_id=%s video_id=%s duration_seconds=%.3f detections=%s tracks=%s accepted_tracks=%s skipped_rows=%s",
            job_id,
            video_id,
            result.duration_seconds,
            len(parsed.detections),
            len(parsed.tracks),
            len(accepted),
            len(parsed.skipped_rows),
        )
    except Exception as exc:
        logger.exception("processing job failed job_id=%s video_id=%s", job_id, video_id)
        with session_factory() as db:
            job = _get_job_for_update(db, job_id)
            if job.status == JobStatus.PROCESSING.value:
                job.status = JobStatus.FAILED.value
                job.finished_at = utc_now()
                job.heartbeat_at = job.finished_at
                job.error_message = _safe_error_message(exc, settings)
                if job.video is not None:
                    job.video.processing_status = VideoProcessingStatus.FAILED.value
                db.commit()


@dataclass
class WorkerRuntimeState:
    current_job_id: uuid.UUID | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_current_job(self, job_id: uuid.UUID | None) -> None:
        with self.lock:
            self.current_job_id = job_id

    def snapshot(self) -> uuid.UUID | None:
        with self.lock:
            return self.current_job_id


def record_worker_heartbeat(
    worker_id: str,
    mode: str,
    current_job_id: uuid.UUID | None,
    session_factory: sessionmaker = SessionLocal,
) -> None:
    observed_at = utc_now()
    with session_factory() as db:
        heartbeat = db.get(WorkerHeartbeat, worker_id)
        if heartbeat is None:
            heartbeat = WorkerHeartbeat(
                worker_id=worker_id,
                mode=mode,
                started_at=observed_at,
                last_seen_at=observed_at,
            )
            db.add(heartbeat)
        heartbeat.mode = mode
        heartbeat.last_seen_at = observed_at
        heartbeat.current_job_id = current_job_id
        if current_job_id is not None:
            job = db.get(ProcessingJob, current_job_id)
            if job is not None and job.status == JobStatus.PROCESSING.value:
                job.heartbeat_at = observed_at
                job.worker_id = worker_id
        db.commit()


def run_heartbeat_loop(
    stop_event: threading.Event,
    state: WorkerRuntimeState,
    worker_id: str,
    settings: Settings,
    session_factory: sessionmaker = SessionLocal,
) -> None:
    while not stop_event.wait(settings.worker_heartbeat_seconds):
        try:
            record_worker_heartbeat(
                worker_id=worker_id,
                mode=settings.worker_mode,
                current_job_id=state.snapshot(),
                session_factory=session_factory,
            )
        except Exception:
            logger.exception("worker heartbeat update failed")


def run_worker_forever() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    if settings.auto_create_tables:
        init_db()

    worker_id = str(uuid.uuid4())
    state = WorkerRuntimeState()
    stop_event = threading.Event()
    record_worker_heartbeat(worker_id, settings.worker_mode, None)
    heartbeat_thread = threading.Thread(
        target=run_heartbeat_loop,
        args=(stop_event, state, worker_id, settings),
        name="worker-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    logger.info(
        "worker started worker_id=%s mode=%s poll_seconds=%s",
        worker_id,
        settings.worker_mode,
        settings.worker_poll_seconds,
    )
    try:
        while True:
            job_id = claim_next_job(
                worker_id=worker_id,
                worker_mode=settings.worker_mode,
            )
            if job_id is None:
                time.sleep(settings.worker_poll_seconds)
                continue
            state.set_current_job(job_id)
            try:
                record_worker_heartbeat(worker_id, settings.worker_mode, job_id)
                process_job(job_id, settings=settings, worker_id=worker_id)
            finally:
                state.set_current_job(None)
                record_worker_heartbeat(worker_id, settings.worker_mode, None)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=settings.worker_heartbeat_seconds + 1)


def _get_job_for_update(db: Session, job_id: uuid.UUID) -> ProcessingJob:
    job = db.scalars(
        select(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .options(selectinload(ProcessingJob.video))
        .with_for_update()
    ).first()
    if job is None:
        raise LookupError(f"Processing job not found: {job_id}")
    if job.video is None:
        raise LookupError(f"Processing job has no video: {job_id}")
    return job


def _replace_video_results(db: Session, video_id: uuid.UUID, settings: Settings) -> None:
    tracks = db.scalars(select(FishTrack).where(FishTrack.video_id == video_id)).all()
    for track in tracks:
        db.delete(track)
    db.flush()
    # Cached fish clips belong to the tracks that are going away with this run.
    remove_video_clips(video_id, settings)


def _safe_error_message(exc: Exception, settings: Settings) -> str:
    message = str(exc) or exc.__class__.__name__
    for path, replacement in (
        (settings.upload_root, "<upload>"),
        (settings.job_root, "<job>"),
        (settings.output_root, "<output>"),
        (settings.viame_root, "<viame>"),
    ):
        path_text = str(path)
        if path_text:
            message = message.replace(path_text, replacement)
    return message[:1000]


if __name__ == "__main__":
    run_worker_forever()
