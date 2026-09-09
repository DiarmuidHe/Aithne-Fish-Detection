from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import JobStatus, ProcessingJob, Video, VideoProcessingStatus
from app.services.processing_config import (
    capture_processing_configuration,
    serialize_processing_configuration,
)


@dataclass(frozen=True)
class VideoMetadata:
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    codec: str | None = None


async def register_uploaded_video(
    db: Session,
    upload: UploadFile,
    settings: Settings,
    camera_id: str | None = None,
) -> Video:
    original_filename = Path(upload.filename or "upload").name
    if len(original_filename) > 512:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video filename is too long",
        )
    extension = Path(original_filename).suffix.lower()
    if extension not in settings.allowed_video_extension_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension. Allowed: {sorted(settings.allowed_video_extension_set)}",
        )

    settings.upload_root.mkdir(parents=True, exist_ok=True)
    storage_path = settings.upload_root / f"{uuid.uuid4()}{extension}"
    bytes_written = 0
    content_digest = hashlib.sha256()

    try:
        with storage_path.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Uploaded video exceeds MAX_UPLOAD_BYTES",
                    )
                content_digest.update(chunk)
                output.write(chunk)
    except Exception:
        if storage_path.exists():
            storage_path.unlink()
        raise

    if bytes_written == 0:
        storage_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    try:
        return register_video_from_path(
            db=db,
            path=storage_path,
            original_filename=original_filename,
            settings=settings,
            camera_id=camera_id,
            size_bytes=bytes_written,
            content_sha256=content_digest.hexdigest(),
        )
    except Exception:
        storage_path.unlink(missing_ok=True)
        raise


def register_video_from_path(
    db: Session,
    path: Path,
    original_filename: str,
    settings: Settings,
    camera_id: str | None = None,
    size_bytes: int | None = None,
    content_sha256: str | None = None,
) -> Video:
    if not path.exists():
        raise FileNotFoundError(path)

    metadata = extract_video_metadata(path)
    video = Video(
        original_filename=Path(original_filename).name,
        storage_path=str(path),
        camera_id=camera_id,
        processing_status=VideoProcessingStatus.UPLOADED.value,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        duration_seconds=metadata.duration_seconds,
        codec=metadata.codec,
        size_bytes=size_bytes if size_bytes is not None else path.stat().st_size,
        content_sha256=content_sha256 or _sha256_file(path),
        viame_version=settings.viame_version,
        model_name=settings.model_name,
        model_version=settings.model_version,
        pipeline_name=str(settings.viame_tracker_pipeline),
        confidence_threshold=settings.min_fish_confidence,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def enqueue_video_processing(db: Session, video_id: uuid.UUID, settings: Settings) -> ProcessingJob:
    video = db.scalars(
        select(Video).where(Video.id == video_id).with_for_update()
    ).first()
    if video is None:
        raise LookupError(f"Video not found: {video_id}")

    active_job = _active_job_for_video(db, video_id)
    if active_job is not None:
        if not _processing_job_is_stale(active_job, settings):
            return active_job
        active_job.status = JobStatus.FAILED.value
        active_job.finished_at = _utc_now()
        active_job.error_message = (
            "Worker stopped reporting progress. This job was closed so it can be retried."
        )
        db.flush()

    configuration = capture_processing_configuration(settings)
    job = ProcessingJob(
        video_id=video.id,
        status=JobStatus.QUEUED.value,
        worker_mode=settings.effective_deployment_mode,
        configuration_json=serialize_processing_configuration(configuration),
    )
    video.processing_status = VideoProcessingStatus.QUEUED.value
    video.pipeline_name = str(settings.viame_tracker_pipeline)
    video.model_name = settings.model_name
    video.model_version = settings.model_version
    video.confidence_threshold = settings.min_fish_confidence
    # A previous render no longer represents the newly requested processing run.
    video.annotated_video_path = None
    video.annotated_at = None
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        # The partial unique index is the final guard if two requests race on a
        # database that does not honor SELECT ... FOR UPDATE (notably SQLite).
        db.rollback()
        active_job = _active_job_for_video(db, video_id)
        if active_job is not None:
            return active_job
        raise
    db.refresh(job)
    return job


def _active_job_for_video(db: Session, video_id: uuid.UUID) -> ProcessingJob | None:
    return db.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.video_id == video_id,
            ProcessingJob.status.in_([JobStatus.QUEUED.value, JobStatus.PROCESSING.value]),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).first()


def _processing_job_is_stale(job: ProcessingJob, settings: Settings) -> bool:
    if job.status != JobStatus.PROCESSING.value:
        return False
    last_progress = job.heartbeat_at or job.started_at
    if last_progress is None:
        return False
    if last_progress.tzinfo is None:
        last_progress = last_progress.replace(tzinfo=UTC)
    return last_progress < _utc_now() - timedelta(seconds=settings.job_stale_after_seconds)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_video_metadata(path: Path) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return VideoMetadata()

    if completed.returncode != 0:
        return VideoMetadata()

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return VideoMetadata()

    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    return VideoMetadata(
        width=_optional_int(stream.get("width")),
        height=_optional_int(stream.get("height")),
        fps=_parse_fps(stream.get("avg_frame_rate")),
        duration_seconds=_optional_float(fmt.get("duration")),
        codec=stream.get("codec_name"),
    )


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
