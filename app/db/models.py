from __future__ import annotations

import enum
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.db.types import GUID


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VideoProcessingStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    camera_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    processing_status: Mapped[str] = mapped_column(
        String(32), default=VideoProcessingStatus.UPLOADED.value, nullable=False, index=True
    )
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    codec: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    viame_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pipeline_name: Mapped[str] = mapped_column(String(512), nullable=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    annotated_video_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    annotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    tracks: Mapped[list["FishTrack"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    @property
    def latest_job(self) -> "ProcessingJob | None":
        if not self.jobs:
            return None
        return max(self.jobs, key=lambda job: (job.created_at, str(job.id)))


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), default=JobStatus.QUEUED.value, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stdout_log_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    stderr_log_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    output_csv_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    worker_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="mock", index=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    configuration_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    video: Mapped[Video] = relationship(back_populates="jobs")

    __table_args__ = (
        Index("ix_processing_jobs_status_created", "status", "created_at"),
        Index(
            "uq_processing_jobs_one_active_per_video",
            "video_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'processing')"),
            sqlite_where=text("status IN ('queued', 'processing')"),
        ),
    )

    @property
    def configuration(self) -> dict[str, Any]:
        try:
            value = json.loads(self.configuration_json)
        except (TypeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    current_job_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)


class FishTrack(Base):
    __tablename__ = "fish_tracks"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    viame_track_id: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_job_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("processing_jobs.id"), nullable=True, index=True
    )
    review_state: Mapped[str] = mapped_column(
        String(32), default="unreviewed", server_default="unreviewed", nullable=False
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    last_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    first_timestamp_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_timestamp_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    max_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    species: Mapped[str | None] = mapped_column(String(256), nullable=True)
    species_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    video: Mapped[Video] = relationship(back_populates="tracks")
    detections: Mapped[list["FishDetection"]] = relationship(
        back_populates="fish_track",
        cascade="all, delete-orphan",
        order_by="FishDetection.frame_number",
    )

    __table_args__ = (
        CheckConstraint(
            "review_state IN ('unreviewed', 'reviewed', 'accepted', 'rejected', 'needs-review')",
            name="ck_fish_tracks_review_state",
        ),
        UniqueConstraint("video_id", "viame_track_id", name="uq_fish_track_video_viame_id"),
        Index("ix_fish_tracks_video_confidence", "video_id", "max_confidence"),
    )


class FishDetection(Base):
    __tablename__ = "fish_detections"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    fish_track_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("fish_tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    class_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    class_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    fish_track: Mapped[FishTrack] = relationship(back_populates="detections")

    __table_args__ = (Index("ix_fish_detections_track_frame", "fish_track_id", "frame_number"),)
