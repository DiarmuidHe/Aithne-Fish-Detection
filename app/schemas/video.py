from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from app.schemas.job import JobRead


class VideoRead(BaseModel):
    id: uuid.UUID
    original_filename: str
    storage_path: str
    camera_id: str | None
    created_at: datetime
    processing_status: str
    fps: float | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    codec: str | None
    size_bytes: int | None
    content_sha256: str | None
    viame_version: str | None
    model_name: str
    model_version: str | None
    pipeline_name: str
    confidence_threshold: float
    annotated_at: datetime | None
    latest_job: JobRead | None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("storage_path", mode="before")
    @classmethod
    def hide_storage_directory(cls, value: str) -> str:
        """Retain API compatibility without disclosing the server directory."""

        return Path(value).name


class VideoSummary(BaseModel):
    video_id: uuid.UUID
    status: str
    fish_tracks: int
    total_detections: int
    mean_track_confidence: float | None
    first_fish_timestamp_seconds: float | None
    last_fish_timestamp_seconds: float | None
    pipeline_name: str
    confidence_threshold: float
    model_name: str
    model_version: str | None
    viame_version: str | None


class VideoAnnotationRead(BaseModel):
    video_id: uuid.UUID
    annotated_at: datetime
    filename: str
    media_type: str
    size_bytes: int
    fps: float
    width: int
    height: int
    frame_count: int
    url: str
