from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrackReviewUpdate(BaseModel):
    review_state: Literal["unreviewed", "reviewed", "accepted", "rejected", "needs-review"]

    model_config = ConfigDict(extra="forbid")


class FishDetectionRead(BaseModel):
    id: uuid.UUID
    frame_number: int
    timestamp_seconds: float | None
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str | None
    class_confidence: float | None

    model_config = ConfigDict(from_attributes=True)


class FishTrackRead(BaseModel):
    id: uuid.UUID
    video_id: uuid.UUID
    processing_job_id: uuid.UUID | None
    review_state: str
    reviewed_at: datetime | None
    viame_track_id: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    detection_count: int
    mean_confidence: float
    max_confidence: float
    species: str | None
    species_confidence: float | None
    detections: list[FishDetectionRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class FishTrackSummaryRead(BaseModel):
    id: uuid.UUID
    video_id: uuid.UUID
    processing_job_id: uuid.UUID | None
    review_state: str
    reviewed_at: datetime | None
    machine_accepted: bool
    viame_track_id: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    detection_count: int
    mean_confidence: float
    max_confidence: float
    species: str | None
    species_confidence: float | None
    accepted: bool

    model_config = ConfigDict(from_attributes=True)


class TrackClipRead(BaseModel):
    """Metadata for a short cropped clip that follows one fish track."""

    track_id: uuid.UUID
    video_id: uuid.UUID
    viame_track_id: str
    species: str | None
    filename: str
    media_type: str
    size_bytes: int
    fps: float
    width: int
    height: int
    frame_count: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    detection_count: int
    max_confidence: float
    generated_at: datetime
    cached: bool
    url: str
