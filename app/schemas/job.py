from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class ProcessingConfigurationRead(BaseModel):
    schema_version: int | None = None
    worker_mode: str | None = None
    pipeline: str | None = None
    downsample_fps: float | None = None
    frame_number_offset: int | None = None
    detector_score_threshold: float | None = None
    tracker_high_threshold: float | None = None
    tracker_low_threshold: float | None = None
    tracker_new_track_threshold: float | None = None
    tracker_buffer_frames: int | None = None
    confidence_threshold: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    viame_version: str | None = None
    mock_fixture: str | None = None
    run_command_sha256: str | None = None


class JobRead(BaseModel):
    id: uuid.UUID
    video_id: uuid.UUID
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    worker_mode: str
    heartbeat_at: datetime | None
    configuration: ProcessingConfigurationRead

    model_config = ConfigDict(from_attributes=True)

    @field_validator("error_message", mode="before")
    @classmethod
    def hide_legacy_log_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return re.sub(
            r";\s*(?:stderr|stdout) log:\s*.+$",
            ". See worker logs.",
            value,
            flags=re.IGNORECASE,
        )
