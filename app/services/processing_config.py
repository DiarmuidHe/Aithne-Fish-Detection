from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import Settings

CONFIGURATION_SCHEMA_VERSION = 1


def capture_processing_configuration(settings: Settings) -> dict[str, Any]:
    """Return the immutable, safe-to-display settings that affect a processing run."""

    return {
        "schema_version": CONFIGURATION_SCHEMA_VERSION,
        "worker_mode": settings.effective_deployment_mode,
        "pipeline": str(settings.viame_tracker_pipeline),
        "downsample_fps": settings.viame_downsample_fps,
        "frame_number_offset": settings.viame_frame_number_offset,
        "detector_score_threshold": settings.viame_detector_score_threshold,
        "tracker_high_threshold": settings.viame_tracker_high_threshold,
        "tracker_low_threshold": settings.viame_tracker_low_threshold,
        "tracker_new_track_threshold": settings.viame_tracker_new_track_threshold,
        "tracker_buffer_frames": settings.viame_tracker_buffer_frames,
        "confidence_threshold": settings.min_fish_confidence,
        "model_name": settings.model_name,
        "model_version": settings.model_version,
        "viame_version": settings.viame_version,
        "mock_fixture": Path(settings.viame_sample_csv).name if settings.viame_mock else None,
        # The configured shell command is never returned by the API. Its digest still
        # makes configuration drift detectable when reproducing a run.
        "run_command_sha256": hashlib.sha256(
            settings.viame_run_command.encode("utf-8")
        ).hexdigest(),
    }


def serialize_processing_configuration(configuration: dict[str, Any]) -> str:
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def settings_for_processing_job(
    settings: Settings, configuration: dict[str, Any]
) -> Settings:
    """Apply a persisted job snapshot to the worker's trusted runtime settings."""

    updates: dict[str, Any] = {}
    mapping = {
        "downsample_fps": "viame_downsample_fps",
        "frame_number_offset": "viame_frame_number_offset",
        "detector_score_threshold": "viame_detector_score_threshold",
        "tracker_high_threshold": "viame_tracker_high_threshold",
        "tracker_low_threshold": "viame_tracker_low_threshold",
        "tracker_new_track_threshold": "viame_tracker_new_track_threshold",
        "tracker_buffer_frames": "viame_tracker_buffer_frames",
        "confidence_threshold": "min_fish_confidence",
        "model_name": "model_name",
        "model_version": "model_version",
        "viame_version": "viame_version",
    }
    for source_name, setting_name in mapping.items():
        if source_name in configuration:
            updates[setting_name] = configuration[source_name]

    pipeline = configuration.get("pipeline")
    if isinstance(pipeline, str) and pipeline:
        updates["viame_tracker_pipeline"] = Path(pipeline)

    return settings.model_copy(update=updates)
