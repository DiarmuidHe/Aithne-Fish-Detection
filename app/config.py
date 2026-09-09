from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Fish Monitor"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://fish:fish@localhost:5432/fish_monitor"
    auto_create_tables: bool = False

    upload_root: Path = Path("data/uploads")
    job_root: Path = Path("data/jobs")
    output_root: Path = Path("data/outputs")
    max_upload_bytes: int = 5 * 1024 * 1024 * 1024
    allowed_video_extensions: str = ".mp4,.mov,.avi,.mkv"

    viame_mock: bool = False
    deployment_mode: Literal["auto", "mock", "gpu"] = "auto"
    viame_root: Path = Path("/opt/noaa/viame")
    viame_setup_script: Path = Path("/opt/noaa/viame/setup_viame.sh")
    viame_tracker_pipeline: Path = Path(
        "/opt/noaa/viame/configs/pipelines/tracker_default_fish_fusion.pipe"
    )
    viame_detector_pipeline: Path = Path(
        "/opt/noaa/viame/configs/pipelines/detector_default_fish_no_motion.pipe"
    )
    viame_timeout_seconds: int = 4 * 60 * 60
    viame_sample_csv: Path = Path("app/fixtures/sample_viame_output.csv")
    viame_downsample_fps: float = Field(default=10.0, gt=0)
    viame_frame_number_offset: int = 0
    viame_detector_score_threshold: float = Field(default=0.10, ge=0, le=1)
    viame_tracker_high_threshold: float = Field(default=0.45, ge=0, le=1)
    viame_tracker_low_threshold: float = Field(default=0.10, ge=0, le=1)
    viame_tracker_new_track_threshold: float = Field(default=0.50, ge=0, le=1)
    viame_tracker_buffer_frames: int = Field(default=30, ge=1)
    viame_run_command: str = Field(
        default=(
            'source "$VIAME_SETUP_SCRIPT" && kwiver runner '
            '-I "$VIAME_ROOT/configs/pipelines" "$VIAME_PIPELINE" '
            '-s "input:video_filename=$VIAME_INPUT_VIDEO" '
            '-s "input:video_reader:type=vidl_ffmpeg" '
            '-s "downsampler:target_frame_rate=$VIAME_DOWNSAMPLE_FPS" '
            '-s "downsampler:renumber_frames=false" '
            '-s "detector:detector:onnx:score_thresh=$VIAME_DETECTOR_SCORE_THRESHOLD" '
            '-s "tracker:track_objects:bytetrack:high_thresh=$VIAME_TRACKER_HIGH_THRESHOLD" '
            '-s "tracker:track_objects:bytetrack:low_thresh=$VIAME_TRACKER_LOW_THRESHOLD" '
            '-s "tracker:track_objects:bytetrack:new_track_thresh=$VIAME_TRACKER_NEW_TRACK_THRESHOLD" '
            '-s "tracker:track_objects:bytetrack:track_buffer=$VIAME_TRACKER_BUFFER_FRAMES" '
            '-s "track_writer:file_name=$VIAME_OUTPUT_CSV"'
        ),
        description="Admin-controlled command executed by bash -lc after trusted env setup.",
    )

    clip_padding_seconds: float = Field(default=0.6, ge=0)
    clip_zoom_margin: float = Field(default=2.2, ge=1.0)
    clip_min_crop_pixels: int = Field(default=128, ge=16)
    clip_output_short_side: int = Field(default=260, ge=64)

    min_fish_confidence: float = 0.60
    model_name: str = "viame-default-fish"
    model_version: str | None = None
    viame_version: str | None = None

    worker_poll_seconds: float = 3.0
    worker_heartbeat_seconds: float = Field(default=5.0, gt=0)
    worker_stale_after_seconds: float = Field(default=30.0, gt=0)
    job_stale_after_seconds: float = Field(default=120.0, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_video_extension_set(self) -> set[str]:
        return {
            extension.strip().lower()
            for extension in self.allowed_video_extensions.split(",")
            if extension.strip()
        }

    @property
    def worker_mode(self) -> Literal["mock", "gpu"]:
        """Mode this process can actually execute."""

        return "mock" if self.viame_mock else "gpu"

    @property
    def effective_deployment_mode(self) -> Literal["mock", "gpu"]:
        if self.deployment_mode == "auto":
            return self.worker_mode
        return self.deployment_mode


@lru_cache
def get_settings() -> Settings:
    return Settings()
