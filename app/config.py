from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LIVE_SOURCE_KEY_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")
DEFAULT_CORAL_CITY_URL = "https://www.coralcitycamera.com/"
PREPROCESS_MODES = ("none", "white_balance", "clahe", "both")
# Blend weights for species_quality.track_quality. Crop size and sharpness dominate
# because they are what Fishial's classifier actually needs; detector confidence is
# weakest because it measures "is this a fish", not "is this crop identifiable".
DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "short_side": 3.0, "sharpness": 2.0, "confidence": 1.0,
    "contrast": 1.5, "colorfulness": 1.0, "frames": 1.5,
}


class LiveCameraSource(BaseModel):
    """One administrator-configured camera.

    The API selects a camera by ``key`` only; URLs never travel from a client into
    the resolver, which would make FFmpeg/yt-dlp an SSRF sink.
    """

    key: str
    label: str
    url: str
    location: str | None = None
    # Optional key into app.services.species_region.REGIONS. Unset means "no
    # regional filtering", so a camera we have no species list for never regresses.
    region: str | None = None


DEFAULT_LIVE_SOURCES: list[dict[str, str]] = [
    # Coral City has no curated species list, so it stays unset: the filter is
    # opt-in per camera and unset means every name is accepted.
    {"key": "coral-city", "label": "Coral City Camera", "location": "Miami, Florida",
     "url": DEFAULT_CORAL_CITY_URL},
    {"key": "smartbay-cam1", "label": "SmartBay Cam 1", "location": "Galway Bay, Ireland",
     "region": "north_east_atlantic",
     "url": "https://live.heanet.ie/261512497d1945a09c73de7f05421dd9/ngrp:_all/playlist.m3u8"},
    {"key": "smartbay-cam2", "label": "SmartBay Cam 2", "location": "Galway Bay, Ireland",
     "region": "north_east_atlantic",
     "url": "https://live.heanet.ie/6c3ea7d05d754f26ba324d2a59c5d26b/ngrp:_all/playlist.m3u8"},
    {"key": "smartbay-cam3", "label": "SmartBay Cam 3 (ANERIS EMUAS)", "location": "Galway Bay, Ireland",
     "region": "north_east_atlantic",
     "url": "https://live.heanet.ie/7b451dbe046149c4bba3d0e93792fae6/ngrp:_all/playlist.m3u8"},
]


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

    # Camera registry, overridable as a JSON array in LIVE_SOURCE_REGISTRY so a
    # deployment can retarget or trim the list without a code change.
    live_source_registry: list[LiveCameraSource] = Field(
        default_factory=lambda: [LiveCameraSource(**entry) for entry in DEFAULT_LIVE_SOURCES],
        description="Ordered list of selectable cameras; administrator configuration only.",
    )
    # Retained so existing .env files keep retargeting the Coral City entry.
    coral_city_url: str = DEFAULT_CORAL_CITY_URL
    live_default_source_key: str = "coral-city"
    live_monitor_enabled: bool = False
    live_lost_track_seconds: float = Field(default=10.0, gt=0)
    # Long chunks amortize VIAME startup. The ceiling only bounds how much footage a
    # single failed chunk can cost, and how far the annotated view trails the camera.
    live_segment_seconds: float = Field(default=2.0, ge=0.5, le=300)
    live_activity_window_seconds: float = Field(default=60.0, ge=10, le=3600)
    live_fps: float = Field(default=5.0, ge=1, le=30)
    live_retry_seconds: float = Field(default=3.0, gt=0)
    live_max_retry_seconds: float = Field(default=30.0, gt=0)
    live_max_retries: int = Field(default=10, ge=0)
    live_read_timeout_seconds: float = Field(default=20.0, gt=0)
    live_detector_timeout_seconds: int = Field(default=60, ge=1)
    live_worker_stale_seconds: float = Field(default=120.0, ge=10)
    live_max_pending_segments: int = Field(default=3, ge=1, le=30)
    # Per-frame crop spooling is write-heavy and short-lived. Keeping it off a slow
    # bound volume (a Windows bind mount costs ~30ms per small file) is what lets
    # inference keep up with the camera. Only finished crops and clips reach OUTPUT_ROOT.
    live_scratch_root: Path = Path(tempfile.gettempdir()) / "fish-live-scratch"

    min_fish_confidence: float = 0.60
    fishial_enabled: bool = False
    fishial_api_base_url: str = "https://api-recognition.fishial.ai/v2"
    fishial_client_id: str | None = Field(default=None, repr=False)
    fishial_client_secret: SecretStr | None = None
    fishial_request_timeout_seconds: float = Field(default=30.0, gt=0)
    fishial_default_frames_per_fish: int = Field(default=5, ge=1, le=20)
    fishial_max_fish_per_session: int = Field(default=20, ge=1, le=200)
    # Safety FLOORS, not selection criteria. Selection is the per-track ranking in
    # LiveTracker._stage_species_crop. Measured over the two reference sessions
    # (5,239 detections, 1280x720): confidence 0.50 is the detector's own floor and
    # rejects 0%; short side 40 px sits at Coral City's 10th percentile (rejects
    # 9.7% there, 0.5% on SmartBay 3). The previous 0.70/96 pair sat ABOVE the 90th
    # percentile of both measurements and rejected 96.8%/85.9% of frames on its own.
    # Any future change to either number must cite a measured percentile.
    fishial_min_frame_confidence: float = Field(default=0.50, ge=0, le=1)
    fishial_min_crop_pixels: int = Field(default=40, ge=16)
    fishial_edge_margin_pixels: int = Field(default=4, ge=0)
    fishial_crop_margin: float = Field(default=1.15, ge=1.0)
    fishial_blur_min_variance: float = Field(default=0.0, ge=0)
    fishial_min_frame_separation_seconds: float = Field(default=0.6, ge=0)
    fishial_min_frames_to_vote: int = Field(default=3, ge=1)
    fishial_min_votes: int = Field(default=3, ge=1)
    fishial_vote_ratio: float = Field(default=0.6, ge=0, le=1)
    fishial_min_species_score: float = Field(default=0.5, ge=0, le=1)
    fishial_min_frame_margin: float = Field(default=0.0, ge=0)
    fishial_max_api_retries: int = Field(default=2, ge=0)
    # --- Unpaid candidate pool (staging is local and free; only calls cost) ---
    # 0 = auto: max(3 * fish_target, fish_target + 8), fixed at session start.
    fishial_candidate_pool_size: int = Field(default=0, ge=0)
    # 0 = auto: frames_per_fish + 2, so the selector has spare frames to rank.
    fishial_max_staged_frames_per_candidate: int = Field(default=0, ge=0)
    fishial_max_staged_bytes: int = Field(default=512 * 1024 * 1024, ge=0)
    # --- Selection ---
    fishial_quality_floor: float = Field(default=0.0, ge=0)
    fishial_quality_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_QUALITY_WEIGHTS),
        description="Relative weights blended by species_quality.track_quality.",
    )
    fishial_late_reserve_fish: int = Field(default=2, ge=0)
    fishial_reserve_after_seconds: float = Field(default=180.0, ge=0)
    # --- Early stopping (0 = disabled) ---
    fishial_max_empty_responses: int = Field(default=2, ge=0)
    # --- Multi-object resolution ---
    fishial_object_match_min_iou: float = Field(default=0.5, ge=0, le=1)
    fishial_object_match_min_margin: float = Field(default=0.2, ge=0, le=1)
    # --- Regional plausibility ---
    fishial_region_filter_enabled: bool = True
    # --- Underwater preprocessing, applied only to the staged identification crop ---
    fishial_preprocess: str = "none"
    fishial_clahe_clip: float = Field(default=2.0, gt=0)
    fishial_upscale_short_side: int = Field(default=0, ge=0)
    # Retains fish imagery on the output volume for offline replay. Off by default.
    fishial_keep_staged_crops: bool = False
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
        hide_input_in_errors=True,
    )

    @model_validator(mode="after")
    def _check_live_sources(self) -> "Settings":
        if self.fishial_enabled and (
            not self.fishial_client_id or not self.fishial_client_id.strip()
            or not self.fishial_client_secret
            or not self.fishial_client_secret.get_secret_value().strip()
        ):
            raise ValueError("Fishial requires a client ID and secret")
        if self.fishial_min_votes > self.fishial_default_frames_per_fish:
            raise ValueError("fishial_min_votes must not exceed default frames per fish")
        if self.fishial_min_frames_to_vote > self.fishial_default_frames_per_fish:
            raise ValueError("fishial_min_frames_to_vote must not exceed default frames per fish")
        if self.fishial_late_reserve_fish >= self.fishial_max_fish_per_session:
            raise ValueError("fishial_late_reserve_fish must be below the per-session fish ceiling")
        if (self.fishial_max_staged_frames_per_candidate
                and self.fishial_max_staged_frames_per_candidate < self.fishial_min_frames_to_vote):
            raise ValueError("fishial_max_staged_frames_per_candidate must not be below "
                             "fishial_min_frames_to_vote")
        if self.fishial_preprocess not in PREPROCESS_MODES:
            raise ValueError(f"fishial_preprocess must be one of {', '.join(PREPROCESS_MODES)}")
        weights = self.fishial_quality_weights
        if set(weights) != set(DEFAULT_QUALITY_WEIGHTS):
            raise ValueError(f"fishial_quality_weights must define exactly "
                             f"{', '.join(sorted(DEFAULT_QUALITY_WEIGHTS))}")
        if any(value < 0 for value in weights.values()) or not any(weights.values()):
            raise ValueError("fishial_quality_weights must be non-negative and not all zero")
        if urlparse(self.fishial_api_base_url).scheme != "https":
            raise ValueError("Fishial API base URL must use HTTPS")
        if not self.live_source_registry:
            # A deployment that passes an empty array (as Docker Compose does for an
            # unset LIVE_SOURCE_REGISTRY) has not overridden anything.
            self.live_source_registry = [LiveCameraSource(**entry) for entry in DEFAULT_LIVE_SOURCES]
        if self.coral_city_url != DEFAULT_CORAL_CITY_URL:
            self.live_source_registry = [
                source.model_copy(update={"url": self.coral_city_url})
                if source.key == "coral-city" else source
                for source in self.live_source_registry
            ]
        keys: set[str] = set()
        for source in self.live_source_registry:
            # A key travels in a URL path and is stored in a String(64) column.
            if not LIVE_SOURCE_KEY_PATTERN.match(source.key):
                raise ValueError(f"Live source key {source.key!r} must match {LIVE_SOURCE_KEY_PATTERN.pattern}")
            if source.key in keys:
                raise ValueError(f"Duplicate live source key {source.key!r}")
            if not source.label.strip():
                raise ValueError(f"Live source {source.key!r} needs a label")
            if urlparse(source.url).scheme not in {"http", "https"}:
                raise ValueError(f"Live source {source.key!r} needs an HTTP(S) URL")
            keys.add(source.key)
        if self.live_default_source_key not in keys:
            raise ValueError(
                f"LIVE_DEFAULT_SOURCE_KEY {self.live_default_source_key!r} is not in the camera registry"
            )
        return self

    def live_sources(self) -> dict[str, LiveCameraSource]:
        """Configured cameras in dropdown order."""

        return {source.key: source for source in self.live_source_registry}

    def live_source(self, key: str | None) -> LiveCameraSource | None:
        """Look up one camera; unknown or malformed keys resolve to None."""

        return self.live_sources().get(key) if key else None

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
