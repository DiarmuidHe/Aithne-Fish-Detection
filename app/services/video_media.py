from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEDIA_TYPE_MP4 = "video/mp4"

TRACK_PALETTE = (
    (66, 135, 245),
    (80, 200, 120),
    (245, 166, 35),
    (235, 87, 87),
    (155, 89, 182),
    (39, 174, 96),
    (47, 128, 237),
    (242, 201, 76),
)


class MediaDependencyError(RuntimeError):
    """OpenCV is not installed in this environment."""


class MediaPathError(RuntimeError):
    """A media path escaped the directory it must stay inside."""


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise MediaDependencyError("OpenCV is required for video rendering") from exc
    return cv2


def open_video_writer(
    cv2: Any, output_path: Path, fps: float, width: int, height: int
) -> Any | None:
    """Return an opened writer, or ``None`` when no available codec accepts the file."""

    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    return None


def track_color(track_label: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(track_label.encode("utf-8")).hexdigest()
    return TRACK_PALETTE[int(digest[:8], 16) % len(TRACK_PALETTE)]


def valid_fps(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return float(value)


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MediaPathError(f"{path} is outside {root}") from exc


def make_browser_compatible(path: Path, timeout_seconds: int) -> None:
    """Prefer H.264/yuv420p so desktop and tablet browsers can play the result.

    OpenCV's broadly available ``mp4v`` writer is retained as a fallback when
    FFmpeg or libx264 is unavailable, keeping local rendering dependency-light.
    """

    transcoded_path = path.with_name(f"{path.stem}.h264{path.suffix}")
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(transcoded_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        logger.warning("FFmpeg unavailable; keeping OpenCV MP4 output")
        remove_file(transcoded_path)
        return

    if completed.returncode != 0 or not transcoded_path.is_file():
        logger.warning("FFmpeg H.264 conversion failed; keeping OpenCV MP4 output")
        remove_file(transcoded_path)
        return
    os.replace(transcoded_path, path)
