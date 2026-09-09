from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import FishDetection, FishTrack, Video, utc_now
from app.services import video_media
from app.services.video_media import MEDIA_TYPE_MP4

logger = logging.getLogger(__name__)


class VideoAnnotationError(RuntimeError):
    public_message = "Annotated video could not be generated"

    def __str__(self) -> str:
        return self.public_message


class VideoAnnotationDependencyError(VideoAnnotationError):
    public_message = "Video annotation requires OpenCV"


class SourceVideoMissingError(VideoAnnotationError):
    public_message = "Source video is not available"


class SourceVideoUnreadableError(VideoAnnotationError):
    public_message = "Source video could not be read as a video"


class NoDetectionsForAnnotationError(VideoAnnotationError):
    public_message = "No completed detections are available for this video"


class AnnotationWriteError(VideoAnnotationError):
    public_message = "Annotated video could not be written"


class AnnotationPathError(VideoAnnotationError):
    public_message = "Annotated video path is invalid"


class AnnotationNotGeneratedError(VideoAnnotationError):
    public_message = "Annotated video has not been generated"


class AnnotatedVideoUnavailableError(VideoAnnotationError):
    public_message = "Annotated video file is not available"


@dataclass(frozen=True)
class FrameAnnotation:
    frame_number: int
    track_label: str
    species_label: str | None
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class AnnotatedVideoResult:
    video_id: uuid.UUID
    output_path: Path
    annotated_at: datetime
    media_type: str
    size_bytes: int
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def filename(self) -> str:
        return self.output_path.name


def generate_annotated_video(
    db: Session,
    video: Video,
    settings: Settings,
    include_species: bool = True,
) -> AnnotatedVideoResult:
    source_path = Path(video.storage_path)
    if not source_path.is_file():
        raise SourceVideoMissingError()

    annotations_by_frame = _load_annotations_by_frame(db, video)
    if not annotations_by_frame:
        raise NoDetectionsForAnnotationError()

    output_root = _resolved_output_root(settings)
    output_path, temp_path = _new_output_paths(output_root)
    old_output_path = _existing_output_path(video, output_root)

    cv2 = _load_cv2()
    try:
        source_fps, width, height, frames_written = _render_video(
            cv2=cv2,
            source_path=source_path,
            temp_path=temp_path,
            video=video,
            annotations_by_frame=annotations_by_frame,
            include_species=include_species,
        )
    except VideoAnnotationError:
        _remove_file(temp_path)
        raise
    except Exception as exc:
        _remove_file(temp_path)
        raise AnnotationWriteError() from exc

    if frames_written == 0:
        _remove_file(temp_path)
        raise SourceVideoUnreadableError()

    _make_browser_compatible(temp_path, settings.viame_timeout_seconds)

    try:
        os.replace(temp_path, output_path)
    except OSError as exc:
        _remove_file(temp_path)
        raise AnnotationWriteError() from exc

    annotated_at = utc_now()
    try:
        video.annotated_video_path = str(output_path)
        video.annotated_at = annotated_at
        db.add(video)
        db.commit()
        db.refresh(video)
    except Exception:
        _remove_file(output_path)
        raise

    if old_output_path is not None and old_output_path != output_path:
        _remove_file(old_output_path)

    return AnnotatedVideoResult(
        video_id=video.id,
        output_path=output_path,
        annotated_at=annotated_at,
        media_type=MEDIA_TYPE_MP4,
        size_bytes=output_path.stat().st_size,
        fps=source_fps,
        width=width,
        height=height,
        frame_count=frames_written,
    )


def get_annotated_video_path(video: Video, settings: Settings) -> Path:
    output_root = _resolved_output_root(settings)
    if video.annotated_video_path is None:
        raise AnnotationNotGeneratedError()

    path = Path(video.annotated_video_path).expanduser().resolve()
    _ensure_under_root(path, output_root)
    if not path.is_file():
        raise AnnotatedVideoUnavailableError()
    return path


def _load_annotations_by_frame(db: Session, video: Video) -> dict[int, list[FrameAnnotation]]:
    rows = db.execute(
        select(FishDetection, FishTrack)
        .join(FishTrack, FishDetection.fish_track_id == FishTrack.id)
        .where(FishTrack.video_id == video.id)
        .order_by(FishDetection.frame_number, FishTrack.viame_track_id)
    ).all()

    annotations_by_frame: dict[int, list[FrameAnnotation]] = defaultdict(list)
    from app.services.fish_counter import is_accepted_track

    for detection, track in rows:
        if not is_accepted_track(track, video.confidence_threshold):
            continue
        annotations_by_frame[detection.frame_number].append(
            FrameAnnotation(
                frame_number=detection.frame_number,
                track_label=track.viame_track_id,
                species_label=detection.class_name or track.species,
                confidence=detection.confidence,
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
            )
        )
    return dict(annotations_by_frame)


def _load_cv2() -> Any:
    try:
        return video_media.load_cv2()
    except video_media.MediaDependencyError as exc:
        raise VideoAnnotationDependencyError() from exc


def _render_video(
    cv2: Any,
    source_path: Path,
    temp_path: Path,
    video: Video,
    annotations_by_frame: dict[int, list[FrameAnnotation]],
    include_species: bool,
) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(source_path))
    writer = None
    try:
        if not capture.isOpened():
            raise SourceVideoUnreadableError()

        source_fps = _valid_fps(capture.get(cv2.CAP_PROP_FPS)) or _valid_fps(video.fps) or 30.0
        success, frame = capture.read()
        if not success or frame is None:
            raise SourceVideoUnreadableError()

        height, width = frame.shape[:2]
        writer = _open_video_writer(cv2, temp_path, source_fps, width, height)

        frame_number = 0
        frames_written = 0
        while success and frame is not None:
            for annotation in annotations_by_frame.get(frame_number, []):
                _draw_annotation(cv2, frame, annotation, include_species)
            writer.write(frame)
            frames_written += 1
            frame_number += 1
            success, frame = capture.read()

        return source_fps, width, height, frames_written
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _open_video_writer(cv2: Any, output_path: Path, fps: float, width: int, height: int):
    writer = video_media.open_video_writer(cv2, output_path, fps, width, height)
    if writer is None:
        raise AnnotationWriteError()
    return writer


def _draw_annotation(cv2: Any, frame, annotation: FrameAnnotation, include_species: bool) -> None:
    height, width = frame.shape[:2]
    bounds = _clamped_bounds(annotation, width, height)
    if bounds is None:
        return

    x1, y1, x2, y2 = bounds
    color = _track_color(annotation.track_label)
    thickness = max(1, round(min(width, height) / 240))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = _label_text(annotation, include_species)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.8, min(width, height) / 720))
    text_thickness = max(1, thickness - 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )

    text_x = min(max(x1, 0), max(0, width - text_width - 6))
    if y1 - text_height - baseline - 8 >= 0:
        text_y = y1 - 6
    else:
        text_y = min(y2 + text_height + baseline + 6, height - baseline - 4)

    background_top = max(0, text_y - text_height - baseline - 4)
    background_bottom = min(height - 1, text_y + baseline + 4)
    background_right = min(width - 1, text_x + text_width + 6)
    cv2.rectangle(
        frame,
        (text_x, background_top),
        (background_right, background_bottom),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_x + 3, text_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _clamped_bounds(
    annotation: FrameAnnotation, width: int, height: int
) -> tuple[int, int, int, int] | None:
    left = int(round(min(annotation.x1, annotation.x2)))
    right = int(round(max(annotation.x1, annotation.x2)))
    top = int(round(min(annotation.y1, annotation.y2)))
    bottom = int(round(max(annotation.y1, annotation.y2)))

    if right < 0 or bottom < 0 or left >= width or top >= height:
        return None

    x1 = _clamp(left, 0, width - 1)
    x2 = _clamp(right, 0, width - 1)
    y1 = _clamp(top, 0, height - 1)
    y2 = _clamp(bottom, 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _label_text(annotation: FrameAnnotation, include_species: bool) -> str:
    if include_species and annotation.species_label:
        return (
            f"track {annotation.track_label} | "
            f"{annotation.species_label} {annotation.confidence:.2f}"
        )
    return f"track {annotation.track_label} | {annotation.confidence:.2f}"


_track_color = video_media.track_color
_valid_fps = video_media.valid_fps
_clamp = video_media.clamp
_remove_file = video_media.remove_file


def _resolved_output_root(settings: Settings) -> Path:
    root = settings.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_output_paths(output_root: Path) -> tuple[Path, Path]:
    output_path = (output_root / f"{uuid.uuid4()}.mp4").resolve()
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}").resolve()
    _ensure_under_root(output_path, output_root)
    _ensure_under_root(temp_path, output_root)
    return output_path, temp_path


def _existing_output_path(video: Video, output_root: Path) -> Path | None:
    if video.annotated_video_path is None:
        return None
    try:
        path = Path(video.annotated_video_path).expanduser().resolve()
        _ensure_under_root(path, output_root)
    except VideoAnnotationError:
        return None
    return path


def _ensure_under_root(path: Path, output_root: Path) -> None:
    try:
        video_media.ensure_under_root(path, output_root)
    except video_media.MediaPathError as exc:
        raise AnnotationPathError() from exc


def _make_browser_compatible(path: Path, timeout_seconds: int) -> None:
    video_media.make_browser_compatible(path, timeout_seconds)
