"""Short cropped clips that follow one annotated fish through the source video.

Clips are derived data: they depend only on the stored detections for a track,
never on review decisions, so they are cached on disk beside the annotated video
and rebuilt only when processing replaces the tracks they were rendered from.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import FishDetection, FishTrack, Video, utc_now
from app.services import video_media
from app.services.video_media import MEDIA_TYPE_MP4

CLIP_DIRECTORY_NAME = "clips"
MAX_UPSCALE = 4.0

logger = logging.getLogger(__name__)


class TrackClipError(RuntimeError):
    public_message = "Fish clip could not be generated"

    def __str__(self) -> str:
        return self.public_message


class TrackClipDependencyError(TrackClipError):
    public_message = "Fish clips require OpenCV"


class ClipSourceVideoMissingError(TrackClipError):
    public_message = "Source video is not available"


class ClipSourceVideoUnreadableError(TrackClipError):
    public_message = "Source video could not be read as a video"


class NoDetectionsForClipError(TrackClipError):
    public_message = "No stored detections are available for this fish track"


class ClipWriteError(TrackClipError):
    public_message = "Fish clip could not be written"


class ClipPathError(TrackClipError):
    public_message = "Fish clip path is invalid"


class ClipNotGeneratedError(TrackClipError):
    public_message = "Fish clip has not been generated"


class ClipUnavailableError(TrackClipError):
    public_message = "Fish clip file is not available"


@dataclass(frozen=True)
class TrackClipResult:
    track_id: uuid.UUID
    video_id: uuid.UUID
    viame_track_id: str
    species: str | None
    path: Path
    media_type: str
    size_bytes: int
    fps: float
    width: int
    height: int
    frame_count: int
    start_seconds: float
    end_seconds: float
    detection_count: int
    max_confidence: float
    generated_at: datetime
    cached: bool

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(frozen=True)
class _ClipPlan:
    """Everything needed to cut one track out of the source video in one pass."""

    track_id: uuid.UUID
    viame_track_id: str
    species: str | None
    detection_count: int
    max_confidence: float
    start_frame: int
    end_frame: int
    crop_width: int
    crop_height: int
    width: int
    height: int
    origins: dict[int, tuple[int, int]]
    boxes: dict[int, tuple[float, float, float, float, float]]
    color: tuple[int, int, int]


def generate_track_clips(
    db: Session,
    video: Video,
    tracks: Sequence[FishTrack],
    settings: Settings,
    refresh: bool = False,
) -> list[TrackClipResult]:
    """Render (or reuse) one cropped clip per track in a single pass over the source."""

    if not tracks:
        return []

    source_path = Path(video.storage_path)
    if not source_path.is_file():
        raise ClipSourceVideoMissingError()

    results: list[TrackClipResult] = []
    pending: list[FishTrack] = []
    for track in tracks:
        cached = None if refresh else existing_clip(track, settings)
        if cached is None:
            pending.append(track)
        else:
            results.append(cached)

    if pending:
        results.extend(_render_clips(db, video, pending, source_path, settings))

    order = {track.id: index for index, track in enumerate(tracks)}
    return sorted(results, key=lambda result: order[result.track_id])


def generate_track_clip(
    db: Session,
    video: Video,
    track: FishTrack,
    settings: Settings,
    refresh: bool = False,
) -> TrackClipResult:
    results = generate_track_clips(db, video, [track], settings, refresh=refresh)
    if not results:
        raise NoDetectionsForClipError()
    return results[0]


def existing_clip(track: FishTrack, settings: Settings) -> TrackClipResult | None:
    """Return the cached clip for a track, or ``None`` when it must be rendered."""

    clip_path, metadata_path = clip_paths(track.video_id, track.id, settings)
    if not clip_path.is_file() or not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return TrackClipResult(
            track_id=track.id,
            video_id=track.video_id,
            viame_track_id=payload["viame_track_id"],
            species=payload["species"],
            path=clip_path,
            media_type=MEDIA_TYPE_MP4,
            size_bytes=clip_path.stat().st_size,
            fps=float(payload["fps"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            frame_count=int(payload["frame_count"]),
            start_seconds=float(payload["start_seconds"]),
            end_seconds=float(payload["end_seconds"]),
            detection_count=int(payload["detection_count"]),
            max_confidence=float(payload["max_confidence"]),
            generated_at=datetime.fromisoformat(payload["generated_at"]),
            cached=True,
        )
    except (KeyError, TypeError, ValueError, OSError):
        return None


def get_clip_path(track: FishTrack, settings: Settings) -> Path:
    clip_path, metadata_path = clip_paths(track.video_id, track.id, settings)
    if not metadata_path.is_file():
        raise ClipNotGeneratedError()
    if not clip_path.is_file():
        raise ClipUnavailableError()
    return clip_path


def remove_video_clips(video_id: uuid.UUID, settings: Settings) -> None:
    """Drop cached clips whose tracks are about to be replaced by a new run."""

    directory = video_clip_directory(video_id, settings)
    if not directory.is_dir():
        return
    try:
        shutil.rmtree(directory)
    except OSError:
        logger.warning("could not remove cached fish clips video_id=%s", video_id)


def clip_root(settings: Settings) -> Path:
    root = (settings.output_root.expanduser().resolve() / CLIP_DIRECTORY_NAME).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def video_clip_directory(video_id: uuid.UUID, settings: Settings) -> Path:
    root = clip_root(settings)
    directory = (root / str(video_id)).resolve()
    _ensure_under_root(directory, root)
    return directory


def clip_paths(
    video_id: uuid.UUID, track_id: uuid.UUID, settings: Settings
) -> tuple[Path, Path]:
    directory = video_clip_directory(video_id, settings)
    clip_path = (directory / f"{track_id}.mp4").resolve()
    metadata_path = (directory / f"{track_id}.json").resolve()
    _ensure_under_root(clip_path, directory)
    _ensure_under_root(metadata_path, directory)
    return clip_path, metadata_path


def _render_clips(
    db: Session,
    video: Video,
    tracks: Sequence[FishTrack],
    source_path: Path,
    settings: Settings,
) -> list[TrackClipResult]:
    cv2 = _load_cv2()
    detections_by_track = _load_detections(db, [track.id for track in tracks])

    capture = cv2.VideoCapture(str(source_path))
    writers: dict[uuid.UUID, Any] = {}
    frames_written: dict[uuid.UUID, int] = defaultdict(int)
    temp_paths: dict[uuid.UUID, Path] = {}
    plans: list[_ClipPlan] = []
    try:
        if not capture.isOpened():
            raise ClipSourceVideoUnreadableError()

        source_fps = (
            video_media.valid_fps(capture.get(cv2.CAP_PROP_FPS))
            or video_media.valid_fps(video.fps)
            or 30.0
        )
        success, frame = capture.read()
        if not success or frame is None:
            raise ClipSourceVideoUnreadableError()
        height, width = frame.shape[:2]

        for track in tracks:
            detections = detections_by_track.get(track.id, [])
            if not detections:
                continue
            plans.append(_build_plan(track, detections, width, height, source_fps, settings))
        if not plans:
            return []

        directory = video_clip_directory(video.id, settings)
        directory.mkdir(parents=True, exist_ok=True)
        for plan in plans:
            clip_path, _ = clip_paths(video.id, plan.track_id, settings)
            temp_paths[plan.track_id] = clip_path.with_name(
                f"{clip_path.stem}.{uuid.uuid4().hex}.tmp{clip_path.suffix}"
            )

        last_frame_needed = max(plan.end_frame for plan in plans)
        frame_number = 0
        while success and frame_number <= last_frame_needed:
            if frame is not None:
                for plan in plans:
                    if plan.start_frame <= frame_number <= plan.end_frame:
                        writer = writers.get(plan.track_id)
                        if writer is None:
                            writer = video_media.open_video_writer(
                                cv2,
                                temp_paths[plan.track_id],
                                source_fps,
                                plan.width,
                                plan.height,
                            )
                            if writer is None:
                                raise ClipWriteError()
                            writers[plan.track_id] = writer
                        writer.write(_clip_frame(cv2, frame, plan, frame_number))
                        frames_written[plan.track_id] += 1
                    elif frame_number > plan.end_frame and plan.track_id in writers:
                        writers.pop(plan.track_id).release()

            frame_number += 1
            if frame_number > last_frame_needed:
                break
            if _frame_is_needed(plans, frame_number):
                success, frame = capture.read()
            else:
                # Decoding is the expensive part; skip frames no clip covers.
                success = capture.grab()
                frame = None
    except TrackClipError:
        _discard_temp_files(temp_paths.values(), writers)
        raise
    except Exception as exc:
        _discard_temp_files(temp_paths.values(), writers)
        raise ClipWriteError() from exc
    finally:
        for writer in writers.values():
            writer.release()
        writers.clear()
        capture.release()

    results: list[TrackClipResult] = []
    for plan in plans:
        temp_path = temp_paths[plan.track_id]
        written = frames_written[plan.track_id]
        if written == 0:
            video_media.remove_file(temp_path)
            continue
        try:
            results.append(
                _finalize_clip(
                    video=video,
                    plan=plan,
                    temp_path=temp_path,
                    fps=source_fps,
                    frame_count=written,
                    settings=settings,
                )
            )
        except TrackClipError:
            video_media.remove_file(temp_path)
            raise
    return results


def _finalize_clip(
    video: Video,
    plan: _ClipPlan,
    temp_path: Path,
    fps: float,
    frame_count: int,
    settings: Settings,
) -> TrackClipResult:
    video_media.make_browser_compatible(temp_path, settings.viame_timeout_seconds)
    clip_path, metadata_path = clip_paths(video.id, plan.track_id, settings)
    try:
        os.replace(temp_path, clip_path)
    except OSError as exc:
        raise ClipWriteError() from exc

    generated_at = utc_now()
    payload = {
        "track_id": str(plan.track_id),
        "video_id": str(video.id),
        "viame_track_id": plan.viame_track_id,
        "species": plan.species,
        "fps": fps,
        "width": plan.width,
        "height": plan.height,
        "frame_count": frame_count,
        "start_seconds": plan.start_frame / fps,
        "end_seconds": (plan.start_frame + frame_count) / fps,
        "detection_count": plan.detection_count,
        "max_confidence": plan.max_confidence,
        "generated_at": generated_at.isoformat(),
    }
    try:
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        video_media.remove_file(clip_path)
        raise ClipWriteError() from exc

    return TrackClipResult(
        track_id=plan.track_id,
        video_id=video.id,
        viame_track_id=plan.viame_track_id,
        species=plan.species,
        path=clip_path,
        media_type=MEDIA_TYPE_MP4,
        size_bytes=clip_path.stat().st_size,
        fps=fps,
        width=plan.width,
        height=plan.height,
        frame_count=frame_count,
        start_seconds=payload["start_seconds"],
        end_seconds=payload["end_seconds"],
        detection_count=plan.detection_count,
        max_confidence=plan.max_confidence,
        generated_at=generated_at,
        cached=False,
    )


def _build_plan(
    track: FishTrack,
    detections: list[FishDetection],
    source_width: int,
    source_height: int,
    fps: float,
    settings: Settings,
) -> _ClipPlan:
    boxes: dict[int, tuple[float, float, float, float, float]] = {}
    centers: dict[int, tuple[float, float]] = {}
    for detection in detections:
        left = min(detection.x1, detection.x2)
        right = max(detection.x1, detection.x2)
        top = min(detection.y1, detection.y2)
        bottom = max(detection.y1, detection.y2)
        boxes[detection.frame_number] = (left, top, right, bottom, detection.confidence)
        centers[detection.frame_number] = ((left + right) / 2, (top + bottom) / 2)

    widest = max(right - left for left, _, right, _, _ in boxes.values())
    tallest = max(bottom - top for _, top, _, bottom, _ in boxes.values())
    crop_width = _even(
        video_media.clamp(
            round(max(widest, 1.0) * settings.clip_zoom_margin),
            min(settings.clip_min_crop_pixels, source_width),
            source_width,
        )
    )
    crop_height = _even(
        video_media.clamp(
            round(max(tallest, 1.0) * settings.clip_zoom_margin),
            min(settings.clip_min_crop_pixels, source_height),
            source_height,
        )
    )
    crop_width = max(2, min(crop_width, _even(source_width)))
    crop_height = max(2, min(crop_height, _even(source_height)))

    padding_frames = max(0, round(settings.clip_padding_seconds * fps))
    detection_frames = sorted(boxes)
    start_frame = max(0, detection_frames[0] - padding_frames)
    end_frame = detection_frames[-1] + padding_frames

    origins: dict[int, tuple[int, int]] = {}
    for frame_number in range(start_frame, end_frame + 1):
        center_x, center_y = _interpolated_center(detection_frames, centers, frame_number)
        origins[frame_number] = (
            video_media.clamp(
                round(center_x - crop_width / 2), 0, max(0, source_width - crop_width)
            ),
            video_media.clamp(
                round(center_y - crop_height / 2), 0, max(0, source_height - crop_height)
            ),
        )

    scale = min(
        MAX_UPSCALE,
        max(1.0, settings.clip_output_short_side / max(1, min(crop_width, crop_height))),
    )
    return _ClipPlan(
        track_id=track.id,
        viame_track_id=track.viame_track_id,
        species=track.species,
        detection_count=track.detection_count,
        max_confidence=track.max_confidence,
        start_frame=start_frame,
        end_frame=end_frame,
        crop_width=crop_width,
        crop_height=crop_height,
        width=_even(round(crop_width * scale)),
        height=_even(round(crop_height * scale)),
        origins=origins,
        boxes=boxes,
        color=video_media.track_color(track.viame_track_id),
    )


def _interpolated_center(
    detection_frames: list[int],
    centers: dict[int, tuple[float, float]],
    frame_number: int,
) -> tuple[float, float]:
    """Follow the fish smoothly across frames VIAME did not report."""

    if frame_number in centers:
        return centers[frame_number]
    index = bisect_left(detection_frames, frame_number)
    if index == 0:
        return centers[detection_frames[0]]
    if index >= len(detection_frames):
        return centers[detection_frames[-1]]
    before = detection_frames[index - 1]
    after = detection_frames[index]
    ratio = (frame_number - before) / (after - before)
    before_x, before_y = centers[before]
    after_x, after_y = centers[after]
    return (
        before_x + (after_x - before_x) * ratio,
        before_y + (after_y - before_y) * ratio,
    )


def _clip_frame(cv2: Any, frame: Any, plan: _ClipPlan, frame_number: int):
    left, top = plan.origins[frame_number]
    region = frame[top : top + plan.crop_height, left : left + plan.crop_width]
    if region.shape[0] != plan.crop_height or region.shape[1] != plan.crop_width:
        # Source frames can be shorter than the first frame in malformed files.
        region = cv2.copyMakeBorder(
            region,
            0,
            max(0, plan.crop_height - region.shape[0]),
            0,
            max(0, plan.crop_width - region.shape[1]),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    if (plan.width, plan.height) == (plan.crop_width, plan.crop_height):
        output = region.copy()
    else:
        output = cv2.resize(
            region, (plan.width, plan.height), interpolation=cv2.INTER_LINEAR
        )

    scale_x = plan.width / plan.crop_width
    scale_y = plan.height / plan.crop_height
    box = plan.boxes.get(frame_number)
    if box is not None:
        box_left, box_top, box_right, box_bottom, _ = box
        x1 = video_media.clamp(round((box_left - left) * scale_x), 0, plan.width - 1)
        y1 = video_media.clamp(round((box_top - top) * scale_y), 0, plan.height - 1)
        x2 = video_media.clamp(round((box_right - left) * scale_x), 0, plan.width - 1)
        y2 = video_media.clamp(round((box_bottom - top) * scale_y), 0, plan.height - 1)
        if x2 > x1 and y2 > y1:
            thickness = max(1, round(min(plan.width, plan.height) / 160))
            cv2.rectangle(output, (x1, y1), (x2, y2), plan.color, thickness)

    _draw_caption(cv2, output, plan, box)
    return output


def _draw_caption(cv2: Any, frame: Any, plan: _ClipPlan, box) -> None:
    label = f"track {plan.viame_track_id}"
    if plan.species:
        label = f"{label} | {plan.species}"
    if box is not None:
        label = f"{label} {box[4]:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.6, min(plan.width, plan.height) / 420))
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (0, 0),
        (min(plan.width - 1, text_width + 8), text_height + baseline + 8),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (4, text_height + 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _frame_is_needed(plans: Sequence[_ClipPlan], frame_number: int) -> bool:
    return any(plan.start_frame <= frame_number <= plan.end_frame for plan in plans)


def _load_detections(
    db: Session, track_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[FishDetection]]:
    rows = db.scalars(
        select(FishDetection)
        .where(FishDetection.fish_track_id.in_(list(track_ids)))
        .order_by(FishDetection.fish_track_id, FishDetection.frame_number)
    ).all()
    detections: dict[uuid.UUID, list[FishDetection]] = defaultdict(list)
    for detection in rows:
        detections[detection.fish_track_id].append(detection)
    return detections


def _discard_temp_files(paths, writers: dict[uuid.UUID, Any]) -> None:
    for writer in writers.values():
        writer.release()
    writers.clear()
    for path in paths:
        video_media.remove_file(path)


def _load_cv2() -> Any:
    try:
        return video_media.load_cv2()
    except video_media.MediaDependencyError as exc:
        raise TrackClipDependencyError() from exc


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        video_media.ensure_under_root(path, root)
    except video_media.MediaPathError as exc:
        raise ClipPathError() from exc


def _even(value: int) -> int:
    return int(value) - (int(value) % 2)
