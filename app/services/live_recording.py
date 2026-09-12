"""Turn a finished live session into an ordinary, reviewable video.

Live monitoring analyses the camera in short chunks and throws each chunk away as
soon as its frames have been tracked, because an unbounded capture directory is
what a session that runs for months cannot afford. Reviewing a session afterwards
needs the opposite: the footage the detections actually describe.

``RecordingWriter`` bridges the two by keeping each chunk *after* the worker has
committed its frames, so the retained recording holds exactly the frames
``LiveTracker`` processed, in the order it processed them. ``frames_processed`` is
the session's global frame counter and is what ``LiveFishDetection.frame_number``
stores, so a detection's frame number indexes the assembled recording directly.
Chunks that were dropped, failed inference, or arrived while reconnecting were
never counted and are never recorded, so the mapping survives every gap without a
correction table.

The assembled recording is then registered as a ``Video`` and the session's live
tracks are copied into ``fish_tracks``/``fish_detections``, which is all the
existing annotation, review, clip and export paths need to work on camera footage.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import (
    FishDetection,
    FishTrack,
    LiveFishDetection,
    LiveFishTrack,
    LiveMonitorSession,
    Video,
    VideoProcessingStatus,
)
from app.services.live_monitor import live_path
from app.services.video_service import register_video_from_path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
# Chunk names sort as they were appended, which is the order they were analyzed.
PART_PATTERN = "[0-9]*.mp4"


class LiveRecordingError(RuntimeError):
    """The recording could not be assembled. The session's data is unaffected."""


@dataclass(frozen=True)
class RecordingPart:
    """One analyzed chunk, and how its detection boxes map onto the recording.

    A reconnect can resolve a rendition with a different aspect ratio, which the
    capture filter scales to a different frame size. Such a chunk is letterboxed
    into the recording's geometry, so the boxes stored against its frames have to
    move with it. Identical geometry - every chunk of an ordinary session - leaves
    the transform as the identity.
    """

    start_frame: int
    frames: int
    width: int
    height: int
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0

    @property
    def identity(self) -> bool:
        return self.scale == 1.0 and not self.offset_x and not self.offset_y

    def place(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        if self.identity:
            return x1, y1, x2, y2
        return (self.offset_x + x1 * self.scale, self.offset_y + y1 * self.scale,
                self.offset_x + x2 * self.scale, self.offset_y + y2 * self.scale)


@dataclass(frozen=True)
class AssembledRecording:
    path: Path
    frames: int
    width: int
    height: int
    fps: float
    parts: tuple[RecordingPart, ...]
    truncated: bool

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def part_for(self, frame_number: int) -> RecordingPart | None:
        starts = [part.start_frame for part in self.parts]
        index = bisect.bisect_right(starts, frame_number) - 1
        if index < 0:
            return None
        part = self.parts[index]
        return part if frame_number < part.start_frame + part.frames else None


def _letterbox(width: int, height: int, target_width: int,
               target_height: int) -> tuple[int, int, int, int]:
    """Centred inner size and offset for fitting one chunk into the recording.

    The inner size is computed here rather than left to FFmpeg's rounding so the
    pad filter and the box transform cannot disagree about where the frame landed.
    """

    ratio = min(target_width / width, target_height / height)
    inner_width = max(2, int(width * ratio) // 2 * 2)
    inner_height = max(2, int(height * ratio) // 2 * 2)
    return (inner_width, inner_height,
            (target_width - inner_width) // 2, (target_height - inner_height) // 2)


class RecordingWriter:
    """Retain analyzed chunks for a running session, bounded by a byte budget."""

    def __init__(self, session_id: uuid.UUID, settings: Settings):
        self.session_id = session_id
        self.settings = settings
        self.directory = live_path(settings, str(session_id), "recording")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parts: list[RecordingPart] = []
        self.frames = 0
        self.bytes = 0
        self.width: int | None = None
        self.height: int | None = None
        self.truncated = False

    @classmethod
    def resume(cls, session_id: uuid.UUID, settings: Settings) -> RecordingWriter:
        """Reopen what a crashed worker left behind, from the persisted manifest."""

        writer = cls(session_id, settings)
        try:
            payload = json.loads((writer.directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return writer
        parts = [RecordingPart(**entry) for entry in payload.get("parts", [])]
        # Trust the files on disk over the manifest: the last append may have been
        # interrupted between moving the chunk and rewriting the manifest.
        available = len(sorted(writer.directory.glob(PART_PATTERN)))
        writer.parts = parts[:available]
        writer.frames = sum(part.frames for part in writer.parts)
        writer.width, writer.height = payload.get("width"), payload.get("height")
        writer.truncated = bool(payload.get("truncated")) or len(writer.parts) < len(parts)
        writer.bytes = sum(path.stat().st_size for path in writer.directory.glob(PART_PATTERN))
        return writer

    def append(self, path: Path, frames: int, width: int, height: int) -> bool:
        """Take one analyzed chunk into the recording. Never raises.

        Returns whether the chunk was retained. A recording failure must never end
        a monitoring session, so every problem here is logged and the session
        simply keeps the footage it already has.
        """

        if frames <= 0 or not path.is_file():
            return False
        try:
            return self._append(path, frames, width, height)
        except (OSError, subprocess.SubprocessError, LiveRecordingError):
            logger.warning("Live chunk could not be added to the recording for session %s",
                           self.session_id, exc_info=True)
            self.truncated = True
            return False

    def _append(self, path: Path, frames: int, width: int, height: int) -> bool:
        if self.width is None:
            self.width, self.height = width, height
        target = self.directory / f"{len(self.parts):08d}.mp4"
        geometry = RecordingPart(self.frames, frames, self.width, self.height)
        if (width, height) != (self.width, self.height):
            inner_width, inner_height, offset_x, offset_y = _letterbox(
                width, height, self.width, self.height)
            geometry = RecordingPart(self.frames, frames, self.width, self.height,
                                     min(inner_width / width, inner_height / height),
                                     offset_x, offset_y)
            self._normalize(path, target, width, height,
                            (inner_width, inner_height, offset_x, offset_y))
        size = target.stat().st_size if target.is_file() else path.stat().st_size
        cap = self.settings.live_recording_max_bytes
        if cap and self.bytes + size > cap:
            target.unlink(missing_ok=True)
            if not self.truncated:
                logger.warning(
                    "Live recording for session %s reached LIVE_RECORDING_MAX_BYTES (%s); "
                    "monitoring continues but no further footage is retained",
                    self.session_id, cap)
            self.truncated = True
            return False
        if not target.is_file():
            os.replace(path, target)
        self.parts.append(geometry)
        self.frames += frames
        self.bytes += size
        self._write_manifest()
        return True

    def _normalize(self, path: Path, target: Path, width: int, height: int,
                   box: tuple[int, int, int, int]) -> None:
        """Letterbox a differently shaped chunk so the recording stays one stream."""

        inner_width, inner_height, offset_x, offset_y = box
        logger.info("Live recording for session %s letterboxes a %sx%s chunk into %sx%s",
                    self.session_id, width, height, self.width, self.height)
        _run_ffmpeg(
            ["-i", str(path), "-an", "-vf",
             (f"scale={inner_width}:{inner_height},"
              f"pad={self.width}:{self.height}:{offset_x}:{offset_y}"),
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(target)],
            timeout=self.settings.live_detector_timeout_seconds,
            failure="A live chunk could not be letterboxed into the recording",
        )

    def _write_manifest(self) -> None:
        payload = {"session_id": str(self.session_id), "fps": self.settings.live_fps,
                   "width": self.width, "height": self.height, "truncated": self.truncated,
                   "parts": [part.__dict__ for part in self.parts]}
        temp = self.directory / f"{MANIFEST_NAME}.tmp"
        temp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temp, self.directory / MANIFEST_NAME)

    def discard(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def assemble(self) -> AssembledRecording:
        """Concatenate the retained chunks into one playable MP4.

        Every chunk carries the capture encoder's settings and, after
        normalization, its geometry, so this is a stream copy: assembling an hour
        of footage costs seconds rather than a re-encode.
        """

        if not self.frames:
            raise LiveRecordingError("This session analyzed no camera frames")
        chunks = sorted(self.directory.glob(PART_PATTERN))[:len(self.parts)]
        if not chunks:
            raise LiveRecordingError("This session's recorded chunks are no longer on disk")
        root = self.settings.upload_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"live-{self.session_id}.mp4"
        listing = self.directory / "concat.txt"
        listing.write_text("".join(f"file '{path.resolve().as_posix()}'\n" for path in chunks),
                           encoding="utf-8")
        fps = self.settings.live_fps
        timeout = int(60 + self.frames / fps) if fps else 600
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                     "-movflags", "+faststart", str(output)],
                    timeout=timeout, failure="The live recording could not be assembled")
        recorded = _probe_frame_count(output, timeout)
        if recorded is not None and recorded != self.frames:
            # Boxes are placed by frame index, so a drift here would misplace every
            # detection after it. Report it rather than annotating silently wrong.
            logger.warning("Live recording for session %s holds %s frames, expected %s",
                           self.session_id, recorded, self.frames)
        return AssembledRecording(
            path=output, frames=self.frames, width=self.width or 0, height=self.height or 0,
            fps=fps, parts=tuple(self.parts), truncated=self.truncated,
        )


def _run_ffmpeg(arguments: list[str], timeout: int, failure: str) -> None:
    try:
        completed = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *arguments],
                                   capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise LiveRecordingError("Live recordings require FFmpeg on PATH") from exc
    except subprocess.SubprocessError as exc:
        raise LiveRecordingError(failure) from exc
    if completed.returncode != 0:
        logger.warning("FFmpeg failed: %s", (completed.stderr or "").strip()[:500])
        raise LiveRecordingError(failure)


def _probe_frame_count(path: Path, timeout: int) -> int | None:
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def register_live_recording_as_video(
    db: Session,
    session: LiveMonitorSession,
    recording: AssembledRecording,
    settings: Settings,
) -> Video:
    """Create the library row for an assembled recording.

    Processing is already done - the session analyzed every retained frame while
    it ran - so the video is registered as completed and never queues a job.
    """

    started = session.started_at or session.created_at
    filename = f"live-{session.source_key}-{started:%Y%m%d-%H%M%S}.mp4"
    video = register_video_from_path(db=db, path=recording.path, original_filename=filename,
                                     settings=settings, camera_id=session.source_key)
    video.is_live_recording = True
    video.source_session_id = session.id
    video.processing_status = VideoProcessingStatus.COMPLETED.value
    video.fps = video.fps or recording.fps
    video.width = video.width or recording.width
    video.height = video.height or recording.height
    video.duration_seconds = video.duration_seconds or recording.duration_seconds
    db.commit()
    db.refresh(video)
    return video


def convert_live_detections_to_video_detections(
    db: Session,
    session: LiveMonitorSession,
    video: Video,
    recording: AssembledRecording,
) -> int:
    """Copy a session's live tracks and observations onto the recording.

    Only observations of frames the recording actually holds are copied: a
    truncated recording must not claim boxes for footage it does not contain.
    Every copied track starts ``unreviewed`` - live monitoring made no review
    decisions, and the library is where those are made.
    """

    # Only a reconnect onto a differently shaped rendition moves boxes, so an
    # ordinary session never pays for the lookup.
    reshaped = any(not part.identity for part in recording.parts)
    fps = recording.fps or 1.0
    labels: set[str] = set()
    buffer = _InsertBuffer(db)
    for live_track in db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == session.id).order_by(
                LiveFishTrack.first_seen_at, LiveFishTrack.id)):
        rows = list(db.scalars(select(LiveFishDetection).where(
            LiveFishDetection.track_id == live_track.id,
            LiveFishDetection.frame_number < recording.frames,
        ).order_by(LiveFishDetection.frame_number, LiveFishDetection.id)))
        if not rows:
            continue
        track_id = uuid.uuid4()
        confidences = [row.confidence for row in rows]
        frames = [row.frame_number for row in rows]
        buffer.track({
            "id": track_id, "video_id": video.id,
            # The label an operator already saw on the live view and its crops.
            "viame_track_id": _unique_label(live_track.id, labels),
            "processing_job_id": None, "review_state": "unreviewed", "reviewed_at": None,
            "first_frame": frames[0], "last_frame": frames[-1],
            "first_timestamp_seconds": frames[0] / fps,
            "last_timestamp_seconds": frames[-1] / fps,
            "detection_count": len(rows),
            "mean_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            # A Fishial identification supersedes the detector's own class name.
            "species": live_track.fishial_species or live_track.species,
            "species_confidence": live_track.fishial_species_confidence,
        })
        for row in rows:
            part = recording.part_for(row.frame_number) if reshaped else None
            x1, y1, x2, y2 = (part.place(row.x1, row.y1, row.x2, row.y2) if part
                              else (row.x1, row.y1, row.x2, row.y2))
            buffer.detection({
                "id": uuid.uuid4(), "fish_track_id": track_id,
                "frame_number": row.frame_number, "timestamp_seconds": row.frame_number / fps,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": row.confidence,
                "class_name": row.species, "class_confidence": None,
            })
    return buffer.finish()


class _InsertBuffer:
    """Write converted rows in batches, so a months-long session still fits in memory."""

    LIMIT = 5_000

    def __init__(self, db: Session):
        self.db = db
        self.tracks: list[dict] = []
        self.detections: list[dict] = []
        self.written = 0

    def track(self, mapping: dict) -> None:
        self.tracks.append(mapping)

    def detection(self, mapping: dict) -> None:
        self.detections.append(mapping)
        if len(self.detections) >= self.LIMIT:
            self.flush()

    def flush(self) -> None:
        if self.tracks:
            # Always ahead of the detections that reference them.
            self.db.bulk_insert_mappings(FishTrack, self.tracks)
            self.tracks = []
        if self.detections:
            self.db.bulk_insert_mappings(FishDetection, self.detections)
            self.written += len(self.detections)
            self.detections = []

    def finish(self) -> int:
        self.flush()
        self.db.commit()
        return self.written


def _discard_failed_registration(db: Session, video: Video | None) -> None:
    if video is None:
        return
    try:
        db.rollback()
        row = db.get(Video, video.id)
        if row is not None:
            db.delete(row)
            db.commit()
    except Exception:
        logger.exception("A failed live recording registration could not be rolled back")


def _unique_label(track_id: uuid.UUID, taken: set[str]) -> str:
    """The live view's 8-character track label, kept unique per recording."""

    label = str(track_id)[:8]
    if label in taken:
        label = str(track_id)
    taken.add(label)
    return label


def finalize_live_recording(
    db: Session,
    session: LiveMonitorSession,
    settings: Settings,
    writer: RecordingWriter | None = None,
) -> Video | None:
    """Assemble, register and convert a finished session's recording.

    Returns the new video, or ``None`` when the session retained no footage.
    Raises only on a genuine failure, which callers report without failing the
    session: the live tracks, crops and clips stand on their own.
    """

    writer = writer or RecordingWriter.resume(session.id, settings)
    if not writer.frames:
        writer.discard()
        return None
    existing = db.scalar(select(Video).where(Video.source_session_id == session.id))
    if existing is not None:
        # A recovery pass must never register the same session twice.
        writer.discard()
        return existing
    recording = writer.assemble()
    video = None
    try:
        video = register_live_recording_as_video(db, session, recording, settings)
        count = convert_live_detections_to_video_detections(db, session, video, recording)
    except Exception:
        # A half-converted recording would show a library row whose boxes are
        # missing, which reads as "the camera saw nothing". Take it back out.
        _discard_failed_registration(db, video)
        recording.path.unlink(missing_ok=True)
        raise
    writer.discard()
    logger.info("Live session %s recorded %s frames (%.1fs) as video %s with %s observations",
                session.id, recording.frames, recording.duration_seconds, video.id, count)
    return video
