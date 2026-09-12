"""One small crop of each fish, so a track table shows the animal and not just a row.

A track's identity is the fish, and until now every screen described that fish only
in numbers. A thumbnail is the cheapest possible answer to "what am I looking at?" -
and it is the same crop the identification path would send, so what an operator
judges by eye is what the classifier is being asked about.

Cost is why these are generated per video rather than per track. Reaching a frame
means grabbing forward through the source (never seeking - a crop taken from the
wrong frame would show the wrong fish), so producing one thumbnail costs almost the
same as producing all of them. One pass covers every track of a video, writes each
crop to the clip cache beside the clips, and later requests are plain file reads.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import FishDetection, FishTrack, Video
from app.services import video_media
from app.services.species_quality import frame_score
from app.services.track_clip import video_clip_directory

logger = logging.getLogger(__name__)

THUMBNAIL_PREFIX = "thumb-"
# Enough to stay crisp on a high-density display at the ~44px a table row gives it,
# and small enough that a video's whole set is a few hundred kilobytes.
THUMBNAIL_SIZE = 160
# Generated in one pass, so a video with thousands of tracks would otherwise hold a
# request open for minutes. The table pages long before this.
MAX_TRACKS_PER_PASS = 600


class ThumbnailError(RuntimeError):
    public_message = "Fish thumbnails could not be generated"

    def __str__(self) -> str:
        return self.public_message


def thumbnail_path(video_id: uuid.UUID, track_id: uuid.UUID, settings: Settings) -> Path:
    directory = video_clip_directory(video_id, settings)
    path = (directory / f"{THUMBNAIL_PREFIX}{track_id}.jpg").resolve()
    video_media.ensure_under_root(path, directory)
    return path


def existing_thumbnail(track: FishTrack, settings: Settings) -> Path | None:
    try:
        path = thumbnail_path(track.video_id, track.id, settings)
    except video_media.MediaPathError:
        return None
    return path if path.is_file() else None


def best_detection(detections: Sequence[FishDetection], settings: Settings):
    """The sighting that shows the fish best, by the same measures selection uses.

    Crop size and detector confidence are the two things knowable without decoding,
    and they are exactly what makes a crop legible. Ranking by the deployment's own
    quality weights keeps "the frame you are shown" and "the frame we would send"
    the same judgement rather than two unrelated ones.
    """

    best, best_score = None, -1.0
    for detection in detections:
        short_side = min(detection.x2 - detection.x1, detection.y2 - detection.y1)
        if short_side <= 0:
            continue
        score = frame_score({"short_side": short_side, "confidence": detection.confidence},
                            settings.fishial_quality_weights)
        if score > best_score:
            best, best_score = detection, score
    return best


def generate_video_thumbnails(db: Session, video: Video, settings: Settings,
                              refresh: bool = False) -> dict[uuid.UUID, Path]:
    """Ensure every track of ``video`` has a thumbnail; return the ones that exist.

    Never raises for a track it cannot render: a missing thumbnail is a blank cell,
    which is a far better outcome than a table that will not load.
    """

    tracks = list(db.scalars(
        select(FishTrack).where(FishTrack.video_id == video.id)
        .order_by(FishTrack.first_frame, FishTrack.id).limit(MAX_TRACKS_PER_PASS)))
    if not tracks:
        return {}

    found: dict[uuid.UUID, Path] = {}
    pending: list[FishTrack] = []
    for track in tracks:
        cached = None if refresh else existing_thumbnail(track, settings)
        if cached is None:
            pending.append(track)
        else:
            found[track.id] = cached
    if pending:
        found.update(_render(db, video, pending, settings))
    return found


def _render(db: Session, video: Video, tracks: Sequence[FishTrack],
            settings: Settings) -> dict[uuid.UUID, Path]:
    source = Path(video.storage_path)
    if not source.is_file():
        return {}
    try:
        cv2 = video_media.load_cv2()
    except video_media.MediaDependencyError:
        return {}

    rows = db.scalars(
        select(FishDetection)
        .where(FishDetection.fish_track_id.in_([track.id for track in tracks]))
        .order_by(FishDetection.frame_number)
    ).all()
    by_track: dict[uuid.UUID, list[FishDetection]] = {}
    for row in rows:
        by_track.setdefault(row.fish_track_id, []).append(row)

    # One frame per track, visited in file order so a single forward pass covers all.
    wanted: dict[int, list[tuple[FishTrack, FishDetection]]] = {}
    for track in tracks:
        detection = best_detection(by_track.get(track.id, []), settings)
        if detection is not None:
            wanted.setdefault(detection.frame_number, []).append((track, detection))
    if not wanted:
        return {}

    directory = video_clip_directory(video.id, settings)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[uuid.UUID, Path] = {}
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            return {}
        position = 0
        for number in sorted(wanted):
            while position < number:
                if not capture.grab():
                    return written
                position += 1
            success, frame = capture.read()
            position += 1
            if not success or frame is None:
                return written
            for track, detection in wanted[number]:
                try:
                    path = _write(cv2, frame, video, track, detection, settings)
                except (OSError, ValueError):
                    # One unreadable crop must never cost the rest of the table.
                    logger.warning("Thumbnail could not be written for track %s", track.id)
                    continue
                if path is not None:
                    written[track.id] = path
    finally:
        capture.release()
    return written


def _write(cv2: Any, frame: Any, video: Video, track: FishTrack, detection: FishDetection,
           settings: Settings) -> Path | None:
    """A square crop centred on the fish, letterboxed so the column stays even.

    Wider than the crop sent for identification: a thumbnail is read at a glance and
    a little surrounding water is what makes a fish recognisable as a shape. The
    aspect ratio is never distorted - a stretched fish is a misleading one.
    """

    import numpy as np

    height, width = frame.shape[:2]
    margin = settings.thumbnail_zoom_margin
    box_width, box_height = detection.x2 - detection.x1, detection.y2 - detection.y1
    side = max(settings.thumbnail_min_crop_pixels,
               round(max(box_width, box_height) * margin))
    side = min(side, max(width, height))
    centre_x = (detection.x1 + detection.x2) / 2
    centre_y = (detection.y1 + detection.y2) / 2
    left = int(max(0, min(width - min(side, width), round(centre_x - side / 2))))
    top = int(max(0, min(height - min(side, height), round(centre_y - side / 2))))
    crop = frame[top:min(height, top + side), left:min(width, left + side)]
    if crop.size == 0:
        return None

    size = THUMBNAIL_SIZE
    output = np.zeros((size, size, 3), dtype="uint8")
    ratio = min(size / crop.shape[1], size / crop.shape[0])
    scaled = cv2.resize(crop, (max(1, round(crop.shape[1] * ratio)),
                               max(1, round(crop.shape[0] * ratio))),
                        interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
    offset_y, offset_x = (size - scaled.shape[0]) // 2, (size - scaled.shape[1]) // 2
    output[offset_y:offset_y + scaled.shape[0], offset_x:offset_x + scaled.shape[1]] = scaled

    path = thumbnail_path(video.id, track.id, settings)
    temp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.jpg")
    if not cv2.imwrite(str(temp), output, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        temp.unlink(missing_ok=True)
        raise OSError("Thumbnail encoding failed")
    os.replace(temp, path)
    return path
