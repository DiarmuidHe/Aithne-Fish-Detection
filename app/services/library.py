"""Library-wide aggregates, filtering and sorting for the video list.

The review state that an operator needs to filter on lives on ``fish_tracks``,
not on ``videos``. Every aggregate here is therefore derived from one pass over
the tracks of the whole candidate set, never a query per listed row.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import FishTrack, Video
from app.services.fish_counter import is_accepted_track
from app.services.reporting import DEFAULT_BORDERLINE_BAND, review_categories, threshold
from app.services.track_clip import count_track_clips

NO_CAMERA = "__none__"

# ``videos.processing_status`` calls a freshly uploaded video "uploaded"; the
# dashboard vocabulary calls the same state "pending". Accept both so a caller
# using either word gets the same rows instead of a 422.
STATUS_ALIASES = {"pending": "uploaded"}

# Highest first, so `sort=review_status&order=desc` puts the work at the top.
REVIEW_STATUS_RANK = {"awaiting": 3, "in-progress": 2, "complete": 1, "n/a": 0}


@dataclass
class VideoAggregate:
    """Everything the list endpoint reports about one video's tracks."""

    track_count: int = 0
    accepted_track_count: int = 0
    detection_count: int = 0
    unreviewed_count: int = 0
    flagged_count: int = 0
    disputed_count: int = 0
    review_status: str = "n/a"
    species: list[str] = field(default_factory=list)
    clip_count: int = 0
    # Union of every track's categories, for the `review` filter.
    categories: set[str] = field(default_factory=set)


def aggregate_videos(
    db: Session,
    videos: list[Video],
    settings: Settings,
    band: float = DEFAULT_BORDERLINE_BAND,
    include_clips: bool = True,
) -> dict[uuid.UUID, VideoAggregate]:
    """Aggregate the tracks of every given video in a single round trip."""

    aggregates = {video.id: VideoAggregate() for video in videos}
    if not videos:
        return aggregates

    rows = db.execute(
        select(
            FishTrack.video_id,
            FishTrack.processing_job_id,
            FishTrack.review_state,
            FishTrack.reviewed_at,
            FishTrack.max_confidence,
            FishTrack.detection_count,
            FishTrack.species,
        ).where(FishTrack.video_id.in_(list(aggregates)))
    ).all()

    grouped: dict[uuid.UUID, list] = {video.id: [] for video in videos}
    for row in rows:
        grouped[row.video_id].append(row)

    for video in videos:
        tracks = grouped[video.id]
        jobs = {job.id: job for job in video.jobs}
        aggregate = aggregates[video.id]
        aggregate.track_count = len(tracks)
        if include_clips:
            aggregate.clip_count = count_track_clips(video.id, settings)
        species = set()
        for track in tracks:
            run_threshold = threshold(video, jobs.get(track.processing_job_id))
            categories = review_categories(track, run_threshold, band)
            aggregate.categories |= categories
            aggregate.unreviewed_count += "unreviewed" in categories
            aggregate.flagged_count += "flagged" in categories
            aggregate.disputed_count += "disputed" in categories
            if is_accepted_track(track, run_threshold):
                aggregate.accepted_track_count += 1
                aggregate.detection_count += track.detection_count
                if track.species:
                    species.add(track.species)
        aggregate.species = sorted(species)
        aggregate.review_status = review_status(video.processing_status, tracks)
    return aggregates


def review_status(processing_status: str, tracks: list) -> str:
    if processing_status != "completed":
        return "n/a"
    if not tracks:
        # A completed run that found nothing leaves no review work outstanding;
        # calling it "awaiting" would pad the review queue with empty videos.
        return "complete"
    if all(track.reviewed_at is None for track in tracks):
        return "awaiting"
    if all(track.review_state not in ("unreviewed", "needs-review") for track in tracks):
        return "complete"
    return "in-progress"


def candidate_query(
    q: str | None = None,
    status: list[str] | None = None,
    camera_id: list[str] | None = None,
    annotated: bool | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> Select:
    """The filters that the videos table can answer on its own."""

    query = select(Video)
    if q:
        pattern = f"%{q.strip()}%"
        query = query.where(
            or_(Video.original_filename.ilike(pattern), Video.camera_id.ilike(pattern))
        )
    if status:
        query = query.where(
            Video.processing_status.in_([STATUS_ALIASES.get(value, value) for value in status])
        )
    if camera_id:
        wanted = [value for value in camera_id if value != NO_CAMERA]
        clauses = []
        if wanted:
            clauses.append(Video.camera_id.in_(wanted))
        if NO_CAMERA in camera_id:
            clauses.append(Video.camera_id.is_(None))
        query = query.where(or_(*clauses))
    if annotated is not None:
        query = query.where(
            Video.annotated_at.is_not(None) if annotated else Video.annotated_at.is_(None)
        )
    if created_after is not None:
        query = query.where(Video.created_at >= created_after)
    if created_before is not None:
        query = query.where(Video.created_at <= created_before)
    if min_duration is not None:
        query = query.where(Video.duration_seconds >= min_duration)
    if max_duration is not None:
        query = query.where(Video.duration_seconds <= max_duration)
    return query


def matches_aggregate_filters(
    aggregate: VideoAggregate,
    review: list[str] | None = None,
    review_status: list[str] | None = None,
    species: list[str] | None = None,
    has_clips: bool | None = None,
    min_fish: int | None = None,
    max_fish: int | None = None,
) -> bool:
    """The filters that only exist once a video's tracks have been aggregated."""

    if review and not aggregate.categories.intersection(review):
        return False
    if review_status and aggregate.review_status not in review_status:
        return False
    if species and not set(aggregate.species).intersection(species):
        return False
    if has_clips is not None and bool(aggregate.clip_count) is not has_clips:
        return False
    if min_fish is not None and aggregate.accepted_track_count < min_fish:
        return False
    return not (max_fish is not None and aggregate.accepted_track_count > max_fish)


def _sort_key(name: str, video: Video, aggregate: VideoAggregate):
    """Return ``(is_missing, value)`` so unknown values sort last in both directions."""

    if name == "filename":
        return (0, video.original_filename.lower())
    if name == "camera_id":
        return (video.camera_id is None, (video.camera_id or "").lower())
    if name == "status":
        return (0, video.processing_status)
    if name == "duration":
        return (video.duration_seconds is None, video.duration_seconds or 0.0)
    if name == "size":
        return (video.size_bytes is None, video.size_bytes or 0)
    if name == "annotated_at":
        return (video.annotated_at is None, video.annotated_at or video.created_at)
    if name == "review_status":
        return (0, REVIEW_STATUS_RANK[aggregate.review_status])
    return (0, {
        "accepted_fish": aggregate.accepted_track_count,
        "detections": aggregate.detection_count,
        "unreviewed": aggregate.unreviewed_count,
        "flagged": aggregate.flagged_count,
    }[name])


def sort_videos(
    rows: list[tuple[Video, VideoAggregate]], sort: str, order: str
) -> list[tuple[Video, VideoAggregate]]:
    descending = order == "desc"
    # Upload order breaks ties, so paging stays stable across equal sort keys.
    ordered = sorted(rows, key=lambda row: (row[0].created_at, str(row[0].id)), reverse=descending)
    if sort == "created_at":
        return ordered
    return sorted(ordered, key=lambda row: _sort_key(sort, row[0], row[1]), reverse=descending)


def facet_counts(db: Session, column) -> dict:
    rows = db.execute(select(column, func.count()).group_by(column)).all()
    return {value: count for value, count in rows if value is not None}
