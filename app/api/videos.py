from __future__ import annotations

import uuid
import mimetypes
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import FishTrack, LiveMonitorSession, Video, VideoProcessingStatus
from app.api.live import session_data
from app.api.tracks import clip_response
from app.schemas.filters import (
    ReviewFilter,
    ReviewState,
    ReviewStatusFilter,
    SortOrder,
    TrackSort,
    VideoSort,
    VideoStatusFilter,
)
from app.schemas.job import JobRead
from app.schemas.track import (
    FishTrackRead,
    FishTrackSummaryRead,
    TrackClipRead,
    TrackThumbnailRead,
)
from app.schemas.video import (
    VideoAnnotationRead,
    VideoFacets,
    VideoListRead,
    VideoRead,
    VideoSummary,
)
from app.services.fish_counter import accepted_tracks, summarize_tracks
from app.services.library import (
    aggregate_videos,
    candidate_query,
    matches_aggregate_filters,
    sort_videos,
)
from app.services.reporting import DEFAULT_BORDERLINE_BAND, track_summary_row
from app.services.track_clip import (
    TrackClipError,
    existing_clip,
    generate_track_clips,
)
from app.services.track_thumbnail import generate_video_thumbnails
from app.services.video_service import enqueue_video_processing, register_uploaded_video
from app.services.video_annotator import (
    AnnotatedVideoResult,
    AnnotatedVideoUnavailableError,
    AnnotationNotGeneratedError,
    AnnotationPathError,
    MEDIA_TYPE_MP4,
    VideoAnnotationError,
    generate_annotated_video,
    get_annotated_video_path,
)

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", response_model=VideoRead, status_code=status.HTTP_201_CREATED)
async def create_video(
    file: UploadFile = File(...),
    camera_id: str | None = Form(None, max_length=128),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Video:
    return await register_uploaded_video(db=db, upload=file, settings=settings, camera_id=camera_id)


@router.get("", response_model=list[VideoListRead])
def list_videos(
    response: Response,
    q: str | None = Query(None, max_length=256,
                          description="Substring of the filename or camera, case-insensitive"),
    status: list[VideoStatusFilter] | None = Query(None),
    camera_id: list[str] | None = Query(None, description='"__none__" selects videos with no camera'),
    review: list[ReviewFilter] | None = Query(None),
    review_status: list[ReviewStatusFilter] | None = Query(None),
    annotated: bool | None = Query(None),
    has_clips: bool | None = Query(None),
    species: list[str] | None = Query(None),
    created_after: datetime | None = Query(None),
    created_before: datetime | None = Query(None),
    min_fish: int | None = Query(None, ge=0),
    max_fish: int | None = Query(None, ge=0),
    min_duration: float | None = Query(None, ge=0),
    max_duration: float | None = Query(None, ge=0),
    band: float = Query(DEFAULT_BORDERLINE_BAND, ge=0, le=1,
                        description="Half-width of the borderline band around the run threshold"),
    sort: VideoSort = Query(VideoSort.CREATED_AT),
    order: SortOrder = Query(SortOrder.DESC),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[VideoListRead]:
    """The library list. Still a bare JSON array; the totals ride in headers."""

    candidates = list(
        db.scalars(
            candidate_query(
                q=q,
                status=[value.value for value in status] if status else None,
                camera_id=camera_id,
                annotated=annotated,
                created_after=created_after,
                created_before=created_before,
                min_duration=min_duration,
                max_duration=max_duration,
            ).options(selectinload(Video.jobs))
        ).all()
    )
    aggregates = aggregate_videos(db, candidates, settings, band)
    review_values = [value.value for value in review] if review else None
    rows = [
        (video, aggregates[video.id])
        for video in candidates
        if matches_aggregate_filters(
            aggregates[video.id],
            review=review_values,
            review_status=[value.value for value in review_status] if review_status else None,
            species=species,
            has_clips=has_clips,
            min_fish=min_fish,
            max_fish=max_fish,
        )
    ]
    rows = sort_videos(rows, sort.value, order.value)
    response.headers["X-Total-Count"] = str(db.scalar(select(func.count(Video.id))) or 0)
    response.headers["X-Filtered-Count"] = str(len(rows))
    return [_video_list_row(video, aggregate) for video, aggregate in rows[offset:offset + limit]]


@router.get("/facets", response_model=VideoFacets)
def get_video_facets(
    band: float = Query(DEFAULT_BORDERLINE_BAND, ge=0, le=1),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> VideoFacets:
    """Filter menu contents and their counts, so the client never loads the library to build a menu."""

    videos = list(db.scalars(select(Video).options(selectinload(Video.jobs))).all())
    # Clip counts are a directory listing per video and no facet needs them.
    aggregates = aggregate_videos(db, videos, settings, band, include_clips=False)
    status_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    cameras, species = set(), set()
    for video in videos:
        aggregate = aggregates[video.id]
        status_counts[video.processing_status] = status_counts.get(video.processing_status, 0) + 1
        review_counts[aggregate.review_status] = review_counts.get(aggregate.review_status, 0) + 1
        if video.camera_id:
            cameras.add(video.camera_id)
        species.update(aggregate.species)
    return VideoFacets(
        cameras=sorted(cameras),
        species=sorted(species),
        status=status_counts,
        review_status=review_counts,
        total=len(videos),
    )


def _video_list_row(video: Video, aggregate) -> VideoListRead:
    return VideoListRead(
        **VideoRead.model_validate(video).model_dump(),
        track_count=aggregate.track_count,
        accepted_track_count=aggregate.accepted_track_count,
        detection_count=aggregate.detection_count,
        unreviewed_count=aggregate.unreviewed_count,
        flagged_count=aggregate.flagged_count,
        disputed_count=aggregate.disputed_count,
        review_status=aggregate.review_status,
        has_annotation=video.annotated_at is not None,
        clip_count=aggregate.clip_count,
        species=aggregate.species,
    )


@router.get("/{video_id}", response_model=VideoRead)
def get_video(video_id: uuid.UUID, db: Session = Depends(get_db)) -> Video:
    return _get_video_or_404(db, video_id)


@router.post("/{video_id}/process", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
def create_processing_job(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    try:
        return enqueue_video_processing(db=db, video_id=video_id, settings=settings)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{video_id}/annotate", response_model=VideoAnnotationRead)
def create_annotated_video(
    video_id: uuid.UUID,
    include_species: bool = Query(True),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> VideoAnnotationRead:
    video = db.scalars(select(Video).where(Video.id == video_id).with_for_update()).first()
    if video is None:
        raise HTTPException(404, "Video not found")
    _require_completed_processing(video)

    try:
        result = generate_annotated_video(
            db=db,
            video=video,
            settings=settings,
            include_species=include_species,
        )
    except VideoAnnotationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return _annotation_response(result)


@router.get("/{video_id}/annotated-video")
def get_annotated_video(
    video_id: uuid.UUID,
    download: bool = Query(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    video = _get_video_or_404(db, video_id)
    _require_completed_processing(video)

    try:
        path = get_annotated_video_path(video, settings)
    except (AnnotationNotGeneratedError, AnnotatedVideoUnavailableError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AnnotationPathError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return FileResponse(
        path=path,
        media_type=MEDIA_TYPE_MP4,
        filename=f"{video.id}-annotated.mp4",
        content_disposition_type="attachment" if download else "inline",
    )


@router.post("/{video_id}/fish-clips", response_model=list[TrackClipRead])
def create_fish_clips(
    video_id: uuid.UUID,
    accepted_only: bool = Query(True),
    refresh: bool = Query(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[TrackClipRead]:
    """Cut one short cropped clip per fish, reusing clips that already exist."""

    video = _get_video_or_404(db, video_id)
    _require_completed_processing(video)
    tracks = _clip_tracks(db, video, accepted_only)
    try:
        results = generate_track_clips(
            db=db, video=video, tracks=tracks, settings=settings, refresh=refresh
        )
    except TrackClipError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    by_id = {track.id: track for track in tracks}
    return [clip_response(result, by_id.get(result.track_id)) for result in results]


@router.get("/{video_id}/fish-clips", response_model=list[TrackClipRead])
def list_fish_clips(
    video_id: uuid.UUID,
    accepted_only: bool = Query(True),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[TrackClipRead]:
    """List the clips already on disk; generation stays an explicit action."""

    video = _get_video_or_404(db, video_id)
    tracks = _clip_tracks(db, video, accepted_only)
    by_id = {track.id: track for track in tracks}
    clips = [existing_clip(track, settings) for track in tracks]
    return [clip_response(clip, by_id.get(clip.track_id)) for clip in clips if clip is not None]


@router.get("/{video_id}/thumbnails", response_model=list[TrackThumbnailRead])
def list_track_thumbnails(
    video_id: uuid.UUID,
    refresh: bool = Query(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[TrackThumbnailRead]:
    """Every fish preview for this video, generating any that are missing.

    Per video rather than per track: reaching a frame means grabbing forward through
    the source, so one pass produces all of them for barely more than the cost of
    one. A track whose crop could not be rendered is simply absent.
    """

    video = _get_video_or_404(db, video_id)
    found = generate_video_thumbnails(db, video, settings, refresh=refresh)
    return [TrackThumbnailRead(track_id=track_id, url=f"/tracks/{track_id}/thumbnail")
            for track_id in found]


@router.get("/{video_id}/source-video")
def get_source_video(video_id: uuid.UUID, db: Session = Depends(get_db),
                     settings: Settings = Depends(get_settings)):
    video = _get_video_or_404(db, video_id)
    path = Path(video.storage_path).resolve()
    if not path.is_relative_to(settings.upload_root.resolve()) or not path.is_file():
        raise HTTPException(404, "Source video unavailable")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        filename=f"{video.id}{path.suffix}", content_disposition_type="inline")


@router.get("/{video_id}/source-session")
def get_source_session(video_id: uuid.UUID, db: Session = Depends(get_db),
                       settings: Settings = Depends(get_settings)):
    """The live session a recording came from, so a reviewer can see its context."""

    video = _get_video_or_404(db, video_id)
    session = (db.get(LiveMonitorSession, video.source_session_id)
               if video.source_session_id else None)
    if session is None:
        raise HTTPException(404, "This video did not come from a live monitoring session")
    return session_data(session, settings, db)


@router.get("/{video_id}/tracks", response_model=list[FishTrackRead])
def get_video_tracks(
    video_id: uuid.UUID,
    accepted_only: bool = Query(True),
    db: Session = Depends(get_db),
) -> list[FishTrack]:
    video = _get_video_or_404(db, video_id)
    tracks = list(
        db.scalars(
            select(FishTrack)
            .where(FishTrack.video_id == video_id)
            .options(selectinload(FishTrack.detections))
            .order_by(FishTrack.first_frame, FishTrack.viame_track_id)
        ).all()
    )
    if not accepted_only:
        return tracks
    return accepted_tracks(tracks, video.confidence_threshold)


@router.get("/{video_id}/track-summaries", response_model=list[FishTrackSummaryRead])
def get_video_track_summaries(
    video_id: uuid.UUID,
    accepted_only: bool = Query(
        True,
        description="Superseded by `review`, and applied in addition to it. "
                    "Pass false when filtering by review state.",
    ),
    review_state: list[ReviewState] | None = Query(None),
    review: list[ReviewFilter] | None = Query(None),
    species: list[str] | None = Query(None),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_detections: int | None = Query(None, ge=0),
    time_from: float | None = Query(None, ge=0),
    time_to: float | None = Query(None, ge=0),
    band: float = Query(DEFAULT_BORDERLINE_BAND, ge=0, le=1),
    sort: TrackSort = Query(TrackSort.FIRST_FRAME),
    order: SortOrder = Query(SortOrder.ASC),
    db: Session = Depends(get_db),
) -> list[dict]:
    video = _get_video_or_404(db, video_id)
    tracks = list(
        db.scalars(
            select(FishTrack)
            .where(FishTrack.video_id == video_id)
            .order_by(FishTrack.first_frame, FishTrack.viame_track_id)
        ).all()
    )
    jobs = {job.id: job for job in video.jobs}
    wanted_states = {value.value for value in review_state} if review_state else None
    wanted_review = {value.value for value in review} if review else None
    rows = []
    for track in tracks:
        row = track_summary_row(track, video, jobs, band)
        if accepted_only and not row["accepted"]:
            continue
        if wanted_states and track.review_state not in wanted_states:
            continue
        if wanted_review and not wanted_review.intersection(row["review_categories"]):
            continue
        if species and track.species not in species:
            continue
        if min_confidence is not None and track.max_confidence < min_confidence:
            continue
        if max_confidence is not None and track.max_confidence > max_confidence:
            continue
        if min_detections is not None and track.detection_count < min_detections:
            continue
        if not _within_window(track, time_from, time_to):
            continue
        rows.append(row)
    return _sort_track_rows(rows, sort.value, order.value)


def _within_window(track: FishTrack, time_from: float | None, time_to: float | None) -> bool:
    """A track is in the window when its observed span overlaps it."""

    if time_from is None and time_to is None:
        return True
    first, last = track.first_timestamp_seconds, track.last_timestamp_seconds
    if first is None or last is None:
        # An untimed track cannot be shown to fall inside a time window.
        return False
    if time_to is not None and first > time_to:
        return False
    return not (time_from is not None and last < time_from)


def _track_sort_key(name: str, row: dict):
    if name == "duration":
        first, last = row["first_timestamp_seconds"], row["last_timestamp_seconds"]
        missing = first is None or last is None
        return (missing, 0.0 if missing else max(0.0, last - first))
    if name == "species":
        return (row["species"] is None, (row["species"] or "").lower())
    return (0, row[name])


def _sort_track_rows(rows: list[dict], sort: str, order: str) -> list[dict]:
    descending = order == "desc"
    # Frame order breaks ties so equal keys keep a stable, meaningful order.
    ordered = sorted(rows, key=lambda row: (row["first_frame"], row["viame_track_id"]),
                     reverse=descending)
    if sort == "first_frame":
        return ordered
    return sorted(ordered, key=lambda row: _track_sort_key(sort, row), reverse=descending)


@router.get("/{video_id}/summary", response_model=VideoSummary)
def get_video_summary(video_id: uuid.UUID, db: Session = Depends(get_db)) -> VideoSummary:
    video = _get_video_or_404(db, video_id)
    tracks = list(
        db.scalars(select(FishTrack).where(FishTrack.video_id == video_id)).all()
    )
    summary = summarize_tracks(
        video_id=video.id,
        status=video.processing_status,
        tracks=tracks,
        min_confidence=video.confidence_threshold,
    )
    payload = asdict(summary)
    payload.update(
        {
            "pipeline_name": video.pipeline_name,
            "confidence_threshold": video.confidence_threshold,
            "model_name": video.model_name,
            "model_version": video.model_version,
            "viame_version": video.viame_version,
        }
    )
    return VideoSummary(**payload)


def _clip_tracks(db: Session, video: Video, accepted_only: bool) -> list[FishTrack]:
    tracks = list(
        db.scalars(
            select(FishTrack)
            .where(FishTrack.video_id == video.id)
            .order_by(FishTrack.first_frame, FishTrack.viame_track_id)
        ).all()
    )
    if not accepted_only:
        return tracks
    return accepted_tracks(tracks, video.confidence_threshold)


def _get_video_or_404(db: Session, video_id: uuid.UUID) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return video


def _require_completed_processing(video: Video) -> None:
    if video.processing_status != VideoProcessingStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Video processing is not completed",
        )


def _annotation_response(result: AnnotatedVideoResult) -> VideoAnnotationRead:
    return VideoAnnotationRead(
        video_id=result.video_id,
        annotated_at=result.annotated_at,
        filename=result.filename,
        media_type=result.media_type,
        size_bytes=result.size_bytes,
        fps=result.fps,
        width=result.width,
        height=result.height,
        frame_count=result.frame_count,
        url=f"/videos/{result.video_id}/annotated-video",
    )
