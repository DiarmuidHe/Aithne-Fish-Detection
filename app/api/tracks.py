from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import FishTrack, Video, VideoProcessingStatus, utc_now
from app.schemas.track import (
    FishTrackRead,
    FishTrackSummaryRead,
    TrackBulkReviewUpdate,
    TrackClipRead,
    TrackReviewUpdate,
)
from app.services.reporting import DEFAULT_BORDERLINE_BAND, track_summary_row
from app.services.track_clip import (
    ClipNotGeneratedError,
    ClipPathError,
    ClipUnavailableError,
    TrackClipError,
    TrackClipResult,
    generate_track_clip,
    get_clip_path,
)
from app.services.video_media import MEDIA_TYPE_MP4

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.post("/review", response_model=list[FishTrackSummaryRead])
def review_tracks(
    payload: TrackBulkReviewUpdate,
    band: float = Query(DEFAULT_BORDERLINE_BAND, ge=0, le=1),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Apply one decision to many tracks. "Accept all visible" and undo need this."""

    track_ids = list(dict.fromkeys(payload.track_ids))
    tracks = list(db.scalars(select(FishTrack).where(FishTrack.id.in_(track_ids))).all())
    if len(tracks) != len(track_ids):
        raise HTTPException(404, "One or more tracks were not found")

    videos = {}
    for video_id in dict.fromkeys(track.video_id for track in tracks):
        # The same lock the single-track path takes, so a bulk decision cannot
        # interleave with processing or annotation of the same video.
        video = db.scalars(select(Video).where(Video.id == video_id).with_for_update()).one()
        if video.processing_status != VideoProcessingStatus.COMPLETED.value:
            raise HTTPException(409, "Review requires completed processing")
        videos[video_id] = video

    changed = False
    for track in tracks:
        db.refresh(track)
        if track.review_state == payload.review_state:
            continue
        track.review_state = payload.review_state
        track.reviewed_at = None if payload.review_state == "unreviewed" else utc_now()
        video = videos[track.video_id]
        video.annotated_at = None
        video.annotated_video_path = None
        changed = True
    if changed:
        db.commit()

    order = {track_id: index for index, track_id in enumerate(track_ids)}
    jobs = {job.id: job for video in videos.values() for job in video.jobs}
    return [
        track_summary_row(track, videos[track.video_id], jobs, band)
        for track in sorted(tracks, key=lambda track: order[track.id])
    ]


@router.patch("/{track_id}/review", response_model=FishTrackRead)
def review_track(track_id: uuid.UUID, payload: TrackReviewUpdate,
                 db: Session = Depends(get_db)) -> FishTrack:
    track = get_track(track_id, db)
    video = db.scalars(select(Video).where(Video.id == track.video_id).with_for_update()).one()
    if video.processing_status != "completed":
        raise HTTPException(409, "Review requires completed processing")
    # Refresh after taking the same video lock used by processing/annotation.
    db.refresh(track)
    if track.review_state != payload.review_state:
        track.review_state = payload.review_state
        track.reviewed_at = None if payload.review_state == "unreviewed" else utc_now()
        video.annotated_at = None
        video.annotated_video_path = None
        db.commit()
    return track


@router.post("/{track_id}/clip", response_model=TrackClipRead)
def create_track_clip(
    track_id: uuid.UUID,
    refresh: bool = Query(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TrackClipRead:
    """Cut a short cropped video that follows this one fish through the source."""

    track = get_track(track_id, db)
    video = _completed_video_for_track(db, track)
    try:
        result = generate_track_clip(db=db, video=video, track=track, settings=settings,
                                     refresh=refresh)
    except TrackClipError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return clip_response(result)


@router.get("/{track_id}/clip")
def get_track_clip(
    track_id: uuid.UUID,
    download: bool = Query(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    track = get_track(track_id, db)
    try:
        path = get_clip_path(track, settings)
    except (ClipNotGeneratedError, ClipUnavailableError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ClipPathError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return FileResponse(
        path=path,
        media_type=MEDIA_TYPE_MP4,
        filename=f"fish-{track.viame_track_id}-{track.id}.mp4",
        content_disposition_type="attachment" if download else "inline",
    )


@router.get("/{track_id}", response_model=FishTrackRead)
def get_track(track_id: uuid.UUID, db: Session = Depends(get_db)) -> FishTrack:
    track = db.scalars(
        select(FishTrack)
        .where(FishTrack.id == track_id)
        .options(selectinload(FishTrack.detections))
    ).first()
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track not found")
    return track


def clip_response(result: TrackClipResult) -> TrackClipRead:
    return TrackClipRead(
        track_id=result.track_id,
        video_id=result.video_id,
        viame_track_id=result.viame_track_id,
        species=result.species,
        filename=result.filename,
        media_type=result.media_type,
        size_bytes=result.size_bytes,
        fps=result.fps,
        width=result.width,
        height=result.height,
        frame_count=result.frame_count,
        start_seconds=result.start_seconds,
        end_seconds=result.end_seconds,
        duration_seconds=result.duration_seconds,
        detection_count=result.detection_count,
        max_confidence=result.max_confidence,
        generated_at=result.generated_at,
        cached=result.cached,
        url=f"/tracks/{result.track_id}/clip",
    )


def _completed_video_for_track(db: Session, track: FishTrack) -> Video:
    video = db.get(Video, track.video_id)
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    if video.processing_status != VideoProcessingStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Video processing is not completed",
        )
    return video
