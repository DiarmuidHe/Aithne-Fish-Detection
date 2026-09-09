from __future__ import annotations

import uuid
import mimetypes
from pathlib import Path
from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import FishTrack, Video, VideoProcessingStatus
from app.api.tracks import clip_response
from app.schemas.job import JobRead
from app.schemas.track import FishTrackRead, FishTrackSummaryRead, TrackClipRead
from app.schemas.video import VideoAnnotationRead, VideoRead, VideoSummary
from app.services.fish_counter import accepted_tracks, is_accepted_track, summarize_tracks
from app.services.track_clip import (
    TrackClipError,
    existing_clip,
    generate_track_clips,
)
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


@router.get("", response_model=list[VideoRead])
def list_videos(db: Session = Depends(get_db)) -> list[Video]:
    return list(
        db.scalars(
            select(Video)
            .options(selectinload(Video.jobs))
            .order_by(Video.created_at.desc())
        ).all()
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
    return [clip_response(result) for result in results]


@router.get("/{video_id}/fish-clips", response_model=list[TrackClipRead])
def list_fish_clips(
    video_id: uuid.UUID,
    accepted_only: bool = Query(True),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[TrackClipRead]:
    """List the clips already on disk; generation stays an explicit action."""

    video = _get_video_or_404(db, video_id)
    clips = [existing_clip(track, settings) for track in _clip_tracks(db, video, accepted_only)]
    return [clip_response(clip) for clip in clips if clip is not None]


@router.get("/{video_id}/source-video")
def get_source_video(video_id: uuid.UUID, db: Session = Depends(get_db),
                     settings: Settings = Depends(get_settings)):
    video = _get_video_or_404(db, video_id)
    path = Path(video.storage_path).resolve()
    if not path.is_relative_to(settings.upload_root.resolve()) or not path.is_file():
        raise HTTPException(404, "Source video unavailable")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        filename=f"{video.id}{path.suffix}", content_disposition_type="inline")


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
    accepted_only: bool = Query(True),
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
    rows = []
    for track in tracks:
        accepted = is_accepted_track(track, video.confidence_threshold)
        if accepted_only and not accepted:
            continue
        rows.append(
            {
                "id": track.id,
                "video_id": track.video_id,
                "processing_job_id": track.processing_job_id,
                "review_state": track.review_state,
                "reviewed_at": track.reviewed_at,
                "machine_accepted": track.max_confidence >= video.confidence_threshold,
                "viame_track_id": track.viame_track_id,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "first_timestamp_seconds": track.first_timestamp_seconds,
                "last_timestamp_seconds": track.last_timestamp_seconds,
                "detection_count": track.detection_count,
                "mean_confidence": track.mean_confidence,
                "max_confidence": track.max_confidence,
                "species": track.species,
                "species_confidence": track.species_confidence,
                "accepted": accepted,
            }
        )
    return rows


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
