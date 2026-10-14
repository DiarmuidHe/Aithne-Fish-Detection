"""Development controls for the local CVAT mirror."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import CvatSync, ProcessingJob, ReviewRegion, ReviewedSpecies, Video
from app.services.cvat_sync import configured, gateway_for, queue_cvat_video, sync_cvat_video
from app.services.review_dataset import (
    LATEST_PILOT_VIDEO, annotation_sha256, candidate_audit, proposal_regions, region_rows,
)
from app.services import species_catalogue

router = APIRouter(prefix="/cvat", tags=["cvat-development"])


class CompleteRegion(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    ids_complete: bool = False
    species_complete: bool = False
    review_minutes: float = Field(gt=0, le=30)


class SpeciesDecision(BaseModel):
    species: str = Field(min_length=1, max_length=256)


def _review_task(video_id: uuid.UUID, db: Session, settings: Settings):
    if not configured(settings):
        raise HTTPException(503, "CVAT review is not configured")
    video = db.get(Video, video_id)
    link = db.get(CvatSync, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    if link is None or link.task_id is None:
        raise HTTPException(409, "Send this video to CVAT first")
    if not video.width or not video.height or not video.fps:
        raise HTTPException(409, "Video dimensions and frame rate are required")
    return video, link


def _region_json(region: ReviewRegion) -> dict:
    return {"id": str(region.id), "start_frame": region.start_frame,
            "end_frame": region.end_frame, "boxes_complete": region.boxes_complete,
            "ids_complete": region.ids_complete, "species_complete": region.species_complete,
            "review_minutes": region.review_minutes,
            "completed_at": region.completed_at.isoformat()}


@router.get("/videos/{video_id}/review")
def review_queue(video_id: uuid.UUID, db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings)) -> dict:
    video, link = _review_task(video_id, db, settings)
    frame_count = max(1, round((video.duration_seconds or 0) * video.fps))
    regions = db.scalars(select(ReviewRegion).where(ReviewRegion.video_id == video_id)).all()
    latest = db.scalars(select(ProcessingJob).where(
        ProcessingJob.video_id == video_id, ProcessingJob.output_csv_path.is_not(None)
    ).order_by(ProcessingJob.finished_at.desc())).first()
    audit = None
    if latest and latest.output_csv_path and Path(latest.output_csv_path).is_file():
        csv_path = Path(latest.output_csv_path)
        audit = candidate_audit(csv_path, csv_path.with_name("computed_detections.csv"),
                                video.fps,
                                frame_offset=int(latest.configuration.get("frame_number_offset", -1)))
    return {"task_id": link.task_id, "fps": video.fps, "frame_count": frame_count,
            "budget_minutes": 30, "used_minutes": sum(r.review_minutes or 0 for r in regions),
            "suggested": proposal_regions(video_id, video.fps, frame_count),
            "hotspots": audit, "completed": [_region_json(region) for region in regions]}


@router.get("/videos/{video_id}/tracks")
def cvat_review_tracks(video_id: uuid.UUID, start_frame: int = Query(ge=0),
                       end_frame: int = Query(ge=0), db: Session = Depends(get_db),
                       settings: Settings = Depends(get_settings)) -> dict:
    video, link = _review_task(video_id, db, settings)
    if end_frame < start_frame or end_frame >= round((video.duration_seconds or 0) * video.fps):
        raise HTTPException(422, "Invalid review bounds")
    with gateway_for(settings) as gateway:
        task = gateway.client.tasks.retrieve(link.task_id)
        rows = region_rows(task, start_frame, end_frame, video.width, video.height)
    track_ids = sorted({row["track_id"] for frame in rows.values() for row in frame
                        if row["track_id"] is not None})
    decisions = {entry.cvat_track_id: entry.species for entry in db.scalars(
        select(ReviewedSpecies).where(ReviewedSpecies.video_id == video_id)).all()}
    return {"tracks": [{"track_id": track_id, "species": decisions.get(track_id)}
                       for track_id in track_ids]}


@router.put("/videos/{video_id}/tracks/{track_id}/species")
def assign_cvat_species(video_id: uuid.UUID, track_id: int, payload: SpeciesDecision,
                        db: Session = Depends(get_db),
                        settings: Settings = Depends(get_settings)) -> dict:
    _, link = _review_task(video_id, db, settings)
    accepted = {name.lower(): name for name, _ in
                species_catalogue.CATALOGUES["north_east_atlantic"]}
    name = payload.species.strip()
    canonical = "unknown" if name.lower() == "unknown" else accepted.get(name.lower())
    if canonical is None:
        raise HTTPException(422, "Choose a Galway Bay catalogue species or unknown")
    with gateway_for(settings) as gateway:
        task = gateway.client.tasks.retrieve(link.task_id)
        if track_id not in {int(track.id) for track in task.get_annotations().tracks
                            if track.id is not None}:
            raise HTTPException(404, "CVAT track not found")
    decision = db.get(ReviewedSpecies, (video_id, track_id))
    if decision is None:
        decision = ReviewedSpecies(video_id=video_id, cvat_track_id=track_id,
                                   species=canonical)
        db.add(decision)
    else:
        decision.species = canonical
        decision.reviewed_at = datetime.now(UTC)
    db.commit()
    return {"track_id": track_id, "species": canonical}


@router.put("/videos/{video_id}/review-regions")
def complete_review_region(video_id: uuid.UUID, payload: CompleteRegion,
                           db: Session = Depends(get_db),
                           settings: Settings = Depends(get_settings)) -> dict:
    video, link = _review_task(video_id, db, settings)
    if (payload.end_frame < payload.start_frame
            or payload.end_frame >= round((video.duration_seconds or 0) * video.fps)
            or payload.species_complete and not payload.ids_complete):
        raise HTTPException(422, "Invalid review bounds or completion flags")
    if video_id == LATEST_PILOT_VIDEO:
        pilot_windows = {(region["start_frame"], region["end_frame"])
                         for region in proposal_regions(video_id, video.fps,
                                                        round((video.duration_seconds or 0) * video.fps))
                         if region["kind"] == "track_window"}
        if ((payload.start_frame, payload.end_frame) in pilot_windows
                and not (payload.ids_complete and payload.species_complete)):
            raise HTTPException(422, "Pilot track windows require checked IDs and species")
    existing = db.scalars(select(ReviewRegion).where(
        ReviewRegion.video_id == video_id,
        ReviewRegion.start_frame == payload.start_frame,
        ReviewRegion.end_frame == payload.end_frame)).first()
    used = sum(region.review_minutes or 0 for region in db.scalars(
        select(ReviewRegion).where(ReviewRegion.video_id == video_id)).all()
               if region.id != getattr(existing, "id", None))
    if used + payload.review_minutes > 30:
        raise HTTPException(422, "The 30 minute review budget would be exceeded")
    with gateway_for(settings) as gateway:
        task = gateway.client.tasks.retrieve(link.task_id)
        rows = region_rows(task, payload.start_frame, payload.end_frame,
                           video.width, video.height)
    if payload.ids_complete and any(row["track_id"] is None for frame in rows.values()
                                    for row in frame):
        raise HTTPException(422, "Convert standalone fish boxes to tracks before checking IDs")
    species_decisions = None
    if payload.species_complete:
        track_ids = {row["track_id"] for frame in rows.values() for row in frame}
        species_decisions = {entry.cvat_track_id: entry.species for entry in db.scalars(
            select(ReviewedSpecies).where(ReviewedSpecies.video_id == video_id)).all()}
        if not track_ids.issubset(species_decisions):
            raise HTTPException(422, "Assign a species or unknown to every reviewed track")
    region = existing or ReviewRegion(video_id=video_id, start_frame=payload.start_frame,
                                      end_frame=payload.end_frame)
    region.boxes_complete = True
    region.ids_complete = payload.ids_complete
    region.species_complete = payload.species_complete
    region.review_minutes = payload.review_minutes
    region.annotation_sha256 = annotation_sha256(rows, species_decisions)
    region.completed_at = datetime.now(UTC)
    db.add(region)
    db.commit()
    return _region_json(region)


@router.delete("/videos/{video_id}/review-regions/{region_id}", status_code=204)
def undo_review_region(video_id: uuid.UUID, region_id: uuid.UUID,
                       db: Session = Depends(get_db),
                       settings: Settings = Depends(get_settings)) -> None:
    _review_task(video_id, db, settings)
    region = db.get(ReviewRegion, region_id)
    if region is None or region.video_id != video_id:
        raise HTTPException(404, "Review region not found")
    db.delete(region)
    db.commit()


@router.get("/status")
def cvat_status(settings: Settings = Depends(get_settings)) -> dict:
    return {"enabled": settings.cvat_dev_mode, "configured": configured(settings),
            "public_url": settings.cvat_public_url if settings.cvat_dev_mode else None}


@router.get("/videos/{video_id}")
def video_cvat_status(video_id: uuid.UUID, db: Session = Depends(get_db),
                      settings: Settings = Depends(get_settings)) -> dict:
    if not settings.cvat_dev_mode:
        raise HTTPException(404, "CVAT development mode is disabled")
    if db.get(Video, video_id) is None:
        raise HTTPException(404, "Video not found")
    link = db.get(CvatSync, video_id)
    return {"status": link.status if link else "not_requested",
            "task_id": link.task_id if link else None,
            "task_url": (f"{settings.cvat_public_url.rstrip('/')}/tasks/{link.task_id}"
                         if link and link.task_id else None),
            "error": link.error_message if link else None}


@router.post("/videos/{video_id}/sync", status_code=202)
def retry_cvat_sync(video_id: uuid.UUID, background: BackgroundTasks,
                    db: Session = Depends(get_db),
                    settings: Settings = Depends(get_settings)) -> dict:
    if not configured(settings):
        raise HTTPException(503, "CVAT development mode or credentials are not configured")
    if db.get(Video, video_id) is None:
        raise HTTPException(404, "Video not found")
    link = queue_cvat_video(db, video_id)
    if link.status in {"uploading", "importing", "review", "needs_review"}:
        return {"status": link.status}
    if link.status != "queued":
        link.status = "queued"
        link.error_message = None
        db.commit()
    background.add_task(sync_cvat_video, video_id, settings)
    return {"status": link.status}
