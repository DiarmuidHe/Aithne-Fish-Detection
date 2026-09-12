from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from datetime import timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db.database import get_db, session_factory_for
from app.db.models import (
    LIVE_OPEN_STATUSES,
    LiveFishDetection,
    LiveFishTrack,
    LiveMonitorSession,
    utc_now,
)
from app.schemas.track import TrackSpeciesUpdate
from app.services.live_monitor import aware, stored_media_path
from app.services.species_quality import review_diagnostics
from app.services.species_request import (
    IdentificationBusyError,
    IdentificationDisabledError,
    claim,
    identification_payload,
    requested_frames,
    run_request,
)
from app.services.video_media import MediaPathError

router = APIRouter(prefix="/live", tags=["live"])


class StartRequest(BaseModel):
    """Cameras are selected by key. Accepting a URL here would make the resolver an SSRF sink."""

    source: str | None = None
    species_id_fish_target: int | None = Field(default=None, ge=0)
    species_id_frames_per_fish: int | None = Field(default=None, ge=1, le=20)


class IdentifyRequest(BaseModel):
    """How many frames of one chosen fish the operator is willing to pay for."""

    frames: int | None = Field(default=None, ge=1, le=50)


def get_source(settings, key):
    source = settings.live_source(key or settings.live_default_source_key)
    if source is None:
        raise HTTPException(404, "Unknown camera")
    return source


def source_label(settings, key):
    source = settings.live_source(key)
    return source.label if source else key


def open_session_for(db, source_key=None):
    query = select(LiveMonitorSession).where(LiveMonitorSession.status.in_(LIVE_OPEN_STATUSES))
    if source_key is not None:
        query = query.where(LiveMonitorSession.source_key == source_key)
    return db.scalar(query.order_by(LiveMonitorSession.created_at))


def get_session(db, session_id):
    session = db.get(LiveMonitorSession, session_id)
    if session is None:
        raise HTTPException(404, "Live session not found")
    return session


def session_data(session, settings, db):
    now = utc_now()
    heartbeat = session.heartbeat_at
    stale = session.status in LIVE_OPEN_STATUSES and heartbeat is not None and (
        now - aware(heartbeat)).total_seconds() > settings.live_worker_stale_seconds
    counts = dict(db.execute(select(LiveFishTrack.fishial_state, func.count()).where(
        LiveFishTrack.session_id == session.id).group_by(LiveFishTrack.fishial_state)).all())
    return {
        "species_id": {"enabled": session.species_id_enabled,
                       "fish_target": session.species_id_fish_target,
                       "frames_per_fish": session.species_id_frames_per_fish,
                       "fish_enrolled": session.species_id_fish_enrolled,
                       "fish_identified": counts.get("identified", 0),
                       "fish_review_required": counts.get("review_required", 0) + counts.get("error", 0),
                       "api_calls": session.species_id_api_calls,
                       "manual_api_calls": session.species_id_manual_api_calls,
                       "candidates": counts.get("candidate", 0),
                       "calls_saved": session.species_id_calls_saved,
                       "region": session.species_id_region},
        "id": str(session.id), "status": "stopping" if session.stop_requested and session.status in LIVE_OPEN_STATUSES else session.status,
        "created_at": aware(session.created_at),
        "started_at": aware(session.started_at) if session.started_at else None,
        "stopped_at": aware(session.stopped_at) if session.stopped_at else None,
        "last_frame_at": aware(session.last_frame_at) if session.last_frame_at else None,
        "heartbeat_at": aware(heartbeat) if heartbeat else None,
        "worker_stale": stale, "error_message": session.error_message,
        "frames_processed": session.frames_processed, "dropped_segments": session.dropped_segments,
        "reconnect_count": session.reconnect_count,
        "lag_seconds": max(0, (now - aware(session.last_frame_at)).total_seconds()) if session.last_frame_at else None,
        "lost_track_seconds": settings.live_lost_track_seconds,
        "annotated_stream_url": f"/live/{session.id}/annotated-stream" if session.snapshot_path else None,
        "source_key": session.source_key, "source_label": source_label(settings, session.source_key),
    }


def track_data(track):
    return {
        "id": str(track.id), "session_id": str(track.session_id), "status": track.status,
        "first_seen_at": aware(track.first_seen_at), "last_seen_at": aware(track.last_seen_at),
        "finalized_at": aware(track.finalized_at) if track.finalized_at else None,
        "finalization_reason": track.finalization_reason, "detection_count": track.detection_count,
        "max_confidence": track.max_confidence, "mean_confidence": track.mean_confidence,
        "species": track.species, "bbox": [track.x1, track.y1, track.x2, track.y2],
        "clip_url": f"/live/tracks/{track.id}/clip" if track.clip_path else None,
        "crop_url": f"/live/tracks/{track.id}/crop" if track.crop_path else None,
        "media_error": track.media_error,
        "fishial_state": track.fishial_state, "fishial_species": track.fishial_species,
        "fishial_species_confidence": track.fishial_species_confidence,
        "fishial_frames_used": track.fishial_frames_used, "fishial_votes": track.fishial_votes,
        "fishial_quality_score": track.fishial_quality_score,
        "manual_species": track.manual_species,
        "manual_species_at": aware(track.manual_species_at) if track.manual_species_at else None,
        "fishial_diagnostics": review_diagnostics(track.fishial_votes)
        if track.fishial_state in {"review_required", "error"} else None,
        "identification": identification_payload(track),
    }


@router.get("/sources")
def sources(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    """Everything the camera selector needs: the configured list and what is running."""

    active = {row.source_key: str(row.id) for row in db.scalars(select(LiveMonitorSession).where(
        LiveMonitorSession.status.in_(LIVE_OPEN_STATUSES)))}
    return {"default_key": settings.live_default_source_key, "enabled": settings.live_monitor_enabled,
            "available": not settings.viame_mock,
            "fishial": {"enabled": settings.fishial_enabled,
                        "max_fish_per_session": settings.fishial_max_fish_per_session,
                        "default_frames_per_fish": settings.fishial_default_frames_per_fish,
                        "request_max_frames": settings.fishial_request_max_frames,
                        "request_default_frames": settings.fishial_request_default_frames},
            "sources": [{"key": source.key, "label": source.label, "location": source.location,
                         "url": source.url, "active_session_id": active.get(source.key)}
                        for source in settings.live_source_registry]}


@router.get("/latest")
def latest(source: str | None = Query(default=None), db: Session = Depends(get_db),
           settings: Settings = Depends(get_settings)):
    configured = get_source(settings, source)
    session = db.scalar(select(LiveMonitorSession).where(
        LiveMonitorSession.source_key == configured.key).order_by(LiveMonitorSession.created_at.desc()).limit(1))
    return {"enabled": settings.live_monitor_enabled, "available": not settings.viame_mock,
            "source_key": configured.key,
            "session": session_data(session, settings, db) if session else None}


@router.post("/start", status_code=202)
def start(body: StartRequest | None = None, db: Session = Depends(get_db),
          settings: Settings = Depends(get_settings)):
    configured = get_source(settings, body.source if body else None)
    target = (body.species_id_fish_target or 0) if body else 0
    frames = (body.species_id_frames_per_fish if body else None)
    frames = settings.fishial_default_frames_per_fish if frames is None else frames
    if target > settings.fishial_max_fish_per_session:
        raise HTTPException(422, "Species fish target exceeds the deployment limit")
    if target > 0 and not settings.fishial_enabled:
        raise HTTPException(409, "Species identification is not configured on this deployment")
    if not settings.live_monitor_enabled:
        raise HTTPException(503, "Live monitoring is disabled; set LIVE_MONITOR_ENABLED=true on the API and live worker")
    if settings.viame_mock:
        raise HTTPException(409, "Live monitoring requires VIAME_MOCK=false and a live worker with VIAME installed")
    session = open_session_for(db, configured.key)
    if session is None:
        # One worker runs one session at a time, so a second camera would queue
        # behind the first with no indication of when it might start.
        running = open_session_for(db)
        if running is not None:
            raise HTTPException(409, f"{source_label(settings, running.source_key)} is already being monitored; "
                                     f"stop it before starting {configured.label}")
        session = LiveMonitorSession(source_url=configured.url, source_key=configured.key,
            species_id_enabled=target > 0, species_id_fish_target=target,
            species_id_frames_per_fish=frames)
        db.add(session)
        try:
            db.commit()
        except IntegrityError:
            # The partial unique index is per camera, so this resolves the race
            # against a concurrent start of this same camera.
            db.rollback()
            session = open_session_for(db, configured.key)
            if session is None:
                raise
    return session_data(session, settings, db)


@router.post("/{session_id}/stop")
def stop(session_id: uuid.UUID, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    session = get_session(db, session_id)
    if session.status in LIVE_OPEN_STATUSES:
        session.stop_requested = True
        if session.status == "queued":
            session.status, session.stopped_at = "stopped", utc_now()
        else:
            session.status = "stopping"
        db.commit()
    return session_data(session, settings, db)


@router.get("/{session_id}/status")
def status(session_id: uuid.UUID, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    return session_data(get_session(db, session_id), settings, db)


@router.get("/{session_id}/species")
def species(session_id: uuid.UUID, db: Session = Depends(get_db)):
    session = get_session(db, session_id)
    rows = db.execute(select(LiveFishTrack.fishial_species, func.count(),
                             func.avg(LiveFishTrack.fishial_species_confidence)).where(
        LiveFishTrack.session_id == session_id, LiveFishTrack.fishial_state == "identified"
    ).group_by(LiveFishTrack.fishial_species).order_by(
        func.count().desc(), LiveFishTrack.fishial_species)).all()
    review = db.scalar(select(func.count()).select_from(LiveFishTrack).where(
        LiveFishTrack.session_id == session_id,
        LiveFishTrack.fishial_state.in_(("review_required", "error"))))
    # "The model would not name these fish" is a different operational problem from
    # "the fish disagreed with each other", so an operator must be able to tell them
    # apart without opening every track's audit.
    reasons = Counter(review_diagnostics(track.fishial_votes)["reason"]
                      for track in db.scalars(select(LiveFishTrack).where(
                       LiveFishTrack.session_id == session_id,
                       LiveFishTrack.fishial_state.in_(("review_required", "error")))))
    return {"session_id": str(session_id), "as_of": utc_now(),
            "api_calls": session.species_id_api_calls, "review_required": review,
            "calls_saved": session.species_id_calls_saved,
            "declined": reasons.get("classifier returned no candidates", 0),
            "review_reasons": dict(reasons),
            "species": [{"species": name, "count": count, "mean_confidence": mean}
                        for name, count, mean in rows]}


@router.get("/{session_id}/activity")
def activity(session_id: uuid.UUID, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    session = get_session(db, session_id)
    # Detections carry capture timestamps, which trail the clock by the chunk and
    # inference time, so window the analyzed timeline rather than the wall clock.
    now = aware(session.last_frame_at) if session.last_frame_at else utc_now()
    since = now - timedelta(seconds=settings.live_activity_window_seconds)
    query = select(LiveFishDetection.observed_at, LiveFishDetection.track_id).join(
        LiveFishTrack, LiveFishTrack.id == LiveFishDetection.track_id).where(
            LiveFishTrack.session_id == session_id, LiveFishDetection.observed_at >= since,
            LiveFishDetection.observed_at <= now)
    rows = db.execute(query).all()
    buckets = [set() for _ in range(20)]
    width = settings.live_activity_window_seconds / len(buckets)
    for timestamp, track_id in rows:
        index = min(len(buckets) - 1, int((aware(timestamp) - since).total_seconds() / width))
        buckets[index].add(str(track_id))
    counts = dict(db.execute(select(LiveFishTrack.status, func.count()).where(
        LiveFishTrack.session_id == session_id).group_by(LiveFishTrack.status)).all())
    return {"session_id": str(session_id), "as_of": now, "window_seconds": settings.live_activity_window_seconds,
        "active_tracks": counts.get("active", 0), "finalized_tracks": counts.get("finalized", 0),
        "window_tracks": len({track for _, track in rows}), "window_detections": len(rows),
        "series": [{"at": since + timedelta(seconds=i * width), "tracks": len(bucket)}
                   for i, bucket in enumerate(buckets)]}


@router.get("/{session_id}/tracks")
def tracks(session_id: uuid.UUID, status: str | None = Query(default=None, pattern="^(active|finalized)$"),
           limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0),
           db: Session = Depends(get_db)):
    get_session(db, session_id)
    query = select(LiveFishTrack).where(LiveFishTrack.session_id == session_id)
    if status:
        query = query.where(LiveFishTrack.status == status)
    return [track_data(t) for t in db.scalars(query.order_by(LiveFishTrack.last_seen_at.desc(), LiveFishTrack.id).limit(limit).offset(offset))]


@router.get("/{session_id}/clips")
def clips(session_id: uuid.UUID, limit: int = Query(default=50, ge=1, le=500),
          offset: int = Query(default=0, ge=0), db: Session = Depends(get_db)):
    get_session(db, session_id)
    return [track_data(t) for t in db.scalars(select(LiveFishTrack).where(
        LiveFishTrack.session_id == session_id, LiveFishTrack.status == "finalized").order_by(
            LiveFishTrack.finalized_at.desc(), LiveFishTrack.id).limit(limit).offset(offset))]


def checked_path(settings, session_id, value):
    try:
        path = stored_media_path(settings, session_id, value)
    except (MediaPathError, OSError, ValueError):
        raise HTTPException(404, "Live media is unavailable")
    if path is None:
        raise HTTPException(404, "Live media has not been generated or is no longer available")
    return path


@router.post("/tracks/{track_id}/identify", status_code=202)
def identify_live_track(track_id: uuid.UUID, background: BackgroundTasks,
                        body: IdentifyRequest | None = None, db: Session = Depends(get_db),
                        settings: Settings = Depends(get_settings)):
    """Name one fish an operator picked out of the live view, on demand.

    Available whether or not the session was started with automatic identification:
    that setting fixes what the session buys on its own, and says nothing about what
    an operator may ask for while watching. Its frames come from the footage the
    session has already retained, so a fish still in view can be identified from the
    moments it has shown so far.
    """

    track = db.get(LiveFishTrack, track_id)
    if track is None:
        raise HTTPException(404, "Live fish track not found")
    frames = requested_frames(body.frames if body else None, settings)
    try:
        claim(db, track, frames, settings)
    except (IdentificationBusyError, IdentificationDisabledError) as exc:
        raise HTTPException(409, str(exc)) from exc
    background.add_task(run_request, session_factory_for(db), LiveFishTrack, track.id, settings)
    return track_data(track)


@router.patch("/tracks/{track_id}/species")
def assign_live_track_species(track_id: uuid.UUID, payload: TrackSpeciesUpdate,
                              db: Session = Depends(get_db)):
    """Name a fish in the live view yourself, without waiting on a classifier.

    The same column and the same rules as a library track: its own field, nothing
    else touched, and clearing it puts the machine's answer back on top. Available
    while the session is still running, because the moment an operator recognises a
    fish is while they are watching it.
    """

    track = db.get(LiveFishTrack, track_id)
    if track is None:
        raise HTTPException(404, "Live fish track not found")
    if track.manual_species != payload.species:
        track.manual_species = payload.species
        track.manual_species_at = utc_now() if payload.species else None
        db.commit()
        db.refresh(track)
    return track_data(track)


@router.get("/tracks/{track_id}/clip")
def clip(track_id: uuid.UUID, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    track = db.get(LiveFishTrack, track_id)
    if track is None:
        raise HTTPException(404, "Live fish track not found")
    return FileResponse(checked_path(settings, track.session_id, track.clip_path), media_type="video/mp4")


@router.get("/tracks/{track_id}/crop")
def crop(track_id: uuid.UUID, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    track = db.get(LiveFishTrack, track_id)
    if track is None:
        raise HTTPException(404, "Live fish track not found")
    return FileResponse(checked_path(settings, track.session_id, track.crop_path), media_type="image/jpeg")


async def annotated_frames(factory, session_id, settings, request):
    previous = None
    while not await request.is_disconnected():
        content = None
        with factory() as db:
            session = db.get(LiveMonitorSession, session_id)
            if session is None:
                return
            try:
                path = stored_media_path(settings, session_id, session.snapshot_path)
                if path:
                    content = path.read_bytes()
            except (MediaPathError, OSError):
                return
            terminal = session.status not in LIVE_OPEN_STATUSES
            stale = session.heartbeat_at and (utc_now() - aware(session.heartbeat_at)).total_seconds() > settings.live_worker_stale_seconds
        # Release the database connection before yielding to a potentially slow viewer.
        if content is not None and content != previous:
            previous = content
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(content)).encode() + b"\r\n\r\n" + content + b"\r\n"
        if terminal or stale:
            yield b"--frame--\r\n"
            return
        await asyncio.sleep(.25)


@router.get("/{session_id}/annotated-stream")
def annotated_stream(session_id: uuid.UUID, request: Request, snapshot: bool = False,
                     db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    session = get_session(db, session_id)
    if not session.snapshot_path:
        raise HTTPException(503, "Waiting for the first analyzed camera frames", headers={"Retry-After": "2"})
    path = checked_path(settings, session_id, session.snapshot_path)
    if snapshot:
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    db.close()
    return StreamingResponse(annotated_frames(factory, session_id, settings, request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
