"""Bounded histograms and observed counts; no inferred detections or frame filling."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.database import get_db
from app.db.models import FishDetection, FishTrack, Video
from app.services.reporting import basename, track_accepted

router = APIRouter(prefix="/analytics", tags=["analytics"])


def histogram(values, bins, upper):
    width = upper / bins
    rows = [{"start": i * width, "end": (i + 1) * width, "count": 0} for i in range(bins)]
    for value in values:
        rows[min(bins - 1, max(0, int(value / width)))]["count"] += 1
    return rows


@router.get("/videos")
def overview(db: Session = Depends(get_db)):
    videos = db.scalars(select(Video).options(selectinload(Video.jobs), selectinload(Video.tracks))
                        .order_by(Video.created_at, Video.id))
    result = []
    for video in videos:
        jobs = {j.id: j for j in video.jobs}
        accepted = [t for t in video.tracks if track_accepted(t, video, jobs)]
        result.append({"video_id": video.id, "filename": basename(video.original_filename),
                       "status": video.processing_status, "accepted_fish_count": len(accepted),
                       "accepted_detections": sum(t.detection_count for t in accepted)})
    return result


@router.get("/videos/{video_id}")
def video_analytics(video_id: uuid.UUID, bins: int = Query(20, ge=1, le=100),
                    db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    tracks = list(db.scalars(select(FishTrack).where(FishTrack.video_id == video.id)))
    jobs = {j.id: j for j in video.jobs}
    accepted_ids = {t.id for t in tracks if track_accepted(t, video, jobs)}
    durations = [max(0, t.last_timestamp_seconds - t.first_timestamp_seconds) for t in tracks
                 if t.first_timestamp_seconds is not None and t.last_timestamp_seconds is not None]
    duration_upper = max(durations, default=0) or 1
    maximum = db.scalar(select(func.max(FishDetection.timestamp_seconds)).join(FishTrack)
                        .where(FishTrack.video_id == video.id)) or 0
    # Include the video tail, with a bounded number of bins even for long recordings.
    upper = max(maximum, video.duration_seconds or 0) or 1
    width = upper / bins
    times = [{"start": i * width, "end": (i + 1) * width,
              "all_detections": 0, "accepted_detections": 0} for i in range(bins)]
    confidence = histogram([], 10, 1)
    unknown_timestamps = 0
    # Fetch aggregated observations by track and bin; acceptance can use each run's snapshot.
    time_bin = case((FishDetection.timestamp_seconds.is_(None), None),
                    else_=func.floor(FishDetection.timestamp_seconds / width))
    conf_bin = func.floor(FishDetection.confidence * 10)
    time_rows = db.execute(select(FishDetection.fish_track_id, time_bin, func.count())
        .select_from(FishDetection).join(FishTrack).where(FishTrack.video_id == video.id)
        .group_by(FishDetection.fish_track_id, time_bin))
    for track_id, bucket, count in time_rows:
        if bucket is None:
            unknown_timestamps += count
            continue
        row = times[min(bins - 1, max(0, int(bucket)))]
        row["all_detections"] += count
        if track_id in accepted_ids:
            row["accepted_detections"] += count
    for bucket, count in db.execute(select(conf_bin, func.count()).select_from(FishDetection)
            .join(FishTrack).where(FishTrack.video_id == video.id).group_by(conf_bin)):
        confidence[min(9, max(0, int(bucket)))]["count"] += count
    return {"video_id": video.id, "status": video.processing_status,
            "time_bins": times, "confidence_distribution": confidence,
            "track_duration_distribution": histogram(durations, 10, duration_upper),
            "unknown_timestamp_detections": unknown_timestamps,
            "unknown_duration_tracks": len(tracks) - len(durations),
            "all_track_count": len(tracks), "accepted_fish_count": len(accepted_ids)}
