import csv
import io
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.database import get_db
from app.db.models import FishDetection, FishTrack, Video
from app.services.reporting import provenance, result_job, track_accepted, track_row

router = APIRouter(tags=["exports"])
ExportKind = Literal["summary", "accepted-tracks", "all-tracks", "detections"]
PROVENANCE = ["video_id", "filename", "video_status", "content_sha256", "video_created_at",
              "video_fps", "job_id", "job_created_at", "processing_started_at",
              "processing_finished_at", "pipeline", "confidence_threshold", "model_name",
              "model_version", "viame_version", "frame_offset", "downsample_fps",
              "processing_mode", "run_command_sha256"]
TRACK_FIELDS = ["track_id", "viame_track_id", "machine_accepted", "accepted", "review_state",
                "reviewed_at", "first_frame", "last_frame", "first_timestamp_seconds",
                "last_timestamp_seconds", "detection_count", "mean_confidence", "max_confidence",
                "species", "species_confidence"]
DETECTION_FIELDS = ["detection_id", "frame_number", "timestamp_seconds", "x1", "y1", "x2", "y2",
                    "confidence", "class_name", "class_confidence"]


def csv_cell(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc).isoformat()
    # Protect spreadsheet consumers from formulas in filenames and model labels.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def export_response(kind, videos, db):
    fields = PROVENANCE + (["accepted_fish_count", "accepted_detections", "all_track_count"]
        if kind == "summary" else TRACK_FIELDS + (DETECTION_FIELDS if kind == "detections" else []))
    exported_at = datetime.now(timezone.utc).isoformat()

    def stream():
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["exported_at", *fields])
        writer.writeheader()
        yield "\ufeff" + buffer.getvalue()
        for video in videos:
            jobs = {job.id: job for job in video.jobs}
            tracks = list(db.scalars(select(FishTrack).where(FishTrack.video_id == video.id)
                                    .order_by(FishTrack.first_frame, FishTrack.viame_track_id)))
            if kind == "summary":
                accepted = [t for t in tracks if track_accepted(t, video, jobs)]
                rows = [{**provenance(video, result_job(video)), "accepted_fish_count": len(accepted),
                         "accepted_detections": sum(t.detection_count for t in accepted),
                         "all_track_count": len(tracks)}]
            elif kind == "detections":
                track_map = {t.id: t for t in tracks}
                observations = db.scalars(select(FishDetection).join(FishTrack)
                    .where(FishTrack.video_id == video.id)
                    .order_by(FishDetection.frame_number, FishDetection.id)
                    .execution_options(yield_per=1000))
                rows = ({**track_row(track_map[d.fish_track_id], video, jobs),
                         "detection_id": d.id,
                         **{key: getattr(d, key) for key in DETECTION_FIELDS if key != "detection_id"}}
                        for d in observations)
            else:
                rows = (track_row(t, video, jobs) for t in tracks
                        if kind == "all-tracks" or track_accepted(t, video, jobs))
            for row in rows:
                buffer.seek(0)
                buffer.truncate(0)
                writer.writerow({"exported_at": exported_at,
                                 **{key: csv_cell(value) for key, value in row.items()}})
                yield buffer.getvalue()

    return StreamingResponse(stream(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="fish-monitor-{kind}.csv"',
                 "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@router.get("/videos/{video_id}/exports/{kind}.csv")
def export_video(video_id: uuid.UUID, kind: ExportKind, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    return export_response(kind, [video], db)


@router.get("/exports/batch.csv")
def export_batch(kind: ExportKind = "summary", video_ids: list[uuid.UUID] | None = Query(None),
                 db: Session = Depends(get_db)):
    if video_ids is not None and len(video_ids) > 100:
        raise HTTPException(422, "Select at most 100 videos or omit video_ids for all videos")
    query = select(Video).options(selectinload(Video.jobs)).order_by(Video.created_at, Video.id)
    if video_ids is not None:
        query = query.where(Video.id.in_(video_ids))
    videos = list(db.scalars(query))
    if video_ids is not None and len(videos) != len(set(video_ids)):
        raise HTTPException(404, "One or more videos not found")
    return export_response(kind, videos, db)
