"""Partial-success batch operations using the same single-video services."""

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.videos import create_annotated_video
from app.config import Settings, get_settings
from app.db.database import get_db
from app.services.video_service import enqueue_video_processing

router = APIRouter(prefix="/batch", tags=["batch"])
logger = logging.getLogger(__name__)


class BatchRequest(BaseModel):
    video_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)


@router.post("/{action}")
def batch_action(action: Literal["process", "annotate"], payload: BatchRequest,
                 db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    results = []
    for video_id in dict.fromkeys(payload.video_ids):
        try:
            if action == "process":
                job = enqueue_video_processing(db, video_id, settings)
                result = {"video_id": video_id, "status": job.status, "job_id": job.id}
                # Release locks even when enqueue returned an existing job.
                db.commit()
            else:
                annotation = create_annotated_video(video_id, True, db, settings)
                result = {"video_id": video_id, "status": "completed", "url": annotation.url}
            results.append({**result, "ok": True})
        except (LookupError, HTTPException) as exc:
            db.rollback()
            results.append({"video_id": video_id, "ok": False, "status": "failed",
                            "error": exc.detail if isinstance(exc, HTTPException) else "Video not found"})
        except Exception:
            db.rollback()
            logger.exception("Batch action failed action=%s video_id=%s", action, video_id)
            results.append({"video_id": video_id, "ok": False, "status": "failed",
                            "error": "Action failed. See server logs."})
    succeeded = sum(row["ok"] for row in results)
    return {"results": results, "total": len(results), "succeeded": succeeded,
            "failed": len(results) - succeeded}
