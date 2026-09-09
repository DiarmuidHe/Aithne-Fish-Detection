from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import JobStatus, ProcessingJob, WorkerHeartbeat
from app.schemas.system import SystemStatusRead

router = APIRouter(prefix="/system", tags=["system"])

MOCK_COMPOSE_COMMAND = "docker compose up -d --build"
GPU_COMPOSE_COMMAND = (
    "docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build"
)


@router.get("/status", response_model=SystemStatusRead)
def get_system_status(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SystemStatusRead:
    mode = settings.effective_deployment_mode
    command = GPU_COMPOSE_COMMAND if mode == "gpu" else MOCK_COMPOSE_COMMAND
    try:
        db.execute(text("SELECT 1"))
        database = {"available": True, "message": "PostgreSQL is available"}
    except SQLAlchemyError:
        db.rollback()
        return SystemStatusRead(
            ready=False,
            processing_mode=mode,
            compose_command=command,
            database={"available": False, "message": "PostgreSQL is unavailable"},
            worker={
                "available": False,
                "expected_mode": mode,
                "active_workers": 0,
                "other_mode_workers": 0,
                "current_jobs": 0,
                "last_seen_at": None,
                "seconds_since_last_seen": None,
                "message": "Worker status is unavailable until the database recovers",
            },
            queue={"queued": 0, "processing": 0, "failed": 0},
        )

    now = datetime.now(UTC)
    cutoff = now - timedelta(seconds=settings.worker_stale_after_seconds)
    heartbeats = list(
        db.scalars(
            select(WorkerHeartbeat)
            .where(WorkerHeartbeat.last_seen_at >= cutoff)
            .order_by(WorkerHeartbeat.last_seen_at.desc())
        ).all()
    )
    matching = [heartbeat for heartbeat in heartbeats if heartbeat.mode == mode]
    other_mode_workers = len(heartbeats) - len(matching)
    latest = matching[0] if matching else None
    last_seen_at = latest.last_seen_at if latest else None
    seconds_since_last_seen = None
    if last_seen_at is not None:
        if last_seen_at.tzinfo is None:
            last_seen_at = last_seen_at.replace(tzinfo=UTC)
        seconds_since_last_seen = max(0.0, (now - last_seen_at).total_seconds())

    queue_counts = dict(
        db.execute(
            select(ProcessingJob.status, func.count(ProcessingJob.id))
            .group_by(ProcessingJob.status)
        ).all()
    )
    worker_available = bool(matching)
    if worker_available:
        worker_message = f"{len(matching)} {mode} worker{'s' if len(matching) != 1 else ''} available"
    elif other_mode_workers:
        worker_message = f"No {mode} worker is available; a worker for another mode is running"
    else:
        worker_message = f"No {mode} worker heartbeat has been received recently"

    return SystemStatusRead(
        ready=worker_available,
        processing_mode=mode,
        compose_command=command,
        database=database,
        worker={
            "available": worker_available,
            "expected_mode": mode,
            "active_workers": len(matching),
            "other_mode_workers": other_mode_workers,
            "current_jobs": sum(1 for heartbeat in matching if heartbeat.current_job_id is not None),
            "last_seen_at": last_seen_at,
            "seconds_since_last_seen": seconds_since_last_seen,
            "message": worker_message,
        },
        queue={
            "queued": queue_counts.get(JobStatus.QUEUED.value, 0),
            "processing": queue_counts.get(JobStatus.PROCESSING.value, 0),
            "failed": queue_counts.get(JobStatus.FAILED.value, 0),
        },
    )
