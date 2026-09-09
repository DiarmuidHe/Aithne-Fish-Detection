from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DatabaseStatusRead(BaseModel):
    available: bool
    message: str


class WorkerStatusRead(BaseModel):
    available: bool
    expected_mode: str
    active_workers: int
    other_mode_workers: int
    current_jobs: int
    last_seen_at: datetime | None
    seconds_since_last_seen: float | None
    message: str


class QueueStatusRead(BaseModel):
    queued: int
    processing: int
    failed: int


class SystemStatusRead(BaseModel):
    ready: bool
    processing_mode: str
    compose_command: str
    database: DatabaseStatusRead
    worker: WorkerStatusRead
    queue: QueueStatusRead
