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


class SpeciesIdentificationRead(BaseModel):
    """Whether an operator may ask Fishial to name one chosen fish, and the limits.

    Served with system status because the control appears on several screens and
    none of them should have to discover the deployment's limits separately.
    """

    available: bool
    max_frames: int
    default_frames: int


class SystemStatusRead(BaseModel):
    ready: bool
    processing_mode: str
    compose_command: str
    database: DatabaseStatusRead
    worker: WorkerStatusRead
    queue: QueueStatusRead
    species_identification: SpeciesIdentificationRead
