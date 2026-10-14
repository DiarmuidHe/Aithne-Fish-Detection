"""Development CVAT mirror. CVAT annotations remain human-owned after first import."""
from __future__ import annotations

import logging
import math
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.database import SessionLocal
from app.db.models import CvatSync, JobStatus, ProcessingJob, Video
from app.services.viame_parser import ParsedVIAMEResult, parse_viame_csv

logger = logging.getLogger(__name__)


def typical_frame_step(parsed: ParsedVIAMEResult) -> int:
    """Infer the source-frame spacing of detector samples, ignoring long gaps."""
    gaps = Counter(
        second.frame_number - first.frame_number
        for track in parsed.tracks
        for first, second in zip(track.observations, track.observations[1:])
        if 0 < second.frame_number - first.frame_number <= 30
    )
    return gaps.most_common(1)[0][0] if gaps else 1


def planned_cvat_shapes(track, width: int, height: int, frame_count: int,
                        frame_step: int | None) -> list[tuple[int, list[float], bool, float | None]]:
    """End visibility at the first missing sample in a long detection gap."""
    result: list[tuple[int, list[float], bool, float | None]] = []
    previous = None
    for observation in track.observations:
        points = [observation.x1, observation.y1, observation.x2, observation.y2]
        if not all(math.isfinite(value) for value in points):
            continue
        if not (0 <= points[0] < points[2] <= width and 0 <= points[1] < points[3] <= height):
            continue
        if not 0 <= observation.frame_number < frame_count:
            continue
        if (frame_step is not None and previous is not None
                and observation.frame_number-previous.frame_number > 2*frame_step):
            result.append((previous.frame_number+frame_step,
                           [previous.x1, previous.y1, previous.x2, previous.y2], True, None))
        result.append((observation.frame_number, points, False, observation.confidence))
        previous = observation
    if previous is not None and previous.frame_number+1 < frame_count:
        result.append((previous.frame_number+1,
                       [previous.x1, previous.y1, previous.x2, previous.y2], True, None))
    return result


def configured(settings: Settings) -> bool:
    token = settings.cvat_access_token.get_secret_value() if settings.cvat_access_token else ""
    password = settings.cvat_password.get_secret_value() if settings.cvat_password else ""
    return bool(settings.cvat_dev_mode and (token or (settings.cvat_username and password)))


def queue_cvat_video(db, video_id: uuid.UUID) -> CvatSync:
    link = db.get(CvatSync, video_id)
    if link is None:
        link = CvatSync(video_id=video_id, status="queued", updated_at=datetime.now(UTC))
        db.add(link)
        db.commit()
    return link


class CvatGateway:
    def __init__(self, client):
        self.client = client

    def create_task(self, video: Video) -> int:
        from cvat_sdk import models
        from cvat_sdk.core.proxies.tasks import ResourceType

        name = f"Aithne {video.id} {video.original_filename}"
        # A retry after a lost response must reuse the remote task.
        for task in self.client.tasks.list(search=str(video.id)):
            if task.name == name:
                return task.id
        attributes = [
            models.AttributeRequest(name="viame_track_id", mutable=False,
                                    input_type="text", values=[]),
            models.AttributeRequest(name="viame_class", mutable=False,
                                    input_type="text", values=[]),
            models.AttributeRequest(name="viame_confidence", mutable=True,
                                    input_type="text", values=[]),
        ]
        task = self.client.tasks.create_from_data(
            spec=models.TaskWriteRequest(
                name=name, labels=[models.PatchedLabelRequest(name="fish", attributes=attributes)]
            ),
            resource_type=ResourceType.LOCAL,
            resources=[Path(video.storage_path)],
            data_params={"image_quality": 80},
        )
        return task.id

    def create_candidate_task(self, video: Video, candidate_name: str) -> int:
        """Keep human corrections in the original task when presenting a new model."""
        from cvat_sdk import models
        from cvat_sdk.core.proxies.tasks import ResourceType

        name = f"Aithne candidate {video.id} {candidate_name}"
        for task in self.client.tasks.list(search=str(video.id)):
            if task.name == name:
                return task.id
        attributes = [
            models.AttributeRequest(name="viame_track_id", mutable=False,
                                    input_type="text", values=[]),
            models.AttributeRequest(name="viame_class", mutable=False,
                                    input_type="text", values=[]),
            models.AttributeRequest(name="viame_confidence", mutable=True,
                                    input_type="text", values=[]),
        ]
        task = self.client.tasks.create_from_data(
            spec=models.TaskWriteRequest(
                name=name, labels=[models.PatchedLabelRequest(name="fish", attributes=attributes)]
            ),
            resource_type=ResourceType.LOCAL,
            resources=[Path(video.storage_path)],
            data_params={"image_quality": 80},
        )
        return task.id

    def has_annotations(self, task_id: int) -> bool:
        annotations = self.client.tasks.retrieve(task_id).get_annotations()
        return bool(annotations.tags or annotations.shapes or annotations.tracks)

    def import_tracks(self, task_id: int, parsed: ParsedVIAMEResult,
                      width: int, height: int, *, break_long_gaps: bool = False) -> int:
        from cvat_sdk import models

        task = self.client.tasks.retrieve(task_id)
        frame_count = task.size
        if not frame_count:
            raise ValueError("CVAT task has no video frames")
        label = next(label for label in task.get_labels() if label.name == "fish")
        attributes = {attribute.name: attribute.id for attribute in label.attributes}
        tracks = []
        frame_step = typical_frame_step(parsed) if break_long_gaps else None
        for source_track in parsed.tracks:
            shapes = []
            for frame, points, outside, confidence in planned_cvat_shapes(
                    source_track, width, height, frame_count, frame_step):
                shapes.append(models.TrackedShapeRequest(
                    type="rectangle", frame=frame,
                    points=points, outside=outside, occluded=False,
                    attributes=[] if confidence is None else [models.AttributeValRequest(
                        spec_id=attributes["viame_confidence"],
                        value=f"{confidence:.6f}")],
                ))
            if not shapes:
                continue
            tracks.append(models.LabeledTrackRequest(
                label_id=label.id, frame=shapes[0].frame, source="auto", shapes=shapes,
                attributes=[
                    models.AttributeValRequest(spec_id=attributes["viame_track_id"],
                                               value=source_track.viame_track_id),
                    models.AttributeValRequest(spec_id=attributes["viame_class"],
                                               value=source_track.species or "fish"),
                ],
            ))
        if tracks:
            task.set_annotations(models.LabeledDataRequest(tracks=tracks))
        return len(tracks)


@contextmanager
def gateway_for(settings: Settings):
    from cvat_sdk import make_client

    url = settings.cvat_url.rstrip("/")
    if settings.cvat_access_token:
        auth = {"access_token": settings.cvat_access_token.get_secret_value()}
    else:
        auth = {"credentials": (
            settings.cvat_username, settings.cvat_password.get_secret_value()
        )}
    with make_client(url, **auth) as client:
        yield CvatGateway(client)


def _set_state(session_factory, video_id, status, *, task_id=None, job_id=None, error=None):
    with session_factory() as db:
        link = db.get(CvatSync, video_id)
        if link is None:
            return
        link.status = status
        link.updated_at = datetime.now(UTC)
        link.error_message = error
        if task_id is not None:
            link.task_id = task_id
        if job_id is not None:
            link.imported_job_id = job_id
        db.commit()


def sync_cvat_video(video_id: uuid.UUID, settings: Settings,
                    session_factory: sessionmaker = SessionLocal,
                    gateway_factory: Callable = gateway_for) -> None:
    """Upload once, then add the first VIAME pass if no CVAT edits exist."""
    if not configured(settings):
        return
    with session_factory() as db:
        link = db.get(CvatSync, video_id)
        video = db.get(Video, video_id)
        if link is None or video is None or link.status in {"uploading", "importing"}:
            return
        task_id, imported_job_id = link.task_id, link.imported_job_id
        video_data = video
    try:
        with gateway_factory(settings) as gateway:
            if task_id is None:
                _set_state(session_factory, video_id, "uploading")
                task_id = gateway.create_task(video_data)
                _set_state(session_factory, video_id, "ready", task_id=task_id)
            if imported_job_id is not None:
                return
            with session_factory() as db:
                job = db.scalars(select(ProcessingJob).where(
                    ProcessingJob.video_id == video_id,
                    ProcessingJob.status == JobStatus.COMPLETED.value,
                    ProcessingJob.output_csv_path.is_not(None),
                ).order_by(ProcessingJob.finished_at.desc())).first()
                if job is None:
                    return
                job_id = job.id
                csv_path = Path(job.output_csv_path)
                offset = int(job.configuration.get("frame_number_offset", 0))
                width, height = video_data.width, video_data.height
            if not width or not height:
                raise ValueError("Source video dimensions are required")
            if gateway.has_annotations(task_id):
                _set_state(session_factory, video_id, "needs_review",
                           error="CVAT already has annotations; suggestions were not imported")
                return
            parsed = parse_viame_csv(csv_path, frame_number_offset=offset)
            _set_state(session_factory, video_id, "importing")
            count = gateway.import_tracks(task_id, parsed, width, height)
            _set_state(session_factory, video_id, "review", job_id=job_id)
            logger.info("CVAT suggestions imported video_id=%s task_id=%s tracks=%s",
                        video_id, task_id, count)
    except Exception as exc:  # noqa: BLE001 - mirror failures must update durable sync state
        logger.warning("CVAT sync failed video_id=%s type=%s", video_id, type(exc).__name__)
        _set_state(session_factory, video_id, "failed",
                   error=f"CVAT sync failed ({type(exc).__name__}); retry from Development")
