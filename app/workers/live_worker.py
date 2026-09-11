"""Dedicated live worker: python -m app.workers.live_worker."""
from __future__ import annotations

import logging
import shutil
import signal
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from sqlalchemy import select, update

from app.config import get_settings
from app.db.database import SessionLocal
from app.db.models import LIVE_OPEN_STATUSES, LiveMonitorSession, utc_now
from app.services.fish_enhancement import initialize_enhancement
from app.services.live_monitor import LiveTracker, live_path, scratch_path
from app.services.live_source import LiveSourceError, SegmentCapture, resolve_stream
from app.services.live_species import SpeciesIdentifier
from app.services.viame_parser import parse_viame_csv
from app.services.viame_runner import build_viame_runner
from app.services.video_media import check_live_video_runtime

logger = logging.getLogger(__name__)


def detect_segment(segment, settings, session_id):
    root = live_path(settings, str(session_id), "inference")
    root.mkdir(parents=True, exist_ok=True)
    configured = settings.model_copy(update={"job_root": root,
        "viame_timeout_seconds": settings.live_detector_timeout_seconds,
        "viame_downsample_fps": settings.live_fps})
    job_id = uuid.uuid4()
    succeeded = False
    try:
        result = build_viame_runner(configured).run(segment.path, job_id)
        parsed = parse_viame_csv(result.output_csv_path, fps=settings.live_fps,
                                 frame_number_offset=settings.viame_frame_number_offset)
        if parsed.skipped_rows:
            raise RuntimeError("VIAME returned malformed live detections")
        succeeded = True
        return parsed.detections
    finally:
        directory = live_path(settings, str(session_id), "inference", str(job_id))
        if succeeded and directory.exists():
            shutil.rmtree(directory)


def claim_session(factory, worker_id):
    with factory() as db:
        candidate = db.scalar(select(LiveMonitorSession.id).where(
            LiveMonitorSession.status == "queued").order_by(LiveMonitorSession.created_at).limit(1))
        if candidate is None:
            return None
        # Compare-and-set also works in SQLite; only one worker can claim a session.
        result = db.execute(update(LiveMonitorSession).where(
            LiveMonitorSession.id == candidate, LiveMonitorSession.status == "queued").values(
                status="starting", worker_id=worker_id, heartbeat_at=utc_now(), started_at=utc_now()))
        db.commit()
        return candidate if result.rowcount else None


def recover_stale_sessions(factory, settings):
    cutoff = utc_now() - timedelta(seconds=settings.live_worker_stale_seconds)
    with factory() as db:
        rows = db.scalars(select(LiveMonitorSession).where(
            LiveMonitorSession.status.in_(LIVE_OPEN_STATUSES),
            LiveMonitorSession.status != "queued", LiveMonitorSession.heartbeat_at < cutoff)).all()
        for session in rows:
            result = db.execute(update(LiveMonitorSession).execution_options(synchronize_session=False).where(
                LiveMonitorSession.id == session.id, LiveMonitorSession.heartbeat_at < cutoff,
                LiveMonitorSession.status.in_(LIVE_OPEN_STATUSES)).values(
                    status="failed", stopped_at=utc_now(), error_message="Live worker heartbeat expired; start a new session"))
            db.commit()
            if result.rowcount:
                db.refresh(session)
                tracker = LiveTracker(db, session, settings)
                tracker.expire(utc_now(), "worker-lost")
                db.commit()
                identifier = SpeciesIdentifier(db, session, settings)
                try:
                    identifier.finish()
                finally:
                    identifier.close()
                    shutil.rmtree(scratch_path(settings, str(session.id)), ignore_errors=True)


def run_session(session_id, settings, factory=SessionLocal, shutdown=None,
                resolver=resolve_stream, capture_factory=SegmentCapture, detector=detect_segment):
    shutdown = shutdown or threading.Event()
    heartbeat_stop = threading.Event()
    with factory() as db:
        session = db.get(LiveMonitorSession, session_id)
        owner = session.worker_id
        configured = settings.live_source(session.source_key)
        # Log the key, never the resolved URL: it can carry signed credentials.
        label = configured.label if configured else session.source_key
        logger.info("Live session %s monitoring camera %s", session_id, session.source_key)
        if session.species_id_enabled and session.species_id_region is None and configured:
            # Freeze the camera's region here so a later edit to the source registry
            # cannot retroactively reinterpret a completed session's filtering.
            session.species_id_region = configured.region
            db.commit()
        logger.info("Live species identification: fish_target=%s frames_per_fish=%s region=%s",
                    session.species_id_fish_target, session.species_id_frames_per_fish,
                    session.species_id_region)
        identifier = SpeciesIdentifier(db, session, settings)

        def heartbeat():
            while not heartbeat_stop.wait(2):
                try:
                    with factory() as pulse:
                        pulse.execute(update(LiveMonitorSession).where(
                            LiveMonitorSession.id == session_id, LiveMonitorSession.worker_id == owner,
                            LiveMonitorSession.status.in_(LIVE_OPEN_STATUSES)).values(heartbeat_at=utc_now()))
                        pulse.commit()
                except Exception:
                    logger.exception("Live worker heartbeat failed")

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        capture, pending, segment = None, None, None
        tracker = None
        retries, retry_at = 0, 0.0
        executor = ThreadPoolExecutor(max_workers=1)
        final_status, error = "stopped", None
        try:
            tracker = LiveTracker(db, session, settings)
            while not shutdown.is_set():
                db.refresh(session)
                if session.status not in LIVE_OPEN_STATUSES or session.worker_id != owner:
                    return
                if session.stop_requested or session.status == "stopping":
                    break
                if pending is not None:
                    if not pending.done():
                        shutdown.wait(.2)
                        continue
                    try:
                        observations = pending.result()
                    except Exception:
                        # A detector crash costs one segment, not the session: VIAME can
                        # fault on a single chunk and run normally on the next one.
                        logger.warning("Live detection failed for a segment", exc_info=True)
                        pending = None
                        segment.path.unlink(missing_ok=True)
                        retries += 1
                        session.dropped_segments += 1
                        session.error_message = "A camera segment could not be analyzed and was skipped"
                        db.commit()
                        if retries > settings.live_max_retries:
                            raise
                        retry_at = time.monotonic() + min(settings.live_max_retry_seconds,
                                                          settings.live_retry_seconds * 2 ** min(retries - 1, 16))
                        shutdown.wait(.2)
                        continue
                    pending = None
                    by_frame = defaultdict(list)
                    for observation in observations:
                        by_frame[observation.frame_number].append(observation)
                    cv2 = tracker.cv2
                    reader = cv2.VideoCapture(str(segment.path))
                    tracker.begin_segment()
                    index = 0
                    try:
                        while True:
                            ok, frame = reader.read()
                            if not ok:
                                break
                            tracker.process_frame(frame, by_frame.get(index, []),
                                segment.started_at + timedelta(seconds=index / settings.live_fps))
                            index += 1
                    finally:
                        reader.release()
                        segment.path.unlink(missing_ok=True)
                    if not index:
                        raise LiveSourceError("Captured camera segment could not be decoded")
                    session.status, session.error_message = "running", None
                    retries = 0
                    db.commit()
                    identifier.run_safely()
                # Expiry follows capture timestamps. In-flight chunks must be applied first;
                # inference latency is exposed by the API rather than rewriting timestamps.
                if time.monotonic() < retry_at:
                    tracker.expire(utc_now())
                    db.commit()
                    identifier.run_safely()
                    shutdown.wait(.2)
                    continue
                try:
                    if capture is None:
                        session.status = "starting" if not retries else "reconnecting"
                        db.commit()
                        url = resolver(session.source_url, settings, label)
                        directory = live_path(settings, str(session_id), "capture", str(uuid.uuid4()))
                        capture = capture_factory(url, directory, settings)
                    segment = capture.next_segment()
                    if capture.dropped:
                        session.dropped_segments += capture.dropped
                        capture.dropped = 0
                        db.commit()
                    if segment:
                        pending = executor.submit(detector, segment, settings, session_id)
                    else:
                        tracker.expire(utc_now())
                        db.commit()
                        identifier.run_safely()
                except LiveSourceError as exc:
                    if capture:
                        capture.close()
                        capture = None
                    retries += 1
                    session.reconnect_count += 1
                    session.status, session.error_message = "reconnecting", str(exc)
                    db.commit()
                    if retries > settings.live_max_retries:
                        raise
                    retry_at = time.monotonic() + min(settings.live_max_retry_seconds,
                                                      settings.live_retry_seconds * 2 ** min(retries - 1, 16))
                shutdown.wait(.2)
        except Exception as exc:
            logger.exception("Live monitoring failed for %s", session_id)
            final_status = "failed"
            error = str(exc) if isinstance(exc, LiveSourceError) else "Live detection or rendering failed; see live worker logs"
        finally:
            # Inference has a bounded timeout. Keep heartbeats alive while it exits.
            executor.shutdown(wait=True, cancel_futures=True)
            if capture:
                capture.close()
            try:
                db.rollback()
                db.refresh(session)
                if session.status in LIVE_OPEN_STATUSES and session.worker_id == owner:
                    if tracker:
                        # Reload after rollback so a partially applied failed frame is discarded.
                        tracker = LiveTracker(db, session, settings)
                        tracker.expire(utc_now(), "stopped" if final_status == "stopped" else "failed")
                    session.status, session.error_message = final_status, error
                    session.stopped_at = utc_now()
                    db.commit()
                    identifier.finish()
            finally:
                identifier.close()
                # Tracks left unrendered by a crash must not strand spool frames.
                shutil.rmtree(scratch_path(settings, str(session_id)), ignore_errors=True)
                heartbeat_stop.set()
                thread.join(timeout=3)


def run_worker_forever():
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if not settings.live_monitor_enabled:
        raise SystemExit("Set LIVE_MONITOR_ENABLED=true to run the live worker")
    if settings.viame_mock:
        raise SystemExit("Live monitoring requires real VIAME (VIAME_MOCK=false); mock detections would misrepresent the camera")
    check_live_video_runtime()
    initialize_enhancement(settings)  # Fail locally before claiming sessions or reserving calls.
    shutdown = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: shutdown.set())
    worker_id = f"live-{uuid.uuid4()}"
    while not shutdown.is_set():
        recover_stale_sessions(SessionLocal, settings)
        session_id = claim_session(SessionLocal, worker_id)
        if session_id:
            run_session(session_id, settings, shutdown=shutdown)
        else:
            shutdown.wait(settings.worker_poll_seconds)


if __name__ == "__main__":
    run_worker_forever()
