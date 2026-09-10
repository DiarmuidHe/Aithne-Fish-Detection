"""Bounded, unpaid collection using the existing live detector and crop ranker."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import LiveFishTrack, LiveMonitorSession, utc_now
from app.services.live_monitor import LiveTracker, scratch_path
from app.services.live_source import SegmentCapture, resolve_stream
from app.services.species_quality import track_quality
from app.workers.live_worker import detect_segment


class CropCollector(LiveTracker):
    """Reuse tracking/staging, omitting gallery media and all identification passes."""

    def _save_crop(self, *args):
        pass

    def _render_clip(self, *args):
        pass

    def _write_jpeg(self, *args):
        pass


def select_crops(db, session, settings, count=2):
    candidates = []
    for track in db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == session.id, LiveFishTrack.fishial_state == "candidate")):
        staged = track.fishial_votes.get("staged") or []
        if len(staged) < settings.fishial_min_frames_to_vote:
            continue
        quality = track_quality(track, settings, session.species_id_frames_per_fish)
        if quality < settings.fishial_quality_floor:
            continue
        best = max(staged, key=lambda entry: (entry["score"], -entry["frame_number"]))
        path = scratch_path(settings, str(session.id), str(track.id), "fishial",
                            f"{best['frame_number']:012d}.jpg")
        if path.is_file():
            candidates.append((quality, str(track.id), path, best))
    return sorted(candidates, key=lambda entry: (-entry[0], entry[1]))[:count]


def collect(directory: Path, source_key, settings, seconds=120, *,
            resolver=resolve_stream, capture_factory=SegmentCapture, detector=detect_segment):
    """Return a reusable manifest. This function never constructs a Fishial client.

    The tracker needs staging enabled, but only this private in-memory session sees
    it: no SpeciesIdentifier or production worker can submit these candidates.
    """
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source") != source_key:
            raise SystemExit("Existing capture belongs to another source")
        return manifest
    if settings.viame_mock:
        raise SystemExit("Crop collection requires real VIAME; use the GPU worker environment")
    source = settings.live_source(source_key)
    if source is None:
        raise SystemExit("Unknown capture source")
    directory.mkdir(parents=True, exist_ok=True)
    configured = settings.model_copy(update={
        "fishial_enabled": True, "fishial_preprocess": "none", "fishial_upscale_short_side": 0,
        "fishial_keep_staged_crops": False, "output_root": directory / "collection",
        "live_scratch_root": directory / "scratch",
        "live_segment_seconds": min(settings.live_segment_seconds, 10),
    })
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    capture = None
    deadline = time.monotonic() + seconds
    try:
        with Session(engine, expire_on_commit=False) as db:
            session = LiveMonitorSession(source_key=source_key, source_url=source.url,
                species_id_enabled=True, species_id_fish_target=2,
                species_id_frames_per_fish=settings.fishial_default_frames_per_fish)
            db.add(session)
            db.commit()
            tracker = CropCollector(db, session, configured)
            url = resolver(source.url, configured, source.label)
            capture = capture_factory(url, directory / "capture", configured)
            selected = []
            while time.monotonic() < deadline:
                segment = capture.next_segment()
                if segment is None:
                    time.sleep(.2)
                    continue
                observations = detector(segment, configured, session.id)
                by_frame = defaultdict(list)
                for observation in observations:
                    by_frame[observation.frame_number].append(observation)
                reader = tracker.cv2.VideoCapture(str(segment.path))
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
                db.commit()
                selected = select_crops(db, session, configured)
                print(f"capture: {session.frames_processed} frames; {len(selected)}/2 eligible fish", flush=True)
                if len(selected) == 2:
                    break
            if len(selected) < 2:
                raise SystemExit("Capture ended without two eligible fish; zero Fishial calls spent")
            entries = []
            for index, (quality, track_id, path, staged) in enumerate(selected):
                name = f"crop-{index + 1}.jpg"
                (directory / name).write_bytes(path.read_bytes())
                entries.append({"file": name, "track": track_id, "track_quality": quality, **staged})
            manifest = {"source": source_key, "region": source.region, "captured_at": utc_now().isoformat(),
                        "preprocess": "none", "crops": entries, "capture_api_calls": 0}
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(manifest_path)
            return manifest
    finally:
        if capture:
            capture.close()
        engine.dispose()
