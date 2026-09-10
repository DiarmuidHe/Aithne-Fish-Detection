"""Incremental track association and media rendering, independent of the input source."""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import LiveFishDetection, LiveFishTrack, LiveMonitorSession
from app.services import video_media
from app.services.species_quality import frame_score, pool_score, preprocess_crop
from app.services.video_annotator import FrameAnnotation, _draw_annotation

logger = logging.getLogger(__name__)

WRITE_ATTEMPTS = 5
WRITE_RETRY_SECONDS = 0.1


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def live_path(settings: Settings, *parts: str) -> Path:
    root = settings.output_root.expanduser().resolve()
    live_root = (root / "live").resolve()
    video_media.ensure_under_root(live_root, root)
    path = live_root.joinpath(*map(str, parts)).resolve()
    video_media.ensure_under_root(path, live_root)
    return path


def scratch_path(settings: Settings, *parts: str) -> Path:
    """Short-lived crop frames. Never served, so it stays off the output volume."""
    root = settings.live_scratch_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root.joinpath(*map(str, parts)).resolve()
    video_media.ensure_under_root(path, root)
    return path


def stored_media_path(settings: Settings, session_id, value: str | None) -> Path | None:
    if not value:
        return None
    root = live_path(settings, str(session_id))
    path = Path(value).resolve()
    video_media.ensure_under_root(path, root)
    return path if path.is_file() else None


def similarity(track, box) -> float:
    x1, y1, x2, y2 = box
    intersection = max(0, min(track.x2, x2) - max(track.x1, x1)) * max(0, min(track.y2, y2) - max(track.y1, y1))
    union = (track.x2 - track.x1) * (track.y2 - track.y1) + (x2 - x1) * (y2 - y1) - intersection
    iou = intersection / union if union > 0 else 0
    distance = math.hypot((x1 + x2 - track.x1 - track.x2) / 2, (y1 + y2 - track.y1 - track.y2) / 2)
    scale = max(x2 - x1, y2 - y1, track.x2 - track.x1, track.y2 - track.y1, 1)
    return iou if iou >= .15 else (.1 * (1 - distance / scale) if distance < scale else 0)


class LiveTracker:
    def __init__(self, db: Session, session: LiveMonitorSession, settings: Settings):
        self.db, self.session, self.settings = db, session, settings
        self.cv2 = video_media.load_cv2()
        self.root = live_path(settings, str(session.id))
        self.root.mkdir(parents=True, exist_ok=True)
        self.active = {t.id: t for t in db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == session.id, LiveFishTrack.status == "active"))}
        self.chunk_ids: dict[str, uuid.UUID] = {}
        self.pool: dict[uuid.UUID, dict] = {}
        self.staged_bytes = 0
        if settings.fishial_enabled and session.species_id_enabled:
            self._load_candidate_pool()

    def _load_candidate_pool(self):
        """Rebuild the unpaid candidate pool from persisted audits after a restart.

        The pool is session-wide, not restricted to currently active tracks: a fish
        that swam out of view still holds its slot until it is selected, completed or
        evicted, because its staged crops are still the evidence we would pay for.
        """

        session = self.session
        if not session.species_id_candidate_pool_size and session.species_id_fish_target:
            target = session.species_id_fish_target
            # Staging is free, so keep many more candidates than we can ever pay for
            # and let selection spend the budget on the best of them. Fixed here so a
            # worker restart cannot silently resize the pool mid-session.
            session.species_id_candidate_pool_size = (
                self.settings.fishial_candidate_pool_size
                or max(3 * target, target + 8))
            self.db.commit()
        weights = self.settings.fishial_quality_weights
        for track in self.db.scalars(select(LiveFishTrack).where(
                LiveFishTrack.session_id == session.id,
                LiveFishTrack.fishial_state == "candidate")):
            staged = track.fishial_votes.get("staged") or []
            self.pool[track.id] = {"score": pool_score(staged, weights), "staged": staged}
        root = scratch_path(self.settings, str(session.id))
        self.staged_bytes = sum(path.stat().st_size for path in root.glob("*/fishial/*.jpg"))

    @property
    def max_staged(self) -> int:
        """Retained frames per candidate. Spare frames are free and give the selector
        and the early-stop rules room to skip weak ones."""

        return (self.settings.fishial_max_staged_frames_per_candidate
                or self.session.species_id_frames_per_fish + 2)

    def begin_segment(self):
        # VIAME IDs restart for every chunk. Never treat them as global identities.
        self.chunk_ids.clear()
        if self.pool:
            self._resync_pool()

    def _resync_pool(self):
        """Release slots the identifier consumed between segments.

        Selection, completion and the scratch wipe all happen in SpeciesIdentifier,
        so the pool this object holds goes stale once a pass runs. Re-reading it costs
        one bounded query per segment and lets a freed slot admit a new fish.
        """

        live = set(self.db.scalars(select(LiveFishTrack.id).where(
            LiveFishTrack.session_id == self.session.id,
            LiveFishTrack.fishial_state == "candidate")))
        for track_id in [key for key in self.pool if key not in live]:
            del self.pool[track_id]
        root = scratch_path(self.settings, str(self.session.id))
        self.staged_bytes = sum(path.stat().st_size for path in root.glob("*/fishial/*.jpg"))

    def process_frame(self, frame, observations, observed_at: datetime):
        observed_at = aware(observed_at)
        self.expire(observed_at)
        frame_number = self.session.frames_processed
        height, width = frame.shape[:2]
        candidates = []
        for obs in observations:
            values = [obs.bbox_left, obs.bbox_top, obs.bbox_right, obs.bbox_bottom, obs.confidence]
            if not all(math.isfinite(v) for v in values) or not self.settings.min_fish_confidence <= obs.confidence <= 1:
                continue
            box = (max(0, obs.bbox_left), max(0, obs.bbox_top),
                   min(width - 1, obs.bbox_right), min(height - 1, obs.bbox_bottom))
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((obs, box))
        assigned, used = {}, set()
        for index, (obs, box) in enumerate(candidates):
            track_id = self.chunk_ids.get(obs.track_id)
            if track_id in self.active and track_id not in used:
                assigned[index] = self.active[track_id]
                used.add(track_id)
        pairs = sorted(
            [(similarity(t, box), index, str(t.id))
             for index, (obs, box) in enumerate(candidates) if index not in assigned
             for t in self.active.values() if t.id not in used
             and (not t.species or not obs.class_name or t.species == obs.class_name)],
            reverse=True,
        )
        for score, index, track_id in pairs:
            key = uuid.UUID(track_id)
            if score > 0 and index not in assigned and key not in used:
                assigned[index] = self.active[key]
                used.add(key)
        annotated = frame.copy()
        for index, (obs, box) in enumerate(candidates):
            track = assigned.get(index)
            if track is None:
                track = LiveFishTrack(id=uuid.uuid4(), session_id=self.session.id,
                    status="active", first_seen_at=observed_at, last_seen_at=observed_at,
                    detection_count=0, max_confidence=0, mean_confidence=0,
                    x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                self.db.add(track)
                self.db.flush()
                self.active[track.id] = track
            self.chunk_ids[obs.track_id] = track.id
            track.mean_confidence = (track.mean_confidence * track.detection_count + obs.confidence) / (track.detection_count + 1)
            track.detection_count += 1
            best = obs.confidence >= track.max_confidence
            track.max_confidence = max(track.max_confidence, obs.confidence)
            track.last_seen_at, track.species = observed_at, obs.class_name or track.species
            track.x1, track.y1, track.x2, track.y2 = box
            self.db.add(LiveFishDetection(track_id=track.id, observed_at=observed_at,
                frame_number=frame_number, confidence=obs.confidence, species=obs.class_name,
                x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
            if self.settings.fishial_enabled and self.session.species_id_enabled:
                try:
                    raw_box = (obs.bbox_left, obs.bbox_top, obs.bbox_right, obs.bbox_bottom)
                    self._stage_species_crop(frame, track, raw_box, obs.confidence,
                                             frame_number, observed_at)
                except OSError:
                    logger.warning("Species crop could not be written for track %s", track.id)
            annotation = FrameAnnotation(frame_number, str(track.id)[:8], track.species, obs.confidence, *box)
            _draw_annotation(self.cv2, annotated, annotation, True)
            try:
                self._save_crop(frame, track, annotation, best)
            except OSError:
                # Losing one crop must not end the session; the detections still stand.
                logger.warning("Live crop could not be written for track %s", track.id, exc_info=True)
                track.media_error = "Some annotated crops could not be written; detections are unaffected"
        self.session.frames_processed += 1
        self.session.last_frame_at = observed_at
        path = live_path(self.settings, str(self.session.id), "annotated.jpg")
        try:
            self._write_jpeg(path, annotated)
        except OSError:
            # The annotated view is a live convenience; the next frame refreshes it.
            logger.warning("Live annotated frame could not be published", exc_info=True)
        else:
            self.session.snapshot_path = str(path)

    def _is_clear_frame(self, frame, box, confidence) -> bool:
        """Safety floor only: reject frames that are genuinely unusable.

        This is NOT selection. The previous 0.70 confidence / 96 px pair sat above the
        90th percentile of both measurements on both reference cameras and rejected
        96.8% (Coral City) and 85.9% (SmartBay 3) of detections conjunctively, which
        is why no track ever reached ``fishial_min_frames_to_vote``. Selection is the
        per-track ranking in ``_stage_species_crop``; this only excludes junk.
        """

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        s = self.settings
        if not all(math.isfinite(v) for v in (*box, confidence)):
            return False
        if (not s.fishial_min_frame_confidence <= confidence <= 1
                or min(x2 - x1, y2 - y1) < s.fishial_min_crop_pixels
                # A fish crossing the frame edge is truncated, so the crop cannot show
                # the whole animal. That is a correctness floor, not a quality one.
                or min(x1, y1, width - 1 - x2, height - 1 - y2) < s.fishial_edge_margin_pixels):
            return False
        if s.fishial_blur_min_variance:
            crop = frame[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
            gray = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2GRAY)
            if self.cv2.Laplacian(gray, self.cv2.CV_64F).var() < s.fishial_blur_min_variance:
                return False
        return True

    def _frame_quality(self, frame, box, confidence) -> dict:
        """Per-frame measurements the ranker and the selector both need.

        Computed once while the frame is in hand, then persisted into the ``staged``
        audit entry, so nothing has to re-decode pixels later. Cheap, but still kept
        off the path that runs for non-candidate detections.
        """

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        crop = frame[max(0, math.floor(y1)):min(height, math.ceil(y2)),
                     max(0, math.floor(x1)):min(width, math.ceil(x2))]
        measures = {"confidence": float(confidence),
                    "short_side": float(min(x2 - x1, y2 - y1)),
                    "sharpness": 0.0, "luminance": 0.0, "contrast": 0.0, "colorfulness": 0.0}
        if crop.size == 0:
            return measures
        gray = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2GRAY)
        measures["sharpness"] = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        lightness = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2LAB)[:, :, 0].astype("float64")
        measures["luminance"] = float(lightness.mean())
        measures["contrast"] = float(lightness.std())  # RMS contrast of L
        blue, green, red = (channel.astype("float64") for channel in self.cv2.split(crop))
        # Hasler-Susstrunk colourfulness: dim green water scores near zero, which is
        # exactly the domain mismatch Fishial's classifier appears to dislike.
        rg, yb = red - green, 0.5 * (red + green) - blue
        measures["colorfulness"] = float(
            math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))
        return measures

    def _preprocess_crop(self, crop):
        """Optional underwater correction, applied ONLY to the staged Fishial crop.

        Never touches the annotated view or the stored ``crop.jpg``. Ships off: see
        the tuning notes in README - enable only if ``scripts/fishial_replay.py``
        shows it produces more non-empty species lists on stored crops.
        """

        return preprocess_crop(crop, self.settings, self.cv2)

    def _drop_candidate(self, track_id, reason):
        """Free an unpaid candidate: its staged crops, its slot and its bytes."""

        entry = self.pool.pop(track_id, None)
        track = self.db.get(LiveFishTrack, track_id)
        directory = scratch_path(self.settings, str(self.session.id), str(track_id), "fishial")
        self.staged_bytes -= sum(path.stat().st_size for path in directory.glob("*.jpg"))
        shutil.rmtree(directory, ignore_errors=True)
        shutil.rmtree(live_path(self.settings, str(self.session.id), str(track_id), "fishial"),
                      ignore_errors=True)
        if track is not None:
            audit = track.fishial_votes
            audit["staged"], audit["reason"] = [], reason
            track.fishial_state = "disabled"
            track.fishial_frames_used = 0
            track.fishial_quality_score = entry["score"] if entry else None
            track.fishial_votes_json = json.dumps(audit)

    def _weakest_candidate(self, exclude=()):
        candidates = [(entry["score"], key) for key, entry in self.pool.items() if key not in exclude]
        return min(candidates)[1] if candidates else None

    def _admit_candidate(self, track, score) -> bool:
        """Admit an unpaid candidate, evicting the weakest one if the pool is full.

        Enrollment no longer commits a paid slot: on SmartBay 3 first-come enrollment
        drew five tracks by arrival order out of 74, and arrival time is uncorrelated
        with identifiability. Admission here is free; the budget is spent at selection.
        """

        if len(self.pool) >= self.session.species_id_candidate_pool_size:
            weakest = self._weakest_candidate()
            if weakest is None or score <= self.pool[weakest]["score"]:
                return False
            self._drop_candidate(weakest, "evicted from candidate pool")
        track.fishial_state = "candidate"
        self.pool[track.id] = {"score": score, "staged": []}
        return True

    def _enforce_byte_cap(self, incoming, track_id) -> bool:
        """Keep the session's staged crops under ``fishial_max_staged_bytes``.

        Generous staging is free in API terms but not in disk terms. Evict the
        weakest candidates first; report whether this track survived.
        """

        cap = self.settings.fishial_max_staged_bytes
        if not cap:
            return True
        while self.staged_bytes + incoming > cap:
            weakest = self._weakest_candidate()
            if weakest is None:
                return False
            self._drop_candidate(weakest, "evicted to stay within the staged byte cap")
            if weakest == track_id:
                return False
        return True

    def _stage_species_crop(self, frame, track, box, confidence, frame_number, observed_at):
        # ready/submitted/terminal tracks have been selected or already paid for and
        # must never be restaged or evicted.
        if track.fishial_state not in {"disabled", "candidate"}:
            return
        if not self._is_clear_frame(frame, box, confidence):
            return
        quality = self._frame_quality(frame, box, confidence)
        score = frame_score(quality, self.settings.fishial_quality_weights)
        entry = self.pool.get(track.id)
        if entry is None and not self._admit_candidate(track, score):
            return
        entry = self.pool[track.id]
        staged = entry["staged"]
        timestamp = observed_at.timestamp()
        audit = track.fishial_votes
        # The persisted origin, not spool order, anchors the separation windows, so
        # they survive a worker restart. Fall back to the earliest retained frame for
        # a track staged before the origin was recorded.
        origin = audit.get("window_origin")
        if origin is None:
            origin = min((entry["timestamp"] for entry in staged), default=timestamp)
        separation = self.settings.fishial_min_frame_separation_seconds
        # Frame separation is a DIVERSITY rule, not a first-come rule. Near-duplicate
        # frames from one moment show the same pose, lighting and occlusion, so they
        # would cast near-identical votes and inflate a consensus that rests on a
        # single observation. Keeping one frame per window makes the votes independent
        # and guarantees the retained set spans as many windows as it holds frames.
        window = int((timestamp - origin) // separation) if separation > 0 else frame_number
        occupant = next((item for item in staged if item.get("window") == window), None)
        if occupant is not None:
            # A better frame may displace a retained one inside the same window.
            if score <= occupant["score"]:
                return
            victim = occupant
        elif len(staged) < self.max_staged:
            victim = None
        else:
            victim = min(staged, key=lambda item: item["score"])
            if score <= victim["score"]:
                return
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]
        dx = (x2 - x1) * (self.settings.fishial_crop_margin - 1) / 2
        dy = (y2 - y1) * (self.settings.fishial_crop_margin - 1) / 2
        left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
        right, bottom = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
        # Rectangular, original-resolution crop; never use the annotated/letterbox spool.
        crop = frame[top:bottom, left:right]
        if crop.shape[:2] == frame.shape[:2]:
            # Even an extreme admin crop margin must never turn this into a full-frame upload.
            return
        # Store the expected box as it actually is, clamping included: at a frame edge
        # the fish is no longer centred, and recomputing a centred box later would
        # match the wrong object.
        expected = [(x1 - left) / (right - left), (y1 - top) / (bottom - top),
                    (x2 - left) / (right - left), (y2 - top) / (bottom - top)]
        ok, encoded = self.cv2.imencode(".jpg", self._preprocess_crop(crop),
                                        [self.cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise OSError("Species crop encoding failed")
        payload = encoded.tobytes()
        if not self._enforce_byte_cap(len(payload), track.id):
            return
        directory = scratch_path(self.settings, str(self.session.id), str(track.id), "fishial")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{frame_number:012d}.jpg"
        temp = path.with_suffix(".tmp")
        temp.write_bytes(payload)
        os.replace(temp, path)
        self.staged_bytes += len(payload)
        if self.settings.fishial_keep_staged_crops:
            self._retain_staged_crop(track, frame_number, payload)
        if victim is not None:
            self._discard_staged(track, staged, victim)
        staged.append({"frame_number": frame_number, "timestamp": timestamp, "window": window,
                       "score": score, "expected_box": expected, "quality": quality,
                       # Recorded so the offline replay harness can resolve the same
                       # multi-object match without re-reading the JPEG.
                       "crop_size": [right - left, bottom - top]})
        staged.sort(key=lambda item: item["frame_number"])
        audit["staged"], audit["window_origin"] = staged, origin
        entry["score"] = pool_score(staged, self.settings.fishial_quality_weights)
        track.fishial_quality_score = entry["score"]
        track.fishial_votes_json = json.dumps(audit)
        # "Frames currently retained", not "frames ever seen".
        track.fishial_frames_used = len(staged)

    def _discard_staged(self, track, staged, victim):
        directory = scratch_path(self.settings, str(self.session.id), str(track.id), "fishial")
        for root in (directory, live_path(self.settings, str(self.session.id), str(track.id), "fishial")):
            path = root / f"{victim['frame_number']:012d}.jpg"
            if root == directory and path.is_file():
                self.staged_bytes -= path.stat().st_size
            path.unlink(missing_ok=True)
        staged.remove(victim)

    def _retain_staged_crop(self, track, frame_number, payload):
        """Copy a staged crop to the output volume for offline replay.

        Off by default: this retains fish imagery on disk past the scratch wipe.
        """

        directory = live_path(self.settings, str(self.session.id), str(track.id), "fishial")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{frame_number:012d}.jpg").write_bytes(payload)

    def _write_jpeg(self, path, frame):
        temp = path.with_name(path.stem + ".tmp.jpg")
        if not self.cv2.imwrite(str(temp), frame):
            raise RuntimeError("Live annotated image could not be written")
        for attempt in range(WRITE_ATTEMPTS):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                # Windows-backed bind mounts refuse the rename while a reader still
                # holds the target open, which the dashboard does on every poll.
                if attempt == WRITE_ATTEMPTS - 1:
                    temp.unlink(missing_ok=True)
                    raise
                time.sleep(WRITE_RETRY_SECONDS * (attempt + 1))

    def _save_crop(self, frame, track, annotation, best):
        height, width = frame.shape[:2]
        side = min(max(width, height), max(self.settings.clip_min_crop_pixels,
                   round(max(track.x2 - track.x1, track.y2 - track.y1) * self.settings.clip_zoom_margin)))
        left = max(0, min(width - min(side, width), round((track.x1 + track.x2 - side) / 2)))
        top = max(0, min(height - min(side, height), round((track.y1 + track.y2 - side) / 2)))
        crop = frame[top:min(height, top + side), left:min(width, left + side)].copy()
        local = FrameAnnotation(annotation.frame_number, annotation.track_label, annotation.species_label,
            annotation.confidence, track.x1 - left, track.y1 - top, track.x2 - left, track.y2 - top)
        _draw_annotation(self.cv2, crop, local, True)
        size = self.settings.clip_output_short_side // 2 * 2
        # Letterbox to a fixed square so varying fish size never distorts the clip.
        import numpy as np
        output = np.zeros((size, size, 3), dtype="uint8")
        ratio = min(size / crop.shape[1], size / crop.shape[0])
        scaled = self.cv2.resize(crop, (max(1, round(crop.shape[1] * ratio)), max(1, round(crop.shape[0] * ratio))))
        y, x = (size - scaled.shape[0]) // 2, (size - scaled.shape[1]) // 2
        output[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
        directory = live_path(self.settings, str(self.session.id), str(track.id))
        directory.mkdir(parents=True, exist_ok=True)
        spool = scratch_path(self.settings, str(self.session.id), str(track.id), "frames")
        spool.mkdir(parents=True, exist_ok=True)
        self._write_jpeg(spool / f"{annotation.frame_number:012d}.jpg", output)
        if best:
            path = directory / "crop.jpg"
            self._write_jpeg(path, output)
            track.crop_path = str(path)

    def expire(self, now: datetime, reason: str | None = None):
        for track in list(self.active.values()):
            if reason or (aware(now) - aware(track.last_seen_at)).total_seconds() >= self.settings.live_lost_track_seconds:
                track.status, track.finalized_at = "finalized", now
                track.finalization_reason = reason or "lost"
                try:
                    self._render_clip(track)
                except Exception:
                    track.media_error = "Clip rendering failed; the annotated crop and stored detections remain available"
                finally:
                    # Scratch frames outlive nothing: drop them even when the clip failed.
                    root = scratch_path(self.settings, str(self.session.id), str(track.id))
                    shutil.rmtree(root / "frames", ignore_errors=True)
                    # Clean identification crops must survive until the between-segment pass.
                    if not (root / "fishial").exists():
                        shutil.rmtree(root, ignore_errors=True)
                del self.active[track.id]

    def _render_clip(self, track):
        directory = live_path(self.settings, str(self.session.id), str(track.id))
        spool = scratch_path(self.settings, str(self.session.id), str(track.id), "frames")
        frames = sorted(spool.glob("*.jpg"))
        if not frames:
            raise RuntimeError("No crop frames available")
        temp, path = directory / "clip.tmp.mp4", directory / "clip.mp4"
        size = self.settings.clip_output_short_side // 2 * 2
        writer = video_media.open_video_writer(self.cv2, temp, self.settings.live_fps, size, size)
        if writer is None:
            raise RuntimeError("No MP4 encoder available")
        try:
            for index, source in enumerate(frames):
                frame = self.cv2.imread(str(source))
                if frame is None:
                    raise RuntimeError("Stored crop is unreadable")
                count = (int(frames[index + 1].stem) - int(source.stem)) if index + 1 < len(frames) else 1
                # Hold the last crop over brief missing observations, up to the loss threshold.
                for _ in range(max(1, min(count, int(self.settings.live_lost_track_seconds * self.settings.live_fps)))):
                    writer.write(frame)
        finally:
            writer.release()
        video_media.make_browser_compatible(temp, 30)
        os.replace(temp, path)
        track.clip_path, track.media_error = str(path), None
