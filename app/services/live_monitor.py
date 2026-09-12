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
from app.services.species_quality import (
    crop_for_fishial,
    crop_measures,
    frame_score,
    is_clear_frame,
    pool_score,
    preprocess_crop,
)
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


def box_iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def center_distance(a, b) -> float:
    return math.hypot((a[0] + a[2] - b[0] - b[2]) / 2,
                      (a[1] + a[3] - b[1] - b[3]) / 2)


class LiveTracker:
    def __init__(self, db: Session, session: LiveMonitorSession, settings: Settings):
        self.db, self.session, self.settings = db, session, settings
        self.cv2 = video_media.load_cv2()
        self.root = live_path(settings, str(session.id))
        self.root.mkdir(parents=True, exist_ok=True)
        self.active = {t.id: t for t in db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == session.id, LiveFishTrack.status == "active"))}
        self.chunk_ids: dict[str, uuid.UUID] = {}
        # One previous observation plus the current DB box gives velocity. Keep it
        # across segments, but never extrapolate old motion after a long absence.
        # A worker reload safely falls back to matching without motion history.
        self.previous: dict[uuid.UUID, tuple[datetime, tuple]] = {}
        self.pool: dict[uuid.UUID, dict] = {}
        self.staged_bytes = 0
        # Creating a directory that already exists still costs a round trip to the
        # output volume, and on a bind mount that is milliseconds per detection.
        self._created: set[Path] = set()
        # Capture time, not wall time, so the published rate is the same whether a
        # segment is rendered live or replayed from storage.
        self._snapshot_at: float | None = None
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

    def _association_cost(self, track, obs, box, observed_at):
        """Return a gated motion/geometry cost; a VIAME ID cannot bypass a gate.

        Distances are in fish-box lengths, times are capture seconds. The short
        recovery horizon is deliberately independent of media finalization: ten
        seconds of retained crops is not ten seconds of reliable identity evidence.
        """
        elapsed = (observed_at - aware(track.last_seen_at)).total_seconds()
        horizon = min(2.0, self.settings.live_lost_track_seconds)
        if elapsed <= 0 or elapsed > horizon:
            return None
        if track.species and obs.class_name and track.species.casefold() != obs.class_name.casefold():
            return None
        last = (track.x1, track.y1, track.x2, track.y2)
        old_w, old_h = last[2] - last[0], last[3] - last[1]
        new_w, new_h = box[2] - box[0], box[3] - box[1]
        if min(old_w, old_h, new_w, new_h) <= 0:
            return None
        area_ratio = new_w * new_h / (old_w * old_h)
        aspect_ratio = (new_w / new_h) / (old_w / old_h)
        if not .5 <= area_ratio <= 2 or not .5 <= aspect_ratio <= 2:
            return None
        scale = max(1, min(max(old_w, old_h), max(new_w, new_h)))
        # Bounded displacement even for a matching local ID; allow startup motion
        # of up to three box lengths/second plus modest detector jitter.
        if center_distance(last, box) / scale > .35 + 3 * elapsed:
            return None
        missed = (elapsed > 1.5 / self.settings.live_fps or
                  (self.session.last_frame_at is not None and
                   aware(self.session.last_frame_at) > aware(track.last_seen_at)))
        prediction = last
        previous = self.previous.get(track.id)
        has_motion = False
        if previous is not None:
            timestamp, prior = previous
            period = (aware(track.last_seen_at) - timestamp).total_seconds()
            if 0 < period <= horizon:
                dx = (last[0] + last[2] - prior[0] - prior[2]) / 2 * elapsed / period
                dy = (last[1] + last[3] - prior[1] - prior[3]) / 2 * elapsed / period
                prediction = (last[0] + dx, last[1] + dy, last[2] + dx, last[3] + dy)
                has_motion = True
        distance = center_distance(prediction, box) / scale
        iou = box_iou(prediction, box)
        if has_motion:
            # Missed observations reduce certainty; do not widen the search into
            # neighbouring fish or fall back to the abandoned last box.
            if distance > (.4 if missed else .75):
                return None
        elif missed:
            if distance > .25 or iou < .5:
                return None
        elif distance > min(1.5, .35 + 3 * elapsed):
            return None
        cost = distance + .25 * (1 - iou) + .1 * (abs(math.log(area_ratio)) + abs(math.log(aspect_ratio)))
        cost += .15 if missed else 0
        # A small hint, less than the ambiguity margin, never absolute identity.
        return cost - (.03 if self.chunk_ids.get(obs.track_id) == track.id else 0)

    def _associate(self, candidates, observed_at):
        by_observation, by_track = {}, {}
        for index, (obs, box) in enumerate(candidates):
            for track in self.active.values():
                cost = self._association_cost(track, obs, box, observed_at)
                if cost is not None:
                    by_observation.setdefault(index, []).append((cost, track.id))
                    by_track.setdefault(track.id, []).append((cost, index))
        for choices in (*by_observation.values(), *by_track.values()):
            choices.sort(key=lambda item: item[0])

        def unambiguous(choices):
            return len(choices) == 1 or choices[1][0] - choices[0][0] >= .15

        assigned = {}
        for index, choices in by_observation.items():
            track_id = choices[0][1]
            reverse = by_track[track_id]
            # Mutual best matches only. Never let greedy removal turn an ambiguous
            # runner-up into a confident identity, in either direction.
            if unambiguous(choices) and unambiguous(reverse) and reverse[0][1] == index:
                assigned[index] = self.active[track_id]
        return assigned

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
        assigned = self._associate(candidates, observed_at)
        # Drop aliases from missed frames and former VIAME IDs. They are only
        # hints for the next observation, never a route back into an old track.
        self.chunk_ids.clear()
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
            if track.detection_count:
                self.previous[track.id] = (aware(track.last_seen_at),
                    (track.x1, track.y1, track.x2, track.y2))
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
        try:
            published = self._publish_snapshot(annotated, observed_at)
        except OSError:
            # The annotated view is a live convenience; the next frame refreshes it.
            logger.warning("Live annotated frame could not be published", exc_info=True)
        else:
            if published:
                self.session.snapshot_path = str(self.root / "annotated.jpg")

    def _is_clear_frame(self, frame, box, confidence) -> bool:
        return is_clear_frame(self.cv2, frame, box, confidence, self.settings)

    def _frame_quality(self, frame, box, confidence) -> dict:
        return crop_measures(self.cv2, frame, box, confidence)

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
        cropped = crop_for_fishial(self.cv2, frame, box, self.settings)
        if cropped is None:
            return
        payload, expected, crop_size = cropped
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
                       "crop_size": crop_size})
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

    def _ensure_directory(self, path: Path) -> Path:
        if path not in self._created:
            path.mkdir(parents=True, exist_ok=True)
            self._created.add(path)
        return path

    def _publish_snapshot(self, frame, observed_at: datetime) -> bool:
        """Publish the annotated view, at most ``live_snapshot_fps`` times a second.

        The dashboard polls this file every two seconds and the detections are
        already recorded, so a skipped frame costs nothing but a little smoothness.
        Downscaling only the published copy keeps the operator's view the size it
        was before capture moved to 1080p, while the detector still sees full detail.
        """

        moment = observed_at.timestamp()
        if self._snapshot_at is not None and moment - self._snapshot_at < 1 / self.settings.live_snapshot_fps:
            return False
        width = frame.shape[1]
        limit = self.settings.live_snapshot_max_width
        if width > limit:
            height = max(1, round(frame.shape[0] * limit / width))
            frame = self.cv2.resize(frame, (limit, height), interpolation=self.cv2.INTER_AREA)
        self._write_jpeg(self.root / "annotated.jpg", frame)
        self._snapshot_at = moment
        return True

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
        spool = scratch_path(self.settings, str(self.session.id), str(track.id), "frames")
        self._ensure_directory(spool)
        self._write_jpeg(spool / f"{annotation.frame_number:012d}.jpg", output)
        if best:
            path = self._ensure_directory(self.root / str(track.id)) / "crop.jpg"
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
                    # The cache records what exists, so forget what was just removed.
                    self._created.discard(root / "frames")
                    # Clean identification crops must survive until the between-segment pass.
                    if not (root / "fishial").exists():
                        shutil.rmtree(root, ignore_errors=True)
                del self.active[track.id]
                self.previous.pop(track.id, None)
        self.chunk_ids = {key: value for key, value in self.chunk_ids.items() if value in self.active}

    def _render_clip(self, track):
        directory = self._ensure_directory(self.root / str(track.id))
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
