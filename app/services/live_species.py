"""Staged live crops -> selection -> durable call reservations -> conservative consensus.

Only run between decoded segments or after finalization, never on the frame hot path.

Accuracy strategy in this module:

* **Pay for the best fish, not the first fish.** Selection is deferred to
  finalization, ordered by measured crop quality, floored, and paced so a fish that
  arrives late in the session can still win a slot.
* **Stop early in both directions.** Decided, unreachable and repeatedly-declined all
  terminate a track without further spend.
* **Reject the impossible for free.** Regional filtering removes whole classes of
  wrong label before voting, at no API cost.
* **The reservation journal is load-bearing.** Selection, the stop rules and the
  regional filter all run *around* it, never through it. A crash must still be unable
  to buy a call.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from functools import partial

from sqlalchemy import select, update

from app.db.models import LiveFishTrack, LiveMonitorSession, utc_now
from app.services.fish_enhancement import (
    EnhancementError,
    enhance_crop,
    failure_metadata,
    validate_enhancement,
)
from app.services.fishial import FishialClient, FishialError
from app.services.live_monitor import scratch_path
from app.services.species_quality import (
    stop_reason,
    submitted_count,
    track_quality,
    verdict,
)
from app.services.species_region import filter_species

logger = logging.getLogger(__name__)
# "candidate" is staged but unpaid; the rest have been selected or already spent on.
IN_PROGRESS = ("candidate", "pending", "ready", "submitted")
SELECTED = ("pending", "ready", "submitted")


class SpeciesIdentifier:
    def __init__(self, db, session, settings, client=None, enhancer=None):
        self.db, self.session, self.settings = db, session, settings
        self.client = client
        self._owns_client = client is None
        self.enhancer = enhancer or enhance_crop

    def close(self):
        if self._owns_client and self.client is not None:
            self.client.close()

    def should_run(self, session=None):
        session = session or self.session
        return bool(self.settings.fishial_enabled and session.species_id_enabled
                    and session.species_id_fish_target > 0 and self.db.scalar(
                        select(LiveFishTrack.id).where(
                            LiveFishTrack.session_id == session.id,
                            LiveFishTrack.fishial_state.in_(IN_PROGRESS)).limit(1)))

    def run_safely(self, final=False):
        try:
            if self.should_run():
                self.run(final)
        except Exception:  # noqa: BLE001 - isolate provider/DB failure without leaking secrets
            # Even injected client exceptions must not leak images or credentials.
            # A submitted journal is resumed on the next pass without replaying calls.
            self.db.rollback()
            logger.warning("Live species identification pass failed; reserved calls remain spent")

    def finish(self):
        """Resolve outstanding work before scratch cleanup, even if the pass fails.

        This is the last chance to spend, so the late reserve is released here: the
        session cannot know its best fish until it ends.
        """

        self.run_safely(final=True)
        for track in self.db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == self.session.id,
            LiveFishTrack.fishial_state.in_(IN_PROGRESS))):
            audit = track.fishial_votes
            if track.fishial_state == "candidate":
                # Never selected: staged for free, but the budget ran out before its
                # rank came up. That is a different operational problem from a track
                # whose frames failed, so it must not collapse into "error".
                audit["reason"] = "budget exhausted"
                track.fishial_state = "review_required"
            else:
                audit["reason"] = "identification interrupted at shutdown"
                track.fishial_state = "error"
            audit.setdefault("tally", {})
            track.fishial_completed_at = utc_now()
            track.fishial_votes_json = json.dumps(audit)
            shutil.rmtree(scratch_path(self.settings, str(self.session.id), str(track.id), "fishial"),
                          ignore_errors=True)
        self.db.commit()

    def run(self, final=False):
        if not self.should_run():
            return
        # Retry allowance is session-wide and survives reconstruction of this object.
        self.retries_used = sum(
            bool(attempt.get("retry"))
            for track in self.db.scalars(select(LiveFishTrack).where(
                LiveFishTrack.session_id == self.session.id))
            for frame in track.fishial_votes.get("frames", [])
            for attempt in frame.get("attempts", [])
        )
        self._select(final)
        tracks = self.db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == self.session.id,
            LiveFishTrack.fishial_state.in_(SELECTED)).order_by(
                LiveFishTrack.fishial_quality_score.desc().nullslast(),
                LiveFishTrack.first_seen_at, LiveFishTrack.id)).all()
        for track in tracks:
            if track.fishial_state == "pending" and track.status != "finalized":
                continue
            self._identify_track(track)

    def _slots(self, final) -> int:
        """How many further fish this pass may select.

        Budget pacing: hold back ``fishial_late_reserve_fish`` slots until the session
        has had time to show us what it has, or until ``finish()`` releases them.
        Without this the whole budget goes to whatever swims past first.
        """

        session = self.session
        target = session.species_id_fish_target
        reserve = max(0, min(self.settings.fishial_late_reserve_fish, target - 1))
        released = final or reserve == 0
        if not released and session.started_at is not None:
            from app.services.live_monitor import aware
            elapsed = (utc_now() - aware(session.started_at)).total_seconds()
            released = elapsed >= self.settings.fishial_reserve_after_seconds
        cap = target if released else target - reserve
        return max(0, cap - session.species_id_fish_enrolled)

    def _select(self, final):
        """Choose which staged candidates are worth paying for. Spends nothing."""

        floor = self.settings.fishial_quality_floor
        frames_per_fish = self.session.species_id_frames_per_fish
        eligible = []
        for track in self.db.scalars(select(LiveFishTrack).where(
                LiveFishTrack.session_id == self.session.id,
                LiveFishTrack.fishial_state == "candidate")):
            # An unfinalized track may still stage a better frame, so never pay for
            # it mid-life; it stays a free candidate until it is lost.
            if track.status != "finalized":
                continue
            audit = track.fishial_votes
            staged = audit.get("staged") or []
            directory = scratch_path(self.settings, str(self.session.id), str(track.id), "fishial")
            if len(staged) < self.settings.fishial_min_frames_to_vote:
                self._complete(track, audit, "review_required", "insufficient clear frames", directory)
                continue
            score = track_quality(track, self.settings, frames_per_fish)
            track.fishial_quality_score = score
            if score < floor:
                # Zero API calls: a crop this poor is not worth an answer we could not
                # trust. Its staged frames are freed immediately.
                self._complete(track, audit, "review_required", "below quality floor", directory)
                continue
            eligible.append((score, track))
        eligible.sort(key=lambda item: (-item[0], item[1].first_seen_at, item[1].id))
        for _, track in eligible[:self._slots(final)]:
            track.fishial_state = "ready"
            # "Fish actually selected for identification", not "fish first sighted".
            self.session.species_id_fish_enrolled += 1
        self.db.commit()

    def _reserve(self, track, audit, record, retry):
        s = self.session
        budget = s.species_id_fish_target * s.species_id_frames_per_fish + self.settings.fishial_max_api_retries
        if retry and self.retries_used >= self.settings.fishial_max_api_retries:
            raise FishialError("retry limit reached")
        # Reserve before the request, not afterwards: a crash after sending must
        # never buy another call. A crash before sending can conservatively waste one.
        result = self.db.execute(update(LiveMonitorSession).where(
            LiveMonitorSession.id == s.id,
            LiveMonitorSession.species_id_api_calls < budget,
        ).values(species_id_api_calls=LiveMonitorSession.species_id_api_calls + 1)
            .execution_options(synchronize_session=False))
        if not result.rowcount:
            raise FishialError("budget exhausted")
        record.setdefault("attempts", []).append({"retry": retry})
        track.fishial_votes_json = json.dumps(audit)
        self.db.commit()  # Reservation and frame journal are atomic.
        self.db.refresh(s)
        if retry:
            self.retries_used += 1

    def _stop_reason(self, frames, remaining) -> str | None:
        """Whether to stop before buying another frame for this track."""

        return stop_reason(frames, remaining, self.settings)

    def _vote(self, record, prediction):
        """Turn one response into a vote, or record why it abstained."""

        record["object_count"] = prediction.object_count
        record["object_index"] = prediction.object_index
        record["object_iou"] = prediction.object_iou
        record["succeeded"] = True
        # The classifier saw our fish and declined to name it. Distinct from a
        # response we could not attribute, and from one we filtered ourselves.
        record["empty"] = prediction.object_index is not None and not prediction.species
        record["reason"] = "no unambiguous fish prediction"
        region = self.session.species_id_region if self.settings.fishial_region_filter_enabled else None
        ranked, dropped = filter_species(prediction.species, region)
        if dropped:
            record["dropped_species"] = dropped
            record["dropped_reason"] = "implausible_for_region"
        if not ranked:
            if dropped:
                # Filtering emptied the list: this frame abstains rather than falling
                # through to a name we already judged impossible here.
                record["reason"] = "implausible_for_region"
            return
        name, score = ranked[0]
        # A single candidate has no runner-up; use zero as second score.
        margin = score - (ranked[1][1] if len(ranked) > 1 else 0)
        record.update(species=name, score=score, margin=margin,
                      voted=margin >= self.settings.fishial_min_frame_margin)
        record["reason"] = None if record["voted"] else "ambiguous frame"

    def _identify_track(self, track):
        audit = track.fishial_votes
        frames_per_fish = self.session.species_id_frames_per_fish
        # Spend on the best staged frames first: a decisive answer then arrives in
        # fewer calls, and the weakest crops are the ones the stop rules skip.
        staged = sorted(audit.get("staged") or [],
                        key=lambda entry: (-(entry.get("score") or 0.0),
                                           entry.get("frame_number", 0)))[:frames_per_fish]
        directory = scratch_path(self.settings, str(self.session.id), str(track.id), "fishial")
        if len(staged) < self.settings.fishial_min_frames_to_vote:
            self._complete(track, audit, "review_required", "insufficient clear frames", directory)
            return
        audit.setdefault("frames", [])
        track.fishial_state = "submitted"
        track.fishial_votes_json = json.dumps(audit)
        self.db.commit()
        attempted = {record["frame_number"] for record in audit["frames"]}
        stopped = None
        for position, staged_frame in enumerate(staged):
            number = staged_frame["frame_number"]
            if number in attempted:
                # A journal record without a result is an uncertain send. Abstain;
                # never replay it after a restart.
                continue
            record = {"frame_number": number, "species": None, "score": None,
                      "margin": None, "voted": False, "reason": "interrupted", "attempts": []}
            audit["frames"].append(record)
            try:
                # Only server-written manifest entries resolve paths. No API body or
                # client-supplied path/URL is accepted here.
                image = (directory / f"{int(number):012d}.jpg").read_bytes()
                try:
                    enhanced = self.enhancer(image, self.settings)
                    validate_enhancement(enhanced, image, self.settings)
                except Exception:  # noqa: BLE001 - injected enhancers must also redact errors
                    raise EnhancementError() from None
                record["preprocessing"] = enhanced.metadata
                image = enhanced.image
                track.fishial_votes_json = json.dumps(audit)
                self.db.commit()  # Pre-send provenance precedes even authentication/reservation.
                if self.client is None:
                    self.client = FishialClient(self.settings)
                expected = staged_frame.get("expected_box")
                if isinstance(self.client, FishialClient):
                    self.client.before_image_call = partial(self._reserve, track, audit, record)
                    prediction = self.client.identify(image, expected)
                else:
                    # Test/source adapters make one image call per identify().
                    self._reserve(track, audit, record, False)
                    prediction = self.client.identify(image, expected)
                record["raw"] = prediction.raw
                self._vote(record, prediction)
            except EnhancementError:
                record["reason"] = "preprocessing failed"
                record["preprocessing"] = failure_metadata(image, self.settings)
            except FishialError as exc:
                record["reason"] = exc.reason
            except Exception:  # noqa: BLE001 - one bad crop/provider result must only abstain
                record["reason"] = "frame unavailable or request failed"
            finally:
                if isinstance(self.client, FishialClient):
                    self.client.before_image_call = None
            track.fishial_votes_json = json.dumps(audit)
            self.db.commit()
            remaining = len(staged) - position - 1
            stopped = self._stop_reason(audit["frames"], remaining) if remaining else None
            if stopped:
                audit["stopped_early"] = stopped
                audit["calls_saved"] = remaining
                self.db.execute(update(LiveMonitorSession).where(
                    LiveMonitorSession.id == self.session.id).values(
                        species_id_calls_saved=LiveMonitorSession.species_id_calls_saved + remaining)
                    .execution_options(synchronize_session=False))
                self.db.commit()
                self.db.refresh(self.session)
                break
        frames = audit["frames"]
        audit["tally"] = dict(Counter(f["species"] for f in frames if f.get("voted")))
        # Each attempted frame contributes to the denominator once, including
        # failed/ambiguous frames. Retries do not manufacture extra votes.
        audit["submitted_frames"] = submitted_count(frames)
        winner, mean = verdict(frames, self.settings)
        if winner is not None:
            track.fishial_species, track.fishial_species_confidence = winner, mean
            self._complete(track, audit, "identified", None, directory)
            return
        exhausted = any(f.get("reason") == "budget exhausted" for f in frames)
        state = "review_required" if exhausted or any(
            f.get("succeeded") or f.get("reason") == "preprocessing failed" for f in frames) else "error"
        # These abstentions are different operational problems and must never
        # collapse into one message: "insufficient clear frames", "below quality
        # floor", "consensus unreachable", "classifier returned no candidates",
        # "consensus not reached", "budget exhausted" and "all frames failed" each
        # point at a different fix.
        reason = ("budget exhausted" if exhausted else
                  "all frames failed" if state == "error" else
                  stopped if stopped and stopped != "decided" else
                  "consensus not reached")
        self._complete(track, audit, state, reason, directory)

    def _complete(self, track, audit, state, reason, directory):
        audit.setdefault("tally", {})
        audit.setdefault("frames", [])
        audit["reason"] = reason
        track.fishial_state, track.fishial_completed_at = state, utc_now()
        if state != "identified":
            track.fishial_species = track.fishial_species_confidence = None
        track.fishial_votes_json = json.dumps(audit)
        self.db.commit()
        shutil.rmtree(directory, ignore_errors=True)
