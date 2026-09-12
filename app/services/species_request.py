"""Operator-requested species identification for one chosen fish.

The automatic live pass buys identifications for whichever fish a session happens
to rank highest, inside a budget fixed before the session started. This is the
other half of the same capability, and the one an operator reaches for: *this*
fish, *this* many frames, now - on a live track or on a library recording.

What the operator chooses and what the system chooses are deliberately split:

* the operator picks the **fish** and **how many frames** to pay for;
* the system picks **which frames those are**, by the same measured crop quality
  the live pass ranks with (:mod:`app.services.species_quality`), so a requested
  identification is never weaker than an automatic one on the same animal.

Frames come from whichever footage actually exists for that fish:

* a library video (including a recorded live session) is read straight from its
  stored file, where a detection's frame number indexes the file;
* a *running* session is read from the chunks ``RecordingWriter`` has retained so
  far, which hold exactly the frames the tracker analyzed, in order.

Both are read as rectangular, source-resolution crops. The annotated view and the
letterboxed gallery clip are never sent: a drawn-on box is not evidence.

Spending is journalled before each request, exactly as the automatic pass does, so
a crash can lose an answer but can never hide a call it already bought.
"""

from __future__ import annotations

import bisect
import json
import logging
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import (
    FishDetection,
    FishTrack,
    LiveFishDetection,
    LiveFishTrack,
    LiveMonitorSession,
    Video,
    utc_now,
)
from app.services import video_media
from app.services.fish_enhancement import (
    EnhancementError,
    enhance_crop,
    failure_metadata,
    validate_enhancement,
)
from app.services.fishial import FishialClient, FishialError
from app.services.live_monitor import aware, live_path
from app.services.live_recording import MANIFEST_NAME, PART_PATTERN, RecordingPart
from app.services.species_quality import (
    crop_for_fishial,
    crop_measures,
    frame_score,
    is_clear_frame,
    review_diagnostics,
    staged_quality,
    stop_reason,
    submitted_count,
    verdict,
)
from app.services.species_region import filter_species

logger = logging.getLogger(__name__)

# States a request may not interrupt: the automatic pass has already claimed the
# fish, or another request holds it.
BUSY = ("pending", "ready", "submitted")

# How many frames to decode and measure for each frame the operator asked to send.
# Quality cannot be known without decoding, so the shortlist has to be wider than
# the request; three candidates per purchased frame was enough on both reference
# sessions to change the selection, without making the read pass noticeably longer.
SHORTLIST_FACTOR = 3


def request_settings(settings: Settings) -> Settings:
    """The same settings, with the request-specific relaxations applied.

    Built once and passed down, so selection, voting, consensus and the stop rules
    all read the ordinary setting names and none of them needs to know it is serving
    a request. That is what keeps this path *the same code* as the automatic pass
    rather than a second, quietly diverging implementation of it.

    They are relaxed because the two paths answer different questions. The automatic
    pass protects an unattended budget from junk crops. A request is one operator
    pointing at one fish and authorising that spend; re-applying floors calibrated
    for unattended spending just returns "no answer" for a fish they can plainly see.
    """

    return settings.model_copy(update={
        "fishial_min_frame_confidence": settings.fishial_request_min_frame_confidence,
        "fishial_min_crop_pixels": settings.fishial_request_min_crop_pixels,
        "fishial_edge_margin_pixels": settings.fishial_request_edge_margin_pixels,
        "fishial_min_votes": settings.fishial_request_min_votes,
        "fishial_min_frames_to_vote": settings.fishial_request_min_frames_to_vote,
        "fishial_vote_ratio": settings.fishial_request_vote_ratio,
        "fishial_min_species_score": settings.fishial_request_min_species_score,
        "fishial_max_empty_responses": settings.fishial_request_max_empty_responses,
        "fishial_region_filter_enabled": settings.fishial_request_region_filter_enabled,
    })


class SpeciesRequestError(RuntimeError):
    """Every message here is written for an operator and is safe to return."""

    public_message = "Species identification could not be requested"

    def __str__(self) -> str:
        return self.public_message


class IdentificationDisabledError(SpeciesRequestError):
    public_message = (
        "Species identification is not configured on this deployment; "
        "set FISHIAL_ENABLED with a client ID and secret"
    )


class IdentificationBusyError(SpeciesRequestError):
    public_message = "This fish is already being identified"


class FootageUnavailableError(SpeciesRequestError):
    public_message = "The footage this fish was seen in is no longer available"


class NoUsableFramesError(SpeciesRequestError):
    public_message = (
        "No frame of this fish could be read: its observations are either too small "
        "to crop or the footage no longer holds the frames they were seen in"
    )


@dataclass(frozen=True)
class Observation:
    """One stored sighting, in the pixel space of the footage we will read."""

    frame_number: int
    box: tuple[float, float, float, float]
    confidence: float


@dataclass
class Chosen:
    """One frame the system picked, already cropped and ready to send."""

    frame_number: int
    score: float
    quality: dict
    expected_box: list[float]
    window: int
    image: bytes = field(repr=False)


# --- Reading frames out of whichever footage holds them ----------------------


class FrameReader:
    """Decode specific frame numbers of one fish's footage, in ascending order."""

    def read(self, numbers: Sequence[int]) -> Iterator[tuple[int, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoFrameReader(FrameReader):
    """A stored video file, where a detection's frame number indexes the file.

    Frames are reached by grabbing forward rather than by seeking. Seeking an
    inter-frame codec lands on a keyframe, and a crop taken from the wrong frame
    would be attributed to this fish with full confidence, which is the one
    failure this feature must not have. Grabbing skipped frames without decoding
    them keeps the pass cheap enough to stay inside a request.
    """

    def __init__(self, path: Path, cv2: Any):
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise FootageUnavailableError()

    def read(self, numbers: Sequence[int]) -> Iterator[tuple[int, Any]]:
        position = 0
        for number in sorted(set(numbers)):
            while position < number:
                if not self.capture.grab():
                    return
                position += 1
            success, frame = self.capture.read()
            position += 1
            if not success or frame is None:
                return
            yield number, frame

    def close(self) -> None:
        self.capture.release()


class LiveRecordingFrameReader(FrameReader):
    """The chunks a running session has retained, addressed by session frame number.

    ``frames_processed`` is the session's global frame counter and is what
    ``LiveFishDetection.frame_number`` stores, so the manifest's part offsets are
    all that is needed to turn one into a chunk and an index inside it. Chunks are
    seconds long, so each is opened and read from its own start.
    """

    def __init__(self, directory: Path, cv2: Any = None):
        # ``cv2`` is only needed to decode; the manifest alone answers "which chunk
        # holds this frame", which is all the observation mapping needs.
        self.cv2 = cv2
        self.directory = directory
        try:
            payload = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
            parts = [RecordingPart(**entry) for entry in payload.get("parts", [])]
        except (OSError, TypeError, ValueError) as exc:
            raise FootageUnavailableError() from exc
        # Trust the files on disk over the manifest, the way RecordingWriter.resume
        # does: the last append may have been interrupted before the rewrite.
        self.chunks = sorted(directory.glob(PART_PATTERN))
        self.parts = parts[: len(self.chunks)]
        if not self.parts:
            raise FootageUnavailableError()
        self.starts = [part.start_frame for part in self.parts]

    def locate(self, frame_number: int) -> int | None:
        """Index of the chunk holding this session frame, or ``None``."""

        index = bisect.bisect_right(self.starts, frame_number) - 1
        if index < 0:
            return None
        part = self.parts[index]
        return index if frame_number < part.start_frame + part.frames else None

    def part_for(self, frame_number: int) -> RecordingPart | None:
        index = self.locate(frame_number)
        return self.parts[index] if index is not None else None

    def read(self, numbers: Sequence[int]) -> Iterator[tuple[int, Any]]:
        wanted: dict[int, list[int]] = {}
        for number in sorted(set(numbers)):
            index = self.locate(number)
            if index is not None:
                wanted.setdefault(index, []).append(number)
        for index, frames in sorted(wanted.items()):
            capture = self.cv2.VideoCapture(str(self.chunks[index]))
            try:
                if not capture.isOpened():
                    continue
                position = 0
                for number in frames:
                    local = number - self.parts[index].start_frame
                    while position < local:
                        if not capture.grab():
                            break
                        position += 1
                    if position != local:
                        break
                    success, frame = capture.read()
                    position += 1
                    if not success or frame is None:
                        break
                    yield number, frame
            finally:
                capture.release()


# --- What there is to read, for either kind of fish --------------------------


@dataclass(frozen=True)
class Footage:
    """One fish's observations plus a way to open the footage that holds them."""

    observations: list[Observation]
    fps: float
    open_reader: Any

    def reader(self, cv2) -> FrameReader:
        return self.open_reader(cv2)


def footage_for_track(db: Session, track: FishTrack, settings: Settings) -> Footage:
    video = db.get(Video, track.video_id)
    if video is None:
        raise FootageUnavailableError()
    path = Path(video.storage_path)
    if not path.is_file():
        raise FootageUnavailableError()
    rows = db.scalars(
        select(FishDetection)
        .where(FishDetection.fish_track_id == track.id)
        .order_by(FishDetection.frame_number)
    ).all()
    observations = [
        Observation(row.frame_number, (row.x1, row.y1, row.x2, row.y2), row.confidence)
        for row in rows
    ]
    return Footage(observations, video_media.valid_fps(video.fps) or 30.0,
                   lambda cv2: VideoFrameReader(path, cv2))


def footage_for_live_track(db: Session, track: LiveFishTrack, settings: Settings) -> Footage:
    """A live track reads from the chunks retained so far, running or not.

    A finished session whose recording has already been published to the library
    has no chunks left; its fish are identified through the recording's own tracks,
    which is the same footage under a library row.
    """

    session = db.get(LiveMonitorSession, track.session_id)
    if session is None:
        raise FootageUnavailableError()
    directory = live_path(settings, str(track.session_id), "recording")
    if not (directory / MANIFEST_NAME).is_file():
        raise FootageUnavailableError()
    rows = db.scalars(
        select(LiveFishDetection)
        .where(LiveFishDetection.track_id == track.id)
        .order_by(LiveFishDetection.frame_number)
    ).all()
    reader = LiveRecordingFrameReader(directory)
    observations = []
    for row in rows:
        part = reader.part_for(row.frame_number)
        if part is None:
            # A chunk the byte cap refused, or one not yet appended. Its frame is
            # not in the recording, so it cannot be read back.
            continue
        box = part.place(row.x1, row.y1, row.x2, row.y2)
        observations.append(Observation(row.frame_number, box, row.confidence))
    reader.close()
    return Footage(observations, video_media.valid_fps(settings.live_fps) or 5.0,
                   lambda cv2: LiveRecordingFrameReader(directory, cv2))


# --- Choosing which frames to buy --------------------------------------------


def shortlist(observations: Sequence[Observation], fps: float, settings: Settings,
              wanted: int) -> list[tuple[int, Observation]]:
    """``(window, observation)`` pairs worth decoding, best first, one per window.

    Two rules, in this order:

    **Independence first.** One frame per separation window is taken before any
    window is used twice. Near-duplicate frames of one moment show the same pose,
    lighting and occlusion, so they cast near-identical votes; spreading the spend
    across moments is what makes the votes worth counting.

    **Then fill.** A fish seen for fewer moments than the operator asked for frames
    used to be sent fewer frames. That is the right instinct for unattended spending
    and the wrong answer for someone who asked a question: a second-best frame of the
    same moment is weaker evidence than a new moment, but it is much better evidence
    than none. So once every window has contributed, the next-best frames fill the
    request, and the audit records how many distinct moments are actually behind it.

    **Quality throughout.** Ranking uses the measurements available without decoding
    - crop size and detector confidence - under the deployment's own quality weights,
    and the shortlist is wider than the request so the full measured score can
    reorder it.
    """

    separation = settings.fishial_min_frame_separation_seconds
    span = max(1, round(separation * fps)) if separation > 0 else 1
    origin = observations[0].frame_number if observations else 0
    windows: dict[int, list[tuple[float, Observation]]] = {}
    for observation in observations:
        x1, y1, x2, y2 = observation.box
        short_side = min(x2 - x1, y2 - y1)
        if (not settings.fishial_min_frame_confidence <= observation.confidence <= 1
                or short_side < settings.fishial_min_crop_pixels):
            continue
        score = frame_score({"short_side": short_side, "confidence": observation.confidence},
                            settings.fishial_quality_weights)
        windows.setdefault((observation.frame_number - origin) // span, []).append(
            (score, observation))
    if not windows:
        return []
    for entries in windows.values():
        entries.sort(key=lambda entry: (-entry[0], entry[1].frame_number))

    # Round-robin across windows: every window gives its best before any gives its
    # second, so the head of this list is always the most independent set available.
    ranked: list[tuple[int, Observation]] = []
    for rank in range(max(len(entries) for entries in windows.values())):
        tier = [(entries[rank][0], window, entries[rank][1])
                for window, entries in windows.items() if rank < len(entries)]
        tier.sort(key=lambda item: (-item[0], item[1]))
        ranked.extend((window, observation) for _, window, observation in tier)
    return ranked[: max(1, wanted) * SHORTLIST_FACTOR]


def choose_frames(footage: Footage, settings: Settings, wanted: int, cv2) -> list[Chosen]:
    """Decode the shortlist, measure it properly, and keep the best ``wanted`` crops.

    One pass over the footage: each shortlisted frame is decoded once, measured,
    cropped and encoded, and the crops are ranked afterwards. Holding a few dozen
    JPEG crops is cheaper than reading the footage twice.
    """

    candidates = shortlist(footage.observations, footage.fps, settings, wanted)
    if not candidates:
        return []
    by_number = {observation.frame_number: (window, observation)
                 for window, observation in candidates}
    reader = footage.reader(cv2)
    measured: list[Chosen] = []
    try:
        for number, frame in reader.read(sorted(by_number)):
            window, observation = by_number[number]
            box = observation.box
            if not is_clear_frame(cv2, frame, box, observation.confidence, settings):
                continue
            try:
                cropped = crop_for_fishial(cv2, frame, box, settings)
            except OSError:
                continue
            if cropped is None:
                continue
            image, expected, _ = cropped
            quality = crop_measures(cv2, frame, box, observation.confidence)
            measured.append(Chosen(number, frame_score(quality, settings.fishial_quality_weights),
                                   quality, expected, window, image))
    finally:
        reader.close()
    return _pick(measured, wanted)


def _pick(measured: list[Chosen], wanted: int) -> list[Chosen]:
    """Best measured crops, spread across moments before any moment repeats.

    Sorting by score alone would hand three views of one instant to a fish that was
    also visible at two other moments. Taking each window's best first, and only then
    filling from what is left, keeps the strongest *independent* evidence at the
    front while still spending everything the operator paid for.
    """

    measured.sort(key=lambda chosen: (-chosen.score, chosen.frame_number))
    windows, taken = set(), set()
    chosen = []
    for entry in measured:
        if entry.window not in windows:
            windows.add(entry.window)
            taken.add(entry.frame_number)
            chosen.append(entry)
    chosen.extend(entry for entry in measured if entry.frame_number not in taken)
    return chosen[:wanted]


# --- The request itself -------------------------------------------------------


def claim(db: Session, track, frames: int, settings: Settings) -> None:
    """Take ownership of this fish's identification, or refuse.

    The claim is a conditional state change committed before anything is read or
    sent, so the live worker's automatic pass (which only ever selects a
    ``candidate``) cannot also start paying for the same fish, and a second
    operator cannot double-spend on it either.
    """

    if not settings.fishial_enabled:
        raise IdentificationDisabledError()
    model = type(track)
    stale = settings.fishial_request_stale_seconds
    requested = track.fishial_requested_at
    if track.fishial_state in BUSY:
        # An API process that died mid-request would otherwise hold the fish
        # forever. Past the staleness horizon the claim may be taken over.
        age = (utc_now() - aware(requested)).total_seconds() if requested else None
        if age is None or age < stale:
            raise IdentificationBusyError()
    # A repeat request starts a fresh journal: its frames are bought again, and
    # mixing them with an earlier attempt's votes would double-count the evidence.
    audit = {"request": {"frames": frames, "at": utc_now().isoformat(), "source": "operator"},
             "staged": [], "frames": [], "tally": {}}
    result = db.execute(
        update(model)
        .where(model.id == track.id, model.fishial_state == track.fishial_state)
        .values(fishial_state="submitted", fishial_requested_at=utc_now(),
                fishial_completed_at=None, fishial_species=None,
                fishial_species_confidence=None, fishial_frames_used=0,
                fishial_votes_json=json.dumps(audit))
        .execution_options(synchronize_session=False)
    )
    if not result.rowcount:
        db.rollback()
        raise IdentificationBusyError()
    db.commit()
    db.refresh(track)


def identify(db: Session, track, settings: Settings, client=None, enhancer=None) -> None:
    """Buy and resolve one claimed request. Never raises; every end state is stored.

    Runs after the response to the operator has already been sent, so its only
    output is the track's own row: a state, a species or a reason, and the audit
    the review panel reads.
    """

    audit = track.fishial_votes
    frames = int((audit.get("request") or {}).get("frames") or 0)
    owns_client = client is None
    enhancer = enhancer or enhance_crop
    # One relaxed view of the settings, passed to everything below: selection, the
    # floors, voting, consensus and the stop rules all read the ordinary names.
    settings = request_settings(settings)
    try:
        footage = (footage_for_track(db, track, settings) if isinstance(track, FishTrack)
                   else footage_for_live_track(db, track, settings))
        chosen = choose_frames(footage, settings, frames, video_media.load_cv2())
        if not chosen:
            _complete(db, track, audit, "review_required", str(NoUsableFramesError()))
            return
        audit["staged"] = [{"frame_number": entry.frame_number, "window": entry.window,
                            "score": entry.score, "quality": entry.quality,
                            "expected_box": entry.expected_box} for entry in chosen]
        # How many distinct moments the evidence actually spans, which is what makes
        # the votes independent - not the same thing as how many frames were bought.
        audit["windows"] = len({entry.window for entry in chosen})
        track.fishial_quality_score = staged_quality(audit["staged"], settings, frames)
        track.fishial_frames_used = len(chosen)
        track.fishial_votes_json = json.dumps(audit)
        db.commit()
        if client is None:
            client = FishialClient(settings)
        _send(db, track, audit, chosen, settings, client, enhancer)
    except SpeciesRequestError as exc:
        _complete(db, track, audit, "error", str(exc))
    except Exception:  # noqa: BLE001 - never let a provider or decoder failure escape
        # The same redaction rule the automatic pass follows: transport and decoder
        # messages can echo credentials or image bytes, so none of them are stored.
        db.rollback()
        db.refresh(track)
        logger.warning("Requested identification failed for fish %s", track.id, exc_info=False)
        _complete(db, track, track.fishial_votes, "error", "identification failed")
    finally:
        if owns_client and client is not None:
            client.close()


def _send(db: Session, track, audit: dict, chosen: list[Chosen], settings: Settings,
          client, enhancer) -> None:
    """Spend on the chosen crops, best first, stopping as soon as the answer is in."""

    region = _region_for(db, track, settings)
    retries_used = 0
    stopped = None
    for position, entry in enumerate(chosen):
        record = {"frame_number": entry.frame_number, "species": None, "score": None,
                  "margin": None, "voted": False, "reason": "interrupted", "attempts": []}
        audit["frames"].append(record)
        image = entry.image
        try:
            try:
                enhanced = enhancer(image, settings)
                validate_enhancement(enhanced, image, settings)
            except Exception:  # noqa: BLE001 - injected enhancers must also redact errors
                raise EnhancementError() from None
            record["preprocessing"] = enhanced.metadata
            image = enhanced.image
            track.fishial_votes_json = json.dumps(audit)
            db.commit()  # Pre-send provenance precedes even authentication.

            def spend(retry: bool, _record=record) -> None:
                nonlocal retries_used
                if retry and retries_used >= settings.fishial_max_api_retries:
                    raise FishialError("retry limit reached")
                # Journalled before the request, not after: a crash after sending
                # must never let a repeat request think the call was free.
                _record.setdefault("attempts", []).append({"retry": retry})
                track.fishial_votes_json = json.dumps(audit)
                _count_manual_call(db, track)
                db.commit()
                if retry:
                    retries_used += 1

            if isinstance(client, FishialClient):
                client.before_image_call = spend
                prediction = client.identify(image, entry.expected_box)
            else:
                # Test and replay adapters make one image call per identify().
                spend(False)
                prediction = client.identify(image, entry.expected_box)
            record["raw"] = prediction.raw
            _vote(record, prediction, region, settings)
        except EnhancementError:
            record["reason"] = "preprocessing failed"
            record["preprocessing"] = failure_metadata(entry.image, settings)
        except FishialError as exc:
            record["reason"] = exc.reason
        except Exception:  # noqa: BLE001 - one bad crop or provider result only abstains
            record["reason"] = "frame unavailable or request failed"
        finally:
            if isinstance(client, FishialClient):
                client.before_image_call = None
        track.fishial_votes_json = json.dumps(audit)
        db.commit()
        remaining = len(chosen) - position - 1
        stopped = stop_reason(audit["frames"], remaining, settings) if remaining else None
        if stopped:
            audit["stopped_early"] = stopped
            audit["calls_saved"] = remaining
            break

    frames = audit["frames"]
    audit["tally"] = dict(Counter(f["species"] for f in frames if f.get("voted")))
    audit["submitted_frames"] = submitted_count(frames)
    winner, mean = verdict(frames, settings)
    if winner is not None:
        # A name the region list does not recognise is still reported, marked, and
        # left to the operator - never silently promoted to a confirmed result.
        if any(winner in (f.get("implausible_species") or []) for f in frames):
            audit["implausible_for_region"] = winner
        track.fishial_species, track.fishial_species_confidence = winner, mean
        _complete(db, track, audit, "identified", None)
        return
    # These abstentions point at different fixes and must never collapse into one
    # message: more frames, a clearer fish, a different region, or a provider fault.
    state = "review_required" if any(
        f.get("succeeded") or f.get("reason") == "preprocessing failed" for f in frames) else "error"
    reason = ("all frames failed" if state == "error" else
              stopped if stopped and stopped != "decided" else
              "consensus not reached")
    _complete(db, track, audit, state, reason)


def _vote(record: dict, prediction, region: str | None, settings: Settings) -> None:
    """Turn one response into a vote, or record why it abstained.

    Regional plausibility **flags** a requested answer rather than discarding it.
    Dropping the only name the classifier offered leaves an operator with an
    unexplained blank; showing them the name next to "not expected in this region"
    leaves them with a decision. The automatic pass still drops, because nobody is
    there to make that decision for an unattended session.
    """

    record["object_count"] = prediction.object_count
    record["object_index"] = prediction.object_index
    record["object_iou"] = prediction.object_iou
    record["succeeded"] = True
    # The classifier saw our fish and declined to name it. Distinct from a response
    # we could not attribute, and from one we filtered ourselves.
    record["empty"] = prediction.object_index is not None and not prediction.species
    record["reason"] = "no unambiguous fish prediction"
    if settings.fishial_region_filter_enabled:
        ranked, dropped = filter_species(prediction.species, region)
        if dropped:
            record["dropped_species"] = dropped
            record["dropped_reason"] = "implausible_for_region"
        if not ranked:
            if dropped:
                # Filtering emptied the list: abstain rather than falling through to
                # a name we already judged impossible here.
                record["reason"] = "implausible_for_region"
            return
    else:
        ranked = list(prediction.species)
        _, implausible = filter_species(ranked, region)
        if implausible:
            record["implausible_species"] = implausible
        if not ranked:
            return
    name, score = ranked[0]
    margin = score - (ranked[1][1] if len(ranked) > 1 else 0)
    record.update(species=name, score=score, margin=margin,
                  voted=margin >= settings.fishial_min_frame_margin)
    record["reason"] = None if record["voted"] else "ambiguous frame"


def _region_for(db: Session, track, settings: Settings) -> str | None:
    """The regional list for this fish, frozen the way the live pass freezes it.

    Resolved whether or not filtering is enabled: with it off the list is still what
    tells an operator a name is out of place, which is the point of flagging.
    """

    if isinstance(track, LiveFishTrack):
        session = db.get(LiveMonitorSession, track.session_id)
        return session.species_id_region if session else None
    video = db.get(Video, track.video_id)
    if video is None:
        return None
    if video.source_session_id is not None:
        session = db.get(LiveMonitorSession, video.source_session_id)
        if session is not None:
            return session.species_id_region
    # An uploaded file carries no camera registry entry, so it is filtered only
    # when its camera id happens to match a configured source.
    source = settings.live_source(video.camera_id)
    return source.region if source else None


def _count_manual_call(db: Session, track) -> None:
    """Record operator spend against the session, separately from its own budget.

    A requested call must not consume the automatic pass's reservation: the two are
    authorised differently, and an operator asking about one fish should not quietly
    starve the fish the session was told to identify.
    """

    if not isinstance(track, LiveFishTrack):
        return
    db.execute(
        update(LiveMonitorSession)
        .where(LiveMonitorSession.id == track.session_id)
        .values(species_id_manual_api_calls=LiveMonitorSession.species_id_manual_api_calls + 1)
        .execution_options(synchronize_session=False)
    )


def _complete(db: Session, track, audit: dict, state: str, reason: str | None) -> None:
    audit.setdefault("tally", {})
    audit.setdefault("frames", [])
    audit["reason"] = reason
    track.fishial_state, track.fishial_completed_at = state, utc_now()
    if state != "identified":
        track.fishial_species = track.fishial_species_confidence = None
    track.fishial_votes_json = json.dumps(audit)
    db.commit()


def identification_payload(track) -> dict:
    """Everything an operator needs to read about one fish's identification.

    Works for a live track and a library track alike, because both carry the same
    identification columns. The diagnostics block is the same one the live review
    panel shows, so an abstention explains itself the same way on both screens.
    """

    audit = track.fishial_votes
    request = audit.get("request") or {}
    frames = audit.get("frames") or []
    requested = request.get("frames")
    return {
        "track_id": track.id,
        "state": track.fishial_state,
        "species": track.fishial_species,
        "confidence": track.fishial_species_confidence,
        "frames_requested": requested if isinstance(requested, int) else None,
        "frames_selected": len(audit.get("staged") or []),
        "frames_submitted": audit.get("submitted_frames") or submitted_count(frames),
        "calls_saved": audit.get("calls_saved") or 0,
        "quality_score": track.fishial_quality_score,
        "requested_at": aware(track.fishial_requested_at) if track.fishial_requested_at else None,
        "completed_at": aware(track.fishial_completed_at) if track.fishial_completed_at else None,
        "stopped_early": audit.get("stopped_early"),
        # Distinct moments behind the evidence, which is what makes votes independent.
        # Fewer than the frames sent means some frames share a moment.
        "windows": audit.get("windows") or 0,
        # A name reported despite the camera's region list not recognising it. The
        # operator decides; the system never quietly promotes or discards it.
        "implausible_for_region": audit.get("implausible_for_region"),
        "tally": audit.get("tally") or {},
        "diagnostics": review_diagnostics(audit)
        if track.fishial_state in {"review_required", "error"} else None,
    }


def requested_frames(value: int | None, settings: Settings) -> int:
    """Clamp an operator's frame count to what this deployment allows."""

    frames = settings.fishial_request_default_frames if value is None else int(value)
    return max(1, min(frames, settings.fishial_request_max_frames))


def run_request(session_factory, model, track_id: uuid.UUID, settings: Settings,
                client=None, enhancer=None) -> None:
    """Background entry point: one claimed request, on its own database session."""

    db = session_factory()
    try:
        track = db.get(model, track_id)
        if track is None or track.fishial_state != "submitted":
            return
        identify(db, track, settings, client=client, enhancer=enhancer)
    except Exception:  # noqa: BLE001 - a background task has nowhere to raise to
        logger.warning("Requested identification task failed for fish %s", track_id, exc_info=False)
        db.rollback()
    finally:
        db.close()
