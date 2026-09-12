"""Pure scoring and consensus arithmetic for live species identification.

Nothing here touches the database, the filesystem or the Fishial client, so every
rule is unit-testable in isolation. That matters most for :func:`consensus_outlook`:
getting its counting wrong silently costs accuracy rather than raising.

Accuracy strategy: rank, do not gate. Absolute thresholds calibrated above the data
reject everything (the 0.70/96 pair rejected 96.8% of Coral City frames). Floors
exclude unusable frames; these blends pick the best of what remains.
"""
from __future__ import annotations

import math
from collections import Counter
from statistics import median


def review_candidates(frames: list[dict]) -> list[dict]:
    """Report tentative names only for the stored matched fish, never other objects.

    These scores describe individual classifier answers, not confirmed track IDs.
    Regional rejections remain visible as rejected suggestions for human review.
    """
    candidates = {}
    for frame in frames:
        found = {}
        raw = frame.get("raw") or {}
        objects = raw.get("objects") if isinstance(raw, dict) else None
        definitions = raw.get("definitions") if isinstance(raw, dict) else None
        index = frame.get("object_index")
        if (type(index) is int and isinstance(objects, list) and 0 <= index < len(objects)
                and isinstance(objects[index], dict) and isinstance(definitions, dict)):
            for entry in objects[index].get("species") or []:
                if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                    continue
                definition = definitions.get(entry["id"])
                if isinstance(definition, dict):
                    name = definition.get("scientificName")
                    if isinstance(name, str):
                        found[name] = (entry.get("certainty"), definition.get("commonName"))
        # Older audits may retain a vote without its raw response or match index.
        if frame.get("species") and isinstance(frame["species"], str):
            found.setdefault(frame["species"], (frame.get("score"), None))
        for name, (score, common) in found.items():
            if (not name.strip() or len(name) > 256 or isinstance(score, bool)
                    or not isinstance(score, (int, float)) or not math.isfinite(score)
                    or not 0 <= score <= 1):
                continue
            rejected = name in (frame.get("dropped_species") or [])
            key = (name, rejected)
            candidate = candidates.setdefault(key, {
                "species": name, "common_name": common if isinstance(common, str) else None,
                "max_score": score, "frames": 0, "rejected_for_region": rejected,
            })
            candidate["max_score"] = max(candidate["max_score"], score)
            candidate["frames"] += 1
    return sorted(candidates.values(), key=lambda c: (
        c["rejected_for_region"], -c["frames"], -c["max_score"], c["species"]))


def review_diagnostics(audit: dict) -> dict:
    """Explain stored abstentions without changing votes or the stopping decision."""

    counts = Counter()
    frames = audit.get("frames") or []
    for frame in frames:
        if frame.get("voted"):
            continue
        reason = ("classifier returned no candidates" if frame.get("empty") else
                  frame.get("reason") or "no unambiguous fish prediction")
        counts[reason] += 1
    stopped = audit.get("reason")
    reason = stopped or "consensus not reached"
    # Keep mixed evidence and operational failures distinct. A single cause can
    # explain an unreachable consensus more usefully than the stopping rule alone.
    if (reason in {"consensus unreachable", "consensus not reached", "all frames failed"}
            and frames and not any(f.get("voted") for f in frames) and len(counts) == 1):
        cause = next(iter(counts))
        if cause in {"classifier returned no candidates", "implausible_for_region", "invalid response"}:
            reason = cause
    return {"reason": reason, "stop_reason": stopped, "frame_reasons": dict(counts),
            "candidates": review_candidates(frames),
            "budget_scope": "fish selection limit" if reason == "budget exhausted"
            and not frames else None}

# Saturating reference values: a measurement at or above its reference is "as good
# as it usefully gets", so the blend stays scale-free and bounded to [0, 1].
#
#   short_side  240 px  - Coral City's 98th and SmartBay 3's 91st percentile over
#                         5,239 reference detections, so under 9% of frames saturate
#                         on either camera. A lower reference looked reasonable but
#                         flattened 35% of SmartBay frames onto the ceiling, and a
#                         flat dominant term makes pool admission degenerate back
#                         into the arrival order this change exists to remove.
#   confidence  0.85    - just under the 0.86 maximum VIAME emitted on either camera.
#   sharpness / contrast / colorfulness are NOT measured against stored data: the
#   reference sessions never persisted per-frame pixel statistics. They are
#   conservative priors on 8-bit LAB/BGR crops. This is exactly why
#   ``fishial_quality_floor`` ships at 0.0 (admit everything) - use
#   ``scripts/fishial_replay.py`` to calibrate before raising it.
REFERENCE: dict[str, float] = {
    "short_side": 240.0,
    "sharpness": 200.0,
    "confidence": 0.85,
    "contrast": 60.0,
    "colorfulness": 40.0,
}


def _normalise(name: str, value) -> float:
    reference = REFERENCE.get(name)
    if not reference or value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0) / reference, 1.0)


def _blend(terms: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.get(name, 0.0) for name in terms)
    if total <= 0:
        return 0.0
    return sum(terms[name] * weights.get(name, 0.0) for name in terms) / total


def frame_score(measures: dict, weights: dict[str, float]) -> float:
    """Rank one staged frame against the others of its own track.

    The per-frame blend deliberately omits the ``frames`` term: it compares crops,
    not tracks. Selection (:func:`track_quality`) adds frame count on top.
    """

    return _blend({name: _normalise(name, measures.get(name)) for name in REFERENCE}, weights)


def pool_score(staged: list[dict], weights: dict[str, float]) -> float:
    """Strength of a candidate for pool admission and eviction.

    Median rather than mean so one lucky frame cannot hold a slot open, and frame
    count is excluded so a brand-new track can be compared fairly against a track
    that has simply been in view longer.
    """

    scores = [frame_score(entry.get("quality") or {}, weights) for entry in staged]
    return median(scores) if scores else 0.0


def track_quality(track, settings, frames_per_fish: int) -> float:
    """Selection score: median crop quality, plus how many independent views we hold.

    Pay for the best fish, not the first fish. Arrival time is uncorrelated with
    identifiability - on SmartBay 3 the five arrival-ordered tracks held 26-77
    detections while later tracks held 151, 113 and 110.
    """

    staged = (track.fishial_votes.get("staged") or []) if hasattr(track, "fishial_votes") else []
    return staged_quality(staged, settings, frames_per_fish)


def staged_quality(staged: list[dict], settings, frames_per_fish: int) -> float:
    """:func:`track_quality` over a bare staged list, for tests and the replay tool."""

    weights = settings.fishial_quality_weights
    if not staged:
        return 0.0
    measures = [entry.get("quality") or {} for entry in staged]
    terms = {name: _normalise(name, median([m.get(name) or 0.0 for m in measures]))
             for name in REFERENCE}
    terms["frames"] = min(len(staged) / max(frames_per_fish, 1), 1.0)
    return _blend(terms, weights)


def _consensus_holds(count: int, runner_up: int, submitted: int, mean_score: float, settings) -> bool:
    return (count > runner_up
            and count >= settings.fishial_min_votes
            and submitted >= settings.fishial_min_frames_to_vote
            and submitted > 0 and count / submitted >= settings.fishial_vote_ratio
            and mean_score >= settings.fishial_min_species_score)


def tally(frames: list[dict]) -> dict[str, list[float]]:
    """Winning scores per species, from frame records that actually voted."""

    votes: dict[str, list[float]] = {}
    for frame in frames:
        if frame.get("voted") and frame.get("species"):
            votes.setdefault(frame["species"], []).append(float(frame.get("score") or 0.0))
    return votes


def submitted_count(frames: list[dict]) -> int:
    """Every attempted frame is in the denominator once, including failures.

    Retries share one frame record, so they cannot manufacture extra votes.
    """

    return sum(bool(frame.get("attempts")) for frame in frames)


def consensus_outlook(frames: list[dict], remaining: int, settings) -> str:
    """``"decided"`` | ``"possible"`` | ``"unreachable"`` for the frames sent so far.

    ``remaining`` is how many further staged frames could still be sent (already
    clamped by the caller to the budget it can actually reserve).

    Stop early in both directions. A decided track spends nothing more; a track whose
    consensus rule can no longer be satisfied even if every remaining frame votes for
    its leader spends nothing more either.
    """

    remaining = max(int(remaining), 0)
    votes = tally(frames)
    submitted = submitted_count(frames)
    counts = {name: len(scores) for name, scores in votes.items()}

    if counts:
        leader = max(counts, key=lambda name: (counts[name], sum(votes[name])))
        count = counts[leader]
        runner_up = max((value for name, value in counts.items() if name != leader), default=0)
        mean_score = sum(votes[leader]) / count
        # Decisive only when the leader is also unassailable: if the runner-up could
        # still catch it, buy the remaining evidence rather than guess.
        if (_consensus_holds(count, runner_up, submitted, mean_score, settings)
                and count > runner_up + remaining):
            return "decided"

    if remaining == 0:
        return "unreachable"

    # Optimistic best case: some single species takes every remaining frame at a
    # perfect score. Consider each species already in play plus a brand-new one.
    for name in [*counts, None]:
        scores = votes.get(name, []) if name else []
        count = len(scores) + remaining
        runner_up = max((value for other, value in counts.items() if other != name), default=0)
        mean_score = (sum(scores) + remaining) / count
        if _consensus_holds(count, runner_up, submitted + remaining, mean_score, settings):
            return "possible"
    return "unreachable"


def stop_reason(frames: list[dict], remaining: int, settings) -> str | None:
    """Why to stop before buying another frame for this fish, or ``None`` to continue.

    ``remaining`` is how many further frames the caller could still send.
    """

    limit = settings.fishial_max_empty_responses
    if limit:
        empty = 0
        for frame in reversed(frames):
            if not frame.get("attempts"):
                continue
            if not frame.get("empty"):
                break
            empty += 1
        # Once a track starts coming back empty, the remaining frames of the same
        # track are near-certain to come back empty too - on SmartBay 3 every
        # consecutive frame of a declining track declined. Stop paying for them.
        if empty >= limit:
            return "classifier returned no candidates"
    outlook = consensus_outlook(frames, remaining, settings)
    if outlook == "decided":
        return "decided"
    if outlook == "unreachable":
        return "consensus unreachable"
    return None


def verdict(frames: list[dict], settings) -> tuple[str | None, float | None]:
    """The species a completed frame set agrees on, and its mean winning score.

    ``(None, None)`` when the consensus rule is not satisfied. This is the single
    definition of "identified" - the live pass, the operator-requested pass and
    :func:`consensus_outlook` must never drift apart on what counts as agreement.
    """

    votes = tally(frames)
    if not votes:
        return None, None
    counts = {name: len(scores) for name, scores in votes.items()}
    leader = max(counts, key=lambda name: (counts[name], sum(votes[name])))
    count = counts[leader]
    runner_up = max((value for name, value in counts.items() if name != leader), default=0)
    mean = sum(votes[leader]) / count
    if _consensus_holds(count, runner_up, submitted_count(frames), mean, settings):
        return leader, mean
    return None, None


def is_clear_frame(cv2, frame, box, confidence, settings) -> bool:
    """Safety floor only: reject frames that are genuinely unusable.

    This is NOT selection. The previous 0.70 confidence / 96 px pair sat above the
    90th percentile of both measurements on both reference cameras and rejected
    96.8% (Coral City) and 85.9% (SmartBay 3) of detections conjunctively, which
    is why no track ever reached ``fishial_min_frames_to_vote``. Selection is the
    ranking by :func:`frame_score`; this only excludes junk.
    """

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in (*box, confidence)):
        return False
    if (not settings.fishial_min_frame_confidence <= confidence <= 1
            or min(x2 - x1, y2 - y1) < settings.fishial_min_crop_pixels
            # A fish crossing the frame edge is truncated, so the crop cannot show
            # the whole animal. That is a correctness floor, not a quality one.
            or min(x1, y1, width - 1 - x2, height - 1 - y2) < settings.fishial_edge_margin_pixels):
        return False
    if settings.fishial_blur_min_variance:
        crop = frame[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < settings.fishial_blur_min_variance:
            return False
    return True


def crop_measures(cv2, frame, box, confidence) -> dict:
    """Per-frame measurements the ranker and the selector both need.

    Computed once while the frame is in hand, then persisted into the staged or
    selected audit entry, so nothing has to re-decode pixels later.
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
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    measures["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lightness = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0].astype("float64")
    measures["luminance"] = float(lightness.mean())
    measures["contrast"] = float(lightness.std())  # RMS contrast of L
    blue, green, red = (channel.astype("float64") for channel in cv2.split(crop))
    # Hasler-Susstrunk colourfulness: dim green water scores near zero, which is
    # exactly the domain mismatch Fishial's classifier appears to dislike.
    rg, yb = red - green, 0.5 * (red + green) - blue
    measures["colorfulness"] = float(
        math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))
    return measures


def crop_for_fishial(cv2, frame, box, settings):
    """``(jpeg, expected_box, crop_size)`` for one detection, or ``None`` to skip.

    The crop is rectangular and at the source resolution: never the annotated view
    and never the letterboxed gallery spool. ``expected_box`` is the detection's
    position *within the crop*, normalised, which is what lets the adapter tell our
    fish apart from a neighbour when the response carries several objects.
    """

    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    dx = (x2 - x1) * (settings.fishial_crop_margin - 1) / 2
    dy = (y2 - y1) * (settings.fishial_crop_margin - 1) / 2
    left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    right, bottom = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
    crop = frame[top:bottom, left:right]
    if crop.size == 0 or crop.shape[:2] == frame.shape[:2]:
        # Even an extreme admin crop margin must never turn this into a full-frame upload.
        return None
    # Store the expected box as it actually is, clamping included: at a frame edge
    # the fish is no longer centred, and recomputing a centred box later would
    # match the wrong object.
    expected = [(x1 - left) / (right - left), (y1 - top) / (bottom - top),
                (x2 - left) / (right - left), (y2 - top) / (bottom - top)]
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("Species crop encoding failed")
    return encoded.tobytes(), expected, [right - left, bottom - top]


def preprocess_crop(crop, settings, cv2):
    """Shared live/replay preprocessing; never changes production settings."""

    import numpy as np
    mode = settings.fishial_preprocess
    if mode in ("white_balance", "both"):
        channels = crop.astype("float64")
        means = channels.reshape(-1, 3).mean(axis=0)
        target = means.mean()
        if all(means > 1):  # Gray-world channel scaling.
            crop = np.clip(channels * (target / means), 0, 255).astype("uint8")
    if mode in ("clahe", "both"):
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=settings.fishial_clahe_clip,
                                     tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    target = settings.fishial_upscale_short_side
    short_side = min(crop.shape[:2])
    if target and short_side and short_side < target:
        # Never downscale, never change the aspect ratio.
        scale = target / short_side
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                      max(1, round(crop.shape[0] * scale))),
                               interpolation=cv2.INTER_CUBIC)
    return crop
