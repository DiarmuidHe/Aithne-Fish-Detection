"""Pure scoring and consensus arithmetic.

``consensus_outlook`` decides when to stop spending. Getting it wrong costs accuracy
silently rather than raising, so it is tested directly and exhaustively at the
boundaries rather than only through the identifier.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.services.species_quality import (
    REFERENCE,
    consensus_outlook,
    frame_score,
    pool_score,
    staged_quality,
    submitted_count,
    tally,
)


@pytest.fixture
def settings():
    return Settings(_env_file=None, fishial_client_id=None, fishial_client_secret=None)


def measures(**overrides):
    base = {"short_side": REFERENCE["short_side"], "sharpness": 200.0, "confidence": .85,
            "luminance": 120.0, "contrast": 60.0, "colorfulness": 40.0}
    return {**base, **overrides}


def frames(pattern, score=.9, submitted=None):
    """``pattern`` is a list of species names, or None for an attempted non-vote."""

    return [{"attempts": [{"retry": False}], "voted": name is not None,
             "species": name, "score": score if name else None}
            for name in pattern] + [{} for _ in range(submitted or 0)]


def test_normalisation_saturates_and_ignores_junk(settings):
    weights = settings.fishial_quality_weights
    assert frame_score(measures(), weights) == pytest.approx(1.0)
    # Above the reference adds nothing, so one huge crop cannot dominate the blend.
    assert frame_score(measures(short_side=10_000), weights) == pytest.approx(1.0)
    assert frame_score(measures(short_side=0, sharpness=0, confidence=0,
                                contrast=0, colorfulness=0), weights) == 0.0
    # Non-finite, non-numeric and boolean measurements score zero rather than raise.
    for bad in (float("nan"), float("inf"), None, True, "120"):
        assert frame_score(measures(short_side=bad), weights) < 1.0
    assert frame_score({}, weights) == 0.0
    assert set(REFERENCE) == {"short_side", "sharpness", "confidence", "contrast", "colorfulness"}


def test_pool_score_uses_the_median_not_one_lucky_frame(settings):
    weights = settings.fishial_quality_weights
    lucky = [{"quality": measures(short_side=4)} for _ in range(4)] + [{"quality": measures()}]
    steady = [{"quality": measures(short_side=60)} for _ in range(5)]
    assert pool_score(steady, weights) > pool_score(lucky, weights)
    assert pool_score([], weights) == 0.0


def test_track_quality_rewards_holding_more_independent_views(settings):
    few = [{"quality": measures()}] * 2
    many = [{"quality": measures()}] * 5
    assert staged_quality(many, settings, 5) > staged_quality(few, settings, 5)
    assert staged_quality(many, settings, 5) == pytest.approx(1.0)
    assert staged_quality([], settings, 5) == 0.0


def test_tally_and_denominator(settings):
    records = frames(["A", "A", None, "B"])
    assert tally(records) == {"A": [.9, .9], "B": [.9]}
    assert submitted_count(records) == 4
    # A record with no attempt never reached the API and is not in the denominator.
    assert submitted_count([*records, {"voted": True, "species": "A"}]) == 4


@pytest.mark.parametrize("pattern,remaining,expected", [
    # Decisive only when the leader is also unassailable by what is left.
    (["A", "A", "A"], 0, "decided"),
    (["A", "A", "A"], 2, "decided"),       # Nothing left can reach 3 and tie it.
    (["A", "A", "A", "B"], 2, "possible"),  # B could tie at 3-3, so buy the rest.
    (["A", "A", "A", "A", "A"], 1, "decided"),
    # Exactly at min_votes with nothing left: the boundary itself.
    (["A", "A"], 0, "unreachable"),        # min_votes = 3.
    (["A", "A"], 1, "possible"),
    # Hopeless: too few frames remain to reach min_votes at all.
    ([None, None], 1, "unreachable"),
    ([None], 2, "unreachable"),            # At most 2 votes left, min_votes is 3.
    ([None], 3, "possible"),
    ([None, None, None], 2, "unreachable"),
    # The vote ratio boundary: 3 of 5 is exactly 0.6 and survives; 3 of 6 does not.
    (["A", None, None], 2, "possible"),
    (["A", None, None, None], 2, "unreachable"),
    # A split vote can still be rescued while the leader can pull clear.
    (["A", "B"], 3, "possible"),
    (["A", "B", "C"], 1, "unreachable"),
])
def test_consensus_outlook_boundaries(settings, pattern, remaining, expected):
    assert consensus_outlook(frames(pattern), remaining, settings) == expected


def test_consensus_outlook_respects_the_species_score_floor(settings):
    # Three low-scoring votes cannot be lifted over min_species_score=0.5 by the one
    # perfect frame that is left, so there is nothing left to buy.
    low = frames(["A", "A", "A"], score=.1)
    assert consensus_outlook(low, 1, settings) == "unreachable"
    assert consensus_outlook(frames(["A", "A", "A"], score=.45), 3, settings) == "possible"


def test_consensus_outlook_never_declares_a_tie_decided(settings):
    tied = frames(["A", "A", "A"]) + frames(["B", "B", "B"])
    assert consensus_outlook(tied, 0, settings) == "unreachable"


def test_consensus_outlook_clamps_negative_remaining(settings):
    assert consensus_outlook(frames(["A", "A", "A"]), -5, settings) == "decided"
