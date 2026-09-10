"""Regional plausibility: reject the impossible for free."""
from __future__ import annotations

import pytest

from app.config import DEFAULT_LIVE_SOURCES, Settings
from app.services.species_region import REGIONS, filter_species, region_names


def test_the_reference_sessions_two_candidates_are_rejected_for_galway_bay():
    """Both Fishial candidates the SmartBay 3 session ever produced were impossible.

    ``Sparisoma aurofrenatum`` is a Caribbean parrotfish; ``Pomatomus saltatrix`` is a
    rare southern vagrant, not part of the Irish coastal assemblage. Neither is on the
    curated list, so both are dropped before voting at no API cost.
    """

    ranked = [("Sparisoma aurofrenatum", .48), ("Pomatomus saltatrix", .46)]
    kept, dropped = filter_species(ranked, "north_east_atlantic")
    assert kept == []
    assert dropped == ["Sparisoma aurofrenatum", "Pomatomus saltatrix"]


def test_local_species_survive_and_matching_is_case_and_whitespace_only():
    for name in ("Gadus morhua", "  gadus MORHUA  ", "Labrus bergylta", "Scomber scombrus"):
        kept, dropped = filter_species([(name, .9)], "north_east_atlantic")
        assert kept == [(name, .9)] and dropped == []
    # No fuzzy or genus-level matching: a near miss abstains rather than guesses.
    for name in ("Gadus", "Gadus morhuaa", "Gadusmorhua", "Gadus  morhua"):
        assert filter_species([(name, .9)], "north_east_atlantic")[0] == []


@pytest.mark.parametrize("region", [None, "", "atlantis", "north_pacific"])
def test_an_unset_or_unknown_region_filters_nothing(region):
    ranked = [("Sparisoma aurofrenatum", .48)]
    assert filter_species(ranked, region) == (ranked, [])
    assert region_names(region) is None


def test_smartbay_cameras_carry_a_region_and_coral_city_does_not():
    settings = Settings(_env_file=None, fishial_client_id=None, fishial_client_secret=None)
    by_key = {source.key: source for source in settings.live_source_registry}
    for key in ("smartbay-cam1", "smartbay-cam2", "smartbay-cam3"):
        assert by_key[key].region == "north_east_atlantic"
    # We have no curated list for Miami, so nothing is filtered there.
    assert by_key["coral-city"].region is None
    assert {entry.get("region") for entry in DEFAULT_LIVE_SOURCES} == {None, "north_east_atlantic"}


def test_region_lists_are_normalised_and_non_trivial():
    names = REGIONS["north_east_atlantic"]
    assert len(names) > 100
    assert all(name == name.strip().lower() for name in names)


def test_latest_smartbay_candidates_keep_saithe_and_reject_spotted_gar():
    ranked = [("Pollachius virens", .51), ("Lepisosteus oculatus", .46)]
    assert filter_species(ranked, "north_east_atlantic") == (
        [("Pollachius virens", .51)], ["Lepisosteus oculatus"])
