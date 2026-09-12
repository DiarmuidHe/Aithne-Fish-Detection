"""Choosing a species by hand: the catalogue, its search, and what gets recorded.

The machine proposes and a person decides, so these tests are mostly about the
decision being *separable* afterwards - a manual name never overwrites a
classifier's, clearing it restores what the machine said, and the search an
operator uses to find the name works offline from a curated list rather than from
whatever a photo service happens to answer.
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.db.models import (
    FishTrack,
    LiveFishTrack,
    LiveMonitorSession,
    Video,
    VideoProcessingStatus,
    utc_now,
)
from app.services import species_catalogue, species_reference


# --- The catalogue ---------------------------------------------------------


def test_every_catalogue_entry_has_a_binomial_and_a_common_name():
    assert len(species_catalogue.ENTRIES) > 100
    for entry in species_catalogue.ENTRIES:
        assert len(entry.species.split()) == 2, entry.species
        assert entry.species[0].isupper() and entry.species.split()[1].islower()
        assert entry.common_name and entry.common_name[0].isupper()


def test_the_region_filter_and_the_catalogue_are_the_same_list():
    """One judgement about what swims here, not two that can drift apart."""

    from app.services.species_region import REGIONS

    assert REGIONS["north_east_atlantic"] == frozenset(
        species.lower() for species, _ in species_catalogue.NORTH_EAST_ATLANTIC
    )


def test_search_ranks_a_whole_name_above_a_name_that_merely_contains_it():
    """Typing "cod" wants Atlantic cod first, not a catshark matched mid-word."""

    names = [entry.species for entry in species_catalogue.search("cod")]
    assert names[0] == "Gadus morhua"
    assert "Trisopterus minutus" in names  # Poor cod: a later word, so it ranks below.
    assert "Scyliorhinus canicula" not in names  # Small-spotted catshark is not a cod.


def test_search_finds_a_scientific_name_a_common_name_and_a_group():
    assert [e.species for e in species_catalogue.search("Gadus mor")] == ["Gadus morhua"]
    assert [e.species for e in species_catalogue.search("john dory")] == ["Zeus faber"]
    # Four of the five local wrasse are called one; Rock cook is not.
    assert len(species_catalogue.search("wrasse")) == 4


def test_search_ignores_case_accents_and_the_punctuation_inside_names():
    for query in ("montagus blenny", "MONTAGU'S BLENNY", "Montagu’s  blenny"):
        assert [e.species for e in species_catalogue.search(query)] == [
            "Coryphoblennius galerita"
        ]
    # A hyphen is a space, but removing the gap entirely is a different word.
    assert species_catalogue.search("small spotted")[0].species == "Scyliorhinus canicula"
    assert species_catalogue.search("smallspotted") == []


def test_search_is_stable_and_bounded():
    assert species_catalogue.search("a", limit=3) == species_catalogue.search("a", limit=3)
    assert len(species_catalogue.search("a", limit=3)) == 3
    assert species_catalogue.search("  ") == []


# --- The search endpoint ---------------------------------------------------


def test_search_endpoint_offers_catalogue_names_with_their_common_names(client):
    rows = client.get("/species/search", params={"q": "ballan"}).json()
    assert rows[0]["species"] == "Labrus bergylta"
    assert rows[0]["common_name"] == "Ballan wrasse"
    assert rows[0]["origin"] == "catalogue"
    # Reference lookups are off in tests, so no photo was fetched to answer this.
    assert rows[0]["image_url"] is None


def test_search_endpoint_refuses_a_query_too_short_to_mean_anything(client):
    assert client.get("/species/search", params={"q": "co"}).json() == []
    assert client.get("/species/search", params={"q": " "}).json() == []


def test_search_endpoint_also_offers_names_this_deployment_has_actually_seen(
    client, db_session_factory
):
    """A classifier can name a fish no curated list holds; that name stays choosable.

    Otherwise an operator who has seen *Pomatomus saltatrix* reported on one fish
    could not put it on a second, which would make the curated list a cage rather
    than a starting point.
    """

    with db_session_factory() as db:
        video = _stored_video(db)
        track = _stored_track(db, video)
        track.fishial_species = "Pomatomus saltatrix"
        db.commit()

    rows = client.get("/species/search", params={"q": "pomatomus"}).json()
    assert [(row["species"], row["origin"]) for row in rows] == [
        ("Pomatomus saltatrix", "observed")
    ]


def test_search_endpoint_lists_a_name_once_even_when_both_sources_hold_it(
    client, db_session_factory
):
    with db_session_factory() as db:
        track = _stored_track(db, _stored_video(db))
        track.fishial_species = "Gadus morhua"
        db.commit()

    rows = client.get("/species/search", params={"q": "gadus morhua"}).json()
    assert [row["species"] for row in rows] == ["Gadus morhua"]
    assert rows[0]["origin"] == "catalogue"


# --- Assigning a name ------------------------------------------------------


def test_assigning_a_species_records_it_beside_the_machines_answers(
    client, db_session_factory
):
    with db_session_factory() as db:
        track = _stored_track(db, _stored_video(db))
        track.species = "fish"
        track.fishial_species = "Pollachius virens"
        track.fishial_state = "identified"
        track.fishial_species_confidence = 0.51
        db.commit()
        track_id = track.id

    response = client.patch(f"/tracks/{track_id}/species", json={"species": "Gadus morhua"})
    assert response.status_code == 200
    body = response.json()
    assert body["manual_species"] == "Gadus morhua"
    assert body["manual_species_at"] is not None
    # Neither machine was overwritten; all three claims are still readable.
    assert body["species"] == "fish"
    assert body["fishial_species"] == "Pollachius virens"
    assert body["fishial_state"] == "identified"


def test_clearing_a_species_restores_the_machine_answer_without_having_lost_it(
    client, db_session_factory
):
    with db_session_factory() as db:
        track = _stored_track(db, _stored_video(db))
        track.fishial_species = "Pollachius virens"
        db.commit()
        track_id = track.id

    client.patch(f"/tracks/{track_id}/species", json={"species": "Gadus morhua"})
    body = client.patch(f"/tracks/{track_id}/species", json={"species": None}).json()
    assert body["manual_species"] is None
    assert body["manual_species_at"] is None
    assert body["fishial_species"] == "Pollachius virens"


def test_an_assigned_name_reaches_the_track_table_and_the_single_track(
    client, db_session_factory
):
    with db_session_factory() as db:
        video = _stored_video(db)
        track = _stored_track(db, video)
        db.commit()
        video_id, track_id = video.id, track.id

    client.patch(f"/tracks/{track_id}/species", json={"species": "Zeus faber"})
    rows = client.get(f"/videos/{video_id}/tracks").json()
    assert [row["manual_species"] for row in rows] == ["Zeus faber"]
    assert client.get(f"/tracks/{track_id}").json()["manual_species"] == "Zeus faber"


@pytest.mark.parametrize(
    "name",
    [
        "  Gadus   morhua  ",  # collapsed, not merely stripped
        "Raja sp.",  # open nomenclature is a real answer
        "Montagu's blenny",
    ],
)
def test_a_name_is_tidied_rather_than_second_guessed(client, db_session_factory, name):
    with db_session_factory() as db:
        track_id = _stored_track(db, _stored_video(db)).id

    stored = client.patch(f"/tracks/{track_id}/species", json={"species": name}).json()
    assert stored["manual_species"] == " ".join(name.split())


@pytest.mark.parametrize("name", ["<script>x</script>", "12345", "a" * 300, "one two three four five"])
def test_something_that_is_not_a_name_is_refused(client, db_session_factory, name):
    with db_session_factory() as db:
        track_id = _stored_track(db, _stored_video(db)).id

    assert client.patch(f"/tracks/{track_id}/species", json={"species": name}).status_code == 422


def test_assigning_a_species_to_a_live_fish_works_the_same_way(client, db_session_factory):
    with db_session_factory() as db:
        session = LiveMonitorSession(source_key="smartbay-cam1", status="running",
                                     source_url="rtsp://example/cam1")
        db.add(session)
        db.flush()
        track = LiveFishTrack(
            session_id=session.id, status="active", first_seen_at=utc_now(),
            last_seen_at=utc_now(), detection_count=3, max_confidence=0.8,
            mean_confidence=0.7, x1=1, y1=1, x2=100, y2=100,
            fishial_state="review_required",
            fishial_votes_json=json.dumps({"reason": "consensus not reached"}),
        )
        db.add(track)
        db.commit()
        track_id = track.id

    body = client.patch(f"/live/tracks/{track_id}/species",
                        json={"species": "Labrus bergylta"}).json()
    assert body["manual_species"] == "Labrus bergylta"
    # The abstention is still there to read: a person deciding is not the classifier
    # having changed its mind.
    assert body["fishial_state"] == "review_required"


def test_assigning_to_a_fish_that_does_not_exist_is_a_404(client):
    missing = uuid.uuid4()
    assert client.patch(f"/tracks/{missing}/species", json={"species": "Zeus faber"}).status_code == 404
    assert client.patch(f"/live/tracks/{missing}/species",
                        json={"species": "Zeus faber"}).status_code == 404


# --- The gallery -----------------------------------------------------------


def test_gallery_of_an_unknown_species_is_empty_rather_than_an_error(client):
    body = client.get("/species/gallery", params={"name": "Gadus morhua"}).json()
    assert body == {
        "species": "Gadus morhua", "slug": "gadus-morhua",
        "common_name": "Atlantic cod", "photos": [],
    }


def test_gallery_leads_with_the_portrait_already_on_screen(test_settings, monkeypatch):
    """Opening the viewer enlarges the photo shown elsewhere, never a different fish."""

    root = species_reference.reference_root(test_settings)
    root.mkdir(parents=True, exist_ok=True)
    (root / "gadus-morhua.jpg").write_bytes(b"\xff\xd8\xffprimary")
    (root / "gadus-morhua-2.jpg").write_bytes(b"\xff\xd8\xffsecond")
    (root / "gadus-morhua.json").write_text(json.dumps({
        "species": "Gadus morhua", "state": "found", "image_filename": "gadus-morhua.jpg",
        "media_type": "image/jpeg", "attribution": "Someone", "licence": "cc-by",
        "gallery": [{"filename": "gadus-morhua-2.jpg", "media_type": "image/jpeg",
                     "attribution": "Someone else", "licence": "cc0"}],
    }), encoding="utf-8")

    photos = species_reference.gallery("Gadus morhua", test_settings)
    assert [photo.filename for photo in photos] == [
        "gadus-morhua.jpg", "gadus-morhua-2.jpg",
    ]
    assert [photo.attribution for photo in photos] == ["Someone", "Someone else"]


def test_a_gallery_entry_whose_file_is_gone_is_skipped_not_served_broken(test_settings):
    root = species_reference.reference_root(test_settings)
    root.mkdir(parents=True, exist_ok=True)
    (root / "zeus-faber.jpg").write_bytes(b"\xff\xd8\xffprimary")
    (root / "zeus-faber.json").write_text(json.dumps({
        "species": "Zeus faber", "state": "found", "image_filename": "zeus-faber.jpg",
        "media_type": "image/jpeg",
        "gallery": [{"filename": "zeus-faber-2.jpg", "media_type": "image/jpeg"}],
    }), encoding="utf-8")

    photos = species_reference.gallery("Zeus faber", test_settings)
    assert [photo.filename for photo in photos] == ["zeus-faber.jpg"]


def test_a_gallery_photo_is_only_served_when_its_own_sidecar_claims_it(test_settings):
    """The filename arrives from a URL, so it is never trusted to belong here."""

    root = species_reference.reference_root(test_settings)
    root.mkdir(parents=True, exist_ok=True)
    (root / "zeus-faber-2.jpg").write_bytes(b"\xff\xd8\xffsecond")
    (root / "zeus-faber.json").write_text(json.dumps({
        "species": "Zeus faber", "state": "found", "image_filename": "zeus-faber.jpg",
        "gallery": [{"filename": "zeus-faber-2.jpg"}],
    }), encoding="utf-8")

    assert species_reference.gallery_image_path(
        "zeus-faber", "zeus-faber-2.jpg", test_settings) is not None
    # Not listed by this species, not inside the root, or not a slug at all.
    for slug, filename in (
        ("zeus-faber", "zeus-faber-3.jpg"),
        ("zeus-faber", "../../etc/passwd"),
        ("gadus-morhua", "zeus-faber-2.jpg"),
        ("../zeus-faber", "zeus-faber-2.jpg"),
    ):
        assert species_reference.gallery_image_path(slug, filename, test_settings) is None


# --- Fixtures --------------------------------------------------------------


def _stored_video(db) -> Video:
    video = Video(
        original_filename="source.mp4", storage_path="source.mp4", fps=10.0,
        model_name="m", pipeline_name="p", confidence_threshold=0.5,
        processing_status=VideoProcessingStatus.COMPLETED.value,
    )
    db.add(video)
    db.commit()
    return video


def _stored_track(db, video: Video) -> FishTrack:
    track = FishTrack(
        video_id=video.id, viame_track_id=uuid.uuid4().hex[:8], first_frame=0,
        last_frame=9, detection_count=4, mean_confidence=0.8, max_confidence=0.9,
    )
    db.add(track)
    db.commit()
    return track
