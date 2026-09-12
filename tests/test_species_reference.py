"""Reference portraits: cached once, openly licensed only, never guessed at.

The feature exists so an operator can disagree with a classifier. Every test here
protects that: a photo must be of the name that was asked about, must be one we are
allowed to keep, and must never cost more than one lookup per name.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.services import species_reference

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PHOTO_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/1/medium.jpg"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        species_reference_enabled=True,
        species_reference_root=tmp_path / "species-reference",
        species_reference_api_base_url="https://taxa.example/v1",
    )


def taxon(name="Gadus morhua", licence="cc-by", url=PHOTO_URL, common="Atlantic cod"):
    return {
        "id": 47178,
        "name": name,
        "preferred_common_name": common,
        "default_photo": {"medium_url": url, "attribution": "(c) someone, CC-BY",
                          "license_code": licence},
    }


def transport(monkeypatch, handler, settings: Settings):
    """Point the service's one connection seam at a scripted transport."""

    def client(_settings: Settings) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(species_reference, "_client", client)
    return settings


def scripted(results, image=JPEG, calls=None, taxon_photos=None):
    """The search endpoint, the taxon detail endpoint and a photo host."""

    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        if request.url.path.endswith("/taxa"):
            return httpx.Response(200, json={"results": results})
        if "/taxa/" in request.url.path:
            detail = dict(results[0], taxon_photos=taxon_photos or [])
            return httpx.Response(200, json={"results": [detail]})
        return httpx.Response(200, content=image)

    return handle


def test_slug_folds_case_accents_and_punctuation():
    assert species_reference.slug_for("Gadus morhua") == "gadus-morhua"
    assert species_reference.slug_for("  GADUS   MORHUA  ") == "gadus-morhua"
    assert species_reference.slug_for("Chelidonichthys lucérna") == "chelidonichthys-lucerna"
    assert species_reference.slug_for("../../etc/passwd") == "etc-passwd"
    assert species_reference.slug_for("") == ""


def test_fetch_stores_photo_and_provenance(monkeypatch, settings):
    calls: list[str] = []
    transport(monkeypatch, scripted([taxon()], calls=calls), settings)

    reference = species_reference.fetch("Gadus morhua", settings)

    assert reference.state == "found"
    assert reference.image_filename == "gadus-morhua.jpg"
    assert reference.common_name == "Atlantic cod"
    assert reference.attribution == "(c) someone, CC-BY"
    assert reference.licence == "cc-by"
    assert reference.source_url == "https://www.inaturalist.org/taxa/47178"

    stored = species_reference.reference_root(settings) / "gadus-morhua.jpg"
    assert stored.read_bytes() == JPEG
    sidecar = json.loads((stored.with_suffix(".json")).read_text(encoding="utf-8"))
    assert sidecar["state"] == "found" and "checked_at" in sidecar
    # No temporary file survives a successful write.
    assert not list(species_reference.reference_root(settings).glob("*.tmp"))
    assert len(calls) == 2


def test_a_cached_species_is_never_looked_up_again(monkeypatch, settings):
    calls: list[str] = []
    transport(monkeypatch, scripted([taxon()], calls=calls), settings)
    species_reference.fetch("Gadus morhua", settings)
    before = len(calls)

    # Case and spacing differences must land on the same cached answer, and the
    # cheap read must not open a connection at all.
    assert species_reference.cached("gadus  MORHUA", settings).state == "found"
    species_reference.resolve(["Gadus morhua", "gadus morhua"], settings)
    assert len(calls) == before


def test_an_all_rights_reserved_photo_is_not_cached(monkeypatch, settings):
    transport(monkeypatch, scripted([taxon(licence=None)]), settings)

    reference = species_reference.fetch("Gadus morhua", settings)

    assert reference.state == "missing" and reference.image_filename is None
    assert not list(species_reference.reference_root(settings).glob("*.jpg"))


def test_an_unusable_default_photo_falls_back_to_the_taxon_photo_list(monkeypatch, settings):
    """Ballan wrasse's top photo is all-rights-reserved; the species still has one."""

    calls: list[str] = []
    photos = [
        {"photo": {"medium_url": PHOTO_URL, "license_code": None,
                   "attribution": "(c) someone, all rights reserved"}},
        {"photo": {"medium_url": PHOTO_URL, "license_code": "cc-by-nc",
                   "attribution": "(c) another, CC BY-NC"}},
    ]
    transport(monkeypatch, scripted([taxon(licence=None)], calls=calls, taxon_photos=photos),
              settings)

    reference = species_reference.fetch("Gadus morhua", settings)

    assert reference.state == "found"
    assert reference.licence == "cc-by-nc"
    assert reference.attribution == "(c) another, CC BY-NC"


def test_the_photo_list_is_only_read_when_the_default_photo_will_not_do(monkeypatch, settings):
    calls: list[str] = []
    transport(monkeypatch, scripted([taxon()], calls=calls), settings)

    assert species_reference.fetch("Gadus morhua", settings).state == "found"
    assert not [url for url in calls if "/taxa/" in url], "no second taxon request was needed"


def test_a_near_match_is_refused_rather_than_illustrated_wrongly(monkeypatch, settings):
    """A search engine's best guess would put the wrong animal beside the name."""

    transport(monkeypatch, scripted([taxon(name="Gadus macrocephalus")]), settings)

    assert species_reference.fetch("Gadus morhua", settings).state == "missing"


def test_a_photo_from_an_unexpected_host_is_not_downloaded(monkeypatch, settings):
    transport(monkeypatch, scripted([taxon(url="https://elsewhere.example/photo.jpg")]), settings)

    assert species_reference.fetch("Gadus morhua", settings).state == "missing"


def test_something_that_is_not_an_image_is_refused(monkeypatch, settings):
    transport(monkeypatch, scripted([taxon()], image=b"<html>not a photo</html>"), settings)

    assert species_reference.fetch("Gadus morhua", settings).state == "missing"
    assert not list(species_reference.reference_root(settings).glob("*.jpg"))


def test_an_oversized_photo_is_refused(monkeypatch, settings):
    settings = settings.model_copy(update={"species_reference_max_bytes": 1024})
    transport(monkeypatch, scripted([taxon()], image=JPEG + b"\x00" * 4096), settings)

    assert species_reference.fetch("Gadus morhua", settings).state == "missing"


def test_a_missing_name_is_remembered_until_the_cool_off_passes(monkeypatch, settings):
    calls: list[str] = []
    transport(monkeypatch, scripted([], calls=calls), settings)

    assert species_reference.fetch("Notaspecies fictus", settings).state == "missing"
    assert len(calls) == 1

    species_reference.fetch("Notaspecies fictus", settings)
    assert len(calls) == 1, "a name known to be missing must not be asked about again"

    monkeypatch.setattr(species_reference.time, "time", lambda: 10**12)
    species_reference.fetch("Notaspecies fictus", settings)
    assert len(calls) == 2, "the cool-off must eventually expire"


def test_a_failing_source_costs_a_gap_and_nothing_else(monkeypatch, settings):
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    transport(monkeypatch, handle, settings)

    assert species_reference.fetch("Gadus morhua", settings).state == "missing"


def test_resolve_bounds_how_many_names_one_request_may_fetch(monkeypatch, settings):
    calls: list[str] = []
    settings = settings.model_copy(update={"species_reference_max_lookups_per_request": 2})
    transport(monkeypatch, scripted([], calls=calls), settings)

    references = species_reference.resolve(
        ["Gadus morhua", "Salmo salar", "Zeus faber", "Conger conger"], settings
    )

    assert [reference.state for reference in references] == [
        "missing", "missing", "unknown", "unknown",
    ]
    assert len(calls) == 2


def test_a_hand_placed_photo_is_used_without_any_network(settings):
    """How a deployment with no egress supplies its own reference set."""

    settings = settings.model_copy(update={"species_reference_enabled": False})
    root = species_reference.reference_root(settings)
    root.mkdir(parents=True)
    (root / "salmo-salar.png").write_bytes(PNG)

    reference = species_reference.resolve(["Salmo salar"], settings)[0]

    assert reference.state == "found"
    assert reference.media_type == "image/png"
    assert reference.source == "local"


def test_a_sidecar_naming_a_file_outside_the_root_serves_nothing(settings):
    root = species_reference.reference_root(settings)
    root.mkdir(parents=True)
    (root / "gadus-morhua.json").write_text(
        json.dumps({"state": "found", "image_filename": "../../secret.jpg"}), encoding="utf-8"
    )

    assert species_reference.image_path("gadus-morhua", "../../secret.jpg", settings) is None
    assert species_reference.cached("Gadus morhua", settings).state == "unknown"


# --- Through the API --------------------------------------------------------


def test_endpoint_reports_names_it_has_not_looked_up_as_unknown(client):
    response = client.get("/species/reference", params={"name": ["Gadus morhua"]})

    assert response.status_code == 200
    body = response.json()
    # The common name comes from the local catalogue, so it is known before any
    # photo is: the name can be printed while the portrait is still unresolved.
    assert body == [{
        "species": "Gadus morhua", "slug": "gadus-morhua", "state": "unknown",
        "common_name": "Atlantic cod", "image_url": None, "attribution": None,
        "licence": None, "source": None, "source_url": None,
    }]


def test_endpoint_serves_a_cached_photo(client, test_settings):
    root = species_reference.reference_root(test_settings)
    root.mkdir(parents=True, exist_ok=True)
    (root / "gadus-morhua.jpg").write_bytes(JPEG)

    listed = client.get("/species/reference", params={"name": "Gadus morhua"}).json()[0]
    assert listed["image_url"] == "/species/reference/gadus-morhua.jpg"

    image = client.get(listed["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content == JPEG


def test_endpoint_refuses_a_path_that_climbs_out_of_the_cache(client):
    assert client.get("/species/reference/gadus-morhua.txt").status_code == 404
    assert client.get("/species/reference/..%2F..%2Fsecret.jpg").status_code == 404
