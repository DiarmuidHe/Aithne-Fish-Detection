"""One reference photo per species, so a name can be checked against an animal.

A classifier returns *Gadus morhua*. An operator who cannot picture a cod has been
given a claim with no way to test it, and the honest reaction to an untestable
claim is to believe it. Putting a photo of the species beside the name turns the
identification back into a judgement: the fish on screen either looks like that or
it does not, and the operator can say which without leaving the page.

What this is not: it is not evidence, and it is never the *detected* fish. It is a
stock portrait of the named species, cached once and shown identically wherever
that name appears. The distinction matters enough that every surface labels it.

Cost and provenance shape the design:

* one lookup per species name, ever - not per fish, per track or per page. The
  answer is written to disk beside a sidecar naming the photographer, the licence
  and the source, and later requests are plain file reads;
* only openly licensed photos are stored. An all-rights-reserved photo is not ours
  to cache and re-serve, so such a species keeps no image rather than a hotlink;
* a name the source does not recognise is remembered as *missing* for a while, so
  a misspelling or a genus-only label costs one request a day and not one a view;
* a deployment with no egress can drop ``<slug>.jpg`` files into the root by hand
  and they are used as-is. Nothing here is required for the dashboard to work.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

# Hosts a photo may actually be downloaded from. The URL comes out of a third
# party's JSON, so it chooses the path but never the destination.
ALLOWED_IMAGE_HOSTS = frozenset({
    "inaturalist-open-data.s3.amazonaws.com",
    "static.inaturalist.org",
})

# Magic bytes to file suffix. What the source calls the file is not evidence of
# what it is; a browser is only ever handed something we have identified ourselves.
IMAGE_TYPES: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
)

SIDECAR_SUFFIX = ".json"
# A sidecar is small and hand-editable; anything larger is not one of ours.
MAX_SIDECAR_BYTES = 64 * 1024


@dataclass(frozen=True)
class SpeciesPhoto:
    """One stored photo of a species, with the credit it must be shown under."""

    filename: str
    media_type: str
    attribution: str | None = None
    licence: str | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class SpeciesReference:
    """What is known about one species name's portrait.

    ``state`` is the whole answer: ``found`` has an image, ``missing`` means the
    source had no openly licensed photo under that name, and ``unknown`` means we
    have not looked yet - a distinction the dashboard needs, because only the last
    one is worth asking about again.
    """

    species: str
    slug: str
    state: str = "unknown"
    common_name: str | None = None
    image_filename: str | None = None
    media_type: str | None = None
    attribution: str | None = None
    licence: str | None = None
    source: str | None = None
    source_url: str | None = None

    @property
    def has_image(self) -> bool:
        return self.state == "found" and bool(self.image_filename)


def slug_for(name: str) -> str:
    """A filesystem name for a species, stable across case and punctuation.

    Accents are folded rather than dropped so *Zeus faber* and a copy of the same
    name carrying a combining mark land on one file instead of two.
    """

    folded = unicodedata.normalize("NFKD", name or "")
    ascii_only = folded.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return cleaned[:96]


def reference_root(settings: Settings) -> Path:
    return settings.species_reference_root.expanduser().resolve()


def allowed_licences(settings: Settings) -> frozenset[str]:
    return frozenset(
        part.strip().lower()
        for part in settings.species_reference_allowed_licences.split(",")
        if part.strip()
    )


def image_path(slug: str, filename: str, settings: Settings) -> Path | None:
    """The stored photo for ``slug``, or ``None`` if the sidecar names anything else.

    The filename is read back off disk, so it is checked against the slug it is
    filed under rather than trusted to stay inside the root on its own.
    """

    if not slug or Path(filename).name != filename:
        return None
    stem, suffix = os.path.splitext(filename)
    if stem != slug or suffix not in {entry[1] for entry in IMAGE_TYPES}:
        return None
    return reference_root(settings) / filename


def cached(name: str, settings: Settings) -> SpeciesReference:
    """What is already on disk for ``name``, without going anywhere near a network."""

    slug = slug_for(name)
    if not slug:
        return SpeciesReference(species=name, slug="", state="missing")

    root = reference_root(settings)
    sidecar = root / f"{slug}{SIDECAR_SUFFIX}"
    record = _read_sidecar(sidecar)
    if record is not None:
        reference = _from_record(name, slug, record)
        if not reference.has_image:
            return reference
        stored = image_path(slug, reference.image_filename or "", settings)
        if stored is not None and stored.is_file():
            return reference
        # The sidecar outlived its photo; treat the pair as never fetched.
        return SpeciesReference(species=name, slug=slug)

    # A photo dropped in by hand, with no sidecar. Deliberately supported: it is
    # how a deployment without egress supplies its own reference set.
    for suffix, media_type in ((entry[1], entry[2]) for entry in IMAGE_TYPES):
        candidate = root / f"{slug}{suffix}"
        if candidate.is_file():
            return SpeciesReference(
                species=name, slug=slug, state="found",
                image_filename=candidate.name, media_type=media_type, source="local",
            )
    return SpeciesReference(species=name, slug=slug)


def resolve(names: list[str], settings: Settings) -> list[SpeciesReference]:
    """Every distinct name's photo, fetching at most the configured few.

    Names already on disk cost nothing. Uncached ones are fetched up to the cap and
    the remainder come back ``unknown``, which the dashboard renders as a nameplate
    with no photo - the next request picks them up. A page is never held open on an
    unbounded number of third-party lookups.
    """

    seen: dict[str, str] = {}
    for name in names:
        slug = slug_for(name)
        if slug and slug not in seen:
            seen[slug] = name.strip()

    budget = settings.species_reference_max_lookups_per_request
    results: list[SpeciesReference] = []
    for name in seen.values():
        reference = cached(name, settings)
        if reference.state == "unknown" and settings.species_reference_enabled and budget > 0:
            budget -= 1
            reference = fetch(name, settings)
        results.append(reference)
    return results


def fetch(name: str, settings: Settings) -> SpeciesReference:
    """Look ``name`` up once and write the answer down, whatever the answer is.

    Never raises. A species with no photo is a cosmetic gap; a dashboard that fails
    to load a track table because a photo service was slow is an outage.
    """

    slug = slug_for(name)
    if not slug or not settings.species_reference_enabled:
        return SpeciesReference(species=name, slug=slug, state="missing")

    root = reference_root(settings)
    sidecar = root / f"{slug}{SIDECAR_SUFFIX}"
    record = _read_sidecar(sidecar)
    if record is not None and not _may_retry(record, settings):
        return _from_record(name, slug, record)

    try:
        found = _lookup(name, settings)
    except (httpx.HTTPError, ValueError, OSError) as exc:
        logger.info("Species reference lookup failed for %r: %s", name, type(exc).__name__)
        found = None

    if found is None:
        _write_sidecar(sidecar, {"species": name, "state": "missing"})
        return SpeciesReference(species=name, slug=slug, state="missing")

    record = {**found, "species": name, "state": "found"}
    _write_sidecar(sidecar, record)
    return _from_record(name, slug, record)


# --- The gallery -----------------------------------------------------------


def gallery_image_path(slug: str, filename: str, settings: Settings) -> Path | None:
    """A stored extra photo, checked against the sidecar that claims to own it.

    The filename arrives from a URL, so it is never trusted to be inside the root
    or to belong to this species: it has to appear in ``slug``'s own sidecar, which
    only this module writes.
    """

    if not slug or slug_for(slug) != slug or Path(filename).name != filename:
        return None
    record = _read_sidecar(reference_root(settings) / f"{slug}{SIDECAR_SUFFIX}")
    known = {entry.get("filename") for entry in (record or {}).get("gallery") or []}
    if filename not in known:
        return None
    return reference_root(settings) / filename


def media_type_for(filename: str) -> str:
    """What a stored photo is, decided by its suffix - which only we ever choose.

    Every file in the root was written with the suffix its own magic bytes earned,
    so reading it back off the name is reading back our own identification.
    """

    suffix = os.path.splitext(filename)[1].lower()
    for _, known, media_type in IMAGE_TYPES:
        if suffix == known:
            return media_type
    return "application/octet-stream"


def gallery(name: str, settings: Settings) -> list[SpeciesPhoto]:
    """Every photo we hold of ``name``, the portrait first, fetching extras once.

    An operator comparing two similar wrasse cannot do it from one stock photo at
    one angle, so the viewer is given the whole set. The extras cost a request and
    some disk, and they are bought only when somebody actually opens the viewer -
    never while drawing a table, which is where the single portrait earns its keep.

    Never raises, for the same reason :func:`fetch` does not: a thin gallery is a
    smaller loss than a screen that fails to open.
    """

    reference = cached(name, settings)
    if not reference.has_image:
        return []
    primary = SpeciesPhoto(
        filename=reference.image_filename or "",
        media_type=reference.media_type or "image/jpeg",
        attribution=reference.attribution,
        licence=reference.licence,
        source_url=reference.source_url,
    )

    slug = reference.slug
    sidecar = reference_root(settings) / f"{slug}{SIDECAR_SUFFIX}"
    record = _read_sidecar(sidecar) or {}
    stored = record.get("gallery")
    if stored is None and settings.species_reference_enabled:
        try:
            stored = _fetch_gallery(slug, record, settings)
        except (httpx.HTTPError, ValueError, OSError) as exc:
            # No answer written: a failed lookup is retried next time the viewer is
            # opened, unlike a genuine "this species has one photo", which is final.
            logger.info("Species gallery lookup failed for %r: %s", name, type(exc).__name__)
            stored = None
        else:
            _write_sidecar(sidecar, {**record, "gallery": stored})

    extras = [
        SpeciesPhoto(
            filename=entry["filename"],
            media_type=entry.get("media_type") or "image/jpeg",
            attribution=_clean(entry.get("attribution")),
            licence=_clean(entry.get("licence")),
            source_url=_clean(entry.get("source_url")) or reference.source_url,
        )
        for entry in stored or []
        if isinstance(entry, dict)
        and isinstance(entry.get("filename"), str)
        # A sidecar can outlive its files; an entry with nothing on disk is skipped
        # rather than served as a broken image.
        and (reference_root(settings) / entry["filename"]).is_file()
    ]
    return [primary, *extras]


def _fetch_gallery(slug: str, record: dict, settings: Settings) -> list[dict]:
    """Download the rest of this taxon's openly licensed photos, and file them.

    Returns the entries to record - an empty list is a real answer, meaning the
    source has nothing more we are entitled to keep.
    """

    taxon_id = _taxon_id(record)
    wanted = settings.species_reference_gallery_size
    if not taxon_id or wanted <= 0:
        return []

    root = reference_root(settings)
    primary = root / (record.get("image_filename") or "")
    primary_bytes = primary.read_bytes() if primary.is_file() else b""

    base = settings.species_reference_api_base_url.rstrip("/")
    licences = allowed_licences(settings)
    entries: list[dict] = []
    with _client(settings) as client:
        detail = _json(client, f"{base}/taxa/{taxon_id}", None)
        results = (detail or {}).get("results") or [{}]
        for photo in _usable_photos(results[0].get("taxon_photos"), licences, wanted + 1):
            try:
                image, media_type, suffix = _download(client, photo["url"], settings)
            except (httpx.HTTPError, ValueError):
                # One unreachable photo does not spoil the rest of the set.
                continue
            # The portrait is already the first entry of every gallery; the source
            # lists it again here, and showing it twice would look like a bug.
            if image == primary_bytes:
                continue
            filename = f"{slug}-{len(entries) + 2}{suffix}"
            _write_image(root / filename, image)
            entries.append({
                "filename": filename,
                "media_type": media_type,
                "attribution": _clean(photo["attribution"]),
                "licence": photo["licence"],
            })
            if len(entries) >= wanted:
                break
    return entries


def _taxon_id(record: dict) -> str | None:
    """The source's id for this species, from the sidecar or its recorded URL.

    Sidecars written before galleries existed carry only the taxon page URL, and
    re-looking-up a name we already resolved to save reading its last path segment
    would be a request spent on nothing.
    """

    value = record.get("taxon_id")
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return str(value)
    source_url = record.get("source_url")
    if not isinstance(source_url, str):
        return None
    tail = urlparse(source_url).path.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.isdigit() else None


# --- The source ------------------------------------------------------------


def _client(settings: Settings) -> httpx.Client:
    """The one place a connection is opened, so a test can supply its own."""

    return httpx.Client(
        timeout=settings.species_reference_timeout_seconds, follow_redirects=True
    )


def _lookup(name: str, settings: Settings) -> dict | None:
    """iNaturalist's taxon record for an exact species name, and a usable photo.

    Exact matching only, on the scientific name the classifier returned. A search
    engine's best guess for a name it does not hold would put a photo of the wrong
    animal beside a name an operator is being asked to check, which is worse than
    no photo at all.

    A taxon's *default* photo is whichever one the community ranked first, and it
    is often all-rights-reserved - Ballan wrasse's is. So a second request reads the
    taxon's full photo list and takes the first openly licensed one, which is why a
    common North Atlantic species has a portrait at all. That request is made only
    when the free one will not do, and only once per species.
    """

    base = settings.species_reference_api_base_url.rstrip("/")
    licences = allowed_licences(settings)
    with _client(settings) as client:
        payload = _json(client, f"{base}/taxa", {
            "q": name, "rank": "species", "per_page": 5, "is_active": "true",
        })
        taxon = _exact_taxon((payload or {}).get("results"), name)
        if taxon is None:
            return None

        photo = _usable_photo([{"photo": taxon.get("default_photo")}], licences)
        taxon_id = taxon.get("id")
        if photo is None and taxon_id:
            detail = _json(client, f"{base}/taxa/{taxon_id}", None)
            results = (detail or {}).get("results") or [{}]
            photo = _usable_photo(results[0].get("taxon_photos"), licences)
        if photo is None:
            return None

        image, media_type, suffix = _download(client, photo["url"], settings)

    filename = f"{slug_for(name)}{suffix}"
    _write_image(reference_root(settings) / filename, image)
    return {
        "common_name": _clean(taxon.get("preferred_common_name")),
        # Kept so a later gallery request can ask about this exact taxon rather
        # than searching the name a second time.
        "taxon_id": taxon_id,
        "image_filename": filename,
        "media_type": media_type,
        "attribution": _clean(photo["attribution"]),
        "licence": photo["licence"],
        "source": "inaturalist",
        "source_url": f"https://www.inaturalist.org/taxa/{taxon_id}" if taxon_id else None,
    }


def _json(client: httpx.Client, url: str, params: dict | None) -> dict | None:
    response = client.get(url, params=params, headers={"Accept": "application/json"})
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def _usable_photo(entries: object, licences: frozenset[str]) -> dict | None:
    """The first photo we are allowed to keep, from a list of ``{"photo": ...}``.

    Order is iNaturalist's own, which is roughly best-first, so taking the first
    acceptable one gives the best portrait we are entitled to rather than the best
    portrait there is.
    """

    found = _usable_photos(entries, licences, 1)
    return found[0] if found else None


def _usable_photos(entries: object, licences: frozenset[str], limit: int) -> list[dict]:
    """Up to ``limit`` photos we are allowed to keep, in the source's own order."""

    if not isinstance(entries, list):
        return []
    usable: list[dict] = []
    for entry in entries:
        photo = entry.get("photo") if isinstance(entry, dict) else None
        if not isinstance(photo, dict):
            continue
        licence = str(photo.get("license_code") or "").lower()
        if licence not in licences:
            continue
        url = _photo_url(photo)
        if url is None:
            continue
        usable.append({"url": url, "licence": licence, "attribution": photo.get("attribution")})
        if len(usable) >= limit:
            break
    return usable


def _exact_taxon(results: object, name: str) -> dict | None:
    wanted = " ".join((name or "").split()).lower()
    if not isinstance(results, list):
        return None
    for entry in results:
        if isinstance(entry, dict) and str(entry.get("name", "")).strip().lower() == wanted:
            return entry
    return None


def _photo_url(photo: dict) -> str | None:
    """The medium rendition, on a host we are willing to fetch from."""

    for key in ("medium_url", "url", "square_url"):
        value = photo.get(key)
        if not isinstance(value, str) or not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme == "https" and parsed.hostname in ALLOWED_IMAGE_HOSTS:
            return value
    return None


def _download(client: httpx.Client, url: str, settings: Settings) -> tuple[bytes, str, str]:
    """The photo's bytes, refused unless they are an image and small enough."""

    limit = settings.species_reference_max_bytes
    with client.stream("GET", url) as response:
        response.raise_for_status()
        chunks, total = [], 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > limit:
                raise ValueError("reference photo is too large")
            chunks.append(chunk)
    image = b"".join(chunks)
    for magic, suffix, media_type in IMAGE_TYPES:
        if image.startswith(magic):
            return image, media_type, suffix
    raise ValueError("reference photo is not an image")


# --- Disk ------------------------------------------------------------------


def _write_image(path: Path, image: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
    temp.write_bytes(image)
    os.replace(temp, path)


def _write_sidecar(path: Path, record: dict) -> None:
    record = {**record, "checked_at": time.time()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        # An unwritable cache means we look the name up again next time, nothing more.
        logger.warning("Species reference sidecar could not be written to %s", path)


def _read_sidecar(path: Path) -> dict | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_SIDECAR_BYTES:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _may_retry(record: dict, settings: Settings) -> bool:
    """Only a past failure is ever retried, and only once the cool-off has passed."""

    if record.get("state") == "found":
        return False
    after = settings.species_reference_retry_after_seconds
    if after <= 0:
        return True
    checked_at = record.get("checked_at")
    if not isinstance(checked_at, (int, float)):
        return True
    return (time.time() - float(checked_at)) >= after


def _from_record(name: str, slug: str, record: dict) -> SpeciesReference:
    state = record.get("state")
    if state != "found":
        return SpeciesReference(species=name, slug=slug, state="missing")
    return SpeciesReference(
        species=name,
        slug=slug,
        state="found",
        common_name=_clean(record.get("common_name")),
        image_filename=_clean(record.get("image_filename")),
        media_type=_clean(record.get("media_type")),
        attribution=_clean(record.get("attribution")),
        licence=_clean(record.get("licence")),
        source=_clean(record.get("source")),
        source_url=_clean(record.get("source_url")),
    )


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text[:400] or None
