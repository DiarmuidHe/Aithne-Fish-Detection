from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import FishTrack, LiveFishTrack
from app.schemas.species import (
    SpeciesGalleryRead,
    SpeciesPhotoRead,
    SpeciesReferenceRead,
    SpeciesSearchRead,
)
from app.services import species_catalogue, species_reference

router = APIRouter(prefix="/species", tags=["species"])

# One request asks about the names on one screen. A caller wanting more than this
# is not drawing a table, and the cap keeps a single URL from being a work queue.
MAX_NAMES_PER_REQUEST = 50

# Below this, a query matches most of the catalogue and the answer is noise. No
# fish here is named in fewer letters, so nothing is lost. The client holds the
# same floor and does not send the request at all; the server repeats it, because
# a floor only the client enforces is not a floor.
MIN_QUERY_LENGTH = 3


@router.get("/reference", response_model=list[SpeciesReferenceRead])
def get_species_references(
    name: list[str] = Query(default_factory=list, description="Scientific species names."),
    settings: Settings = Depends(get_settings),
) -> list[SpeciesReferenceRead]:
    """What each named species looks like, so an operator can check the naming.

    Cached names are read from disk; a few uncached ones are looked up now and the
    rest come back ``unknown`` for the next request to pick up. See
    :mod:`app.services.species_reference` for why the work is bounded this way.
    """

    references = species_reference.resolve(name[:MAX_NAMES_PER_REQUEST], settings)
    return [_read(reference) for reference in references]


@router.get("/search", response_model=list[SpeciesSearchRead])
def search_species(
    q: str = Query(description="Part of a common or scientific name."),
    limit: int = Query(default=24, ge=1, le=MAX_NAMES_PER_REQUEST),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[SpeciesSearchRead]:
    """Species an operator may assign to a fish, best match first.

    Two sources, in that order of trust: the curated list for these waters, and
    names this deployment has already recorded. The second matters because a
    classifier can return a species no curated list holds, and an operator who has
    seen that name on one fish must be able to put it on another rather than being
    told it does not exist.

    Portraits ride along where they are already cached, and the same per-request
    lookup budget as ``/species/reference`` applies, so a search never turns into a
    burst of third-party calls.
    """

    query = q.strip()
    if len(species_catalogue.fold(query)) < MIN_QUERY_LENGTH:
        return []

    found = [(entry.species, "catalogue") for entry in species_catalogue.search(query, limit)]
    known = {species.lower() for species, _ in found}
    for species in _observed_names(db, query, limit - len(found)):
        if species.lower() not in known:
            found.append((species, "observed"))

    references = {
        reference.slug: reference
        for reference in species_reference.resolve([species for species, _ in found], settings)
    }
    results: list[SpeciesSearchRead] = []
    for species, origin in found:
        reference = references.get(species_reference.slug_for(species))
        if reference is None:
            continue
        # `resolve` answers per slug, so the name echoed back is the one asked for
        # rather than whichever spelling happened to reach the cache first.
        record = _read(reference).model_dump() | {"species": species}
        results.append(SpeciesSearchRead(**record, origin=origin))
    return results


def _observed_names(db: Session, query: str, limit: int) -> list[str]:
    """Distinct species names this deployment has already recorded, matching ``query``.

    Matched in Python rather than SQL so the ranking and the fold are the ones the
    catalogue uses; the candidate set is the distinct names a deployment has ever
    seen, which is tens of rows, not a table scan of every track.
    """

    if limit <= 0:
        return []
    columns = (
        FishTrack.species, FishTrack.fishial_species, FishTrack.manual_species,
        LiveFishTrack.species, LiveFishTrack.fishial_species, LiveFishTrack.manual_species,
    )
    names: dict[str, None] = {}
    for column in columns:
        for value in db.scalars(select(column).where(column.is_not(None)).distinct()).all():
            names.setdefault(" ".join(str(value).split()), None)

    wanted = species_catalogue.fold(query)
    matched = sorted(name for name in names if wanted in species_catalogue.fold(name))
    return matched[:limit]


@router.get("/gallery", response_model=SpeciesGalleryRead)
def get_species_gallery(
    name: str = Query(description="One scientific species name."),
    settings: Settings = Depends(get_settings),
) -> SpeciesGalleryRead:
    """Every reference photo we hold of one species, for a closer look.

    Asked for only when an operator opens the viewer, because this is the one call
    that may buy several photos. The portrait comes first and is always the one
    already shown elsewhere, so opening the viewer enlarges what was on screen
    rather than replacing it with a different animal.
    """

    species = " ".join(name.split())
    slug = species_reference.slug_for(species)
    if not slug:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "That is not a species name")
    return SpeciesGalleryRead(
        species=species,
        slug=slug,
        common_name=(
            species_catalogue.common_name(species)
            or species_reference.cached(species, settings).common_name
        ),
        photos=[
            SpeciesPhotoRead(
                url=f"/species/reference/{slug}/photo/{photo.filename}",
                media_type=photo.media_type,
                attribution=photo.attribution,
                licence=photo.licence,
                source_url=photo.source_url,
            )
            for photo in species_reference.gallery(species, settings)
        ],
    )


@router.get("/reference/{slug}/photo/{filename}")
def get_species_gallery_image(
    slug: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """One photo from a species' gallery. The slug names the sidecar that owns it."""

    path = species_reference.gallery_image_path(slug, filename, settings)
    if path is None or not path.is_file():
        # The primary portrait is served by the route below; a gallery URL that
        # names it would otherwise 404 confusingly.
        path = species_reference.image_path(slug, filename, settings)
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such reference photo")
    return FileResponse(
        path,
        media_type=species_reference.media_type_for(filename),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/reference/{filename}")
def get_species_reference_image(
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """One cached reference photo. Filenames are slugs, so they are stable."""

    stem, _ = os.path.splitext(filename)
    path = species_reference.image_path(stem, filename, settings)
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No reference photo for this species")

    record = species_reference.cached(stem, settings)
    return FileResponse(
        path,
        media_type=record.media_type or "image/jpeg",
        # A species' portrait is refreshed only by clearing the cache on the server,
        # so a browser holding one for a day costs nothing and saves every page view.
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _read(reference: species_reference.SpeciesReference) -> SpeciesReferenceRead:
    return SpeciesReferenceRead(
        species=reference.species,
        slug=reference.slug,
        state=reference.state,  # type: ignore[arg-type]
        # The curated common name wins over the photo source's, everywhere a
        # reference is read. The source gives whichever name its own community
        # prefers, and for a fish named differently on either side of the Atlantic
        # that is not the name the operator reading this screen uses: it calls
        # Pollachius virens "Pollock", which on these coasts is saithe and is one
        # letter from Pollachius pollachius, a different fish on the same list.
        common_name=(
            species_catalogue.common_name(reference.species) or reference.common_name
        ),
        image_url=(
            f"/species/reference/{reference.image_filename}" if reference.has_image else None
        ),
        attribution=reference.attribution,
        licence=reference.licence,
        source=reference.source,
        source_url=reference.source_url,
    )
