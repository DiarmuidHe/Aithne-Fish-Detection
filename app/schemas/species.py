from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SpeciesReferenceRead(BaseModel):
    """A stock photo of what a named species looks like - never a detected fish.

    ``state`` separates "there is no photo for this name" from "we have not looked
    yet", because only the second is worth asking about again. A client showing
    either one draws the name on its own.
    """

    species: str
    slug: str
    state: Literal["found", "missing", "unknown"]
    common_name: str | None = None
    image_url: str | None = None
    attribution: str | None = None
    licence: str | None = None
    source: str | None = None
    source_url: str | None = None


class SpeciesSearchRead(SpeciesReferenceRead):
    """A species an operator may choose, with whatever portrait we already hold.

    The same shape as a reference so one component draws both, plus where the name
    came from: ``catalogue`` is a curated local species, ``observed`` is a name
    this deployment has already recorded and can therefore be chosen again even
    though no curated list contains it.
    """

    origin: Literal["catalogue", "observed"] = "catalogue"


class SpeciesPhotoRead(BaseModel):
    """One photo in a species' gallery, with the credit it must be shown under."""

    url: str
    media_type: str
    attribution: str | None = None
    licence: str | None = None
    source_url: str | None = None


class SpeciesGalleryRead(BaseModel):
    """Every reference photo held for one species, the portrait first.

    Never the detected fish - these are stock photos of the named animal, and the
    viewer that shows them says so.
    """

    species: str
    slug: str
    common_name: str | None = None
    photos: list[SpeciesPhotoRead] = []
