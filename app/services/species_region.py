"""Local regional plausibility lists. No network, no dependency, no new egress.

Reject the impossible for free. A regional filter removes a whole class of wrong
label before voting, at zero API cost: the SmartBay 3 reference session's only two
Fishial candidates were *Pomatomus saltatrix* and *Sparisoma aurofrenatum* - a rare
southern vagrant and a Caribbean parrotfish - on a Galway Bay camera.

Matching is on ``scientificName``, normalised for case and surrounding whitespace
only. No fuzzy or genus-level matching: a near miss must abstain, not guess.
"""
from __future__ import annotations

from app.services.species_catalogue import CATALOGUES

# The names themselves live in :mod:`app.services.species_catalogue`, beside the
# common names an operator searches by. One list, two uses: what a camera here
# could plausibly see is both what a classifier may be believed about and what a
# person may choose from, and keeping two copies of that judgement would let them
# disagree.

REGIONS: dict[str, frozenset[str]] = {
    region: frozenset(species.strip().lower() for species, _ in catalogue)
    for region, catalogue in CATALOGUES.items()
}


def region_names(region: str | None) -> frozenset[str] | None:
    """Accepted names for ``region``; ``None`` means "accept everything".

    An unset or unknown region never filters, so a camera we have no curated list
    for cannot regress.
    """

    return REGIONS.get(region) if region else None


def filter_species(ranked, region: str | None) -> tuple[list, list[str]]:
    """Split ``[(scientific_name, score), ...]`` into kept pairs and dropped names."""

    accepted = region_names(region)
    if accepted is None:
        return list(ranked), []
    kept, dropped = [], []
    for name, score in ranked:
        if str(name).strip().lower() in accepted:
            kept.append((name, score))
        else:
            dropped.append(name)
    return kept, dropped
