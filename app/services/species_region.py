"""Local regional plausibility lists. No network, no dependency, no new egress.

Reject the impossible for free. A regional filter removes a whole class of wrong
label before voting, at zero API cost: the SmartBay 3 reference session's only two
Fishial candidates were *Pomatomus saltatrix* and *Sparisoma aurofrenatum* - a rare
southern vagrant and a Caribbean parrotfish - on a Galway Bay camera.

Matching is on ``scientificName``, normalised for case and surrounding whitespace
only. No fuzzy or genus-level matching: a near miss must abstain, not guess.
"""
from __future__ import annotations

# Source: the Irish Specimen Fish Committee's list of recognised species combined
# with FishBase's Ireland country checklist, restricted to coastal and shelf species
# a fixed camera in Galway Bay could plausibly see. Deliberately generous - the
# filter exists to reject tropical and other-ocean labels, not to second-guess a
# plausible local record.
NORTH_EAST_ATLANTIC = (
    # Gadoids
    "Gadus morhua", "Melanogrammus aeglefinus", "Pollachius pollachius",
    "Pollachius virens", "Merlangius merlangus", "Trisopterus luscus",
    "Trisopterus minutus", "Trisopterus esmarkii", "Micromesistius poutassou",
    "Molva molva", "Merluccius merluccius", "Raniceps raninus",
    "Ciliata mustela", "Ciliata septentrionalis", "Gaidropsarus vulgaris",
    "Gaidropsarus mediterraneus",
    # Flatfish
    "Pleuronectes platessa", "Platichthys flesus", "Limanda limanda",
    "Microstomus kitt", "Solea solea", "Scophthalmus maximus", "Scophthalmus rhombus",
    "Hippoglossoides platessoides", "Hippoglossus hippoglossus",
    "Buglossidium luteum", "Arnoglossus laterna", "Zeugopterus punctatus",
    "Phrynorhombus norvegicus",
    # Wrasse
    "Labrus bergylta", "Labrus mixtus", "Ctenolabrus rupestris",
    "Symphodus melops", "Centrolabrus exoletus",
    # Gobies, blennies and their relatives
    "Pomatoschistus minutus", "Pomatoschistus pictus", "Pomatoschistus microps",
    "Gobius niger", "Gobius paganellus", "Gobiusculus flavescens",
    "Thorogobius ephippiatus", "Lipophrys pholis", "Parablennius gattorugine",
    "Coryphoblennius galerita", "Chirolophis ascanii", "Pholis gunnellus",
    "Zoarces viviparus", "Anarhichas lupus",
    # Sculpins, gurnards, lumpsuckers and other scorpaeniforms
    "Myoxocephalus scorpius", "Taurulus bubalis", "Agonus cataphractus",
    "Cyclopterus lumpus", "Liparis liparis", "Liparis montagui",
    "Chelidonichthys lucerna", "Chelidonichthys cuculus", "Eutrigla gurnardus",
    "Sebastes norvegicus", "Sebastes viviparus",
    # Pelagic shoaling species
    "Scomber scombrus", "Clupea harengus", "Sprattus sprattus",
    "Sardina pilchardus", "Engraulis encrasicolus", "Trachurus trachurus",
    "Belone belone", "Scomberesox saurus", "Atherina presbyter",
    "Ammodytes tobianus", "Ammodytes marinus", "Hyperoplus lanceolatus",
    # Bass, mullet, breams
    "Dicentrarchus labrax", "Chelon labrosus", "Chelon auratus", "Chelon ramada",
    "Spondyliosoma cantharus", "Pagellus bogaraveo", "Pagrus pagrus",
    "Sparus aurata", "Boops boops", "Mullus surmuletus",
    # Sticklebacks, pipefish and seahorses
    "Gasterosteus aculeatus", "Spinachia spinachia", "Syngnathus acus",
    "Syngnathus typhle", "Syngnathus rostellatus", "Entelurus aequoreus",
    "Nerophis lumbriciformis", "Nerophis ophidion",
    "Hippocampus hippocampus", "Hippocampus guttulatus",
    # Eels, salmonids and diadromous species
    "Anguilla anguilla", "Conger conger", "Salmo salar", "Salmo trutta",
    "Osmerus eperlanus", "Alosa alosa", "Alosa fallax",
    "Petromyzon marinus", "Lampetra fluviatilis",
    # Sharks, skates and rays
    "Scyliorhinus canicula", "Scyliorhinus stellaris", "Mustelus asterias",
    "Galeorhinus galeus", "Squalus acanthias", "Cetorhinus maximus",
    "Prionace glauca", "Lamna nasus", "Squatina squatina",
    "Raja clavata", "Raja brachyura", "Raja montagui", "Raja microocellata",
    "Leucoraja naevus", "Dipturus batis", "Amblyraja radiata",
    "Torpedo nobiliana", "Dasyatis pastinaca",
    # Remaining coastal residents and regular visitors
    "Zeus faber", "Lophius piscatorius", "Callionymus lyra", "Callionymus maculatus",
    "Echiichthys vipera", "Trachinus draco", "Mola mola", "Balistes capriscus",
    "Trigla lyra", "Capros aper", "Maurolicus muelleri",
)

REGIONS: dict[str, frozenset[str]] = {
    "north_east_atlantic": frozenset(name.strip().lower() for name in NORTH_EAST_ATLANTIC),
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
