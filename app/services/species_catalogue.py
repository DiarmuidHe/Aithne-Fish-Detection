"""The names an operator may choose from, and what each one is called in English.

A classifier proposes; a person decides. Deciding needs a list, and the list has
to be one an operator can search the way they think - "cod", "wrasse", "ray" -
rather than by recalling a binomial. So each scientific name is carried with its
common name, and both are searched.

The list is the same curated Irish coastal and shelf assemblage that
:mod:`app.services.species_region` filters against, and it lives here because the
two uses are one claim: these are the fish a camera in these waters could
plausibly be looking at. A name added for searching is therefore a name that
becomes regionally plausible, which is the honest consequence - a species an
operator may assign is a species we are saying could be there.

Local, and deliberately so. No network, no index, no service to be down: search
is a scan of a few hundred short strings, which at this size is faster than any
cleverness and cannot fail separately from the process.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Source: the Irish Specimen Fish Committee's list of recognised species combined
# with FishBase's Ireland country checklist, restricted to coastal and shelf species
# a fixed camera in Galway Bay could plausibly see. Deliberately generous - the
# regional filter exists to reject tropical and other-ocean labels, not to
# second-guess a plausible local record.
#
# Common names are the ones used on these coasts. Where a species is widely known
# by two, the one an operator is likelier to type is given.
NORTH_EAST_ATLANTIC: tuple[tuple[str, str], ...] = (
    # Gadoids
    ("Gadus morhua", "Atlantic cod"),
    ("Melanogrammus aeglefinus", "Haddock"),
    ("Pollachius pollachius", "Pollack"),
    ("Pollachius virens", "Saithe"),
    ("Merlangius merlangus", "Whiting"),
    ("Trisopterus luscus", "Pouting"),
    ("Trisopterus minutus", "Poor cod"),
    ("Trisopterus esmarkii", "Norway pout"),
    ("Micromesistius poutassou", "Blue whiting"),
    ("Molva molva", "Ling"),
    ("Merluccius merluccius", "European hake"),
    ("Raniceps raninus", "Tadpole fish"),
    ("Ciliata mustela", "Five-bearded rockling"),
    ("Ciliata septentrionalis", "Northern rockling"),
    ("Gaidropsarus vulgaris", "Three-bearded rockling"),
    ("Gaidropsarus mediterraneus", "Shore rockling"),
    # Flatfish
    ("Pleuronectes platessa", "European plaice"),
    ("Platichthys flesus", "European flounder"),
    ("Limanda limanda", "Common dab"),
    ("Microstomus kitt", "Lemon sole"),
    ("Solea solea", "Common sole"),
    ("Scophthalmus maximus", "Turbot"),
    ("Scophthalmus rhombus", "Brill"),
    ("Hippoglossoides platessoides", "Long rough dab"),
    ("Hippoglossus hippoglossus", "Atlantic halibut"),
    ("Buglossidium luteum", "Solenette"),
    ("Arnoglossus laterna", "Scaldfish"),
    ("Zeugopterus punctatus", "Topknot"),
    ("Phrynorhombus norvegicus", "Norwegian topknot"),
    # Wrasse
    ("Labrus bergylta", "Ballan wrasse"),
    ("Labrus mixtus", "Cuckoo wrasse"),
    ("Ctenolabrus rupestris", "Goldsinny wrasse"),
    ("Symphodus melops", "Corkwing wrasse"),
    ("Centrolabrus exoletus", "Rock cook"),
    # Gobies, blennies and their relatives
    ("Pomatoschistus minutus", "Sand goby"),
    ("Pomatoschistus pictus", "Painted goby"),
    ("Pomatoschistus microps", "Common goby"),
    ("Gobius niger", "Black goby"),
    ("Gobius paganellus", "Rock goby"),
    ("Gobiusculus flavescens", "Two-spotted goby"),
    ("Thorogobius ephippiatus", "Leopard-spotted goby"),
    ("Lipophrys pholis", "Shanny"),
    ("Parablennius gattorugine", "Tompot blenny"),
    ("Coryphoblennius galerita", "Montagu's blenny"),
    ("Chirolophis ascanii", "Yarrell's blenny"),
    ("Pholis gunnellus", "Rock gunnel"),
    ("Zoarces viviparus", "Viviparous eelpout"),
    ("Anarhichas lupus", "Atlantic wolffish"),
    # Sculpins, gurnards, lumpsuckers and other scorpaeniforms
    ("Myoxocephalus scorpius", "Shorthorn sculpin"),
    ("Taurulus bubalis", "Long-spined sea scorpion"),
    ("Agonus cataphractus", "Hooknose"),
    ("Cyclopterus lumpus", "Lumpsucker"),
    ("Liparis liparis", "Common seasnail"),
    ("Liparis montagui", "Montagu's seasnail"),
    ("Chelidonichthys lucerna", "Tub gurnard"),
    ("Chelidonichthys cuculus", "Red gurnard"),
    ("Eutrigla gurnardus", "Grey gurnard"),
    ("Sebastes norvegicus", "Golden redfish"),
    ("Sebastes viviparus", "Norway redfish"),
    # Pelagic shoaling species
    ("Scomber scombrus", "Atlantic mackerel"),
    ("Clupea harengus", "Atlantic herring"),
    ("Sprattus sprattus", "European sprat"),
    ("Sardina pilchardus", "European pilchard"),
    ("Engraulis encrasicolus", "European anchovy"),
    ("Trachurus trachurus", "Atlantic horse mackerel"),
    ("Belone belone", "Garfish"),
    ("Scomberesox saurus", "Atlantic saury"),
    ("Atherina presbyter", "Sand smelt"),
    ("Ammodytes tobianus", "Small sandeel"),
    ("Ammodytes marinus", "Lesser sandeel"),
    ("Hyperoplus lanceolatus", "Greater sandeel"),
    # Bass, mullet, breams
    ("Dicentrarchus labrax", "European sea bass"),
    ("Chelon labrosus", "Thick-lipped grey mullet"),
    ("Chelon auratus", "Golden grey mullet"),
    ("Chelon ramada", "Thin-lipped grey mullet"),
    ("Spondyliosoma cantharus", "Black seabream"),
    ("Pagellus bogaraveo", "Blackspot seabream"),
    ("Pagrus pagrus", "Red porgy"),
    ("Sparus aurata", "Gilthead seabream"),
    ("Boops boops", "Bogue"),
    ("Mullus surmuletus", "Striped red mullet"),
    # Sticklebacks, pipefish and seahorses
    ("Gasterosteus aculeatus", "Three-spined stickleback"),
    ("Spinachia spinachia", "Sea stickleback"),
    ("Syngnathus acus", "Greater pipefish"),
    ("Syngnathus typhle", "Broad-nosed pipefish"),
    ("Syngnathus rostellatus", "Nilsson's pipefish"),
    ("Entelurus aequoreus", "Snake pipefish"),
    ("Nerophis lumbriciformis", "Worm pipefish"),
    ("Nerophis ophidion", "Straight-nosed pipefish"),
    ("Hippocampus hippocampus", "Short-snouted seahorse"),
    ("Hippocampus guttulatus", "Long-snouted seahorse"),
    # Eels, salmonids and diadromous species
    ("Anguilla anguilla", "European eel"),
    ("Conger conger", "European conger"),
    ("Salmo salar", "Atlantic salmon"),
    ("Salmo trutta", "Sea trout"),
    ("Osmerus eperlanus", "European smelt"),
    ("Alosa alosa", "Allis shad"),
    ("Alosa fallax", "Twaite shad"),
    ("Petromyzon marinus", "Sea lamprey"),
    ("Lampetra fluviatilis", "River lamprey"),
    # Sharks, skates and rays
    ("Scyliorhinus canicula", "Small-spotted catshark"),
    ("Scyliorhinus stellaris", "Nursehound"),
    ("Mustelus asterias", "Starry smooth-hound"),
    ("Galeorhinus galeus", "Tope shark"),
    ("Squalus acanthias", "Spiny dogfish"),
    ("Cetorhinus maximus", "Basking shark"),
    ("Prionace glauca", "Blue shark"),
    ("Lamna nasus", "Porbeagle"),
    ("Squatina squatina", "Angelshark"),
    ("Raja clavata", "Thornback ray"),
    ("Raja brachyura", "Blonde ray"),
    ("Raja montagui", "Spotted ray"),
    ("Raja microocellata", "Small-eyed ray"),
    ("Leucoraja naevus", "Cuckoo ray"),
    ("Dipturus batis", "Common skate"),
    ("Amblyraja radiata", "Starry ray"),
    ("Torpedo nobiliana", "Great torpedo ray"),
    ("Dasyatis pastinaca", "Common stingray"),
    # Remaining coastal residents and regular visitors
    ("Zeus faber", "John dory"),
    ("Lophius piscatorius", "Angler"),
    ("Callionymus lyra", "Common dragonet"),
    ("Callionymus maculatus", "Spotted dragonet"),
    ("Echiichthys vipera", "Lesser weever"),
    ("Trachinus draco", "Greater weever"),
    ("Mola mola", "Ocean sunfish"),
    ("Balistes capriscus", "Grey triggerfish"),
    ("Trigla lyra", "Piper gurnard"),
    ("Capros aper", "Boarfish"),
    ("Maurolicus muelleri", "Pearlside"),
)

CATALOGUES: dict[str, tuple[tuple[str, str], ...]] = {
    "north_east_atlantic": NORTH_EAST_ATLANTIC,
}


@dataclass(frozen=True)
class CatalogueEntry:
    """One choosable species: the name that is recorded, and the name that is read."""

    species: str
    common_name: str


# dict over every catalogue, so a species shared by two regions is offered once
# rather than twice.
ENTRIES: tuple[CatalogueEntry, ...] = tuple(
    CatalogueEntry(species=species, common_name=common)
    for species, common in dict(
        pair for catalogue in CATALOGUES.values() for pair in catalogue
    ).items()
)

BY_NAME: dict[str, CatalogueEntry] = {entry.species.lower(): entry for entry in ENTRIES}


# An apostrophe is dropped so *Montagu's blenny* is found by someone who did not
# type one; every other separator becomes a space, so "small-spotted" and "small
# spotted" are one query while "smallspotted" is not - collapsing that far starts
# matching across words the operator never typed.
_DROPPED = "'’ʼ"


def fold(value: str) -> str:
    """Case, accents and punctuation flattened, and nothing else."""

    ascii_only = unicodedata.normalize("NFKD", value or "")
    for char in _DROPPED:
        ascii_only = ascii_only.replace(char, "")
    ascii_only = ascii_only.encode("ascii", "ignore").decode("ascii")
    return " ".join(
        "".join(char if char.isalnum() else " " for char in ascii_only.lower()).split()
    )


def common_name(species: str | None) -> str | None:
    """What this scientific name is called in English, if the catalogue knows."""

    entry = BY_NAME.get(" ".join((species or "").split()).lower())
    return entry.common_name if entry else None


def search(query: str, limit: int = 24) -> list[CatalogueEntry]:
    """Catalogue entries matching ``query``, best first.

    Ranked rather than filtered, because the ranking is the value: an operator
    typing "cod" wants *Atlantic cod* before *Poor cod*, and both before
    *Small-spotted catshark*, which matches only mid-word. A name that starts with
    the query beats one whose later word starts with it, which beats a mid-word
    hit; ties fall back to alphabetical order so the list stays stable between
    keystrokes rather than reshuffling under the operator's cursor.
    """

    wanted = fold(query)
    if not wanted:
        return []

    ranked: list[tuple[int, str, CatalogueEntry]] = []
    for entry in ENTRIES:
        score = min(_rank(fold(entry.common_name), wanted), _rank(fold(entry.species), wanted))
        if score < _NO_MATCH:
            ranked.append((score, entry.common_name.lower(), entry))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return [entry for _, _, entry in ranked[:limit]]


# Rank values, best to worst. Plain numbers, because the order between them is the
# entire specification and naming each step would not add to it.
_NO_MATCH = 3


def _rank(haystack: str, needle: str) -> int:
    if haystack.startswith(needle):
        return 0
    if f" {needle}" in f" {haystack}":
        return 1
    if needle in haystack:
        return 2
    return _NO_MATCH
