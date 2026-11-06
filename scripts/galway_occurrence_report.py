"""Query iNaturalist research-grade occurrence evidence for Irish target species."""

import argparse
import csv
from pathlib import Path

import yaml

from galway_dataset.common import Http
from galway_dataset.inat_source import API, resolve_taxon

# Broad regional envelopes for prioritisation; neither polygon is a SmartBay footprint.
REGIONS = {
    "ireland_region": {"swlat": 51.0, "swlng": -12.0, "nelat": 56.0, "nelng": -5.0},
    "galway_bay_area": {"swlat": 53.08, "swlng": -9.85, "nelat": 53.52, "nelng": -8.75},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/galway_species.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/galway_fish_dataset"))
    parser.add_argument("--species", action="append", help="Exact scientific name; repeat as needed")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    names = [item["scientific_name"] for item in yaml.safe_load(
        args.config.read_text(encoding="utf-8"))["species"] if item.get("enabled", True)]
    selected = args.species or names
    invalid = set(selected) - set(names)
    if invalid:
        parser.error(f"species absent from config: {', '.join(sorted(invalid))}")
    http = Http(delay=args.delay)
    output = args.dataset / "metadata" / "regional_occurrence.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scientific_name", "region", "reported_research_grade_observations",
              "count_status", "interpretation", "notes"]
    prior = {}
    if output.exists():
        with output.open(newline="", encoding="utf-8") as fh:
            prior = {(r["scientific_name"], r["region"]): r for r in csv.DictReader(fh)}
    for species in selected:
        try:
            taxon_id = resolve_taxon(http, species)
        except Exception as exc:
            taxon_id = None
            error = str(exc)
        else:
            error = ""
        for region, bounds in REGIONS.items():
            record = {"scientific_name": species, "region": region,
                      "reported_research_grade_observations": "", "count_status": "unknown",
                      "interpretation": "not SmartBay sighting evidence", "notes": error}
            if not error and taxon_id is None:
                record.update(reported_research_grade_observations=0, count_status="verified",
                              notes="exact active taxon absent")
            elif taxon_id is not None:
                try:
                    result = http.get(f"{API}/observations", params={
                        "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
                        "per_page": 1, **bounds}).json()
                    count = result.get("total_results")
                    if isinstance(count, int):
                        record.update(reported_research_grade_observations=count,
                                      count_status="verified",
                                      notes="bounding-box observations; descendant taxa may be included")
                except Exception as exc:
                    record["notes"] = str(exc)
            prior[(species, region)] = record
            print(f"{species}, {region}: {record['reported_research_grade_observations'] or ('0' if record['reported_research_grade_observations'] == 0 else 'unknown')}")
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for record in sorted(prior.values(), key=lambda r: (r["scientific_name"], r["region"])):
                writer.writerow(record)
    print(output)


if __name__ == "__main__":
    main()
