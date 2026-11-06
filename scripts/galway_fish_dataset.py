"""Download species-labelled fish images from FathomNet and iNaturalist.

Run ``python scripts/galway_fish_dataset.py --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import yaml

from galway_dataset import fathom_source, inat_source
from galway_dataset.common import Http, Store

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "galway_species.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("data/galway_fish_dataset"))
    parser.add_argument("--species", action="append", help="Exact scientific name; repeat to select several")
    parser.add_argument("--sources", nargs="+", choices=["fathomnet", "inaturalist"],
                        default=["fathomnet", "inaturalist"])
    parser.add_argument("--limit", type=int, default=20, help="Total usable images per source and species")
    parser.add_argument("--min-side", type=int, default=64)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between HTTP requests")
    parser.add_argument("--image-size", choices=["original", "large", "medium"], default="original")
    parser.add_argument("--max-observations", type=int, default=10000)
    parser.add_argument("--include-nc", action="store_true", help="Include CC BY-NC and CC BY-NC-SA; noncommercial use only")
    parser.add_argument("--allow-unverified-fathomnet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Search and report counts without image downloads")
    args = parser.parse_args()
    if args.limit < 0 or args.min_side < 1 or args.delay < 0 or args.max_observations < 1:
        parser.error("limit/delay must be nonnegative; min-side/max-observations must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured = [item["scientific_name"] for item in config["species"] if item.get("enabled", True)]
    selected = args.species or configured
    unknown = set(selected) - set(configured)
    if unknown:
        parser.error(f"species absent from config: {', '.join(sorted(unknown))}")
    args.output.mkdir(parents=True, exist_ok=True)
    config_dir = args.output / "configs"
    config_dir.mkdir(exist_ok=True)
    shutil.copyfile(args.config, config_dir / "species.yaml")
    guide = Path(__file__).resolve().parents[1] / "docs" / "galway_fish_dataset.md"
    if guide.is_file():
        shutil.copyfile(guide, args.output / "README.md")
    store = Store(args.output, min_side=args.min_side)
    http = Http(delay=args.delay)
    availability: dict[tuple[str, str], dict] = {}
    prior = args.output / "metadata" / "species_counts.csv"
    if prior.exists():
        with prior.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                availability[row["scientific_name"], row["source"]] = {
                    "observations": row["available_observations"],
                    "images": row["available_images"], "status": row["count_status"],
                    "notes": row["notes"]}
    counts = Counter((r["scientific_name"], r["source"]) for r in store.rows
                     if r["status"] == "accepted" and (args.output / r["image_path"]).exists())
    for name in selected:
        for source in args.sources:
            remaining = max(0, args.limit - counts[name, source])
            try:
                if source == "fathomnet":
                    result = fathom_source.run(
                        name, store, http, remaining, args.include_nc, dry_run=args.dry_run,
                        verified_only=not args.allow_unverified_fathomnet)
                else:
                    result = inat_source.run(
                        name, store, http, remaining, args.include_nc, dry_run=args.dry_run,
                        image_size=args.image_size, max_observations=args.max_observations)
            except Exception as exc:
                print(f"ERROR {source} {name}: {exc}")
                result = {"status": "unknown", "notes": f"search or source error: {exc}"}
            availability[name, source] = result
            store.report(configured, availability)
    print(f"Report: {args.output / 'metadata' / 'species_counts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
