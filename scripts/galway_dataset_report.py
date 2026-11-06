"""Report usable downloaded images by species and source from saved metadata."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/galway_fish_dataset"))
    args = parser.parse_args()
    path = args.dataset / "metadata" / "images.csv"
    if not path.is_file():
        parser.error(f"missing {path}; run the downloader first")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "accepted"
                and (args.dataset / r["image_path"]).is_file()]
    counts = Counter((r["scientific_name"], r["source"]) for r in rows)
    print("scientific_name,source,usable_images")
    for (name, source), count in sorted(counts.items()):
        print(f"{name},{source},{count}")
    print(f"TOTAL,,{len(rows)}")
    totals = Counter(r["scientific_name"] for r in rows)
    if totals:
        print(f"class_imbalance_max_over_min={max(totals.values()) / min(totals.values()):.2f}")
    print(f"near_duplicate_flags={sum(bool(r.get('near_duplicate_of')) for r in rows)}")


if __name__ == "__main__":
    main()
