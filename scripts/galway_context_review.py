"""Create or extend a human review sheet for image context and label confidence."""

import argparse
import csv
from pathlib import Path

FIELDS = ["image_path", "context", "underwater_status", "species_id_status",
          "include_training", "review_notes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/galway_fish_dataset"))
    args = parser.parse_args()
    metadata = args.dataset / "metadata"
    review_path = metadata / "context_review.csv"
    existing = {}
    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8") as fh:
            existing = {r["image_path"]: r for r in csv.DictReader(fh)}
    with (metadata / "images.csv").open(newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    with review_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            if record["status"] != "accepted":
                continue
            writer.writerow(existing.get(record["image_path"], {
                "image_path": record["image_path"], "context": "unknown",
                "underwater_status": "unknown", "species_id_status": "unreviewed",
                "include_training": "no", "review_notes": "Needs context review"}))
    print(review_path)


if __name__ == "__main__":
    main()
