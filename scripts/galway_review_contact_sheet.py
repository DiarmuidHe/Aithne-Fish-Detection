"""Make a deterministic visual sample of downloaded species photos for review."""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/galway_fish_dataset"))
    parser.add_argument("--per-species", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with (args.dataset / "metadata" / "images.csv").open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "accepted"
                and (args.dataset / r["image_path"]).is_file()]
    by_species = defaultdict(list)
    for row in rows:
        by_species[row["scientific_name"]].append(row)
    rng = random.Random(args.seed)
    selected = [row for species in sorted(by_species)
                for row in rng.sample(by_species[species], min(args.per_species, len(by_species[species])))]
    columns, cell_w, cell_h = 4, 360, 280
    sheet = Image.new("RGB", (columns * cell_w, ((len(selected) + columns - 1) // columns) * cell_h), "#111111")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        with Image.open(args.dataset / row["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((cell_w - 12, cell_h - 48))
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 28 + (cell_h - 48 - image.height) // 2))
        draw.text((x + 8, y + 6), f"{row['scientific_name']}  #{row['source_id']}", fill="white")
    output = args.dataset / "metadata" / "review_contact_sheet.jpg"
    sheet.save(output, quality=90)
    print(output)


if __name__ == "__main__":
    main()
