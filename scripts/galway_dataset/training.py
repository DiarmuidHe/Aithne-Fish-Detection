"""PyTorch classification loader and group-aware manifest builder."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def create_manifest(root: Path, val_fraction: float = 0.2, seed: int = 42,
                    underwater_only: bool = True) -> dict:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    with (root / "metadata" / "images.csv").open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "accepted"
                and (root / r["image_path"]).is_file()]
    review_path = root / "metadata" / "context_review.csv"
    reviews = {}
    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8") as fh:
            reviews = {r["image_path"]: r for r in csv.DictReader(fh)}
    rows = [r for r in rows if reviews.get(r["image_path"], {}).get("include_training", "no").lower() == "yes"
            and (not underwater_only or reviews.get(r["image_path"], {}).get(
                "underwater_status", r["underwater_status"]).lower() in {"yes", "verified"})]
    species = sorted({r["scientific_name"] for r in rows})
    mapping = {name: i for i, name in enumerate(species)}
    # Union observation/recording groups linked by visually near-identical images.
    parent = {r["image_path"]: r["image_path"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["source"], row["group_id"])].append(row["image_path"])
    for paths in by_group.values():
        for path in paths[1:]:
            union(paths[0], path)
    for row in rows:
        near = row.get("near_duplicate_of")
        if near in parent:
            union(row["image_path"], near)
    group_rows = defaultdict(list)
    for row in rows:
        group_rows[find(row["image_path"])].append(row)
    rng = random.Random(seed)
    # Assign each connected group once. A mixed-species group is never split.
    groups = list(group_rows.values())
    rng.shuffle(groups)
    total = Counter(r["scientific_name"] for r in rows)
    target = {name: max(1, round(n * val_fraction)) if n > 1 and val_fraction else 0
              for name, n in total.items()}
    val_count = Counter()
    split_by_path = {}
    for group in groups:
        group_count = Counter(r["scientific_name"] for r in group)
        improvement = sum(abs(target[n] - val_count[n]) - abs(target[n] - val_count[n] - c)
                          for n, c in group_count.items())
        use_val = improvement > 0 and val_fraction > 0 and all(
            group_count[name] < total[name] for name in group_count)
        for row in group:
            split_by_path[row["image_path"]] = "val" if use_val else "train"
        if use_val:
            val_count.update(group_count)
    fields = ["image_path", "scientific_name", "class_index", "split", "source",
              "group_id", "license", "underwater_status", "context", "near_duplicate_of"]
    manifest = root / "metadata" / "training_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            review = reviews.get(row["image_path"], {})
            writer.writerow({**row,
                             "underwater_status": review.get("underwater_status", row["underwater_status"]),
                             "context": review.get("context", row["context"]),
                             "class_index": mapping[row["scientific_name"]],
                             "split": split_by_path[row["image_path"]]})
    (root / "metadata" / "species_to_index.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8")
    report = {"total": len(rows), "species_to_index": mapping,
              "train": dict(Counter(r["scientific_name"] for r in rows
                                    if split_by_path[r["image_path"]] == "train")),
              "val": dict(Counter(r["scientific_name"] for r in rows
                                  if split_by_path[r["image_path"]] == "val")),
              "groups": len(groups), "near_duplicate_flags": sum(bool(r.get("near_duplicate_of")) for r in rows)}
    (root / "metadata" / "training_stats.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


class FishSpeciesDataset:
    """Load one manifest split for a PyTorch species classifier."""

    def __init__(self, root: Path | str, split: str = "train", transform=None):
        self.root = Path(root)
        with (self.root / "metadata" / "training_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as fh:
            self.rows = [r for r in csv.DictReader(fh) if r["split"] == split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as opened:
            image = opened.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["class_index"])


def classification_transforms(train: bool = True, size: int = 224):
    """Mild underwater augmentation; retain dark/turbid source examples."""
    from torchvision import transforms as T

    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.75, 1.0)),
                          T.RandomHorizontalFlip(),
                          T.ColorJitter(brightness=0.15, contrast=0.15,
                                        saturation=0.12, hue=0.025),
                          T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return T.Compose([T.Resize(size + 32), T.CenterCrop(size), T.ToTensor(),
                      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
