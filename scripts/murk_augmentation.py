"""Seeded darkness/turbidity degradation for an imported YOLO fish dataset.

``augment-train`` writes one degraded copy of each **train** image (about 1x the train
set) as a training-only folder for ``fish_model.py train --extra-train``. Validation and
test images are never augmented. ``synthetic-test`` writes a separate, clearly labelled
evaluation set: degraded copies of the test split, using ranges disjoint from training so
the robustness check is not circular. Labels are copied unchanged (geometry is untouched).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Image formation: I = J*t + B*(1 - t), after a water tint and lower brightness/contrast,
# then mild blur and sensor noise. BGR order throughout, values in [0, 255].
# A first, harsher draft (brightness down to 0.35, transmission 0.45) left some fish
# invisible in its contact sheet; labels on invisible fish are noise, so training stays
# within what a viewer could still annotate.
TRAIN_RANGES = {
    "brightness": (0.45, 0.90), "contrast": (0.60, 0.95),
    "tint_blue": (0.90, 1.10), "tint_green": (0.95, 1.15), "tint_red": (0.60, 0.90),
    "transmission": (0.55, 0.90), "transmission_gradient": (-0.10, 0.10),
    "veil_blue": (60.0, 130.0), "veil_green": (90.0, 160.0), "veil_red": (30.0, 90.0),
    "blur_sigma": (0.0, 1.2), "noise_sigma": (2.0, 6.0),
}
# A different mix from training rather than simply more of it: darker, bluer, blurrier
# and noisier, with less backscatter, so fish stay visible. (A first draft that was
# harsher on every axis left test frames near-uniform green, which would measure nothing.)
# Each range is disjoint from TRAIN_RANGES except the gradient sign, which only tilts
# transmission top to bottom.
TEST_RANGES = {
    "brightness": (0.25, 0.45), "contrast": (0.95, 1.00),
    "tint_blue": (1.10, 1.25), "tint_green": (0.80, 0.95), "tint_red": (0.35, 0.60),
    "transmission": (0.90, 0.98), "transmission_gradient": (-0.15, 0.15),
    "veil_blue": (20.0, 60.0), "veil_green": (40.0, 90.0), "veil_red": (5.0, 30.0),
    "blur_sigma": (1.2, 2.2), "noise_sigma": (6.0, 12.0),
}
WORKERS = 6  # Bounded: an all-core pool beside VIAME ran the host low on memory.


def sample_params(seed: int, key: str, ranges: dict) -> dict:
    """Per-image parameters from (seed, key) alone, so order and parallelism don't matter."""
    import numpy as np

    rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16))
    params = {name: float(rng.uniform(*ranges[name])) for name in sorted(ranges)}
    params["noise_seed"] = int(rng.integers(0, 2**31 - 1))
    return params


def degrade(image, params: dict):
    import cv2
    import numpy as np

    j = image.astype(np.float32) / 255
    j *= np.array([params["tint_blue"], params["tint_green"], params["tint_red"]], dtype=np.float32)
    mean = j.mean(axis=(0, 1), keepdims=True)
    j = ((j - mean) * params["contrast"] + mean) * params["brightness"]
    rows = np.linspace(-1, 1, j.shape[0], dtype=np.float32)[:, None, None]
    t = np.clip(params["transmission"] + params["transmission_gradient"] * rows, .05, 1)
    veil = np.array([params["veil_blue"], params["veil_green"], params["veil_red"]], dtype=np.float32) / 255
    out = j * t + veil * (1 - t)
    if params["blur_sigma"] > .05:
        out = cv2.GaussianBlur(out, (0, 0), params["blur_sigma"])
    noise = np.random.default_rng(params["noise_seed"]).normal(0, params["noise_sigma"] / 255, out.shape)
    return np.clip(np.rint((out + noise.astype(np.float32)) * 255), 0, 255).astype(np.uint8)


def _write_copy(job: tuple) -> None:
    import cv2

    source, target, params = job
    image = cv2.imread(source)
    if image is None:
        raise ValueError(f"Cannot decode {source}")
    if not cv2.imwrite(target, degrade(image, params), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise ValueError(f"Cannot write {target}")


def _degraded_copies(dataset: Path, output: Path, split: str, out_split: str, seed: int,
                     ranges: dict, suffix: str, extra: dict) -> list[dict]:
    rows = [json.loads(line) for line in (dataset / "metadata.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} rows in {dataset}")
    (output / "images" / out_split).mkdir(parents=True)
    (output / "labels" / out_split).mkdir(parents=True)
    jobs, metadata = [], []
    for row in rows:
        image = dataset / row["output_image"]
        stem = Path(row["output_image"]).stem
        params = sample_params(seed, stem, ranges)
        target = output / "images" / out_split / f"{stem}-{suffix}.jpg"
        shutil.copyfile(dataset / row["output_label"], output / "labels" / out_split / f"{stem}-{suffix}.txt")
        jobs.append((str(image), str(target), params))
        metadata.append({**{k: row[k] for k in ("source_sequence", "source_frame", "reviewed",
                                                "reviewed_for_smartbay") if k in row},
                         "split": out_split, "unverified_empty": row.get("unverified_empty", False),
                         "output_image": str(target.relative_to(output)),
                         "output_label": f"labels/{out_split}/{stem}-{suffix}.txt",
                         "augmented_from": row["output_image"], "params": params, "seed": seed,
                         "synthetic": True, "review_scope": "labels inherited from the source frame",
                         **extra})
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(_write_copy, jobs, chunksize=32))
    (output / "metadata.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metadata), encoding="utf-8")
    shutil.copyfile(dataset / "classes.json", output / "classes.json")
    return metadata


def augment_train(dataset: Path, output: Path, seed: int, ranges: dict = TRAIN_RANGES) -> dict:
    """One degraded copy per train image; a training-only folder for --extra-train."""
    dataset, output = dataset.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata = _degraded_copies(dataset, output, "train", "train", seed, ranges, "murk",
                                {"purpose": "training augmentation"})
    report = {"source_dataset": str(dataset), "source_report_sha256": _digest(dataset / "report.json"),
              "seed": seed, "ranges": ranges, "copies": len(metadata), "splits_augmented": ["train"],
              "model": "I = J*t + B*(1-t) after tint and brightness/contrast; then blur and noise"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    contact_sheet(dataset, output, metadata, output / "contact-sheet.jpg")
    return report


def synthetic_test(dataset: Path, output: Path, seed: int, ranges: dict = TEST_RANGES) -> dict:
    """Degraded copies of the test split for a synthetic robustness check only."""
    dataset, output = dataset.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata = _degraded_copies(dataset, output, "test", "test", seed, ranges, "dark",
                                {"purpose": "synthetic robustness evaluation only"})
    (output / "dataset.yaml").write_text(
        f"path: '{output.as_posix()}'\ntrain: '{(dataset / 'images/train').as_posix()}'\n"
        f"val: '{(dataset / 'images/val').as_posix()}'\ntest: images/test\nnames:\n  0: fish\n",
        encoding="utf-8")
    report = {"source_dataset": str(dataset), "seed": seed, "ranges": ranges,
              "frames": {"train": 0, "val": 0, "test": len(metadata)},
              "note": "Synthetic darkened copy of the test split; not real murky water",
              "ranges_disjoint_from_training": True}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    contact_sheet(dataset, output, metadata, output / "contact-sheet.jpg")
    return report


def contact_sheet(dataset: Path, output: Path, metadata: list[dict], path: Path, pairs: int = 12) -> None:
    """Original | degraded pairs, chosen by stable hash, at half size."""
    import cv2
    import numpy as np

    chosen = sorted(metadata, key=lambda m: hashlib.sha256(m["output_image"].encode()).hexdigest())[:pairs]
    tiles = []
    for row in chosen:
        before = cv2.resize(cv2.imread(str(dataset / row["augmented_from"])), (480, 270))
        after = cv2.resize(cv2.imread(str(output / row["output_image"])), (480, 270))
        tiles.append(np.hstack([before, after]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("augment-train", "synthetic-test"):
        command = commands.add_parser(name)
        command.add_argument("dataset", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "augment-train":
        report = augment_train(args.dataset, args.output, args.seed)
    else:
        report = synthetic_test(args.dataset, args.output, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
