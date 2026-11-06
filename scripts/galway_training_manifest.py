"""Create group-aware training/validation manifest and class balance report."""

import argparse
import json
from pathlib import Path

from galway_dataset.training import create_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/galway_fish_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--underwater-only", action="store_true")
    parser.add_argument("--all-contexts", action="store_true",
                        help="Include reviewed images marked for training even when not underwater")
    args = parser.parse_args()
    if args.underwater_only and args.all_contexts:
        parser.error("choose either --underwater-only or --all-contexts")
    print(json.dumps(create_manifest(args.dataset, args.val_fraction, args.seed,
                                     not args.all_contexts), indent=2))


if __name__ == "__main__":
    main()
