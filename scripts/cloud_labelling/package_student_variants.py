"""Build the one file a new instance needs for ``scripts/student_variants.py``. From the repository
root on the PC:

    .venv\\Scripts\\python.exe -m scripts.cloud_labelling.package_student_variants

The same allowlist as the hard-negatives bundle (``package_hard_negatives``), plus the variants
script. The bigger sizes' COCO checkpoints aren't in it: the instance's ``weights`` stage fetches
them from RF-DETR's bucket and checks them, which is quicker than uploading them from here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.cloud_labelling import package_hard_negatives as base

OUTPUT = Path("data/outputs/smartbay-rounds/round-1-variants-upload/aithne-student-variants.tgz")
EXTRA = ("scripts/student_variants.py",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=Path, default=base.ROUND, help="Round 1's results on this PC")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(base.build(Path("."), args.round, args.output, EXTRA), indent=2))


if __name__ == "__main__":
    main()
