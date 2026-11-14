"""Build the code bundle for the base instance that becomes the round 2 image. From the repository
root on the PC:

    .venv\\Scripts\\python.exe -m scripts.cloud_labelling.package_base_image

The bundle carries code and the two package locks only: everything the archive scan, SAM 3
labelling, the round checks and student training import. The checkpoints travel beside it by
scp, as ``bootstrap_labelling.sh`` and ``verify_rfdetr`` check them by size and SHA256 on the
instance. No round data, ``.env``, key or video is in it. Later runs upload a round bundle that
overwrites this code, so the image never has to be rebuilt for a code change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.cloud_labelling import package_hard_negatives as base

OUTPUT = Path("data/outputs/smartbay-rounds/round-2-upload/aithne-base-code.tgz")
CODE = (*base.CODE, "scripts/student_variants.py", "scripts/cloud-labelling-archive-requirements.txt")
MODELS = ("data/models/sam3/sam3.pt", "data/models/sam3/LICENSE", "data/models/rfdetr/rf-detr-nano.pth",
          "data/models/rfdetr/LICENSE")


def contents(repo: Path) -> dict[str, Path]:
    """Name inside the bundle to its file here: the code, with no model or round file."""
    wanted = {name: repo / name for name in CODE}
    for folder in base.CODE_FOLDERS:
        wanted |= {p.relative_to(repo).as_posix(): p for p in sorted((repo / folder).rglob("*"))
                   if p.suffix in base.CODE_SUFFIXES and "__pycache__" not in p.parts}
    missing = [name for name, path in wanted.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing: {missing}")
    bad = [name for name, path in wanted.items() if name.endswith(".sh") and b"\r" in path.read_bytes()]
    if bad:
        raise SystemExit(f"Windows line endings in {bad}")
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = base.build(Path("."), Path("."), args.output, files=contents(Path(".")))
    missing = [m for m in MODELS if not Path(m).is_file()]
    print(json.dumps(summary | {"models_to_scp": list(MODELS), "missing_models": missing}, indent=2))


if __name__ == "__main__":
    main()
