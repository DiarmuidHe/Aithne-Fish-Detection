"""Build the one file a new instance needs for ``scripts/hard_negatives.py``. From the repository
root on the PC, after the round's results were copied back:

    .venv\\Scripts\\python.exe -m scripts.cloud_labelling.package_hard_negatives

The bundle unpacks into an empty folder that then works as the repository root. It holds an
allowlist and nothing else, so no ``.env``, key or other local file can travel with it: the code
these stages import, the RF-DETR COCO checkpoint, the round's trained student, and SAM 3's finished
output. It holds neither SAM 3's checkpoint, as nothing here runs SAM 3, nor any video: ``restore``
fetches those from the public archive. ``SHA256SUMS`` inside lets the instance check every file.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROUND = Path("data/outputs/smartbay-rounds/round-1/cloud")
INSTANCE_ROUND = "data/outputs/smartbay-rounds/round-1"
OUTPUT = Path("data/outputs/smartbay-rounds/round-1-hn-upload/aithne-hard-negatives.tgz")
CODE = ("app/__init__.py", "app/services/box_tracking.py", "app/services/external_datasets.py",
        "app/services/static_filter.py", "app/services/viame_parser.py", "scripts/cloud-labelling-train-requirements.txt",
        "scripts/hard_negatives.py", "scripts/sam3_autolabel.py", "scripts/sam3_botsort.py", "scripts/sam3_stages.py",
        "scripts/student_detector.py", "scripts/viame_zero_label_benchmark.py")
CODE_FOLDERS = ("scripts/cloud_labelling", "scripts/smartbay_archive_scan")
MODELS = ("data/models/rfdetr/rf-detr-nano.pth", "data/models/rfdetr/LICENSE", "data/models/sam3/LICENSE")
ROUND_FILES = ("journal.json", "checks/quality_report.json", "selection/day_roles.json", "selection/to_label.csv",
               "student/round_dataset.json", "student/round_eval.json", "student/rfdetr-nano/*/frames.jsonl", "windows/*.json",
               "videos/*/video.json", "videos/*/labels/fish/rows.jsonl", "videos/*/sam3/fish/frames.jsonl",
               "videos/*/static/sam3-fish/removed.jsonl")
CODE_SUFFIXES = {".py", ".sh"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contents(repo: Path, round_dir: Path, extra: tuple[str, ...] = ()) -> dict[str, Path]:
    """Name inside the bundle to its file here, plus ``extra`` repository files. Refuses a missing
    file, a shell script with Windows line endings, which won't run on the instance, and a student
    that isn't the one the round's journal recorded."""
    wanted = {name: repo / name for name in (*CODE, *MODELS, *extra)}
    for folder in CODE_FOLDERS:
        wanted |= {p.relative_to(repo).as_posix(): p for p in sorted((repo / folder).rglob("*"))
                   if p.suffix in CODE_SUFFIXES and "__pycache__" not in p.parts}
    for pattern in ROUND_FILES:
        found = sorted(round_dir.glob(pattern))
        if not found:
            raise SystemExit(f"Nothing matches {pattern} under {round_dir}")
        wanted |= {f"{INSTANCE_ROUND}/{p.relative_to(round_dir).as_posix()}": p for p in found}
    checkpoint = json.loads((round_dir / "journal.json").read_text(encoding="utf-8"))["decisions"]["student_model"]["checkpoint"]
    relative = checkpoint["path"].removeprefix(INSTANCE_ROUND + "/")
    student = round_dir / relative
    if not student.is_file() or sha256_of(student) != checkpoint["sha256"]:
        raise SystemExit(f"{student} is missing or isn't the student the journal recorded")
    wanted[checkpoint["path"]] = student
    missing = [name for name, path in wanted.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing: {missing}")
    bad = [name for name, path in wanted.items() if name.endswith(".sh") and b"\r" in path.read_bytes()]
    if bad:
        raise SystemExit(f"Windows line endings in {bad}")
    return wanted


def build(repo: Path, round_dir: Path, output: Path, extra: tuple[str, ...] = (),
          files: dict[str, Path] | None = None) -> dict:
    """Write the bundle through a temporary name, so a stopped run leaves no file to upload by
    mistake. ``files`` (name inside to file here) replaces the hard-negatives allowlist."""
    files = files if files is not None else contents(repo, round_dir, extra)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    sums = []

    def add(archive, name, info_size, stream):
        info = tarfile.TarInfo(name)
        info.size, info.mode = info_size, 0o755 if name.endswith(".sh") else 0o644
        archive.addfile(info, stream)

    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for name, path in sorted(files.items()):
            sums.append(f"{sha256_of(path)}  {name}\n")
            with path.open("rb") as stream:
                add(archive, name, path.stat().st_size, stream)
        manifest = "".join(sums).encode()
        add(archive, "SHA256SUMS", len(manifest), io.BytesIO(manifest))
    temporary.replace(output)
    digest = sha256_of(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8", newline="\n")
    return {"bundle": output.as_posix(), "bytes": output.stat().st_size, "sha256": digest, "files": len(files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=Path, default=ROUND, help="The round's results on this PC")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(Path("."), args.round, args.output), indent=2))


if __name__ == "__main__":
    main()
