"""Export only explicitly reviewed CVAT regions to detector and track truth.

Run inside the API container so database, CVAT and /data video paths agree:
  python scripts/reviewed_dataset.py manifest.json /data/training/reviewed-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
from sqlalchemy import select

from app.config import get_settings
from app.db.database import SessionLocal
from app.db.models import CvatSync, ReviewRegion, ReviewedSpecies, Video
from app.services.cvat_sync import gateway_for
from app.services.review_dataset import annotation_sha256, region_rows


def export(manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])
    if not sources:
        raise ValueError("Manifest needs at least one video source")
    groups: dict[str, str] = {}
    prepared = []
    settings = get_settings()
    with SessionLocal() as db, gateway_for(settings) as gateway:
        for source in sources:
            video_id = uuid.UUID(source["video_id"])
            split = source["split"]
            group = source["group"]
            if split not in {"train", "val", "test"} or not group:
                raise ValueError("Each source needs a group and train, val or test split")
            if group in groups and groups[group] != split:
                raise ValueError(f"Recording group {group} crosses splits")
            groups[group] = split
            video = db.get(Video, video_id)
            link = db.get(CvatSync, video_id)
            regions = db.scalars(select(ReviewRegion).where(
                ReviewRegion.video_id == video_id)).all()
            if not video or not link or not link.task_id or not regions:
                raise ValueError(f"{video_id} needs a CVAT task and completed review regions")
            if not video.width or not video.height or not video.fps:
                raise ValueError(f"{video_id} lacks video dimensions or fps")
            species = {row.cvat_track_id: row.species for row in db.scalars(
                select(ReviewedSpecies).where(ReviewedSpecies.video_id == video_id)).all()}
            task = gateway.client.tasks.retrieve(link.task_id)
            frame_rows: dict[int, list[dict]] = {}
            ids_frames: set[int] = set()
            species_frames: set[int] = set()
            for region in regions:
                rows = region_rows(task, region.start_frame, region.end_frame,
                                   video.width, video.height)
                if annotation_sha256(rows, species if region.species_complete else None) != region.annotation_sha256:
                    raise ValueError(f"CVAT changed in {video_id} frames {region.start_frame}-"
                                     f"{region.end_frame}; re-certify this region")
                frame_rows.update(rows)
                if region.ids_complete:
                    ids_frames.update(rows)
                if region.species_complete:
                    species_frames.update(rows)
                    missing = {row["track_id"] for entries in rows.values() for row in entries
                               if row["track_id"] not in species}
                    if missing:
                        raise ValueError(f"Species decisions missing in {video_id}: {sorted(missing)}")
            prepared.append((source, video, link, frame_rows, ids_frames,
                             species_frames, species))

    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    metadata, track_truth, species_truth = [], [], []
    counts = defaultdict(int)
    for source, video, link, frame_rows, ids_frames, species_frames, species in prepared:
        split = source["split"]
        cap = cv2.VideoCapture(video.storage_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video.storage_path}")
        try:
            for frame, boxes in sorted(frame_rows.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok or image.shape[:2] != (video.height, video.width):
                    raise RuntimeError(f"Cannot decode {video.id} frame {frame}")
                stem = f"{video.id.hex}_{frame:08d}"
                image_path = output / "images" / split / f"{stem}.jpg"
                label_path = output / "labels" / split / f"{stem}.txt"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Cannot write {image_path}")
                labels = []
                for row in boxes:
                    x1, y1, x2, y2 = row["box"]
                    labels.append(f"0 {(x1+x2)/2/video.width:.8f} "
                                  f"{(y1+y2)/2/video.height:.8f} "
                                  f"{(x2-x1)/video.width:.8f} {(y2-y1)/video.height:.8f}")
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                row_meta = {"image": str(image_path.relative_to(output)),
                            "video_id": str(video.id), "video": video.storage_path,
                            "source_sha256": video.content_sha256, "cvat_task_id": link.task_id,
                            "group": source["group"], "split": split, "frame": frame,
                            "reviewed": True, "ids_complete": frame in ids_frames,
                            "species_complete": frame in species_frames,
                            "fish_count": len(boxes)}
                metadata.append(row_meta)
                counts[split] += 1
                counts["negative"] += not boxes
                if frame in ids_frames:
                    track_truth.append({**row_meta, "tracks": boxes})
                if frame in species_frames:
                    for row in boxes:
                        name = species[row["track_id"]]
                        species_truth.append({"video_id": str(video.id), "group": source["group"],
                                              "split": split, "frame": frame,
                                              "cvat_track_id": row["track_id"],
                                              "species": name, "box": row["box"]})
        finally:
            cap.release()
    for filename, rows in (("metadata.jsonl", metadata), ("tracks_truth.jsonl", track_truth),
                           ("species_truth.jsonl", species_truth)):
        (output / filename).write_text("".join(json.dumps(row) + "\n" for row in rows),
                                       encoding="utf-8")
    (output / "classes.json").write_text('["fish"]\n', encoding="utf-8")
    (output / "dataset.yaml").write_text(
        f"path: {json.dumps(str(output.resolve()))}\ntrain: images/train\n"
        "val: images/val\ntest: images/test\nnames: [fish]\n", encoding="utf-8")
    report = {"frames": {split: counts[split] for split in ("train", "val", "test")},
              "verified_empty_frames": counts["negative"],
              "identity_checked_frames": len(track_truth),
              "species_decisions": len(species_truth),
              "source_sessions": len(groups),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "note": "Only certified source frames are exported. Unknown species remain fish labels."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.manifest, args.output), indent=2))
