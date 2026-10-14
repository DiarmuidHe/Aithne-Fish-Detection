"""Create a separate CVAT task for unreviewed custom-model track suggestions.

Run inside the API container:
  python scripts/import_model_suggestions.py VIDEO_ID NAME /data/outputs/run/tracks.csv
The original corrected CVAT task is never replaced.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from app.config import get_settings
from app.db.database import SessionLocal
from app.db.models import Video
from app.services.cvat_sync import gateway_for
from app.services.viame_parser import parse_viame_csv


def import_suggestions(video_id: uuid.UUID, name: str, csv_path: Path, *, frame_offset: int = 0,
                       break_long_gaps: bool = False) -> dict:
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name):
        raise ValueError("Candidate name must contain only letters, digits, hyphens or underscores")
    parsed = parse_viame_csv(csv_path, strict=True, frame_number_offset=frame_offset)
    with SessionLocal() as db:
        video = db.get(Video, video_id)
        if not video or not video.width or not video.height:
            raise ValueError("Video is missing or has no dimensions")
        settings = get_settings()
        with gateway_for(settings) as gateway:
            task_id = gateway.create_candidate_task(video, name)
            if gateway.has_annotations(task_id):
                return {"task_id": task_id, "status": "already_has_annotations",
                        "note": "Existing task was preserved"}
            imported = gateway.import_tracks(task_id, parsed, video.width, video.height,
                                             break_long_gaps=break_long_gaps)
    return {"task_id": task_id, "status": "unreviewed_suggestions",
            "tracks": imported, "source_csv": str(csv_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", type=uuid.UUID)
    parser.add_argument("name")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--break-long-gaps", action="store_true")
    args = parser.parse_args()
    print(json.dumps(import_suggestions(args.video_id, args.name, args.csv,
                                        frame_offset=args.frame_offset,
                                        break_long_gaps=args.break_long_gaps), indent=2))
