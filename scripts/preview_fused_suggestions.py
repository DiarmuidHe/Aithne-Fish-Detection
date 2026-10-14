"""Create a contact sheet of unreviewed Fishial-only tracks from a fused CSV."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2


def preview(video: Path, candidates: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    with candidates.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    extras = defaultdict(list)
    for row in rows:
        if row["track_id"].startswith("fishial-extra-"):
            extras[row["track_id"]].append(row)
    if not extras:
        raise ValueError("No Fishial-only tracks in candidate CSV")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    tiles = []
    try:
        for track_id, track_rows in sorted(extras.items(), key=lambda item: int(item[0].split("-")[-1])):
            row = track_rows[len(track_rows)//2]
            frame_number = int(row["frame"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {frame_number}")
            box = tuple(round(float(row[key])) for key in ("tl_x", "tl_y", "br_x", "br_y"))
            cv2.rectangle(frame, box[:2], box[2:], (0, 255, 255), 5)
            cv2.putText(frame, f"{track_id} frame {frame_number} ({len(track_rows)} boxes)",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            tiles.append(cv2.resize(frame, (640, 360)))
    finally:
        capture.release()
    import numpy as np

    columns = 3
    rows_count = math.ceil(len(tiles)/columns)
    sheet = np.zeros((rows_count*360, columns*640, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        sheet[y*360:(y+1)*360, x*640:(x+1)*640] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Cannot write {output}")
    return len(tiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Wrote {preview(args.video, args.candidates, args.output)} unreviewed track previews")
