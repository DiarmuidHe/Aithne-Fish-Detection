import csv
from pathlib import Path

import pytest

from scripts.fuse_fish_suggestions import fuse

HEADER = ["track_id", "video_id", "frame", "tl_x", "tl_y", "br_x", "br_y",
          "confidence", "target_length", "class", "score"]


def write_csv(path: Path, rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)


def row(track: str, frame: int, x: int, confidence: float = .8) -> list:
    return [track, "video.mp4", frame, x, 10, x+20, 30, confidence, -1, "fish", confidence]


def test_keeps_baseline_and_adds_only_persistent_new_fish(tmp_path: Path) -> None:
    baseline, fishial, output = (tmp_path / name for name in ("baseline.csv", "fishial.csv", "out"))
    write_csv(baseline, [row("1", 1, 10), row("1", 4, 12)])
    write_csv(fishial, [row("f0", 0, 10), row("f1", 3, 12),
                        row("f2", 0, 100), row("f3", 3, 102),
                        row("f4", 20, 200)])
    report = fuse(baseline, fishial, output)
    with (output / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["frame"]) for row in rows] == [0, 3, 0, 3]
    assert [row["track_id"] for row in rows] == ["viame-1", "viame-1",
                                                "fishial-extra-1", "fishial-extra-1"]
    assert report["fishial_matched_to_viame"] == 2
    assert report["fishial_unconfirmed_boxes"] == 1
    assert report["fishial_added_tracks"] == 1
    with pytest.raises(FileExistsError):
        fuse(baseline, fishial, output)


def test_gap_and_score_gate_prevent_unstable_suggestions(tmp_path: Path) -> None:
    baseline, fishial, output = (tmp_path / name for name in ("baseline.csv", "fishial.csv", "out"))
    write_csv(baseline, [])
    write_csv(fishial, [row("a", 0, 100), row("b", 3, 102, .2),
                        row("c", 9, 104), row("d", 20, 200), row("e", 23, 300)])
    report = fuse(baseline, fishial, output)
    assert report["fishial_added_boxes"] == 0
    assert report["fishial_below_score"] == 1
    with pytest.raises(ValueError):
        fuse(baseline, fishial, tmp_path / "bad", min_support=1)


def test_wide_grass_like_boxes_do_not_become_tracks(tmp_path: Path) -> None:
    baseline, fishial, output = (tmp_path / name for name in ("baseline.csv", "fishial.csv", "out"))
    write_csv(baseline, [])
    write_csv(fishial, [["a", "video.mp4", frame, 20, 10, 220, 30,
                         .9, -1, "fish", .9] for frame in (0, 3, 6)])
    report = fuse(baseline, fishial, output)
    assert report["fishial_implausible_shape"] == 3
    assert report["fishial_added_tracks"] == 0
