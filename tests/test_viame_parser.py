from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from app.services.viame_parser import (
    VIAMEParseError,
    frame_to_timestamp_seconds,
    parse_viame_csv,
)

FIXTURE = Path("tests/fixtures/sample_viame_output.csv")


def test_viame_csv_parsing_and_track_grouping():
    parsed = parse_viame_csv(FIXTURE, fps=30.0)

    assert len(parsed.detections) == 7
    assert len(parsed.tracks) == 3
    assert len(parsed.skipped_rows) == 1

    first_track = parsed.tracks[0]
    assert first_track.viame_track_id == "1"
    assert first_track.first_frame == 0
    assert first_track.last_frame == 2
    assert first_track.detection_count == 3
    assert first_track.max_confidence == pytest.approx(0.94)
    assert first_track.mean_confidence == pytest.approx(0.92)
    assert first_track.species == "fish"


def test_timestamp_calculation_handles_fractional_fps():
    fps = float(Fraction(30000, 1001))

    assert frame_to_timestamp_seconds(30, fps) == pytest.approx(1.001, rel=1e-6)


def test_frame_number_offset_normalizes_one_based_viame_frames(tmp_path: Path):
    csv_path = tmp_path / "one_based_tracks.csv"
    csv_path.write_text(
        "1,input.mp4,1,10,20,30,40,0.9,-1,fish,0.9\n",
        encoding="utf-8",
    )

    parsed = parse_viame_csv(csv_path, fps=25.0, frame_number_offset=-1)

    assert len(parsed.detections) == 1
    assert parsed.detections[0].frame_number == 0
    assert parsed.detections[0].timestamp_seconds == pytest.approx(0.0)
    assert parsed.tracks[0].first_frame == 0


def test_strict_mode_rejects_malformed_rows():
    with pytest.raises(VIAMEParseError):
        parse_viame_csv(FIXTURE, fps=None, strict=True)
