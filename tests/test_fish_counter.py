from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.services.fish_counter import accepted_tracks, summarize_tracks
from app.services.viame_parser import parse_viame_csv


def test_confidence_filtering_uses_track_max_confidence():
    parsed = parse_viame_csv(Path("tests/fixtures/sample_viame_output.csv"), fps=30.0)

    accepted = accepted_tracks(parsed.tracks, min_confidence=0.60)

    assert [track.viame_track_id for track in accepted] == ["1", "3"]


def test_summary_uses_accepted_tracks_not_frame_detections():
    parsed = parse_viame_csv(Path("tests/fixtures/sample_viame_output.csv"), fps=30.0)

    summary = summarize_tracks(
        video_id=uuid4(),
        status="completed",
        tracks=parsed.tracks,
        min_confidence=0.60,
    )

    assert summary.fish_tracks == 2
    assert summary.total_detections == 5
    assert summary.mean_track_confidence == pytest.approx(0.90)
    assert summary.first_fish_timestamp_seconds == pytest.approx(0.0)
    assert summary.last_fish_timestamp_seconds == pytest.approx(31 / 30)

