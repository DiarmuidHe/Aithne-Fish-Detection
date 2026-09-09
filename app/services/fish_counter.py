from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, TypeVar


class TrackLike(Protocol):
    max_confidence: float
    mean_confidence: float
    detection_count: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None


TTrack = TypeVar("TTrack", bound=TrackLike)


@dataclass(frozen=True)
class FishCountSummary:
    video_id: uuid.UUID
    status: str
    fish_tracks: int
    total_detections: int
    mean_track_confidence: float | None
    first_fish_timestamp_seconds: float | None
    last_fish_timestamp_seconds: float | None


def is_accepted_track(track: TrackLike, min_confidence: float) -> bool:
    """Human decisions override the threshold without modifying detector evidence."""
    review = getattr(track, "review_state", "unreviewed")
    if review == "accepted":
        return True
    if review in {"rejected", "needs-review"}:
        return False
    return track.max_confidence >= min_confidence


def accepted_tracks(tracks: list[TTrack], min_confidence: float) -> list[TTrack]:
    return [track for track in tracks if is_accepted_track(track, min_confidence)]


def summarize_tracks(
    video_id: uuid.UUID,
    status: str,
    tracks: list[TrackLike],
    min_confidence: float,
) -> FishCountSummary:
    accepted = accepted_tracks(tracks, min_confidence)
    first_timestamps = [
        track.first_timestamp_seconds
        for track in accepted
        if track.first_timestamp_seconds is not None
    ]
    last_timestamps = [
        track.last_timestamp_seconds
        for track in accepted
        if track.last_timestamp_seconds is not None
    ]

    mean_track_confidence = None
    if accepted:
        mean_track_confidence = sum(track.mean_confidence for track in accepted) / len(accepted)

    return FishCountSummary(
        video_id=video_id,
        status=status,
        fish_tracks=len(accepted),
        total_detections=sum(track.detection_count for track in accepted),
        mean_track_confidence=mean_track_confidence,
        first_fish_timestamp_seconds=min(first_timestamps) if first_timestamps else None,
        last_fish_timestamp_seconds=max(last_timestamps) if last_timestamps else None,
    )
