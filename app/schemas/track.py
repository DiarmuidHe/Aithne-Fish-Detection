from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrackReviewUpdate(BaseModel):
    review_state: Literal["unreviewed", "reviewed", "accepted", "rejected", "needs-review"]

    model_config = ConfigDict(extra="forbid")


class TrackBulkReviewUpdate(BaseModel):
    """One decision applied to many tracks. The queue needs "accept all visible"."""

    track_ids: list[uuid.UUID] = Field(min_length=1, max_length=200)
    review_state: Literal["unreviewed", "reviewed", "accepted", "rejected", "needs-review"]

    model_config = ConfigDict(extra="forbid")


class TrackSpeciesUpdate(BaseModel):
    """The name a person decided this fish carries, or ``null`` to take it back.

    Not restricted to the curated catalogue. The catalogue is what the dashboard
    offers, and an operator who knows better than a list built for one bay must be
    able to say so; what is enforced is only that the value looks like a name -
    letters and the punctuation names actually contain - so the field cannot be
    used to store something that is not one.
    """

    species: str | None = Field(default=None, max_length=256)

    model_config = ConfigDict(extra="forbid")

    @field_validator("species")
    @classmethod
    def _looks_like_a_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Whitespace is collapsed rather than merely stripped, so the same name
        # pasted from two places is stored once and matches itself later.
        name = " ".join(value.split())
        if not name:
            return None
        if len(name.split()) > 4 or not NAME_PATTERN.fullmatch(name):
            raise ValueError("That does not look like a species name")
        return name


# Letters, and the punctuation that appears inside real names: the hyphen of
# *Trisopterus luscus*'s English name, the full stop of an open nomenclature
# "Raja sp.", the apostrophe of *Montagu's blenny*.
NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z.'’-]*(?: [A-Za-z][A-Za-z.'’-]*)*")


class SpeciesIdentificationRequest(BaseModel):
    """Ask Fishial to name one chosen fish.

    ``frames`` is how many frames the operator is willing to pay for; the system
    picks which frames those are. Omitted means the deployment's default.
    """

    frames: int | None = Field(default=None, ge=1, le=50)

    model_config = ConfigDict(extra="forbid")


class TrackThumbnailRead(BaseModel):
    """Where to find one fish's preview crop."""

    track_id: uuid.UUID
    url: str


class TrackIdentificationRead(BaseModel):
    """One fish's identification, whether unasked, running, or finished."""

    track_id: uuid.UUID
    state: Literal["none", "submitted", "identified", "review_required", "error"]
    species: str | None = None
    confidence: float | None = None
    # What was asked for, what the selector could actually find, and what was sent.
    # They differ when the fish was in view for fewer independent moments than the
    # operator asked for frames, which is worth saying rather than hiding.
    frames_requested: int | None = None
    frames_selected: int = 0
    frames_submitted: int = 0
    # Distinct moments the sent frames span. Fewer than frames_submitted means some
    # of them share a moment, which is weaker evidence and is shown as such.
    windows: int = 0
    calls_saved: int = 0
    quality_score: float | None = None
    requested_at: datetime | None = None
    completed_at: datetime | None = None
    stopped_early: str | None = None
    # Set when the reported name is not on the camera's regional species list.
    implausible_for_region: str | None = None
    tally: dict[str, int] = Field(default_factory=dict)
    diagnostics: dict | None = None


class FishDetectionRead(BaseModel):
    id: uuid.UUID
    frame_number: int
    timestamp_seconds: float | None
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str | None
    class_confidence: float | None

    model_config = ConfigDict(from_attributes=True)


class FishTrackRead(BaseModel):
    id: uuid.UUID
    video_id: uuid.UUID
    processing_job_id: uuid.UUID | None
    review_state: str
    reviewed_at: datetime | None
    viame_track_id: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    detection_count: int
    mean_confidence: float
    max_confidence: float
    species: str | None
    species_confidence: float | None
    # The Fishial answer is reported beside the detector's own class name, never
    # in place of it: the two were produced by different models from different
    # evidence, and an operator reviewing a name needs to know which said what.
    fishial_state: str = "none"
    fishial_species: str | None = None
    fishial_species_confidence: float | None = None
    # And a person's own answer beside both machines'. Present means somebody
    # decided; absent means nobody has, which is not the same as agreeing.
    manual_species: str | None = None
    manual_species_at: datetime | None = None
    detections: list[FishDetectionRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class FishTrackSummaryRead(BaseModel):
    id: uuid.UUID
    video_id: uuid.UUID
    processing_job_id: uuid.UUID | None
    review_state: str
    reviewed_at: datetime | None
    machine_accepted: bool
    viame_track_id: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    detection_count: int
    mean_confidence: float
    max_confidence: float
    species: str | None
    species_confidence: float | None
    # The Fishial answer is reported beside the detector's own class name, never
    # in place of it: the two were produced by different models from different
    # evidence, and an operator reviewing a name needs to know which said what.
    fishial_state: str = "none"
    fishial_species: str | None = None
    fishial_species_confidence: float | None = None
    # And a person's own answer beside both machines'; see FishTrackRead.
    manual_species: str | None = None
    manual_species_at: datetime | None = None
    accepted: bool
    # The threshold this track's own run used, and the review words that follow
    # from it. Sent so the client never has to reclassify a track itself.
    run_threshold: float
    review_categories: list[str]

    model_config = ConfigDict(from_attributes=True)


class TrackClipRead(BaseModel):
    """Metadata for a short cropped clip that follows one fish track."""

    track_id: uuid.UUID
    video_id: uuid.UUID
    viame_track_id: str
    species: str | None
    filename: str
    media_type: str
    size_bytes: int
    fps: float
    width: int
    height: int
    frame_count: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    detection_count: int
    max_confidence: float
    generated_at: datetime
    cached: bool
    url: str
    # Read from the track, never from the clip's cached metadata: a clip on disk
    # long outlives the identification state it was cut under.
    fishial_state: str = "none"
    fishial_species: str | None = None
    fishial_species_confidence: float | None = None
    # And a person's own answer beside both machines'; see FishTrackRead.
    manual_species: str | None = None
    manual_species_at: datetime | None = None
