"""Query vocabularies shared by the video list, the facets and the track table.

These are enums rather than free strings so a typo in a saved view is a 422 the
operator can see, not a filter that silently matches everything.
"""

from __future__ import annotations

import enum


class VideoStatusFilter(str, enum.Enum):
    # "pending" is the dashboard's word for a video that has been uploaded but
    # never queued; the column stores "uploaded". Both select the same rows.
    PENDING = "pending"
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewFilter(str, enum.Enum):
    FLAGGED = "flagged"
    UNREVIEWED = "unreviewed"
    BORDERLINE = "borderline"
    DISPUTED = "disputed"
    DONE = "done"


class ReviewStatusFilter(str, enum.Enum):
    """The per-video roll-up that /videos/facets counts, so it can also be filtered."""

    NA = "n/a"
    AWAITING = "awaiting"
    IN_PROGRESS = "in-progress"
    COMPLETE = "complete"


class ReviewState(str, enum.Enum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs-review"


class VideoSort(str, enum.Enum):
    CREATED_AT = "created_at"
    FILENAME = "filename"
    CAMERA_ID = "camera_id"
    STATUS = "status"
    DURATION = "duration"
    SIZE = "size"
    ACCEPTED_FISH = "accepted_fish"
    DETECTIONS = "detections"
    UNREVIEWED = "unreviewed"
    FLAGGED = "flagged"
    REVIEW_STATUS = "review_status"
    ANNOTATED_AT = "annotated_at"


class TrackSort(str, enum.Enum):
    FIRST_FRAME = "first_frame"
    DURATION = "duration"
    DETECTION_COUNT = "detection_count"
    MEAN_CONFIDENCE = "mean_confidence"
    MAX_CONFIDENCE = "max_confidence"
    SPECIES = "species"
    REVIEW_STATE = "review_state"


class SortOrder(str, enum.Enum):
    ASC = "asc"
    DESC = "desc"
