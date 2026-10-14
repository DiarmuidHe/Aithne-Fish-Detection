"""Ultralytics BoT-SORT or ByteTrack over boxes from any detector (VIAME, SAM 3, ...).

Ultralytics trackers take a Results-like object, so ``Detections`` wraps plain arrays.
Ultralytics is AGPL-3.0 (or an Ultralytics Enterprise licence); importing this module's
trackers pulls it in, so keep it to experiments unless that licence is acceptable.

Two choices make the output comparable with VIAME's own ByteTrack:

- A track's first box is reported. Ultralytics, like VIAME, only confirms a new track when
  the next frame matches it, and then outputs it from that frame on; the unconfirmed first
  box is held and emitted once the track is confirmed (VIAME's CSV keeps it too).
- Boxes and scores are the detector's own, picked by detection index, not the Kalman
  filter's smoothed box.

``track_buffer`` counts tracker calls (frames fed), as VIAME's does: 30 calls are 3 s at
10 FPS and 7.5 s at 4 FPS. With ``gmc_method`` "none", or no image passed, BoT-SORT does
no camera-motion compensation, which suits a fixed camera. ReID stays off: with external
boxes the "auto" ReID encoder has no detector features to use, and a separate ReID model
is not an approved download.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class TrackerSettings:
    kind: str = "botsort"  # botsort | bytetrack
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "none"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False

    def tag(self) -> str:
        """Compact (Windows paths are short): bot-h45n50l10-b30-m80-f1-gnone."""
        percent = lambda value: f"{round(value * 100):02d}"
        return (f"{self.kind[:4]}-h{percent(self.track_high_thresh)}n{percent(self.new_track_thresh)}"
                f"l{percent(self.track_low_thresh)}-b{self.track_buffer}-m{percent(self.match_thresh)}"
                f"-f{int(self.fuse_score)}-g{self.gmc_method[:6]}")


# Ultralytics' shipped defaults (cfg/trackers/botsort.yaml in 8.4) with GMC off, and
# VIAME's production ByteTrack thresholds (high .45, new track .50, low .10, buffer 30).
ULTRALYTICS_DEFAULT = TrackerSettings()
VIAME_PRODUCTION_THRESHOLDS = TrackerSettings(track_high_thresh=.45, new_track_thresh=.50,
                                              track_low_thresh=.10, track_buffer=30)


class Detections:
    """The Results-like view Ultralytics trackers read: conf, cls, xyxy and xywh arrays
    that support boolean indexing."""

    def __init__(self, xyxy, conf, cls=None):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = (np.zeros(len(self.conf), dtype=np.float32) if cls is None
                    else np.asarray(cls, dtype=np.float32).reshape(-1))
        if not len(self.xyxy) == len(self.conf) == len(self.cls):
            raise ValueError("Boxes, scores and classes differ in length")

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> Detections:
        return Detections(self.xyxy[index], self.conf[index], self.cls[index])


def make_tracker(settings: TrackerSettings):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker

    if settings.with_reid:
        raise ValueError("ReID needs a separate model, which is not approved here")
    args = SimpleNamespace(**{k: v for k, v in asdict(settings).items() if k != "kind"},
                           tracker_type=settings.kind, model="auto")
    if settings.kind == "botsort":
        return BOTSORT(args)
    if settings.kind == "bytetrack":
        return BYTETracker(args)
    raise ValueError(f"Unknown tracker {settings.kind!r}")


class BoxTracker:
    """One tracker over a stream of frames; keeps its state between ``update`` calls, so
    a caller can carry tracks across video segments or reset them."""

    def __init__(self, settings: TrackerSettings = ULTRALYTICS_DEFAULT):
        self.settings = settings
        self.tracker = make_tracker(settings)
        self.pending: dict[int, dict] = {}

    def reset(self) -> None:
        self.tracker.reset()
        self.pending.clear()

    def update(self, frame: int, boxes, scores, image=None, **extra) -> list[dict]:
        """Rows ``{"id", "frame", "box", "confidence", "det_index"}`` reported for this
        frame, including the held first box of any track confirmed now. ``extra`` keys
        are copied onto each row (for example a timestamp)."""
        detections = Detections(boxes, scores)
        output = self.tracker.update(detections, image if self.settings.gmc_method != "none" else None)
        rows = []
        for result in output:
            track_id, index = int(result[4]), int(result[7])
            held = self.pending.pop(track_id, None)
            if held is not None:
                rows.append(held)
            rows.append(self._row(track_id, frame, detections, index, extra))
        # New tracks are unconfirmed until the next frame matches them: hold their box.
        for track in self.tracker.tracked_stracks:
            if not track.is_activated and track.start_frame == self.tracker.frame_id:
                self.pending[track.track_id] = self._row(track.track_id, frame, detections,
                                                         int(track.idx), extra)
        live = {t.track_id for t in self.tracker.tracked_stracks}
        self.pending = {k: v for k, v in self.pending.items() if k in live}
        return rows

    @staticmethod
    def _row(track_id: int, frame: int, detections: Detections, index: int, extra: dict) -> dict:
        return {"id": track_id, "frame": frame, "box": [float(v) for v in detections.xyxy[index]],
                "confidence": float(detections.conf[index]), "det_index": index} | extra


def track_stream(frames: Iterable[tuple[int, np.ndarray, np.ndarray]],
                 settings: TrackerSettings = ULTRALYTICS_DEFAULT) -> Iterator[dict]:
    """Track ``(frame, boxes_xyxy, scores)`` items fed in order; frames with no boxes
    must still be fed (with empty arrays) so lost tracks age correctly."""
    tracker = BoxTracker(settings)
    for frame, boxes, scores in frames:
        yield from tracker.update(frame, boxes, scores)


def with_overrides(settings: TrackerSettings, **values) -> TrackerSettings:
    return replace(settings, **values)
