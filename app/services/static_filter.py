"""Suppress detections of things that never move, for a fixed camera.

Works on any detector's output: rows are ``{"id", "frame", "t", "box", "confidence"}``
dicts (the shape ``scripts/viame_zero_label_benchmark.observations`` returns), with
``id`` None for untracked boxes, ``frame`` the zero-based source frame, ``t`` seconds and
``box`` ``[x1, y1, x2, y2]`` in source pixels.

Three independent flags:

- **still track**: a track that stays in place for a long time (rule A);
- **static zone**: a box shape that recurs in the same place across much of the video
  (rule B), and tracks that sit in such a zone without moving;
- **background match**: a still track (3 s or more) whose pixels match a median
  background image of the video, in which anything present for less than half the video
  (a resting fish) has dropped out.

The filter is deliberately conservative. A track is only removed when it never leaves
its spot: three or more observations away from its median position keep the whole track,
so a fish that rests and then swims away survives, and a few seconds of rest are far
below the still-track and zone durations. A resting fish is absent from the median
background, so its crops do not match it. The thresholds in ``StaticSettings`` were fixed
from the scene's geometry before any detector comparison was looked at; change them only
with a new, documented decision.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StaticSettings:
    # Rule A: a still track. 20 s is several times a resting fish's "few seconds".
    min_still_seconds: float = 20.0
    # An observation is an excursion when its centre is this many median-box diagonals
    # from the track's median centre; this many excursions mean the track moved.
    excursion_diagonals: float = 0.5
    max_excursions: int = 2
    # Median IoU of a still track's boxes with its own median box.
    min_median_iou: float = 0.6
    # Rule B: a static zone is a box shape (IoU >= zone_iou with the zone's median box)
    # seen on at least this share of analysed frames, for at least this many seconds in
    # total, and in at least this share of equal time bins across the video.
    zone_iou: float = 0.6
    min_zone_frame_share: float = 0.25
    min_zone_seconds: float = 20.0
    zone_time_bins: int = 10
    min_zone_bin_share: float = 0.6
    # A track is removed by rule B when this share of its boxes is in static zones and it
    # does not move (same excursion rule as A).
    min_track_zone_share: float = 0.8
    # Zone search reads at most this many frames, evenly spaced (shares are unchanged).
    zone_sample_frames: int = 1500
    # Background match: median normalized cross-correlation of a still track's crops
    # with the median background, for tracks of at least min_background_seconds. It
    # removes since 2026-10-14 (decision static_background_removes): on SAM 3's SmartBay
    # tracks the A and B rules missed fixture fragments shorter than 20 s; every still
    # track matching at >= 0.93 was a fixture, every visible fish (resting ones too) was
    # <= 0.80. Before that it only flagged.
    background_ncc: float = 0.9
    min_background_seconds: float = 3.0
    background_removes: bool = True


DEFAULT_SETTINGS = StaticSettings()


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _ious(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box against an (N, 4) array."""
    w = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    h = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    intersection = w * h
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - intersection)
    return np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """(L, R) IoU matrix."""
    w = np.clip(np.minimum(left[:, None, 2], right[None, :, 2])
                - np.maximum(left[:, None, 0], right[None, :, 0]), 0, None)
    h = np.clip(np.minimum(left[:, None, 3], right[None, :, 3])
                - np.maximum(left[:, None, 1], right[None, :, 1]), 0, None)
    intersection = w * h
    area_l = (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1])
    area_r = (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1])
    union = area_l[:, None] + area_r[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)


def track_motion(rows: list[dict], settings: StaticSettings) -> dict:
    """Stillness measures for one track's observations."""
    boxes = np.array([r["box"] for r in rows], dtype=float)
    median_box = np.median(boxes, axis=0)
    centres = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    median_centre = np.median(centres, axis=0)
    diagonal = max(math.hypot(median_box[2] - median_box[0], median_box[3] - median_box[1]), 1.0)
    distance = np.hypot(*(centres - median_centre).T) / diagonal
    times = [r["t"] for r in rows]
    return {"observations": len(rows), "seconds": max(times) - min(times),
            "median_box": [float(v) for v in median_box],
            "median_iou_with_median_box": float(np.median(_ious(median_box, boxes))),
            "excursions": int((distance > settings.excursion_diagonals).sum()),
            "max_centre_shift_diagonals": float(distance.max())}


def _still(motion: dict, settings: StaticSettings) -> bool:
    return (motion["excursions"] <= settings.max_excursions
            and motion["median_iou_with_median_box"] >= settings.min_median_iou)


def static_zones(rows: list[dict], analysed_frames: list[int], fps: float,
                 settings: StaticSettings = DEFAULT_SETTINGS) -> list[dict]:
    """Box shapes that recur in one place across a large share of the video.

    Greedy: the box with the most same-shape neighbours (IoU >= zone_iou) seeds a zone;
    its members' median box is the zone box; members are taken out and the search
    repeats. ``analysed_frames`` lists the frames the detector looked at, so frames it
    never saw are not counted as absences.
    """
    analysed = sorted(set(analysed_frames))
    if not rows or not analysed:
        return []
    step = max(1, math.ceil(len(analysed) / settings.zone_sample_frames))
    sampled = set(analysed[::step])
    sample = [r for r in rows if r["frame"] in sampled]
    if not sample:
        return []
    boxes = np.array([r["box"] for r in sample], dtype=float)
    frame_of = np.array([r["frame"] for r in sample])
    # Neighbour counts by chunks keep memory bounded (512 x N floats at a time).
    support = np.zeros(len(boxes), dtype=int)
    for start in range(0, len(boxes), 512):
        support[start:start + 512] = (_pairwise_iou(boxes[start:start + 512], boxes)
                                      >= settings.zone_iou).sum(axis=1)
    first, span = analysed[0], analysed[-1] - analysed[0] + 1
    seconds_per_frame = span / fps / len(analysed)  # Time each analysed frame stands for
    taken = np.zeros(len(boxes), dtype=bool)
    zones = []
    for seed in np.argsort(-support, kind="stable"):
        if support[seed] < settings.min_zone_frame_share * len(sampled):
            break
        if taken[seed]:
            continue
        members = (_ious(boxes[seed], boxes) >= settings.zone_iou) & ~taken
        zone_box = np.median(boxes[members], axis=0)
        members = (_ious(zone_box, boxes) >= settings.zone_iou) & ~taken
        if not members.any():
            continue
        taken |= members
        zone_frames = set(frame_of[members].tolist())
        share = len(zone_frames) / len(sampled)
        seconds = share * len(analysed) * seconds_per_frame
        bins = {min(settings.zone_time_bins - 1, (f - first) * settings.zone_time_bins // span)
                for f in zone_frames}
        bin_share = len(bins) / settings.zone_time_bins
        if (share >= settings.min_zone_frame_share and seconds >= settings.min_zone_seconds
                and bin_share >= settings.min_zone_bin_share):
            zones.append({"box": [float(v) for v in zone_box], "frame_share": share,
                          "seconds": seconds, "time_bin_share": bin_share,
                          "sampled_boxes": int(members.sum())})
    return zones


def background_ncc(crop: np.ndarray, background: np.ndarray) -> float | None:
    """Normalized cross-correlation of two equal grey crops; None for a flat crop."""
    a = crop.astype(np.float64).ravel()
    b = background.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    norm = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / norm) if norm > 1e-6 else None


def filter_static(rows: list[dict], *, analysed_frames: list[int], fps: float,
                  settings: StaticSettings = DEFAULT_SETTINGS,
                  background_scores: dict | None = None) -> dict:
    """Split ``rows`` into kept and removed and explain every removal.

    ``background_scores`` optionally maps a track id to the median background NCC of
    its sampled crops (see ``scripts/sam3_botsort.py``); a still track that matches is
    removed while ``settings.background_removes`` is set, and only flagged otherwise.
    """
    zones = static_zones(rows, analysed_frames, fps, settings)
    zone_boxes = np.array([z["box"] for z in zones], dtype=float).reshape(-1, 4)

    def in_zone(row) -> bool:
        if not len(zone_boxes):
            return False
        box = np.asarray(row["box"], dtype=float)
        return bool((_pairwise_iou(box[None], zone_boxes)[0] >= settings.zone_iou).any())

    tracks = defaultdict(list)
    untracked = []
    for row in rows:
        (tracks[row["id"]] if row.get("id") is not None else untracked).append(row)
    decisions = {}
    removed_ids = set()
    for track_id, members in tracks.items():
        members.sort(key=lambda r: r["frame"])
        motion = track_motion(members, settings)
        zone_share = sum(map(in_zone, members)) / len(members)
        still = _still(motion, settings)
        flags = []
        if still and motion["seconds"] >= settings.min_still_seconds:
            flags.append("still_track")
        if still and zone_share >= settings.min_track_zone_share:
            flags.append("static_zone")
        ncc = (background_scores or {}).get(track_id)
        background = (ncc is not None and still and ncc >= settings.background_ncc
                      and motion["seconds"] >= settings.min_background_seconds)
        if background:
            flags.append("background_match")
        remove = ("still_track" in flags or "static_zone" in flags
                  or (background and settings.background_removes))
        decisions[str(track_id)] = motion | {"zone_share": zone_share, "flags": flags,
                                             "background_ncc": ncc, "removed": remove}
        if remove:
            removed_ids.add(track_id)
    kept, removed = [], []
    for track_id, members in tracks.items():
        (removed if track_id in removed_ids else kept).extend(members)
    for row in untracked:  # A lone box can only be judged by where it is.
        (removed if in_zone(row) else kept).append(row)
    order = lambda r: (r["frame"], str(r.get("id")))
    return {"kept": sorted(kept, key=order), "removed": sorted(removed, key=order),
            "zones": zones, "tracks": decisions,
            "counts": {"boxes": len(rows), "kept_boxes": len(kept), "removed_boxes": len(removed),
                       "tracks": len(tracks), "removed_tracks": len(removed_ids),
                       "untracked_boxes": len(untracked),
                       "removed_untracked_boxes": sum(1 for r in removed if r.get("id") is None),
                       "flags": {flag: sum(flag in d["flags"] for d in decisions.values())
                                 for flag in ("still_track", "static_zone", "background_match")}},
            "settings": asdict(settings)}
