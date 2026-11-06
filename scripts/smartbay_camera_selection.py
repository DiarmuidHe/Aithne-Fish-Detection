"""Compare SmartBay camera screening and choose a fixed, day-disjoint window manifest.

The archive scan contains unreviewed SAM 3 screening evidence, not truth. This script is
CPU-only: it reads completed Cam 1 and sampled Cam 2 days, writes condition/yield reports,
assigns whole days to train/validation/test/expert roles, and chooses a camera-balanced
set of 30-second windows for contact-sheet review before any paid pseudo-labelling.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from scripts.smartbay_archive_scan.core import nearby_images
from scripts.smartbay_label_selection import (
    ATTRIBUTION,
    BAD,
    CLIP_FPS,
    CLIP_FRAMES,
    HOUR_BINS,
    MIN_CLIP_GAP,
    TERTILES,
    WINDOW_FRAMES,
    density_bin,
    evidence_of,
    tertile,
    tertile_cuts,
)

CAMERAS = ("SmartBayCam1", "SmartBayCam2")
CAMERA_BUDGETS = {"SmartBayCam1": 120, "SmartBayCam2": 40}
ROLE_SHARES = {"train": 0.60, "validation": 0.15, "test": 0.15, "expert": 0.10}
EVIDENCE_SHARES = {"strong": 0.55, "weak": 0.15, "none": 0.15, "murky": 0.15}
MAX_WINDOWS_PER_DAY = 4
SEED = 17
MANIFEST_FIELDS = [
    "window_id", "camera", "day", "split", "labelling_route", "source_mp4_url",
    "clip_time", "start_frame", "frame_count", "fps", "evidence", "light",
    "light_mean", "clarity", "contrast", "hour_bin", "density", "screening_rank",
    "rank_reason", "png_max_score", "png_detections", "nearby_pngs", "attribution",
]


def largest_remainder(total: int, shares: dict[str, float]) -> dict[str, int]:
    exact = {key: total * share for key, share in shares.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    order = {key: index for index, key in enumerate(shares)}
    for key in sorted(shares, key=lambda item: (-(exact[item] - result[item]), order[item]))[
        : total - sum(result.values())
    ]:
        result[key] += 1
    return result


def completed_days(scan: Path, camera: str, year: int = 2025) -> list[str]:
    path = scan / f"{camera}_{year}" / "year_progress.csv"
    with path.open(encoding="utf-8") as stream:
        return sorted(row["day"] for row in csv.DictReader(stream) if row["status"] == "completed")


def load_candidates(scan: Path) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    clips, images, days_by_camera = [], [], {}
    for camera in CAMERAS:
        days = completed_days(scan, camera)
        days_by_camera[camera] = days
        for day in days:
            directory = scan / f"{camera}_{day}"
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            with (directory / "review_manifest.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            selected = [
                url for url in state.get("selected_png_urls", [])
                if "thumbnail" in state["images"].get(url, {})
            ]
            thumb_index = {url: index for index, url in enumerate(selected)}
            day_images = []
            for entry in state["entries"]:
                if entry["kind"] != "png" or entry["url"] not in state["images"]:
                    continue
                item = {
                    **entry,
                    **state["images"][entry["url"]],
                    "camera": camera,
                    "day": day,
                    "thumb_index": thumb_index.get(entry["url"]),
                }
                day_images.append(item)
                images.append(item)
            entries = {entry["url"]: entry for entry in state["entries"] if entry["kind"] == "mp4"}
            for row in rows:
                entry = entries[row["source_mp4_url"]]
                nearby = nearby_images(entry, day_images)
                sampled = state["clips"].get(entry["url"], {})
                evidence, excluded = evidence_of(
                    row["screening_rank"], row["rank_reason"], nearby,
                    row.get("scan_error", ""), sampled.get("frame_count"),
                )
                seen = [item for item in nearby if item.get("quality") not in BAD and "metrics" in item]
                usable = [item for item in nearby if item.get("quality") == "usable"]
                detections = max((item.get("detection_count", 0) for item in usable), default=0)
                clips.append({
                    "camera": camera,
                    "day": day,
                    "url": entry["url"],
                    "stem": Path(entry["name"]).stem,
                    "clip_time": entry["timestamp"],
                    "rank": row["screening_rank"],
                    "rank_reason": row["rank_reason"],
                    "evidence": evidence,
                    "excluded": excluded,
                    "nearby": nearby,
                    "light_mean": statistics.median(item["metrics"]["mean"] for item in seen) if seen else None,
                    "contrast": statistics.median(item["metrics"]["contrast"] for item in seen) if seen else None,
                    "hour_bin": HOUR_BINS[int(entry["timestamp"][11:13]) // 6],
                    "density": density_bin(detections) if evidence == "strong" else "",
                    "png_detections": detections,
                    "png_max_score": max((item.get("max_score", 0) for item in usable), default=0),
                })
    return clips, images, days_by_camera


def add_condition_bins(clips: list[dict], images: list[dict]) -> dict:
    measured = [item for item in images if item.get("quality") not in BAD and "metrics" in item]
    light_cuts = tertile_cuts([item["metrics"]["mean"] for item in measured])
    clarity_cuts = tertile_cuts([item["metrics"]["contrast"] for item in measured])
    for item in images:
        if "metrics" in item:
            item["light"] = tertile(item["metrics"]["mean"], light_cuts)
            item["clarity"] = tertile(item["metrics"]["contrast"], clarity_cuts)
    for clip in clips:
        if clip["light_mean"] is not None:
            clip["light"] = tertile(clip["light_mean"], light_cuts)
            clip["clarity"] = tertile(clip["contrast"], clarity_cuts)
    return {"light": [round(value, 2) for value in light_cuts],
            "clarity": [round(value, 2) for value in clarity_cuts]}


def _spread_pick(days: list[str], count: int, rng: random.Random) -> list[str]:
    if count > len(days):
        raise ValueError(f"need {count} role days but only {len(days)} are available")
    selected = []
    for index in range(count):
        start = index * len(days) // count
        stop = (index + 1) * len(days) // count
        selected.append(rng.choice(days[start:stop]))
    return selected


def assign_day_roles(
    days: list[str], role_budget: dict[str, int], seed: int,
    *, never_expert: set[str] | None = None,
) -> dict[str, str]:
    """Assign complete days before windows; enough days exist for <=4 windows each."""
    never_expert = never_expert or set()
    required = {
        role: math.ceil(role_budget[role] / MAX_WINDOWS_PER_DAY)
        for role in ("expert", "test", "validation")
    }
    if sum(required.values()) + math.ceil(role_budget["train"] / MAX_WINDOWS_PER_DAY) > len(days):
        raise ValueError("not enough days for the split budgets without exceeding the daily cap")
    rng = random.Random(seed)
    roles: dict[str, str] = {}
    for role in ("expert", "test", "validation"):
        pool = [day for day in days if day not in roles and (role != "expert" or day not in never_expert)]
        for day in _spread_pick(pool, required[role], rng):
            roles[day] = role
    for day in days:
        roles.setdefault(day, "train")
    return dict(sorted(roles.items()))


def _stable_tie(seed: int, *parts: str) -> int:
    raw = "|".join((str(seed), *parts)).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def choose_windows(
    clips: list[dict], roles: dict[str, str], camera: str, budget: int, seed: int
) -> tuple[list[dict], dict]:
    role_budget = largest_remainder(budget, ROLE_SHARES)
    pool = [
        clip for clip in clips
        if clip["camera"] == camera and clip["evidence"] and clip["day"] in roles
        and clip.get("light") in TERTILES and clip.get("clarity") in TERTILES
    ]
    used: set[str] = set()
    per_day: dict[str, list[dict]] = defaultdict(list)
    selected: list[dict] = []
    shortfalls: dict[str, dict[str, int]] = defaultdict(dict)

    def spaced(clip: dict) -> bool:
        stamp = datetime.fromisoformat(clip["clip_time"])
        return all(
            abs(stamp - datetime.fromisoformat(other["clip_time"])) >= MIN_CLIP_GAP
            for other in per_day[clip["day"]]
        )

    for role, role_total in role_budget.items():
        evidence_budget = largest_remainder(role_total, EVIDENCE_SHARES)
        counts = {key: Counter() for key in ("hour", "light", "clarity", "day")}

        def take(clip: dict) -> None:
            available = max(0, CLIP_FRAMES - WINDOW_FRAMES)
            start = random.Random(_stable_tie(seed, clip["url"])).randint(0, available)
            window = {
                **clip,
                "split": role,
                "start_frame": start,
                "frame_count": WINDOW_FRAMES,
                "window_id": f"{clip['stem']}_f{start:04d}",
                "labelling_route": "human_review_only" if role == "expert" else "sam3_then_human_review",
            }
            selected.append(window)
            per_day[clip["day"]].append(window)
            used.add(clip["url"])
            for key, value in (("hour", clip["hour_bin"]), ("light", clip["light"]),
                               ("clarity", clip["clarity"]), ("day", clip["day"])):
                counts[key][value] += 1

        def best(options: list[dict]) -> dict | None:
            viable = [
                clip for clip in options if clip["url"] not in used
                and len(per_day[clip["day"]]) < MAX_WINDOWS_PER_DAY and spaced(clip)
            ]
            if not viable:
                return None
            return min(viable, key=lambda clip: (
                counts["day"][clip["day"]], counts["hour"][clip["hour_bin"]],
                counts["light"][clip["light"]], counts["clarity"][clip["clarity"]],
                _stable_tie(seed, role, clip["url"]),
            ))

        role_pool = [clip for clip in pool if roles[clip["day"]] == role]
        for evidence, wanted in evidence_budget.items():
            for _ in range(wanted):
                clip = best([item for item in role_pool if item["evidence"] == evidence])
                if clip is None:
                    shortfalls[role][evidence] = shortfalls[role].get(evidence, 0) + 1
                    break
                take(clip)
        while sum(window["split"] == role for window in selected) < role_total:
            clip = best(role_pool)
            if clip is None:
                break
            take(clip)

    return sorted(selected, key=lambda item: (item["camera"], item["day"], item["clip_time"])), {
        "target": budget,
        "achieved": len(selected),
        "target_by_split": role_budget,
        "achieved_by_split": dict(Counter(item["split"] for item in selected)),
        "evidence": dict(Counter(item["evidence"] for item in selected)),
        "shortfalls": {key: value for key, value in shortfalls.items() if value},
    }


def metric_row(camera: str, dimension: str, value: str, members: list[dict]) -> dict:
    usable = [item for item in members if item.get("quality") == "usable"]
    strong = [item for item in usable if item.get("max_score", 0) >= 0.5]
    weak = [item for item in usable if 0.25 <= item.get("max_score", 0) < 0.5]
    negative = [item for item in usable if item.get("max_score", 0) < 0.25]
    unusable = [item for item in members if item.get("quality") != "usable"]
    measured = [item for item in members if "metrics" in item]
    return {
        "camera": camera, "dimension": dimension, "value": value, "images": len(members),
        "days": len({item["day"] for item in members}), "usable": len(usable),
        "strong_fish_candidates": len(strong), "weak_fish_candidates": len(weak),
        "negative_candidates": len(negative), "unusable_images": len(unusable),
        "strong_rate_all": round(len(strong) / len(members), 6) if members else 0,
        "strong_rate_usable": round(len(strong) / len(usable), 6) if usable else 0,
        "mean_detections_usable": round(statistics.mean(item.get("detection_count", 0) for item in usable), 4)
        if usable else 0,
        "mean_light": round(statistics.mean(item["metrics"]["mean"] for item in measured), 3)
        if measured else "",
        "mean_contrast": round(statistics.mean(item["metrics"]["contrast"] for item in measured), 3)
        if measured else "",
    }


def condition_metrics(images: list[dict]) -> list[dict]:
    rows = []
    for camera in CAMERAS:
        camera_images = [item for item in images if item["camera"] == camera]
        dimensions = {
            "all": lambda item: "all",
            "hour": lambda item: item["timestamp"][11:13],
            "hour_bin": lambda item: HOUR_BINS[int(item["timestamp"][11:13]) // 6],
            "month": lambda item: item["day"][:7],
            "quality": lambda item: item.get("quality", "unknown"),
            "light": lambda item: item.get("light", "unknown"),
            "clarity": lambda item: item.get("clarity", "unknown"),
            "murk": lambda item: (
                item.get("quality", "unknown")
                if item.get("quality") != "usable"
                else "low-clarity" if item.get("clarity") == "low" else "clearer"
            ),
        }
        for dimension, getter in dimensions.items():
            groups: dict[str, list[dict]] = defaultdict(list)
            for item in camera_images:
                groups[str(getter(item))].append(item)
            rows.extend(metric_row(camera, dimension, value, members)
                        for value, members in sorted(groups.items()))
    return rows


@functools.lru_cache(maxsize=32)
def _sheet(path: str):
    import cv2
    return cv2.imread(path)


def thumbnail(scan: Path, camera: str, day: str, index: int | None):
    if index is None:
        return None
    name = "png_pilot.jpg" if index < 40 else f"png_page_{index // 40 + 1:03d}.jpg"
    sheet = _sheet(str(scan / f"{camera}_{day}" / "contact_sheets" / name))
    if sheet is None:
        return None
    row, column = divmod(index % 40, 4)
    return sheet[row * 252:(row + 1) * 252, column * 400:(column + 1) * 400].copy()


def write_contact_sheet(windows: list[dict], scan: Path, path: Path) -> None:
    import cv2
    import numpy as np
    tiles = []
    for window in windows:
        start = datetime.fromisoformat(window["clip_time"]) + timedelta(
            seconds=window["start_frame"] / CLIP_FPS
        )
        nearest = min(window["nearby"], key=lambda item: abs(datetime.fromisoformat(item["timestamp"]) - start))
        tile = thumbnail(scan, window["camera"], window["day"], nearest.get("thumb_index"))
        if tile is None:
            tile = np.full((252, 400, 3), 60, np.uint8)
        cv2.rectangle(tile, (0, 0), (400, 42), (30, 30, 30), -1)
        cv2.putText(tile, window["window_id"].replace("SmartBay", ""), (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"{window['split']} | {window['evidence']} | {window['hour_bin']} | "
                          f"L {window['light']} C {window['clarity']}", (5, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 220, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    columns = 4
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([
        np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)
    ]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def manifest_row(window: dict) -> dict:
    return {
        "window_id": window["window_id"], "camera": window["camera"], "day": window["day"],
        "split": window["split"], "labelling_route": window["labelling_route"],
        "source_mp4_url": window["url"], "clip_time": window["clip_time"],
        "start_frame": window["start_frame"], "frame_count": window["frame_count"],
        "fps": f"{CLIP_FPS:.6f}", "evidence": window["evidence"], "light": window["light"],
        "light_mean": window["light_mean"], "clarity": window["clarity"],
        "contrast": window["contrast"], "hour_bin": window["hour_bin"],
        "density": window["density"], "screening_rank": window["rank"],
        "rank_reason": window["rank_reason"], "png_max_score": window["png_max_score"],
        "png_detections": window["png_detections"], "nearby_pngs": len(window["nearby"]),
        "attribution": ATTRIBUTION,
    }


def build(scan: Path, output: Path, budgets: dict[str, int], seed: int = SEED) -> dict:
    clips, images, days_by_camera = load_candidates(scan)
    cuts = add_condition_bins(clips, images)
    metrics = condition_metrics(images)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    all_windows, roles_by_camera, selection = [], {}, {}
    touched = {"2025-07-17", "2025-08-01"}
    for index, camera in enumerate(CAMERAS):
        role_budget = largest_remainder(budgets[camera], ROLE_SHARES)
        candidate_days = sorted({clip["day"] for clip in clips if clip["camera"] == camera and clip["evidence"]})
        roles = assign_day_roles(candidate_days, role_budget, seed + index, never_expert=touched)
        windows, result = choose_windows(clips, roles, camera, budgets[camera], seed + index)
        roles_by_camera[camera] = roles
        selection[camera] = result
        all_windows.extend(windows)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_row(window) for window in all_windows)
    review_fields = [
        "window_id", "camera", "day", "split", "evidence", "contact_sheet",
        "decision", "replacement_reason", "review_notes",
    ]
    with (output / "contact_sheet_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        for window in all_windows:
            writer.writerow({
                "window_id": window["window_id"], "camera": window["camera"],
                "day": window["day"], "split": window["split"],
                "evidence": window["evidence"],
                "contact_sheet": f"contact_sheets/{window['camera']}-{window['split']}.jpg",
                "decision": "", "replacement_reason": "", "review_notes": "",
            })
    (output / "day_roles.json").write_text(json.dumps({
        "seed": seed, "roles": roles_by_camera,
        "rule": "whole days assigned before windows; no day crosses train/validation/test/expert",
        "labelling": {"expert": "human review only", "other": "SAM 3 suggestions then human review"},
    }, indent=2), encoding="utf-8")
    for camera in CAMERAS:
        for role in ROLE_SHARES:
            write_contact_sheet(
                [window for window in all_windows if window["camera"] == camera and window["split"] == role],
                scan, output / "contact_sheets" / f"{camera}-{role}.jpg",
            )

    missing = {}
    date_coverage = {}
    for camera in CAMERAS:
        camera_images = [item for item in images if item["camera"] == camera]
        hour_counts = Counter(item["timestamp"][11:13] for item in camera_images)
        completed_days = sorted(days_by_camera[camera])
        date_coverage[camera] = {
            "first_day": completed_days[0] if completed_days else None,
            "last_day": completed_days[-1] if completed_days else None,
            "completed_days": len(completed_days),
            "days_with_screened_pngs": len({item["day"] for item in camera_images}),
            "months": sorted({day[:7] for day in completed_days}),
        }
        missing[camera] = {
            "hours_with_fewer_than_5_images": [f"{hour:02d}" for hour in range(24)
                                                if hour_counts[f"{hour:02d}"] < 5],
            "candidate_evidence_counts": dict(Counter(
                clip["evidence"] for clip in clips if clip["camera"] == camera and clip["evidence"]
            )),
            "selection_shortfalls": selection[camera]["shortfalls"],
        }
    report = {
        "interpretation": "SAM 3 screening evidence and quality metrics; not ground truth",
        "scan_days": {camera: len(days) for camera, days in days_by_camera.items()},
        "screened_pngs": dict(Counter(item["camera"] for item in images)),
        "indexed_clips": dict(Counter(clip["camera"] for clip in clips)),
        "tertile_cuts_global": cuts,
        "budgets": budgets,
        "selection": selection,
        "manifest_windows": len(all_windows),
        "windows_by_camera_split": {
            camera: dict(Counter(window["split"] for window in all_windows if window["camera"] == camera))
            for camera in CAMERAS
        },
        "windows_by_camera_evidence": {
            camera: dict(Counter(window["evidence"] for window in all_windows if window["camera"] == camera))
            for camera in CAMERAS
        },
        "date_coverage": date_coverage,
        "missing_conditions": missing,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = {
        camera: next(row for row in metrics if row["camera"] == camera and
                     row["dimension"] == "all")
        for camera in CAMERAS
    }

    def rows_for(camera: str, dimension: str) -> list[dict]:
        return [row for row in metrics if row["camera"] == camera and row["dimension"] == dimension]

    lines = [
        "# SmartBay camera comparison and fixed-budget selection", "",
        "Screening evidence is unreviewed and is not ground truth.", "",
        f"- Manifest: **{len(all_windows)}** windows ({budgets['SmartBayCam1']} Cam 1, "
        f"{budgets['SmartBayCam2']} Cam 2).",
        f"- Screened PNGs: Cam 1 {report['screened_pngs'].get('SmartBayCam1', 0):,}; "
        f"Cam 2 {report['screened_pngs'].get('SmartBayCam2', 0):,}.",
        f"- Global light tertile cuts: {cuts['light']}; clarity cuts: {cuts['clarity']}.", "",
        "## Camera comparison", "",
        "| Camera | Images | Strong | Weak | Negative | Unusable | Strong/all | Strong/usable | Mean light | Mean contrast |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for camera in CAMERAS:
        row = overall[camera]
        lines.append(
            f"| {camera} | {row['images']:,} | {row['strong_fish_candidates']:,} | "
            f"{row['weak_fish_candidates']:,} | {row['negative_candidates']:,} | "
            f"{row['unusable_images']:,} | {row['strong_rate_all']:.1%} | "
            f"{row['strong_rate_usable']:.1%} | {row['mean_light']} | {row['mean_contrast']} |"
        )
    lines += ["", "Cam 2 is a small, seasonally spread sample, so its rates are less stable than Cam 1.", "",
              "## Hourly candidate rate", ""]
    for camera in CAMERAS:
        hourly = [row for row in rows_for(camera, "hour") if row["images"] >= 5]
        best = sorted(hourly, key=lambda row: (-row["strong_rate_all"], row["value"]))[:5]
        worst = sorted(hourly, key=lambda row: (row["strong_rate_all"], row["value"]))[:5]
        lines += [f"### {camera}", "",
                  "- Highest: " + ", ".join(
                      f"{row['value']}:00 {row['strong_rate_all']:.1%} (n={row['images']})" for row in best
                  ),
                  "- Lowest: " + ", ".join(
                      f"{row['value']}:00 {row['strong_rate_all']:.1%} (n={row['images']})" for row in worst
                  ), ""]
    lines += ["## Environment and image quality", "",
              "Rates below are strong-candidate rate over all screened images in each bin.", ""]
    for camera in CAMERAS:
        lines += [f"### {camera}", ""]
        for dimension in ("light", "clarity", "murk"):
            values = ", ".join(
                f"{row['value']} {row['strong_rate_all']:.1%} (n={row['images']})"
                for row in rows_for(camera, dimension)
            )
            lines.append(f"- {dimension.title()}: {values}")
        lines += [f"- Mean light / contrast: {overall[camera]['mean_light']} / "
                  f"{overall[camera]['mean_contrast']}", ""]
    lines += ["## Seasonal and date coverage", ""]
    for camera in CAMERAS:
        coverage = date_coverage[camera]
        month_rows = rows_for(camera, "month")
        lines += [f"### {camera}", "",
                  f"- Completed scan days: {coverage['completed_days']} from "
                  f"{coverage['first_day']} to {coverage['last_day']}.",
                  f"- Months: {', '.join(coverage['months'])}.",
                  "- Monthly strong rates: " + ", ".join(
                      f"{row['value']} {row['strong_rate_all']:.1%} (n={row['images']})"
                      for row in month_rows
                  ), ""]
    lines += ["## Conditions still missing or thin", ""]
    for camera in CAMERAS:
        hours = missing[camera]["hours_with_fewer_than_5_images"]
        lines += [f"- {camera}: hours with fewer than five screened images: "
                  f"{', '.join(hours) if hours else 'none'}; selection stratum shortfalls: "
                  f"{missing[camera]['selection_shortfalls'] or 'none'}."]
    lines += ["", "Cam 2 especially needs more low-light/low-clarity negatives and murky footage; "
              "the current 12-day sample is sparse in several hours. Cam 1 has no sampled September-December "
              "coverage in this result set.", "", "## Selection", ""]
    for camera in CAMERAS:
        lines += [f"### {camera}", "", f"- Splits: {selection[camera]['achieved_by_split']}",
                  f"- Evidence: {selection[camera]['evidence']}",
                  f"- Shortfalls: {selection[camera]['shortfalls'] or 'none'}", ""]
    lines += [
        "## Review gate", "",
        "Inspect every contact sheet before downloading or pseudo-labelling videos. Reject or replace obvious",
        "mesh, vegetation, glare, duplicate-session, black, or unusable candidates. Expert windows are routed",
        "to human review only; model suggestions must not be treated as verified labels.", "",
        "See `condition_metrics.csv` for camera/hour/month/quality/light/clarity candidate rates and",
        "`report.json` for missing-condition and provenance details.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=Path("data/archive_scan"))
    parser.add_argument("--output", type=Path, default=Path("data/outputs/smartbay-camera-balanced-20261029"))
    parser.add_argument("--cam1-windows", type=int, default=CAMERA_BUDGETS["SmartBayCam1"])
    parser.add_argument("--cam2-windows", type=int, default=CAMERA_BUDGETS["SmartBayCam2"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    budgets = {"SmartBayCam1": args.cam1_windows, "SmartBayCam2": args.cam2_windows}
    if any(value < 1 for value in budgets.values()):
        parser.error("camera window budgets must be positive")
    print(json.dumps(build(args.scan, args.output, budgets, args.seed), indent=2))


if __name__ == "__main__":
    main()
