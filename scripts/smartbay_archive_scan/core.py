"""Pure archive parsing, matching, quality, ranking, and pilot selection logic."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlparse

ROOT = "https://smartbay.marine.ie/data/video/"
NAME = re.compile(
    r"^(SmartBayCam[12])_(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.(png|mp4)$", re.IGNORECASE
)
RANK_ORDER = {"likely_fish": 0, "uncertain": 1, "candidate_clear_and_empty": 2, "unusable": 3}


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def day_url(camera: str, day: str) -> str:
    if camera not in {"SmartBayCam1", "SmartBayCam2"}:
        raise ValueError("camera must be SmartBayCam1 or SmartBayCam2")
    parsed_day = date.fromisoformat(day)
    if parsed_day.isoformat() != day:
        raise ValueError("day must be YYYY-MM-DD")
    return f"{ROOT}{camera}/{parsed_day:%Y/%m/%d}/"


def year_url(camera: str, year: int) -> str:
    day_url(camera, f"{year:04d}-01-01")
    return f"{ROOT}{camera}/{year:04d}/"


def parse_calendar_index(html: str, camera: str, year: int, month: int | None = None) -> list:
    """Read direct month or day folders, rejecting off-origin and invalid dates."""
    base = year_url(camera, year) if month is None else f"{year_url(camera, year)}{month:02d}/"
    if month is not None and not 1 <= month <= 12:
        raise ValueError("month must be 1..12")
    parser = _Links()
    parser.feed(html)
    values = set()
    for href in parser.hrefs:
        parsed = urlparse(urljoin(base, href))
        if (parsed.scheme, parsed.netloc, parsed.query, parsed.fragment) != (
            "https",
            "smartbay.marine.ie",
            "",
            "",
        ):
            continue
        if not parsed.path.startswith(urlparse(base).path):
            continue
        child = parsed.path[len(urlparse(base).path) :]
        if not re.fullmatch(r"\d{2}/", child):
            continue
        number = int(child[:2])
        if month is None:
            if 1 <= number <= 12:
                values.add(number)
        else:
            try:
                values.add(date(year, month, number).isoformat())
            except ValueError:
                continue
    return sorted(values)


def parse_index(html: str, camera: str, day: str) -> list[dict]:
    """Accept only direct PNG/MP4 files of the requested camera and day."""
    base = day_url(camera, day)
    parser = _Links()
    parser.feed(html)
    entries = {}
    for href in parser.hrefs:
        url = urljoin(base, href)
        parsed = urlparse(url)
        if (parsed.scheme, parsed.netloc, parsed.query, parsed.fragment) != (
            "https",
            "smartbay.marine.ie",
            "",
            "",
        ):
            continue
        if not parsed.path.startswith(urlparse(base).path):
            continue
        name = unquote(parsed.path.rsplit("/", 1)[-1])
        if parsed.path != urlparse(base).path + name:
            continue
        match = NAME.fullmatch(name)
        if not match or match.group(1) != camera or match.group(2) != day:
            continue
        hour, minute = int(match.group(3)), int(match.group(4))
        if hour > 23 or minute > 59:
            continue
        stamp = datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00")
        entries[url] = {
            "url": url,
            "name": name,
            "kind": match.group(5).lower(),
            "timestamp": stamp.isoformat(),
        }
    return sorted(entries.values(), key=lambda e: (e["timestamp"], e["kind"], e["url"]))


def choose_pngs(entries: list[dict], limit: int | None = 30) -> list[dict]:
    """Sample short runs across the day, or return every PNG when limit is None."""
    images = [e for e in entries if e["kind"] == "png"]
    if limit is None:
        return images
    if not 1 <= limit <= 30:
        raise ValueError("sampled PNG limit must be 1..30")
    if len(images) <= limit:
        return images
    groups = (limit + 2) // 3
    starts = [round(i * (len(images) - 3) / max(groups - 1, 1)) for i in range(groups)]
    chosen = {j for start in starts for j in range(start, min(start + 3, len(images)))}
    return [images[i] for i in sorted(chosen)[:limit]]


def nearby_images(clip: dict, images: list[dict], minutes: int = 3) -> list[dict]:
    """A clip is associated with PNGs from 1 minute before to 3 minutes after start."""
    start = datetime.fromisoformat(clip["timestamp"])
    return [
        image
        for image in images
        if -timedelta(minutes=1)
        <= datetime.fromisoformat(image["timestamp"]) - start
        <= timedelta(minutes=minutes)
    ]


def rank_clip(nearby: list[dict]) -> tuple[str, str]:
    usable = [p for p in nearby if p.get("quality") == "usable"]
    if any(p.get("max_score", 0) >= 0.5 for p in usable):
        return "likely_fish", "nearby PNG has SAM 3 score >= 0.50"
    if not nearby or any(p.get("error") for p in nearby):
        return "uncertain", "no reliable nearby PNG coverage"
    if not usable:
        return "unusable", "all nearby PNGs fail quality screening"
    if any(p.get("max_score", 0) >= 0.25 for p in usable):
        return "uncertain", "nearby PNG has weaker SAM 3 evidence"
    if len(usable) >= 2:
        return (
            "candidate_clear_and_empty",
            "two usable PNGs have no SAM 3 detections; manual review required",
        )
    return "uncertain", "one usable PNG without detection is insufficient"


def select_clips(clips: list[dict], limit: int = 3, seed: int = 17) -> list[dict]:
    """Reserve one slot for a seeded random clip without nearby detections."""
    import random

    if not 1 <= limit <= 3:
        raise ValueError("pilot MP4 limit must be 1..3")
    candidates = [c for c in clips if c["rank"] != "unusable" and c.get("nearby_png_count", 0)]
    no_detection = [c for c in candidates if c.get("png_detection_count", 0) == 0]
    if not no_detection:
        no_detection = [
            c for c in clips if c["rank"] == "uncertain" and c.get("png_detection_count", 0) == 0
        ]
    random_pick = random.Random(seed).choice(no_detection) if no_detection else None
    chosen = []
    if random_pick:
        chosen.append({**random_pick, "selection_reason": "random_no_detection_review"})
    for clip in sorted(
        candidates,
        key=lambda c: (RANK_ORDER[c["rank"]], -c.get("png_max_score", 0), c["timestamp"]),
    ):
        if len(chosen) >= limit:
            break
        if clip["url"] not in {c["url"] for c in chosen}:
            chosen.append({**clip, "selection_reason": "ranked_sample"})
    return chosen


def random_no_detection_controls(
    clips: list[dict], selected_urls: set[str], count: int = 3, seed: int = 17
) -> set[str]:
    """Random manual-review links with no associated PNG detections.

    Zero PNG coverage is kept distinct from two usable zero-detection PNGs in the
    manifest. These are review controls, never evidence of an empty clip.
    """
    import random

    pool = [
        c for c in clips if c["url"] not in selected_urls and c.get("png_detection_count", 0) == 0
    ]
    covered = [c for c in pool if c.get("nearby_png_count", 0) > 0]
    source = covered or pool
    return {c["url"] for c in random.Random(seed).sample(source, min(count, len(source)))}


def image_quality(image, previous=None) -> tuple[str, dict]:
    """Conservative visual heuristics; flags are triage, not human labels."""
    import cv2
    import numpy as np

    small = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    contrast = float(gray.std())
    dark = float(np.mean(gray < 18))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics = {
        "mean": round(mean, 2),
        "contrast": round(contrast, 2),
        "dark_fraction": round(dark, 4),
        "sharpness": round(sharpness, 2),
    }
    if mean < 14 or dark > 0.98:
        return "black", metrics
    if contrast < 5 and sharpness < 2:
        return "camera_off", metrics
    if previous is not None:
        prev = cv2.cvtColor(
            cv2.resize(previous, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
        )
        delta = cv2.absdiff(gray, prev)
        metrics["previous_mean_difference"] = round(float(delta.mean()), 3)
        metrics["previous_static_fraction"] = round(float(np.mean(delta <= 2)), 4)
        if (
            metrics["previous_mean_difference"] < 0.4
            and metrics["previous_static_fraction"] > 0.995
        ):
            return "frozen", metrics
    if contrast < 12 or sharpness < 4:
        return "obscured", metrics
    return "usable", metrics
