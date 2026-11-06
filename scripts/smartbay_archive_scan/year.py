"""Oldest-first traversal of one SmartBay camera/year with a durable day ledger."""

from __future__ import annotations

import csv
import html
import json
import random
import time
from pathlib import Path

from .core import parse_calendar_index, year_url
from .workflow import OUTPUT, _download, _save, run


def load_year_state(
    path: Path,
    camera: str,
    year: int,
    png_limit: int | None,
    mp4_limit: int,
    seed: int,
    sample_days: int | None = None,
) -> dict:
    source = year_url(camera, year)
    settings = {"png_limit_per_day": png_limit, "mp4_limit_per_day": mp4_limit, "seed": seed}
    if sample_days is not None:
        settings["sample_days"] = sample_days
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state["camera"], state["year"], state["settings"]) != (camera, year, settings):
            raise ValueError("year resume requires the original camera, year, and scan limits")
        return state
    return {
        "version": 1,
        "camera": camera,
        "year": year,
        "source_index": source,
        "settings": settings,
        "discovered_days": [],
        "selected_days": [],
        "completed_days": {},
        "current_day": None,
        "last_completed_day": None,
        "errors": [],
        "created_temp_files": [],
        "usage": {"bytes_transferred": 0, "peak_temp_bytes": 0},
    }


def select_sample_days(days: list[str], count: int, seed: int) -> list[str]:
    """Choose one seeded day from each consecutive slice of the available season."""
    if count < 1:
        raise ValueError("sample_days must be positive")
    if count >= len(days):
        return list(days)
    rng = random.Random(seed)
    selected = []
    for index in range(count):
        start = index * len(days) // count
        stop = (index + 1) * len(days) // count
        selected.append(rng.choice(days[start:stop]))
    return selected


def discover_days(state: dict, path: Path) -> list[str]:
    """Re-read folder indexes so later runs can find days added since the last run."""
    camera, year = state["camera"], state["year"]
    root = year_url(camera, year)
    year_html = _download(root, state, path, max_bytes=1_000_000).decode("utf-8")
    months = parse_calendar_index(year_html, camera, year)
    if not months:
        raise RuntimeError(f"No month folders found at {root}")
    days = []
    for month in months:
        month_url = f"{root}{month:02d}/"
        month_html = _download(month_url, state, path, max_bytes=1_000_000).decode("utf-8")
        days.extend(parse_calendar_index(month_html, camera, year, month))
    state["discovered_days"] = sorted(set(days))
    _save(path, state)
    return state["discovered_days"]


def _progress(state: dict, path: Path) -> None:
    rows = []
    with (path.parent / "year_progress.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["day", "status", "day_report"])
        for day in state["discovered_days"]:
            status = (
                "completed"
                if day in state["completed_days"]
                else "in_progress"
                if day == state["current_day"]
                else "pending"
            )
            report = f"../{state['camera']}_{day}/index.html" if status == "completed" else ""
            writer.writerow([day, status, report])
            rows.append((day, status, report))
    page = [
        "<!doctype html><meta charset='utf-8'><title>SmartBay year scan</title>",
        f"<h1>{html.escape(state['camera'])} {state['year']} archive scan</h1>",
        "<p>Local analyst review only. SAM 3 outputs are unreviewed.</p>",
        f"<p>Last completed day: {html.escape(state['last_completed_day'] or 'none')}</p>",
        (
            "<p><a href='year_progress.csv'>Progress CSV</a> · "
            "<a href='year_summary.json'>Resource summary</a></p>"
        ),
        "<table border='1'><tr><th>Day</th><th>Status</th><th>Review</th></tr>",
    ]
    for day, status, report in rows:
        link = f"<a href='{html.escape(report, quote=True)}'>Open review</a>" if report else ""
        page.append(f"<tr><td>{html.escape(day)}</td><td>{status}</td><td>{link}</td></tr>")
    page.append("</table>")
    (path.parent / "index.html").write_text("\n".join(page), encoding="utf-8")


def _summary(state: dict, path: Path) -> dict:
    complete = state["completed_days"]
    target_days = state.get("selected_days") or state["discovered_days"]
    summaries = [item["summary"] for item in complete.values()]
    current_day = state["current_day"]
    partial = None
    if current_day and current_day not in complete:
        day_path = OUTPUT / f"{state['camera']}_{current_day}" / "state.json"
        if day_path.is_file():
            partial = json.loads(day_path.read_text(encoding="utf-8"))
    partial_entries = partial["entries"] if partial else []
    partial_images = partial["images"] if partial else {}
    partial_clips = partial["clips"] if partial else {}
    partial_png_urls = partial["selected_png_urls"] if partial else []
    partial_clip_urls = partial["selected_clip_urls"] if partial else []
    partial_usage = partial["usage"] if partial else {}
    result = {
        "camera": state["camera"],
        "year": state["year"],
        "png_limit_per_day": state["settings"]["png_limit_per_day"],
        "mp4_limit_per_day": state["settings"]["mp4_limit_per_day"],
        "discovered_days": len(state["discovered_days"]),
        "selected_days": len(target_days),
        "unselected_days": len(set(state["discovered_days"]) - set(target_days)),
        "completed_days": len(complete),
        "last_completed_day": state["last_completed_day"],
        "current_day": state["current_day"],
        "pending_days": len(set(target_days) - set(complete)),
        "indexed_pngs": sum(s["indexed_pngs"] for s in summaries)
        + sum(e["kind"] == "png" for e in partial_entries),
        "indexed_mp4s": sum(s["indexed_mp4s"] for s in summaries)
        + sum(e["kind"] == "mp4" for e in partial_entries),
        "processed_pngs": sum(s["processed_pngs"] for s in summaries)
        + sum(url in partial_images and "error" not in partial_images[url] for url in partial_png_urls),
        "processed_mp4s": sum(s["processed_mp4s"] for s in summaries)
        + sum("frames" in partial_clips.get(url, {}) for url in partial_clip_urls),
        "bytes_transferred": (
            state["usage"]["bytes_transferred"]
            + sum(s["usage"]["bytes_transferred"] for s in summaries)
            + partial_usage.get("bytes_transferred", 0)
        ),
        "gpu_inference_seconds": sum(s["usage"]["gpu_inference_seconds"] for s in summaries)
        + partial_usage.get("gpu_inference_seconds", 0),
        "index_errors": len(state["errors"]),
        "current_day_file_errors": len(partial["errors"]) if partial else 0,
        "progress_file": str(path.parent / "year_progress.csv"),
    }
    (path.parent / "year_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_year(
    camera: str,
    year: int,
    png_limit: int | None = 30,
    mp4_limit: int = 3,
    seed: int = 17,
    max_days: int | None = None,
    retry_incomplete: int = 0,
    retry_delay_seconds: float = 5.0,
    sample_days: int | None = None,
) -> dict:
    """Process every listed day in ascending order; resume the first unfinished day."""
    if (png_limit is not None and not 1 <= png_limit <= 30) or not 1 <= mp4_limit <= 3:
        raise ValueError("per-day sample limits are 1..30 PNGs and 1..3 MP4s")
    if max_days is not None and max_days < 1:
        raise ValueError("max_days must be positive")
    if retry_incomplete < 0:
        raise ValueError("retry_incomplete must be non-negative")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    if sample_days is not None and sample_days < 1:
        raise ValueError("sample_days must be positive")
    path = OUTPUT / f"{camera}_{year:04d}" / "year_state.json"
    state = load_year_state(path, camera, year, png_limit, mp4_limit, seed, sample_days)
    _save(path, state)
    days = discover_days(state, path)
    if sample_days is not None:
        selected = select_sample_days(days, sample_days, seed)
        if state.get("selected_days") and state["selected_days"] != selected:
            raise ValueError("year resume requires the original sampled days")
        state["selected_days"] = selected
        _save(path, state)
        days = selected
    _progress(state, path)
    _summary(state, path)
    detector_cache = {}
    newly_completed = 0
    for day in days:
        if day in state["completed_days"]:
            continue
        state["current_day"] = day
        _save(path, state)
        _progress(state, path)
        print(f"{camera} {day}: scanning", flush=True)
        result = None
        for attempt in range(retry_incomplete + 1):
            try:
                result = run(camera, day, png_limit, mp4_limit, seed, detector_cache)
            except Exception as error:
                state["errors"].append({"day": day, "message": str(error)})
                _save(path, state)
                _progress(state, path)
                _summary(state, path)
                raise
            complete = (
                result["processed_pngs"] == result["selected_pngs"]
                and result["processed_mp4s"] == result["selected_mp4s"]
            )
            if complete or attempt == retry_incomplete:
                break
            print(
                f"{camera} {day}: incomplete; retry {attempt + 1}/{retry_incomplete} "
                f"after {retry_delay_seconds:g}s",
                flush=True,
            )
            if retry_delay_seconds:
                time.sleep(retry_delay_seconds)
        assert result is not None
        if (
            result["processed_pngs"] != result["selected_pngs"]
            or result["processed_mp4s"] != result["selected_mp4s"]
        ):
            state["errors"].append(
                {"day": day, "message": "incomplete per-file scan; retry this day"}
            )
            _save(path, state)
            _progress(state, path)
            _summary(state, path)
            break
        state["completed_days"][day] = {"summary": result}
        state["last_completed_day"] = day
        state["current_day"] = None
        _save(path, state)
        _progress(state, path)
        _summary(state, path)
        print(f"{camera} {day}: complete", flush=True)
        newly_completed += 1
        if max_days is not None and newly_completed >= max_days:
            break
    return _summary(state, path)
