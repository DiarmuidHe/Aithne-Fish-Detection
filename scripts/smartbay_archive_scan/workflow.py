"""Resumable, analyst-only SmartBay archive scan for one day or camera/year."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .core import (
    choose_pngs,
    day_url,
    image_quality,
    nearby_images,
    parse_index,
    random_no_detection_controls,
    rank_clip,
    select_clips,
)

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "data" / "archive_scan"
AGENT = "Aithne local dataset archive pilot/1.0"


class StateSaveError(OSError):
    """The resume ledger could not be persisted after transient-lock retries."""


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    payload = json.dumps(state, indent=2, sort_keys=True)
    for attempt in range(10):
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError as error:
            if attempt == 9:
                raise StateSaveError(f"could not save resume state at {path}: {error}") from error
            time.sleep(min(0.1 * 2**attempt, 1.0))


def load_state(path: Path, camera: str, day: str) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state["camera"], state["day"]) != (camera, day):
            raise ValueError("existing pilot state is for a different camera/day")
        return state
    return {
        "version": 1,
        "camera": camera,
        "day": day,
        "source_index": day_url(camera, day),
        "entries": [],
        "indexed": False,
        "selected_png_urls": [],
        "selected_clip_urls": [],
        "images": {},
        "clips": {},
        "errors": [],
        "created_temp_files": [],
        "usage": {
            "bytes_transferred": 0,
            "gpu_inference_seconds": 0.0,
            "gpu_model_load_seconds": 0.0,
            "peak_temp_bytes": 0,
        },
    }


def _download(
    url: str,
    state: dict,
    state_path: Path,
    *,
    target: Path | None = None,
    max_bytes: int = 80_000_000,
) -> bytes | None:
    """Count every received byte, including interrupted transfers."""
    received = 0
    last_checkpoint_bytes = 0
    last_checkpoint_at = time.monotonic()
    chunks = []
    if target is not None:
        relative = str(target.relative_to(state_path.parent))
        if relative not in state["created_temp_files"]:
            state["created_temp_files"].append(relative)
            _save(state_path, state)
        if target.exists():
            target.unlink()  # exact path registered as created by this pilot
        target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(Request(url, headers={"User-Agent": AGENT}), timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError(f"remote file exceeds {max_bytes:,} byte safety cap: {url}")
            stream = target.open("wb") if target else None
            try:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > max_bytes:
                        raise ValueError(f"transfer exceeded {max_bytes:,} byte safety cap: {url}")
                    state["usage"]["bytes_transferred"] += len(chunk)
                    if stream:
                        stream.write(chunk)
                        state["usage"]["peak_temp_bytes"] = max(
                            state["usage"]["peak_temp_bytes"], received
                        )
                    else:
                        chunks.append(chunk)
                    now = time.monotonic()
                    if (
                        last_checkpoint_bytes == 0
                        or received - last_checkpoint_bytes >= 4 * 1024 * 1024
                        or now - last_checkpoint_at >= 5
                    ):
                        _save(state_path, state)
                        last_checkpoint_bytes = received
                        last_checkpoint_at = now
            finally:
                if stream:
                    stream.close()
        return None if target else b"".join(chunks)
    finally:
        _save(state_path, state)


def _thumbnail(image, caption: str, boxes=None) -> object:
    import cv2

    thumb = cv2.resize(image, (400, 225), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    for box in (boxes or [])[:20]:
        x1, y1, x2, y2 = box
        cv2.rectangle(
            thumb,
            (round(x1 * 400 / w), round(y1 * 225 / h)),
            (round(x2 * 400 / w), round(y2 * 225 / h)),
            (0, 165, 255),
            1,
        )
    canvas = cv2.copyMakeBorder(thumb, 0, 27, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(
        canvas,
        caption[:59],
        (5, 244),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _contact_sheet(paths: list[Path], destination: Path) -> None:
    import cv2
    import numpy as np

    thumbnails = [cv2.imread(str(p)) for p in paths if p.is_file()]
    thumbnails = [t for t in thumbnails if t is not None]
    if not thumbnails:
        return
    width, height, cols = 400, 252, 4
    rows = (len(thumbnails) + cols - 1) // cols
    sheet = np.full((rows * height, cols * width, 3), 30, dtype=np.uint8)
    for index, thumb in enumerate(thumbnails):
        y, x = divmod(index, cols)
        sheet[y * height : (y + 1) * height, x * width : (x + 1) * width] = thumb
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _detector(state: dict, state_path: Path, cached: dict):
    if "value" not in cached:
        from .detector import FishDetector

        started = time.perf_counter()
        cached["value"] = FishDetector()
        state["usage"]["gpu_model_load_seconds"] += time.perf_counter() - started
        _save(state_path, state)
    return cached["value"]


def _record_error(state: dict, path: Path, stage: str, url: str, error: Exception) -> None:
    state["errors"].append(
        {
            "stage": stage,
            "url": url,
            "message": str(error),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _save(path, state)


def index(state: dict, path: Path) -> None:
    if state.get("indexed") or state["entries"]:
        return
    raw = _download(state["source_index"], state, path, max_bytes=5_000_000)
    listing = raw.decode("utf-8")
    entries = parse_index(listing, state["camera"], state["day"])
    if not entries and f"Index of {urlparse(state['source_index']).path}" not in listing:
        raise RuntimeError("day response is not a recognized archive index")
    if not entries and re.search(
        r'href="SmartBayCam[12][^\"]*\.(?:png|mp4)"', listing, re.IGNORECASE
    ):
        raise RuntimeError("day contains camera media with unrecognized names")
    state["entries"] = entries
    state["indexed"] = True
    _save(path, state)


def scan_images(state: dict, path: Path, limit: int | None, cached: dict) -> None:
    import cv2
    import numpy as np

    desired = [e["url"] for e in choose_pngs(state["entries"], limit)]
    if not state["selected_png_urls"] or (limit is None and state["selected_png_urls"] != desired):
        state["selected_png_urls"] = desired
        _save(path, state)
    entries = {e["url"]: e for e in state["entries"]}
    previous = None
    previous_stamp = None
    for url in state["selected_png_urls"]:
        if url in state["images"] and "error" not in state["images"][url]:
            saved = state["images"][url]
            reference = (
                path.parent / saved["quality_reference"]
                if saved.get("quality_reference")
                else None
            )
            previous = (
                cv2.imread(str(reference))
                if reference is not None and reference.is_file()
                else None
            )
            from datetime import datetime

            previous_stamp = datetime.fromisoformat(entries[url]["timestamp"])
            continue
        entry = entries[url]
        try:
            data = _download(url, state, path, max_bytes=10_000_000)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("PNG decode failed")
            from datetime import datetime

            stamp = datetime.fromisoformat(entry["timestamp"])
            adjacent = (
                previous is not None
                and previous_stamp is not None
                and (stamp - previous_stamp).total_seconds() <= 180
            )
            quality, metrics = image_quality(image, previous if adjacent else None)
            boxes, scores, gpu_seconds = [], [], 0.0
            if quality == "usable":
                boxes, scores, gpu_seconds = _detector(state, path, cached).detect(image)
                state["usage"]["gpu_inference_seconds"] += gpu_seconds
                _save(path, state)
            thumb = path.parent / "thumbs" / (entry["name"] + ".jpg")
            thumb.parent.mkdir(parents=True, exist_ok=True)
            reference = path.parent / "quality_refs" / entry["name"]
            reference.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(reference), cv2.resize(image, (320, 180)))
            cv2.imwrite(
                str(thumb),
                _thumbnail(image, f"{entry['name']} | {quality} | SAM3 {len(scores)}", boxes),
            )
            state["images"][url] = {
                "quality": quality,
                "metrics": metrics,
                "detection_count": len(scores),
                "max_score": max(scores, default=0),
                "gpu_seconds": gpu_seconds,
                "thumbnail": thumb.relative_to(path.parent).as_posix(),
                "quality_reference": reference.relative_to(path.parent).as_posix(),
            }
            previous, previous_stamp = image, stamp
            _save(path, state)
        except StateSaveError:
            raise
        except Exception as error:  # noqa: BLE001 - record per-file failure and continue pilot
            state["images"][url] = {"error": str(error), "quality": "unknown"}
            _record_error(state, path, "png", url, error)
            previous = None


def _ranked_clips(state: dict) -> list[dict]:
    images = [
        {**entry, **state["images"][entry["url"]]}
        for entry in state["entries"]
        if entry["url"] in state["images"]
    ]
    clips = []
    for entry in state["entries"]:
        if entry["kind"] != "mp4":
            continue
        nearby = nearby_images(entry, images)
        rank, reason = rank_clip(nearby)
        clips.append(
            {
                **entry,
                "rank": rank,
                "reason": reason,
                "nearby_png_count": len(nearby),
                "png_detection_count": sum(p.get("detection_count", 0) for p in nearby),
                "png_max_score": max((p.get("max_score", 0) for p in nearby), default=0),
            }
        )
    return clips


def scan_clips(state: dict, path: Path, limit: int, seed: int, cached: dict) -> None:
    import cv2

    ranked = _ranked_clips(state)
    if not state["selected_clip_urls"]:
        selected = select_clips(ranked, limit, seed)
        state["selected_clip_urls"] = [c["url"] for c in selected]
        state["selection_reasons"] = {c["url"]: c["selection_reason"] for c in selected}
        _save(path, state)
    entries = {e["url"]: e for e in state["entries"]}
    for url in state["selected_clip_urls"]:
        if url in state["clips"] and "error" not in state["clips"][url]:
            continue
        entry = entries[url]
        temporary = path.parent / "temp" / (entry["name"] + ".part")
        capture = None
        try:
            _download(url, state, path, target=temporary, max_bytes=80_000_000)
            capture = cv2.VideoCapture(str(temporary))
            if not capture.isOpened():
                raise ValueError("MP4 decode failed")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if count <= 0 or fps <= 0:
                raise ValueError("MP4 has no valid frames or FPS")
            indices = sorted({min(count - 1, round((i + 0.5) * count / 6)) for i in range(6)})
            results, thumbs = [], []
            (path.parent / "thumbs").mkdir(parents=True, exist_ok=True)
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    results.append({"frame": frame_index, "error": "frame decode failed"})
                    continue
                quality, metrics = image_quality(frame)
                boxes, scores, gpu_seconds = [], [], 0.0
                if quality == "usable":
                    boxes, scores, gpu_seconds = _detector(state, path, cached).detect(frame)
                    state["usage"]["gpu_inference_seconds"] += gpu_seconds
                    _save(path, state)
                thumb = path.parent / "thumbs" / (entry["name"] + f"_{frame_index}.jpg")
                cv2.imwrite(
                    str(thumb),
                    _thumbnail(
                        frame,
                        f"{entry['name']} {frame_index / fps:.1f}s | {quality} | SAM3 {len(scores)}",
                        boxes,
                    ),
                )
                thumbs.append(thumb)
                results.append(
                    {
                        "frame": frame_index,
                        "seconds": round(frame_index / fps, 2),
                        "quality": quality,
                        "metrics": metrics,
                        "detection_count": len(scores),
                        "max_score": max(scores, default=0),
                        "gpu_seconds": gpu_seconds,
                    }
                )
            sheet = path.parent / "contact_sheets" / (entry["name"] + ".jpg")
            _contact_sheet(thumbs, sheet)
            state["clips"][url] = {
                "frames": results,
                "frame_count": count,
                "fps": fps,
                "download_bytes": temporary.stat().st_size,
                "contact_sheet": sheet.relative_to(path.parent).as_posix(),
            }
            _save(path, state)
        except StateSaveError:
            raise
        except Exception as error:  # noqa: BLE001 - record per-file failure and continue pilot
            state["clips"][url] = {"error": str(error)}
            _record_error(state, path, "mp4", url, error)
        finally:
            if capture is not None:
                capture.release()
            if (
                temporary.exists()
                and str(temporary.relative_to(path.parent)) in state["created_temp_files"]
            ):
                temporary.unlink()
            _save(path, state)


def report(state: dict, path: Path) -> dict:
    directory = path.parent
    ranked = _ranked_clips(state)
    selected = set(state["selected_clip_urls"])
    controls = random_no_detection_controls(ranked, selected, seed=state.get("seed", 17))
    png_entries = {e["url"]: e for e in state["entries"] if e["kind"] == "png"}
    rows = []
    for clip in ranked:
        result = state["clips"].get(clip["url"], {})
        frames = result.get("frames", [])
        usable = [f for f in frames if f.get("quality") == "usable"]
        rank = clip["rank"]
        reason = clip["reason"]
        if any(f.get("max_score", 0) >= 0.5 for f in usable):
            rank, reason = "likely_fish", "sampled MP4 frame has SAM 3 score >= 0.50"
        elif frames and not usable:
            rank, reason = "unusable", "all sampled MP4 frames fail quality screening"
        elif (
            frames and any(f.get("max_score", 0) >= 0.25 for f in usable) and rank != "likely_fish"
        ):
            rank, reason = "uncertain", "sampled MP4 has weaker SAM 3 evidence"
        rows.append(
            {
                "camera": state["camera"],
                "day": state["day"],
                "clip_time": clip["timestamp"],
                "source_mp4_url": clip["url"],
                "screening_rank": rank,
                "rank_reason": reason,
                "selected_for_frame_sample": clip["url"] in selected,
                "selection_reason": state.get("selection_reasons", {}).get(clip["url"], ""),
                "random_manual_review_control": clip["url"] in controls,
                "control_coverage": (
                    "nearby_png_no_detection" if clip["nearby_png_count"] else "no_nearby_png"
                )
                if clip["url"] in controls
                else "",
                "nearby_png_count": clip["nearby_png_count"],
                "sampled_frame_count": len(frames),
                "contact_sheet": result.get("contact_sheet", "").replace("\\", "/"),
                "scan_error": result.get("error", ""),
                "human_fish_label": "",
                "human_quality_label": "",
                "human_notes": "",
            }
        )
    manifest = directory / "review_manifest.csv"
    old_reviews = {}
    prior_fields = []
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            prior_fields = reader.fieldnames or []
            old_reviews = {
                row["source_mp4_url"]: row for row in reader if row.get("source_mp4_url")
            }
    human_fields = ("human_fish_label", "human_quality_label", "human_notes")
    for row in rows:
        prior = old_reviews.get(row["source_mp4_url"], {})
        for field in human_fields:
            row[field] = prior.get(field, "")
    review_fields = [
        "camera",
        "day",
        "clip_time",
        "source_mp4_url",
        "screening_rank",
        "rank_reason",
        "selected_for_frame_sample",
        "selection_reason",
        "random_manual_review_control",
        "control_coverage",
        "nearby_png_count",
        "sampled_frame_count",
        "contact_sheet",
        "scan_error",
        *human_fields,
    ]
    review_fields.extend(field for field in prior_fields if field not in review_fields)
    for row in rows:
        prior = old_reviews.get(row["source_mp4_url"], {})
        for field in review_fields:
            if field not in row:
                row[field] = prior.get(field, "")
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "sam3_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "source_url",
            "kind",
            "frame",
            "quality",
            "sam3_detection_count",
            "sam3_max_score",
            "gpu_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for url in state["selected_png_urls"]:
            item = state["images"].get(url, {})
            writer.writerow(
                {
                    "source_url": url,
                    "kind": "png",
                    "frame": "",
                    "quality": item.get("quality", "unknown"),
                    "sam3_detection_count": item.get("detection_count", ""),
                    "sam3_max_score": item.get("max_score", ""),
                    "gpu_seconds": item.get("gpu_seconds", ""),
                }
            )
        for url in state["selected_clip_urls"]:
            for frame in state["clips"].get(url, {}).get("frames", []):
                writer.writerow(
                    {
                        "source_url": url,
                        "kind": "mp4_frame",
                        "frame": frame["frame"],
                        "quality": frame.get("quality", "unknown"),
                        "sam3_detection_count": frame.get("detection_count", ""),
                        "sam3_max_score": frame.get("max_score", ""),
                        "gpu_seconds": frame.get("gpu_seconds", ""),
                    }
                )
    thumb_paths = [
        directory / state["images"][url]["thumbnail"]
        for url in state["selected_png_urls"]
        if "thumbnail" in state["images"].get(url, {})
    ]
    png_sheets = []
    for offset in range(0, len(thumb_paths), 40):
        name = "png_pilot.jpg" if offset == 0 else f"png_page_{offset // 40 + 1:03d}.jpg"
        sheet_path = directory / "contact_sheets" / name
        _contact_sheet(thumb_paths[offset : offset + 40], sheet_path)
        png_sheets.append(sheet_path.relative_to(directory).as_posix())
    from datetime import datetime

    periods = []
    preceding = None
    for url in state["selected_png_urls"]:
        entry = png_entries[url]
        item = state["images"].get(url, {})
        stamp = datetime.fromisoformat(entry["timestamp"])
        quality = item.get("quality")
        if quality in {"black", "camera_off", "frozen", "obscured"}:
            start = (
                preceding["timestamp"] if quality == "frozen" and preceding else entry["timestamp"]
            )
            if (
                periods
                and periods[-1]["quality"] == quality
                and (
                    stamp - datetime.fromisoformat(periods[-1]["last_sample_time"])
                ).total_seconds()
                <= 180
            ):
                periods[-1]["last_sample_time"] = entry["timestamp"]
                periods[-1]["sample_count"] += 1
            else:
                periods.append(
                    {
                        "quality": quality,
                        "first_sample_time": start,
                        "last_sample_time": entry["timestamp"],
                        "sample_count": 1,
                        "note": "suspected period from sampled PNGs only",
                    }
                )
        preceding = entry
    with (directory / "suspected_quality_periods.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["quality", "first_sample_time", "last_sample_time", "sample_count", "note"],
        )
        writer.writeheader()
        writer.writerows(periods)
    links = [
        "<!doctype html><meta charset='utf-8'><title>SmartBay local archive review</title>",
        "<h1>SmartBay archive pilot — local analyst review</h1>",
        "<p>SAM 3 outputs are unreviewed. No detection is not proof of no fish.</p>",
        (
            "<p><a href='review_manifest.csv'>Review manifest</a> · "
            "<a href='sam3_predictions.csv'>SAM 3 predictions</a> · "
            "<a href='suspected_quality_periods.csv'>Quality periods</a></p>"
        ),
        "<h2>PNG contact sheets</h2><ul>",
    ]
    for sheet in png_sheets:
        links.append(
            f"<li><a href='{html.escape(sheet, quote=True)}'>{html.escape(sheet)}</a></li>"
        )
    links.append("</ul><h2>Sampled MP4s</h2><ul>")
    for row in rows:
        if not row["selected_for_frame_sample"]:
            continue
        url = html.escape(row["source_mp4_url"], quote=True)
        sheet = html.escape(row["contact_sheet"], quote=True)
        links.append(
            f"<li>{html.escape(row['screening_rank'])} — "
            f"<a href='{url}'>source MP4</a> — "
            f"<a href='{sheet}'>contact sheet</a><br><a href='{url}'>"
            f"<img src='{sheet}' width='800' alt='sampled frames'></a></li>"
        )
    links.append("</ul><h2>Random manual review controls</h2><ul>")
    for row in rows:
        if row["random_manual_review_control"]:
            url = html.escape(row["source_mp4_url"], quote=True)
            links.append(
                f"<li><a href='{url}'>{html.escape(row['source_mp4_url'].rsplit('/', 1)[-1])}</a>"
                f" — {html.escape(row['control_coverage'])}; no fish absence label</li>"
            )
    links.append("</ul>")
    (directory / "index.html").write_text("\n".join(links), encoding="utf-8")
    summary = {
        "camera": state["camera"],
        "day": state["day"],
        "indexed_pngs": len(png_entries),
        "indexed_mp4s": len(rows),
        "selected_pngs": len(state["selected_png_urls"]),
        "png_selection_mode": "all" if state.get("all_pngs") else "sample",
        "png_contact_sheets": len(png_sheets),
        "processed_pngs": sum(
            u in state["images"] and "error" not in state["images"][u]
            for u in state["selected_png_urls"]
        ),
        "selected_mp4s": len(selected),
        "random_manual_review_controls": len(controls),
        "random_controls_without_png_coverage": sum(
            row["random_manual_review_control"] and row["control_coverage"] == "no_nearby_png"
            for row in rows
        ),
        "processed_mp4s": sum("frames" in state["clips"].get(u, {}) for u in selected),
        "frame_sample_new_likely_fish": sum(
            clip["rank"] != "likely_fish"
            and any(
                f.get("quality") == "usable" and f.get("max_score", 0) >= 0.5
                for f in state["clips"].get(clip["url"], {}).get("frames", [])
            )
            for clip in ranked
            if clip["url"] in selected
        ),
        "quality_counts": {
            q: sum(i.get("quality") == q for i in state["images"].values())
            for q in ("usable", "black", "camera_off", "frozen", "obscured")
        },
        "suspected_quality_periods": len(periods),
        "rank_counts": {
            r: sum(row["screening_rank"] == r for row in rows)
            for r in ("likely_fish", "uncertain", "candidate_clear_and_empty", "unusable")
        },
        "usage": state["usage"],
        "error_count": len(state["errors"]),
        "interpretation": "Unreviewed screening only. No detection is not proof of no fish.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _cleanup_completed_day(state: dict, path: Path, summary: dict) -> None:
    """Discard scan intermediates only after the durable review artifacts exist."""
    if (
        summary["processed_pngs"] != summary["selected_pngs"]
        or summary["processed_mp4s"] != summary["selected_mp4s"]
    ):
        return

    directory = path.parent
    png_count = sum(
        "thumbnail" in state["images"].get(url, {}) for url in state["selected_png_urls"]
    )
    if png_count != summary["selected_pngs"]:
        raise RuntimeError("cannot clean completed day; PNG thumbnails are missing from state")
    sheets = [
        directory / "contact_sheets" / (
            "png_pilot.jpg" if offset == 0 else f"png_page_{offset // 40 + 1:03d}.jpg"
        )
        for offset in range(0, png_count, 40)
    ]
    sheets.extend(
        directory / state["clips"][url]["contact_sheet"]
        for url in state["selected_clip_urls"]
    )
    required = [
        directory / name
        for name in (
            "state.json",
            "summary.json",
            "index.html",
            "review_manifest.csv",
            "sam3_predictions.csv",
            "suspected_quality_periods.csv",
        )
    ]
    missing = [item for item in (*required, *sheets) if not item.is_file()]
    if missing:
        raise RuntimeError(f"cannot clean completed day; missing review artifact: {missing[0]}")

    for name in ("quality_refs", "thumbs"):
        intermediate = directory / name
        if intermediate.is_dir():
            for attempt in range(3):
                try:
                    shutil.rmtree(intermediate)
                    break
                except FileNotFoundError:
                    break  # removed by another process after is_dir()
                except OSError as error:
                    if attempt == 2:
                        warnings.warn(
                            f"could not remove scan intermediate {intermediate}: {error}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        time.sleep(0.2 * (attempt + 1))


def run(
    camera: str,
    day: str,
    max_pngs: int | None = 30,
    max_mp4s: int = 3,
    seed: int = 17,
    cached_detector: dict | None = None,
) -> dict:
    if (max_pngs is not None and not 1 <= max_pngs <= 30) or not 1 <= max_mp4s <= 3:
        raise ValueError("sample limits are at most 30 PNGs and 3 MP4s; use --all-pngs explicitly")
    day_url(camera, day)
    path = OUTPUT / f"{camera}_{day}" / "state.json"
    state = load_state(path, camera, day)
    if (
        max_pngs is not None
        and len(state["selected_png_urls"]) > max_pngs
        or len(state["selected_clip_urls"]) > max_mp4s
    ):
        raise ValueError("existing pilot selection exceeds the requested limits")
    if "seed" in state and state["seed"] != seed:
        raise ValueError("resume requires the original random seed")
    state["seed"] = seed
    state["all_pngs"] = max_pngs is None
    _save(path, state)
    cached = cached_detector if cached_detector is not None else {}
    index(state, path)
    scan_images(state, path, max_pngs, cached)
    scan_clips(state, path, max_mp4s, seed, cached)
    summary = report(state, path)
    _cleanup_completed_day(state, path, summary)
    return summary


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        choices=["1", "2", "SmartBayCam1", "SmartBayCam2"],
        default="2",
        help="camera number (1 or 2); default: 2",
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument("--day", help="scan one day (default: 2025-07-17)")
    date_group.add_argument("--year", type=int, help="scan all listed days in a year, oldest first")
    png_group = parser.add_mutually_exclusive_group()
    png_group.add_argument(
        "--max-pngs", type=int, default=30, help="sample up to this many PNGs per day (default: 30)"
    )
    png_group.add_argument("--all-pngs", action="store_true", help="screen every PNG on each day")
    parser.add_argument("--max-mp4s", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-days", type=int, help="stop after this many new days; resume later")
    parser.add_argument(
        "--sample-days",
        type=int,
        help="scan a seeded, seasonally spread sample of this many archive days",
    )
    parser.add_argument(
        "--retry-incomplete",
        type=int,
        default=0,
        help="retry an incomplete day this many times before stopping (default: 0)",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
        help="delay between incomplete-day retries (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="scan-state directory (default: repository data/archive_scan)",
    )
    args = parser.parse_args()
    if args.output is not None:
        OUTPUT = args.output.resolve()
    camera = f"SmartBayCam{args.camera}" if args.camera in {"1", "2"} else args.camera
    png_limit = None if args.all_pngs else args.max_pngs
    if args.year is not None:
        from .year import run_year

        result = run_year(
            camera,
            args.year,
            png_limit,
            args.max_mp4s,
            args.seed,
            args.max_days,
            args.retry_incomplete,
            args.retry_delay_seconds,
            args.sample_days,
        )
    else:
        if args.max_days is not None:
            parser.error("--max-days requires --year")
        result = run(camera, args.day or "2025-07-17", png_limit, args.max_mp4s, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
