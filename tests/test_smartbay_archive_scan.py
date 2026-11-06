import json
from pathlib import Path

import pytest

from scripts.smartbay_archive_scan import workflow, year
from scripts.smartbay_archive_scan.core import (
    choose_pngs,
    image_quality,
    nearby_images,
    parse_calendar_index,
    parse_index,
    random_no_detection_controls,
    rank_clip,
    select_clips,
)


def entry(name, kind, stamp):
    return {
        "url": f"https://smartbay.marine.ie/data/video/SmartBayCam2/2025/07/17/{name}",
        "name": name,
        "kind": kind,
        "timestamp": f"2025-07-17T{stamp}:00",
    }


def test_archive_parser_accepts_only_exact_day_camera_and_files():
    html = """<a href="../">../</a>
    <a href="SmartBayCam2_2025-07-17_0001.mp4">video</a>
    <a href="SmartBayCam2_2025-07-17_0001.png">image</a>
    <a href="SmartBayCam2_2025_07_17_quick.mp4">quick</a>
    <a href="SmartBayCam2_2025-07-18_0001.png">wrong day</a>
    <a href="SmartBayCam1_2025-07-17_0001.png">wrong camera</a>
    <a href="https://evil.test/SmartBayCam2_2025-07-17_0001.png">external</a>
    <a href="SmartBayCam2_2025-07-17_2460.png">invalid time</a>"""
    results = parse_index(html, "SmartBayCam2", "2025-07-17")
    assert [(e["kind"], e["timestamp"]) for e in results] == [
        ("mp4", "2025-07-17T00:01:00"),
        ("png", "2025-07-17T00:01:00"),
    ]


def test_calendar_parser_orders_months_and_days_and_rejects_bad_links():
    months = '<a href="12/">12</a><a href="07/">07</a><a href="13/">bad</a>'
    assert parse_calendar_index(months, "SmartBayCam2", 2025) == [7, 12]
    days = (
        '<a href="31/">invalid</a><a href="02/">2</a><a href="01/">1</a>'
        '<a href="https://evil.test/03/">external</a><a href="../">up</a>'
    )
    assert parse_calendar_index(days, "SmartBayCam2", 2025, 2) == ["2025-02-01", "2025-02-02"]


def test_empty_archive_day_is_indexed_once(monkeypatch, tmp_path: Path):
    path = tmp_path / "state.json"
    state = workflow.load_state(path, "SmartBayCam1", "2017-10-27")
    calls = []

    def fake_download(url, *_args, **_kwargs):
        calls.append(url)
        return b"<title>Index of /data/video/SmartBayCam1/2017/10/27/</title>"

    monkeypatch.setattr(workflow, "_download", fake_download)
    workflow.index(state, path)
    workflow.index(state, path)
    assert state["indexed"] and state["entries"] == []
    assert len(calls) == 1
    workflow.scan_images(state, path, 30, {})
    workflow.scan_clips(state, path, 3, 17, {})
    assert workflow.report(state, path)["indexed_pngs"] == 0


def test_unrecognized_media_is_not_silently_marked_empty(monkeypatch, tmp_path: Path):
    path = tmp_path / "state.json"
    state = workflow.load_state(path, "SmartBayCam1", "2017-10-27")
    listing = (
        b"<title>Index of /data/video/SmartBayCam1/2017/10/27/</title>"
        b'<a href="SmartBayCam1-other.mp4">video</a>'
    )
    monkeypatch.setattr(workflow, "_download", lambda *_args, **_kwargs: listing)
    with pytest.raises(RuntimeError, match="unrecognized names"):
        workflow.index(state, path)


def test_timestamp_matching_uses_filename_time_and_short_window():
    clip = entry("SmartBayCam2_2025-07-17_1200.mp4", "mp4", "12:00")
    images = [
        entry(f"SmartBayCam2_2025-07-17_{stamp.replace(':', '')}.png", "png", stamp)
        for stamp in ("11:58", "11:59", "12:00", "12:02", "12:03", "12:04")
    ]
    assert [e["timestamp"][11:16] for e in nearby_images(clip, images)] == [
        "11:59",
        "12:00",
        "12:02",
        "12:03",
    ]
    assert len(choose_pngs(images * 10, 30)) <= 30
    assert choose_pngs(images, None) == images


def test_ranking_never_calls_no_detection_empty_proof_and_reserves_random_review():
    assert rank_clip([])[0] == "uncertain"
    assert rank_clip([{"quality": "usable", "max_score": 0}])[0] == "uncertain"
    rank, reason = rank_clip([{"quality": "usable", "max_score": 0}] * 2)
    assert rank == "candidate_clear_and_empty"
    assert "manual review" in reason
    assert rank_clip([{"quality": "usable", "max_score": 0.7}])[0] == "likely_fish"
    assert rank_clip([{"quality": "black"}])[0] == "unusable"
    clips = [
        {
            **entry(f"SmartBayCam2_2025-07-17_120{i}.mp4", "mp4", f"12:0{i}"),
            "rank": "likely_fish" if i < 2 else "candidate_clear_and_empty",
            "nearby_png_count": 2,
            "png_detection_count": 2 if i < 2 else 0,
            "png_max_score": 0.8 if i < 2 else 0,
        }
        for i in range(4)
    ]
    chosen = select_clips(clips, 3, 23)
    assert len(chosen) == 3
    assert sum(c["selection_reason"] == "random_no_detection_review" for c in chosen) == 1
    assert any(c["rank"] == "likely_fish" for c in chosen)
    controls = random_no_detection_controls(clips, {c["url"] for c in chosen}, 3, 23)
    assert len(controls) == 1
    assert all(c["png_detection_count"] == 0 for c in clips if c["url"] in controls)


def test_image_quality_flags_black_camera_off_and_frozen():
    import numpy as np

    black = np.zeros((180, 320, 3), dtype=np.uint8)
    flat = np.full((180, 320, 3), 80, dtype=np.uint8)
    textured = np.random.default_rng(3).integers(40, 200, (180, 320, 3), dtype=np.uint8)
    assert image_quality(black)[0] == "black"
    assert image_quality(flat)[0] == "camera_off"
    assert image_quality(textured, textured)[0] == "frozen"


def test_resume_skips_completed_index_and_png(monkeypatch, tmp_path: Path):
    import cv2
    import numpy as np

    state_path = tmp_path / "state.json"
    state = workflow.load_state(state_path, "SmartBayCam2", "2025-07-17")
    image_name = "SmartBayCam2_2025-07-17_1200.png"
    second_image = "SmartBayCam2_2025-07-17_1201.png"
    video_name = "SmartBayCam2_2025-07-17_1200.mp4"
    listing = (
        f'<a href="{image_name}">png</a><a href="{second_image}">png</a>'
        f'<a href="{video_name}">mp4</a>'
    ).encode()
    ok, encoded = cv2.imencode(".png", np.full((100, 100, 3), 90, dtype=np.uint8))
    assert ok
    calls = []

    def fake_download(url, *_args, **_kwargs):
        calls.append(url)
        return listing if url.endswith("/") else encoded.tobytes()

    class FakeDetector:
        def detect(self, _image):
            return [], [], 0.1

    monkeypatch.setattr(workflow, "_download", fake_download)
    monkeypatch.setattr(workflow, "_detector", lambda *_args: FakeDetector())
    monkeypatch.setattr(workflow, "image_quality", lambda *_args: ("usable", {}))
    workflow.index(state, state_path)
    workflow.scan_images(state, state_path, 1, {})
    reloaded = workflow.load_state(state_path, "SmartBayCam2", "2025-07-17")
    workflow.index(reloaded, state_path)
    workflow.scan_images(reloaded, state_path, 1, {})
    assert len(calls) == 2  # one listing and one PNG; resume fetches neither
    assert reloaded["images"][next(iter(reloaded["images"]))]["detection_count"] == 0
    workflow.scan_images(reloaded, state_path, None, {})
    assert len(calls) == 3  # explicit all-PNG mode expands selection without refetching the first
    assert len(reloaded["selected_png_urls"]) == 2
    with pytest.raises(ValueError):
        workflow.load_state(state_path, "SmartBayCam1", "2025-07-17")


def test_interrupted_transfer_persists_received_bytes(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "state.json"
    state = workflow.load_state(state_path, "SmartBayCam2", "2025-07-17")

    class BrokenResponse:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if not hasattr(self, "received"):
                self.received = True
                return b"abc"
            raise OSError("connection dropped")

    monkeypatch.setattr(workflow, "urlopen", lambda *_args, **_kwargs: BrokenResponse())
    with pytest.raises(OSError, match="connection dropped"):
        workflow._download(state["source_index"], state, state_path)
    resumed = workflow.load_state(state_path, "SmartBayCam2", "2025-07-17")
    assert resumed["usage"]["bytes_transferred"] == 3


def test_large_transfer_limits_resume_file_writes_and_saves_final_byte_count(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "state.json"
    state = workflow.load_state(path, "SmartBayCam2", "2025-07-17")
    chunk = b"x" * (256 * 1024)

    class Response:
        def __init__(self):
            self.headers = {}
            self.remaining = 32

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if self.remaining == 0:
                return b""
            self.remaining -= 1
            return chunk

    monkeypatch.setattr(workflow, "urlopen", lambda *_args, **_kwargs: Response())
    original_save = workflow._save
    writes = []

    def counted_save(*args):
        writes.append(1)
        original_save(*args)

    monkeypatch.setattr(workflow, "_save", counted_save)
    target = tmp_path / "temp" / "clip.mp4.part"
    workflow._download("https://smartbay.marine.ie/clip.mp4", state, path, target=target)
    assert target.stat().st_size == 8 * 1024 * 1024
    assert workflow.load_state(path, "SmartBayCam2", "2025-07-17")["usage"][
        "bytes_transferred"
    ] == 8 * 1024 * 1024
    assert len(writes) <= 5


def test_resume_save_retries_transient_windows_lock(monkeypatch, tmp_path: Path):
    path = tmp_path / "state.json"
    old = {"value": "old"}
    workflow._save(path, old)
    original_replace = Path.replace
    attempts = []

    def sometimes_locked(source, target):
        attempts.append(source)
        if len(attempts) <= 2:
            raise PermissionError(5, "Access is denied")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", sometimes_locked)
    monkeypatch.setattr(workflow.time, "sleep", lambda _seconds: None)
    workflow._save(path, {"value": "new"})
    assert len(attempts) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "new"}


def test_resume_save_failure_preserves_previous_state(monkeypatch, tmp_path: Path):
    path = tmp_path / "state.json"
    workflow._save(path, {"value": "old"})

    def always_locked(_source, _target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(workflow.time, "sleep", lambda _seconds: None)
    with pytest.raises(workflow.StateSaveError, match="could not save resume state"):
        workflow._save(path, {"value": "new"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "old"}


def test_year_traversal_resumes_oldest_unfinished_day(monkeypatch, tmp_path: Path):
    import json

    monkeypatch.setattr(year, "OUTPUT", tmp_path)
    listings = {
        "https://smartbay.marine.ie/data/video/SmartBayCam2/2025/": b'<a href="08/">08</a><a href="07/">07</a>',
        "https://smartbay.marine.ie/data/video/SmartBayCam2/2025/07/": b'<a href="12/">12</a><a href="10/">10</a><a href="11/">11</a>',
        "https://smartbay.marine.ie/data/video/SmartBayCam2/2025/08/": b'<a href="01/">01</a>',
    }
    monkeypatch.setattr(year, "_download", lambda url, *_args, **_kwargs: listings[url])
    visited = []
    fail_once = {"value": True}

    def fake_day(_camera, day, _png_limit, _mp4_limit, _seed, _cache):
        visited.append(day)
        if day == "2025-07-11" and fail_once["value"]:
            fail_once["value"] = False
            raise OSError("interrupted")
        return {
            "indexed_pngs": 2,
            "indexed_mp4s": 1,
            "selected_pngs": 2,
            "processed_pngs": 2,
            "selected_mp4s": 1,
            "processed_mp4s": 1,
            "usage": {"bytes_transferred": 100, "gpu_inference_seconds": 1},
        }

    monkeypatch.setattr(year, "run", fake_day)
    with pytest.raises(OSError, match="interrupted"):
        year.run_year("SmartBayCam2", 2025)
    state_path = tmp_path / "SmartBayCam2_2025" / "year_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_completed_day"] == "2025-07-10"
    assert state["current_day"] == "2025-07-11"
    summary = year.run_year("SmartBayCam2", 2025)
    assert visited == ["2025-07-10", "2025-07-11", "2025-07-11", "2025-07-12", "2025-08-01"]
    assert summary["completed_days"] == 4
    assert summary["last_completed_day"] == "2025-08-01"
    assert summary["pending_days"] == 0
    assert year.run_year("SmartBayCam2", 2025)["completed_days"] == 4
    assert len(visited) == 5


def test_year_retries_incomplete_day_without_reloading_detector(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(year, "OUTPUT", tmp_path)
    listings = {
        "https://smartbay.marine.ie/data/video/SmartBayCam1/2025/": b'<a href="07/">07</a>',
        "https://smartbay.marine.ie/data/video/SmartBayCam1/2025/07/": b'<a href="01/">01</a>',
    }
    monkeypatch.setattr(year, "_download", lambda url, *_args, **_kwargs: listings[url])
    monkeypatch.setattr(year.time, "sleep", lambda _seconds: None)
    calls = []

    def fake_day(_camera, _day, _png_limit, _mp4_limit, _seed, cache):
        calls.append(id(cache))
        complete = len(calls) == 2
        return {
            "indexed_pngs": 1,
            "indexed_mp4s": 0,
            "selected_pngs": 1,
            "processed_pngs": int(complete),
            "selected_mp4s": 0,
            "processed_mp4s": 0,
            "usage": {"bytes_transferred": 1, "gpu_inference_seconds": 1},
        }

    monkeypatch.setattr(year, "run", fake_day)
    summary = year.run_year("SmartBayCam1", 2025, retry_incomplete=2, retry_delay_seconds=0)
    assert summary["completed_days"] == 1
    assert len(calls) == 2
    assert len(set(calls)) == 1


def test_year_sample_days_are_deterministic_spread_and_only_targets_are_visited(
    monkeypatch, tmp_path: Path
):
    days = [f"2025-07-{day:02d}" for day in range(1, 13)]
    first = year.select_sample_days(days, 4, 17)
    assert first == year.select_sample_days(days, 4, 17)
    assert first != year.select_sample_days(days, 4, 18)
    assert all(day in days[index * 3 : (index + 1) * 3] for index, day in enumerate(first))

    monkeypatch.setattr(year, "OUTPUT", tmp_path)
    listings = {
        "https://smartbay.marine.ie/data/video/SmartBayCam2/2025/": b'<a href="07/">07</a>',
        "https://smartbay.marine.ie/data/video/SmartBayCam2/2025/07/": b"".join(
            f'<a href="{day:02d}/">{day:02d}</a>'.encode() for day in range(1, 13)
        ),
    }
    monkeypatch.setattr(year, "_download", lambda url, *_args, **_kwargs: listings[url])
    visited = []

    def fake_day(_camera, day, *_args):
        visited.append(day)
        return {
            "indexed_pngs": 1,
            "indexed_mp4s": 0,
            "selected_pngs": 1,
            "processed_pngs": 1,
            "selected_mp4s": 0,
            "processed_mp4s": 0,
            "usage": {"bytes_transferred": 1, "gpu_inference_seconds": 1},
        }

    monkeypatch.setattr(year, "run", fake_day)
    summary = year.run_year("SmartBayCam2", 2025, sample_days=4)
    assert visited == first
    assert summary["completed_days"] == summary["selected_days"] == 4
    assert summary["pending_days"] == 0
    assert summary["unselected_days"] == 8


def test_year_summary_counts_in_progress_day_usage(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(year, "OUTPUT", tmp_path)
    state = year.load_year_state(
        tmp_path / "SmartBayCam1_2025" / "year_state.json", "SmartBayCam1", 2025, None, 3, 17
    )
    state["discovered_days"] = ["2025-06-19", "2025-06-20"]
    state["current_day"] = "2025-06-20"
    day_path = tmp_path / "SmartBayCam1_2025-06-20" / "state.json"
    partial = workflow.load_state(day_path, "SmartBayCam1", "2025-06-20")
    partial["entries"] = [
        {"kind": "png", "url": "image"},
        {"kind": "mp4", "url": "clip"},
    ]
    partial["selected_png_urls"] = ["image"]
    partial["selected_clip_urls"] = ["clip"]
    partial["images"]["image"] = {"detection_count": 0}
    partial["clips"]["clip"] = {"error": "retry"}
    partial["errors"] = [{"message": "retry"}]
    partial["usage"]["bytes_transferred"] = 123
    partial["usage"]["gpu_inference_seconds"] = 2.5
    workflow._save(day_path, partial)
    (tmp_path / "SmartBayCam1_2025").mkdir()
    summary = year._summary(state, tmp_path / "SmartBayCam1_2025" / "year_state.json")
    assert summary["completed_days"] == 0
    assert summary["indexed_pngs"] == summary["indexed_mp4s"] == 1
    assert summary["processed_pngs"] == 1
    assert summary["processed_mp4s"] == 0
    assert summary["bytes_transferred"] == 123
    assert summary["gpu_inference_seconds"] == 2.5
    assert summary["current_day_file_errors"] == 1


def test_report_preserves_human_review_fields(tmp_path: Path):
    import csv

    path = tmp_path / "state.json"
    state = workflow.load_state(path, "SmartBayCam2", "2025-07-17")
    state["entries"] = [entry("SmartBayCam2_2025-07-17_1200.mp4", "mp4", "12:00")]
    workflow._save(path, state)
    workflow.report(state, path)
    manifest = tmp_path / "review_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["human_fish_label"] = "fish_present"
    rows[0]["analyst_extra"] = "keep"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    workflow.report(state, path)
    with manifest.open(newline="", encoding="utf-8") as stream:
        reviewed = next(csv.DictReader(stream))
        assert reviewed["human_fish_label"] == "fish_present"
        assert reviewed["analyst_extra"] == "keep"


def test_completed_day_removes_intermediates_after_report(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)

    def fake_index(state, path):
        state["indexed"] = True
        for name in ("quality_refs", "thumbs"):
            intermediate = path.parent / name
            intermediate.mkdir(parents=True)
            (intermediate / "sample.jpg").write_bytes(b"sample")

    monkeypatch.setattr(workflow, "index", fake_index)
    summary = workflow.run("SmartBayCam2", "2025-07-17")
    directory = tmp_path / "SmartBayCam2_2025-07-17"
    assert summary["processed_pngs"] == summary["selected_pngs"] == 0
    assert not (directory / "quality_refs").exists()
    assert not (directory / "thumbs").exists()
    for name in ("state.json", "summary.json", "index.html", "review_manifest.csv"):
        assert (directory / name).is_file()


def test_completed_day_survives_locked_intermediate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)

    def fake_index(state, path):
        state["indexed"] = True
        for name in ("quality_refs", "thumbs"):
            (path.parent / name).mkdir(parents=True)

    monkeypatch.setattr(workflow, "index", fake_index)
    monkeypatch.setattr(workflow.time, "sleep", lambda _seconds: None)
    original_rmtree = workflow.shutil.rmtree
    attempts = []

    def locked_rmtree(path):
        if path.name == "quality_refs":
            attempts.append(path)
            raise PermissionError(5, "Access is denied", str(path))
        original_rmtree(path)

    monkeypatch.setattr(workflow.shutil, "rmtree", locked_rmtree)
    with pytest.warns(RuntimeWarning, match="could not remove scan intermediate"):
        summary = workflow.run("SmartBayCam2", "2025-07-17")

    directory = tmp_path / "SmartBayCam2_2025-07-17"
    assert summary["processed_pngs"] == summary["selected_pngs"] == 0
    assert len(attempts) == 3
    assert (directory / "quality_refs").is_dir()
    assert not (directory / "thumbs").exists()
    assert (directory / "summary.json").is_file()


def test_cleanup_keeps_incomplete_day_and_requires_contact_sheets(tmp_path: Path):
    path = tmp_path / "state.json"
    state = workflow.load_state(path, "SmartBayCam2", "2025-07-17")
    url = "https://smartbay.marine.ie/image.png"
    state["selected_png_urls"] = [url]
    state["images"][url] = {"thumbnail": "thumbs/image.jpg"}
    workflow._save(path, state)
    for name in ("quality_refs", "thumbs"):
        intermediate = tmp_path / name
        intermediate.mkdir()
        (intermediate / "image.jpg").write_bytes(b"sample")

    workflow._cleanup_completed_day(
        state, path, {"processed_pngs": 0, "selected_pngs": 1, "processed_mp4s": 0, "selected_mp4s": 0}
    )
    assert (tmp_path / "quality_refs").is_dir()
    assert (tmp_path / "thumbs").is_dir()

    complete = {"processed_pngs": 1, "selected_pngs": 1, "processed_mp4s": 0, "selected_mp4s": 0}
    with pytest.raises(RuntimeError, match="missing review artifact"):
        workflow._cleanup_completed_day(state, path, complete)
    assert (tmp_path / "quality_refs").is_dir()
    assert (tmp_path / "thumbs").is_dir()

    (tmp_path / "contact_sheets").mkdir()
    (tmp_path / "contact_sheets" / "png_pilot.jpg").write_bytes(b"sheet")
    for name in (
        "summary.json",
        "index.html",
        "review_manifest.csv",
        "sam3_predictions.csv",
        "suspected_quality_periods.csv",
    ):
        (tmp_path / name).write_text("saved", encoding="utf-8")
    workflow._cleanup_completed_day(state, path, complete)
    assert not (tmp_path / "quality_refs").exists()
    assert not (tmp_path / "thumbs").exists()
    assert (tmp_path / "contact_sheets" / "png_pilot.jpg").is_file()
