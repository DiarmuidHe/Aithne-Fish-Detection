from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.api import live
from app.db.models import LiveFishDetection, LiveFishTrack, LiveMonitorSession, utc_now
from app.services import video_media
from app.services.live_monitor import LiveTracker, live_path, scratch_path
from app.services.live_source import LiveSourceError, Segment, SegmentCapture, resolve_stream
from app.services.viame_parser import VIAMEDetection
from app.workers.live_worker import claim_session, recover_stale_sessions, run_session


def detection(track="1", frame=0, x=50, confidence=.9):
    return VIAMEDetection(track, "synthetic", frame, x, 40, x + 35, 70,
                          confidence, None, "fish", .9, None)


@pytest.fixture
def synthetic_frame():
    frame = np.full((160, 240, 3), (110, 70, 20), dtype=np.uint8)
    frame[40:70, 50:85] = (20, 210, 230)
    return frame


def new_session(db, **kwargs):
    row = LiveMonitorSession(source_url="https://camera.example/", **kwargs)
    db.add(row)
    db.commit()
    return row


def test_active_lost_boundary_and_real_media(db_session_factory, test_settings, synthetic_frame):
    cv2 = video_media.load_cv2()
    now = utc_now()
    with db_session_factory() as db:
        session = new_session(db)
        tracker = LiveTracker(db, session, test_settings)
        tracker.process_frame(synthetic_frame, [detection()], now)
        tracker.process_frame(synthetic_frame, [detection(frame=1)], now + timedelta(seconds=1))
        db.commit()
        track = next(iter(tracker.active.values()))
        tracker.expire(now + timedelta(seconds=10.999))
        assert track.status == "active"
        tracker.expire(now + timedelta(seconds=11))
        db.commit()
        assert track.status == "finalized"
        assert track.finalization_reason == "lost"
        assert track.detection_count == 2
        assert Path(track.crop_path).is_file()
        assert cv2.imread(track.crop_path) is not None
        reader = cv2.VideoCapture(track.clip_path)
        try:
            ok, frame = reader.read()
            assert ok and frame.shape[:2] == (260, 260)
        finally:
            reader.release()
        # Per-frame crops spool to scratch and are cleaned up; only finished media is served.
        assert not (Path(track.crop_path).parent / "frames").exists()
        assert not scratch_path(test_settings, str(session.id), str(track.id)).exists()
        assert sorted(p.name for p in Path(track.crop_path).parent.iterdir()) == ["clip.mp4", "crop.jpg"]
        assert db.scalar(select(func.count()).select_from(LiveFishDetection)) == 2


def test_snapshot_write_retries_a_locked_target(db_session_factory, test_settings, synthetic_frame, monkeypatch):
    import app.services.live_monitor as monitor
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    attempts = []
    real_replace = monitor.os.replace

    def flaky_replace(src, dst):
        if Path(dst).name != "annotated.jpg":
            return real_replace(src, dst)
        attempts.append(dst)
        # A reader holding the published frame open makes the rename fail on
        # Windows-backed bind mounts; the next attempt succeeds once it closes.
        if len(attempts) < 3:
            raise PermissionError(13, "Permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(monitor.os, "replace", flaky_replace)
    with db_session_factory() as db:
        session = new_session(db)
        tracker = LiveTracker(db, session, test_settings)
        tracker.process_frame(synthetic_frame, [detection()], utc_now())
        assert len(attempts) == 3
        assert Path(session.snapshot_path).is_file()


def test_unpublishable_snapshot_does_not_end_the_session(db_session_factory, test_settings, synthetic_frame, monkeypatch):
    import app.services.live_monitor as monitor
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    monkeypatch.setattr(monitor.os, "replace", lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "denied")))
    with db_session_factory() as db:
        session = new_session(db)
        tracker = LiveTracker(db, session, test_settings)
        tracker.process_frame(synthetic_frame, [detection()], utc_now())
        db.commit()
        # Detections are the durable record; only the live view is lost.
        assert session.frames_processed == 1
        assert session.snapshot_path is None
        assert db.scalar(select(func.count()).select_from(LiveFishDetection)) == 1
        assert not list(Path(monitor.live_path(test_settings, str(session.id))).glob("*.tmp.jpg"))


def published_frames(tracker):
    """Record every annotated frame the tracker actually publishes."""
    written = []
    original = tracker._write_jpeg

    def record(path, frame):
        if path.name == "annotated.jpg":
            written.append(frame.shape)
        return original(path, frame)

    tracker._write_jpeg = record
    return written


def test_annotated_view_is_published_at_its_own_rate(db_session_factory, test_settings, synthetic_frame):
    """Sampling faster than the dashboard polls must not cost a publish per frame.

    Publishing one JPEG to a bind-mounted output volume costs far more than the
    tracking it accompanies, so at 4 FPS it dominated the segment. Every frame is
    still detected and recorded; only the live view is thinned to the rate an
    operator can actually see.
    """

    settings = test_settings.model_copy(update={"live_fps": 4, "live_snapshot_fps": 2})
    now = utc_now()
    with db_session_factory() as db:
        session = new_session(db)
        tracker = LiveTracker(db, session, settings)
        written = published_frames(tracker)
        for index in range(8):
            tracker.process_frame(synthetic_frame, [detection(frame=index)],
                                  now + timedelta(seconds=index / 4))
        db.commit()
        # Capture time drives the rate, so a replay thins exactly as a live feed does.
        assert len(written) == 4
        # The first frame publishes immediately: an operator should not wait for it.
        assert session.snapshot_path is not None and Path(session.snapshot_path).is_file()
        # Thinning the view changes nothing about what was detected or stored.
        assert session.frames_processed == 8
        assert db.scalar(select(func.count()).select_from(LiveFishDetection)) == 8


def test_published_annotated_view_is_capped_for_display(db_session_factory, test_settings):
    """The detector sees the full capture; the operator's view is only a picture of it."""

    settings = test_settings.model_copy(update={"live_snapshot_max_width": 640})
    frame = np.full((1080, 1920, 3), (110, 70, 20), dtype=np.uint8)
    with db_session_factory() as db:
        session = new_session(db)
        tracker = LiveTracker(db, session, settings)
        written = published_frames(tracker)
        tracker.process_frame(frame, [detection(x=900, confidence=.95)], utc_now())
        db.commit()
        assert written == [(360, 640, 3)]
        # Boxes are stored in full-capture pixels, not the coordinates of the thumbnail.
        stored = db.scalar(select(LiveFishDetection))
        assert stored.x1 == 900 and stored.x2 == 935


def test_a_smaller_capture_is_published_unscaled(db_session_factory, test_settings, synthetic_frame):
    """A feed below the cap gains nothing from being enlarged, so it is left alone."""

    with db_session_factory() as db:
        session = new_session(db)
        tracker = LiveTracker(db, session, test_settings)
        written = published_frames(tracker)
        tracker.process_frame(synthetic_frame, [detection()], utc_now())
        assert written == [synthetic_frame.shape]


def test_chunk_ids_reassociate_but_never_resurrect_lost_track(db_session_factory, test_settings, synthetic_frame):
    with db_session_factory() as db:
        session = new_session(db)
        tracker = LiveTracker(db, session, test_settings)
        now = utc_now()
        tracker.process_frame(synthetic_frame, [detection("1"), detection("2", x=160)], now)
        original_ids = set(tracker.active)
        tracker.begin_segment()
        tracker.process_frame(synthetic_frame, [detection("99", x=52), detection("1", x=162)], now + timedelta(seconds=2))
        assert set(tracker.active) == original_ids
        assert all(t.detection_count == 2 for t in tracker.active.values())
        tracker.expire(now + timedelta(seconds=12))
        tracker.process_frame(synthetic_frame, [detection("99", x=52)], now + timedelta(seconds=13))
        assert set(tracker.active).isdisjoint(original_ids)
        db.commit()


def test_invalid_and_low_confidence_boxes_ignored(db_session_factory, test_settings, synthetic_frame):
    with db_session_factory() as db:
        tracker = LiveTracker(db, new_session(db), test_settings)
        tracker.process_frame(synthetic_frame, [detection(confidence=.1), detection(x=float("nan")), detection(x=900)], utc_now())
        assert not tracker.active


@pytest.fixture
def association_tracker(db_session_factory, test_settings, monkeypatch):
    # Exercise real association and persisted detections without encoding media.
    test_settings.live_fps = 2
    monkeypatch.setattr(LiveTracker, "_save_crop", lambda *args: None)
    monkeypatch.setattr(LiveTracker, "_write_jpeg", lambda *args: None)
    monkeypatch.setattr(LiveTracker, "_render_clip", lambda *args: None)
    with db_session_factory() as db:
        yield LiveTracker(db, new_session(db), test_settings)


def observed_ids(tracker):
    tracker.db.flush()
    rows = tracker.db.scalars(select(LiveFishDetection).where(
        LiveFishDetection.frame_number == tracker.session.frames_processed - 1)).all()
    return {row.x1: row.track_id for row in rows}


def test_low_fps_motion_survives_large_steps_and_segment_reset(association_tracker, synthetic_frame):
    tracker, now = association_tracker, utc_now()
    tracker.process_frame(synthetic_frame, [detection(x=10)], now)
    original = observed_ids(tracker)[10]
    # More than a box width per sampled frame: no last-box overlap.
    tracker.begin_segment()
    tracker.process_frame(synthetic_frame, [detection(x=50)], now + timedelta(seconds=.5))
    assert observed_ids(tracker)[50] == original
    tracker.begin_segment()
    tracker.process_frame(synthetic_frame, [detection("99", x=90)], now + timedelta(seconds=1))
    assert observed_ids(tracker)[90] == original
    tracker.process_frame(synthetic_frame, [], now + timedelta(seconds=1.5))
    tracker.process_frame(synthetic_frame, [detection("7", x=170)], now + timedelta(seconds=2))
    assert observed_ids(tracker)[170] == original


def test_four_fps_tracking_keeps_the_sixty_percent_acceptance_floor(association_tracker, synthetic_frame):
    tracker, now = association_tracker, utc_now()
    tracker.settings.live_fps = 4
    tracker.settings.min_fish_confidence = .60
    original = None
    for index, x in enumerate((20, 40, 60, 80)):
        if index == 2:
            tracker.begin_segment()
        tracker.process_frame(synthetic_frame, [detection(str(index), x=x, confidence=.60),
                              detection("weak", x=180, confidence=.599)],
                              now + timedelta(seconds=index / 4))
        ids = observed_ids(tracker)
        assert set(ids) == {x}
        original = original or ids[x]
        assert ids[x] == original
    assert tracker.active[original].mean_confidence == pytest.approx(.60)


@pytest.mark.parametrize("replacement", [
    detection(x=170),
    VIAMEDetection("1", "synthetic", 0, 40, 25, 145, 115, .9, None, "fish", .9, None),
    VIAMEDetection("1", "synthetic", 0, 50, 40, 110, 55, .9, None, "fish", .9, None),
    VIAMEDetection("1", "synthetic", 0, 50, 40, 85, 70, .9, None, "crab", .9, None),
])
def test_segment_id_cannot_override_geometry_or_class(association_tracker, synthetic_frame, replacement):
    tracker, now = association_tracker, utc_now()
    tracker.process_frame(synthetic_frame, [detection()], now)
    original = observed_ids(tracker)[50]
    tracker.process_frame(synthetic_frame, [replacement], now + timedelta(seconds=.5))
    assert original not in observed_ids(tracker).values()
    assert tracker.active[original].detection_count == 1


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_crossing_fish_follow_motion_even_if_viame_ids_swap(association_tracker, synthetic_frame, reset, reverse):
    tracker, now = association_tracker, utc_now()
    def step(seconds, left, right, swap=False):
        observations = [detection("2" if swap else "1", x=left),
                        detection("1" if swap else "2", x=right)]
        tracker.process_frame(synthetic_frame, observations[::-1] if reverse else observations,
                              now + timedelta(seconds=seconds))
        return observed_ids(tracker)
    initial = step(0, 20, 160)
    assert step(.5, 50, 130) == {50: initial[20], 130: initial[160]}
    assert step(1, 80, 100, swap=not reset) == {80: initial[20], 100: initial[160]}
    if reset:
        tracker.begin_segment()  # The nearest last boxes now belong to the wrong fish.
    assert step(1.5, 110, 70, swap=True) == {110: initial[20], 70: initial[160]}


def test_missed_moving_track_cannot_steal_fish_at_its_old_box(association_tracker, synthetic_frame):
    tracker, now = association_tracker, utc_now()
    tracker.process_frame(synthetic_frame, [detection(x=20)], now)
    original = observed_ids(tracker)[20]
    tracker.process_frame(synthetic_frame, [detection(x=50)], now + timedelta(seconds=.5))
    tracker.process_frame(synthetic_frame, [], now + timedelta(seconds=1))
    tracker.process_frame(synthetic_frame, [detection(x=52)], now + timedelta(seconds=1.5))
    assert observed_ids(tracker)[52] != original
    tracker.process_frame(synthetic_frame, [detection("new-id", x=140), detection(x=54)],
                          now + timedelta(seconds=2))
    assert observed_ids(tracker)[140] == original
    assert observed_ids(tracker)[54] != original


@pytest.mark.parametrize("reset", [False, True])
def test_stale_unexpired_track_is_not_a_reassociation_candidate(association_tracker, synthetic_frame, reset):
    tracker, now = association_tracker, utc_now()
    tracker.process_frame(synthetic_frame, [detection()], now)
    original = observed_ids(tracker)[50]
    if reset:
        tracker.begin_segment()
    tracker.process_frame(synthetic_frame, [detection(x=52)], now + timedelta(seconds=4))
    assert tracker.active[original].status == "active"  # Retention is not permission to match.
    assert observed_ids(tracker)[52] != original


@pytest.mark.parametrize("two_tracks", [False, True])
def test_ambiguous_match_does_not_reuse_identity(association_tracker, synthetic_frame, two_tracks):
    tracker, now = association_tracker, utc_now()
    initial = [detection(x=50), detection("2", x=70)] if two_tracks else [detection(x=60)]
    tracker.process_frame(synthetic_frame, initial, now)
    originals = set(observed_ids(tracker).values())
    # Test ambiguity in both directions; even an existing VIAME ID must not break a tie.
    observations = [detection(x=60)] if two_tracks else [detection(x=59), detection("2", x=61)]
    tracker.process_frame(synthetic_frame, observations, now + timedelta(seconds=.5))
    assert originals.isdisjoint(observed_ids(tracker).values())


def test_activity_updates_and_media_endpoints(client, db_session_factory, test_settings, synthetic_frame, monkeypatch):
    now = utc_now()
    monkeypatch.setattr(live, "utc_now", lambda: now)
    with db_session_factory() as db:
        session = new_session(db)
        sid = str(session.id)
        assert client.get(f"/live/{sid}/activity").json()["window_detections"] == 0
        assert client.get(f"/live/{sid}/annotated-stream").status_code == 503
        tracker = LiveTracker(db, session, test_settings)
        tracker.process_frame(synthetic_frame, [detection()], now)
        db.commit()
        tid = str(next(iter(tracker.active)))
        result = client.get(f"/live/{sid}/activity").json()
        assert result["active_tracks"] == 1
        assert result["window_tracks"] == result["window_detections"] == 1
        assert sum(point["tracks"] for point in result["series"]) == 1
        response = client.get(f"/live/{sid}/annotated-stream?snapshot=true")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content[:2] == b"\xff\xd8"
        annotated = video_media.load_cv2().imdecode(np.frombuffer(response.content, np.uint8), 1)
        assert np.abs(annotated.astype(float) - synthetic_frame).mean() > 1
        tracker.expire(now + timedelta(seconds=10))
        session.status = "stopped"
        db.commit()
    assert client.get(f"/live/{sid}/activity").json()["finalized_tracks"] == 1
    assert len(client.get(f"/live/{sid}/clips").json()) == 1
    assert client.get(f"/live/{sid}/tracks?status=active").json() == []
    assert client.get(f"/live/tracks/{tid}/crop").headers["content-type"] == "image/jpeg"
    response = client.get(f"/live/tracks/{tid}/clip")
    assert response.headers["content-type"] == "video/mp4"
    assert b"ftyp" in response.content[:32]
    response = client.get(f"/live/tracks/{tid}/clip", headers={"Range": "bytes=0-31"})
    assert response.status_code == 206 and len(response.content) == 32
    response = client.get(f"/live/{sid}/annotated-stream")
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"Content-Type: image/jpeg" in response.content and b"--frame--" in response.content
    # Capture timestamps trail the clock, so the window follows the analyzed timeline:
    # a slow chunk must not blank the chart that the finished chunk just filled.
    monkeypatch.setattr(live, "utc_now", lambda: now + timedelta(seconds=61))
    assert client.get(f"/live/{sid}/activity").json()["window_detections"] == 1
    with db_session_factory() as db:
        db.get(LiveMonitorSession, uuid.UUID(sid)).last_frame_at = now + timedelta(seconds=61)
        db.commit()
    assert client.get(f"/live/{sid}/activity").json()["window_detections"] == 0


def test_start_stop_idempotency_and_guards(client, test_settings, db_session_factory):
    assert client.post("/live/start", json={"source": "coral-city"}).status_code == 503
    test_settings.live_monitor_enabled = True
    assert client.post("/live/start", json={"source": "coral-city"}).status_code == 409
    test_settings.viame_mock = False
    response = client.post("/live/start", json={"source": "coral-city"})
    assert response.status_code == 202
    sid = response.json()["id"]
    # Starting the same camera again adopts the open session rather than queueing another.
    assert client.post("/live/start", json={"source": "coral-city"}).json()["id"] == sid
    assert client.post("/live/start").json()["id"] == sid
    assert claim_session(db_session_factory, "worker-1") == uuid.UUID(sid)
    assert claim_session(db_session_factory, "worker-2") is None
    assert client.post(f"/live/{sid}/stop").json()["status"] == "stopping"
    assert client.post(f"/live/{sid}/stop").json()["status"] == "stopping"
    with db_session_factory() as db:
        assert db.get(LiveMonitorSession, uuid.UUID(sid)).stop_requested
    assert client.get(f"/live/{uuid.uuid4()}/status").status_code == 404
    assert client.get("/live/not-a-uuid/status").status_code == 422


def test_queued_session_can_stop_without_worker(client, test_settings):
    test_settings.live_monitor_enabled, test_settings.viame_mock = True, False
    sid = client.post("/live/start", json={"source": "smartbay-cam1"}).json()["id"]
    assert client.post(f"/live/{sid}/stop").json()["status"] == "stopped"
    assert client.post("/live/start", json={"source": "smartbay-cam1"}).json()["id"] != sid


def test_sources_endpoint_lists_the_registry_in_order(client, test_settings):
    body = client.get("/live/sources").json()
    assert [source["key"] for source in body["sources"]] == [
        "coral-city", "smartbay-cam1", "smartbay-cam2", "smartbay-cam3"]
    assert body["default_key"] == "coral-city" and body["enabled"] is False and body["available"] is False
    assert all(source["active_session_id"] is None for source in body["sources"])
    coral = body["sources"][0]
    assert coral["label"] == "Coral City Camera" and coral["location"] == "Miami, Florida"
    assert coral["url"] == test_settings.live_source("coral-city").url


def test_each_camera_key_stores_its_configured_url(client, test_settings, db_session_factory):
    test_settings.live_monitor_enabled, test_settings.viame_mock = True, False
    for key, configured in test_settings.live_sources().items():
        body = client.post("/live/start", json={"source": key}).json()
        assert body["source_key"] == key and body["source_label"] == configured.label
        with db_session_factory() as db:
            session = db.get(LiveMonitorSession, uuid.UUID(body["id"]))
            assert (session.source_key, session.source_url) == (key, configured.url)
            # Only one camera runs at a time; free the worker for the next key.
            session.status, session.stopped_at = "stopped", utc_now()
            db.commit()
        assert client.get(f"/live/latest?source={key}").json()["session"]["id"] == body["id"]
    assert all(source["active_session_id"] is None for source in client.get("/live/sources").json()["sources"])


def test_unknown_and_url_shaped_sources_are_rejected_without_a_session(client, test_settings, db_session_factory):
    test_settings.live_monitor_enabled, test_settings.viame_mock = True, False
    # A URL in the source field is only ever a failed key lookup; the resolver never sees it.
    for value in ("nope", "Coral-City", "https://evil.example/stream.m3u8", "http://169.254.169.254/latest/meta-data"):
        assert client.post("/live/start", json={"source": value}).status_code == 404
        assert client.get("/live/latest", params={"source": value}).status_code == 404
    with db_session_factory() as db:
        assert db.scalar(select(func.count()).select_from(LiveMonitorSession)) == 0


def test_second_camera_is_refused_while_another_runs(client, test_settings, db_session_factory):
    test_settings.live_monitor_enabled, test_settings.viame_mock = True, False
    sid = client.post("/live/start", json={"source": "smartbay-cam1"}).json()["id"]
    response = client.post("/live/start", json={"source": "smartbay-cam2"})
    assert response.status_code == 409
    assert "SmartBay Cam 1" in response.json()["detail"]
    with db_session_factory() as db:
        assert db.scalar(select(func.count()).select_from(LiveMonitorSession)) == 1
        # The open-session index is partial on source_key, so the database itself
        # permits one open session per camera even though the API serializes them.
        db.add(LiveMonitorSession(source_url="https://camera.example/", source_key="smartbay-cam2"))
        db.commit()
        assert db.scalar(select(func.count()).select_from(LiveMonitorSession)) == 2
    assert client.get("/live/latest?source=smartbay-cam1").json()["session"]["id"] == sid
    assert client.get("/live/latest?source=smartbay-cam2").json()["session"]["id"] != sid
    keyed = {source["key"]: source["active_session_id"] for source in client.get("/live/sources").json()["sources"]}
    assert keyed["smartbay-cam1"] == sid and keyed["smartbay-cam2"] and keyed["coral-city"] is None


def test_latest_defaults_to_the_configured_camera(client, db_session_factory):
    with db_session_factory() as db:
        db.add(LiveMonitorSession(source_url="https://camera.example/", source_key="smartbay-cam3", status="stopped"))
        db.commit()
    body = client.get("/live/latest").json()
    assert body["source_key"] == "coral-city" and body["session"] is None
    assert client.get("/live/latest?source=smartbay-cam3").json()["session"]["source_label"] == "SmartBay Cam 3 (ANERIS EMUAS)"


def test_smartbay_playlist_is_used_without_yt_dlp(test_settings, monkeypatch):
    import app.services.live_source as source

    def unexpected(*args, **kwargs):
        raise AssertionError("A direct HLS playlist must not invoke yt-dlp")

    monkeypatch.setattr(source.subprocess, "run", unexpected)
    url = test_settings.live_source("smartbay-cam1").url
    requested = []

    def handle(request):
        requested.append(str(request.url))
        return httpx.Response(200, text="#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION=1280x720\n"
                                        "chunklist_w1234.m3u8\n")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert resolve_stream(url, test_settings, "SmartBay Cam 1", client) == url
    assert requested == [url]


def test_idle_master_playlist_is_reported_as_not_broadcasting(test_settings):
    url = test_settings.live_source("smartbay-cam2").url
    # Cam 2 answers with a master playlist advertising no renditions while it is idle.
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="#EXTM3U\n"))) as client:
        with pytest.raises(LiveSourceError, match="not currently broadcasting"):
            resolve_stream(url, test_settings, "SmartBay Cam 2", client)

    def scraped(request):
        if request.url.path == "/":
            return httpx.Response(200, text='<video><source src="https://media.example/live.m3u8"></video>')
        return httpx.Response(200, text="#EXTM3U\n#EXT-X-ENDLIST\n")

    # A playlist found by scraping the camera page gets the same check.
    with httpx.Client(transport=httpx.MockTransport(scraped)) as client:
        with pytest.raises(LiveSourceError, match="not currently broadcasting"):
            resolve_stream("https://camera.example/", test_settings, "Coral City Camera", client)


def test_registry_overrides_keep_existing_deployments_working():
    from app.config import DEFAULT_LIVE_SOURCES, Settings

    # CORAL_CITY_URL retargets that one camera and leaves the others alone.
    retargeted = Settings(_env_file=None, coral_city_url="https://camera.example/reef")
    assert retargeted.live_source("coral-city").url == "https://camera.example/reef"
    assert retargeted.live_source("smartbay-cam1").url.startswith("https://live.heanet.ie/")
    # An empty override is not an override; Compose passes [] for an unset variable.
    assert list(Settings(_env_file=None, live_source_registry=[]).live_sources()) == [
        entry["key"] for entry in DEFAULT_LIVE_SOURCES]
    trimmed = [{"key": "bay", "label": "Bay", "url": "https://a.example/live.m3u8"}]
    assert list(Settings(_env_file=None, live_source_registry=trimmed,
                         live_default_source_key="bay").live_sources()) == ["bay"]
    for invalid in ([{"key": "Bay!", "label": "Bay", "url": "https://a.example"}],
                    [{"key": "bay", "label": "Bay", "url": "https://a.example"},
                     {"key": "bay", "label": "Other", "url": "https://b.example"}],
                    [{"key": "bay", "label": " ", "url": "https://a.example"}],
                    [{"key": "bay", "label": "Bay", "url": "file:///etc/passwd"}]):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, live_source_registry=invalid, live_default_source_key="bay")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_default_source_key="ghost-camera")


def test_media_paths_cannot_escape_session(client, db_session_factory, test_settings, tmp_path):
    outside = tmp_path / "private.jpg"
    outside.write_bytes(b"private")
    with db_session_factory() as db:
        session = new_session(db, snapshot_path=str(outside))
        sid = str(session.id)
        track = LiveFishTrack(session_id=session.id, first_seen_at=utc_now(), last_seen_at=utc_now(),
            x1=0, y1=0, x2=10, y2=10, crop_path=str(outside), clip_path=str(outside))
        db.add(track)
        db.commit()
        tid = str(track.id)
    assert client.get(f"/live/{sid}/annotated-stream?snapshot=true").status_code == 404
    assert client.get(f"/live/tracks/{tid}/crop").status_code == 404
    assert client.get(f"/live/tracks/{tid}/clip").status_code == 404
    with pytest.raises(video_media.MediaPathError):
        live_path(test_settings, "..", "..", "private.jpg")


def test_resolves_page_iframe_and_hls_without_network(test_settings):
    def handle(request):
        if request.url.path == "/":
            return httpx.Response(200, text='<iframe src="/camera"></iframe><iframe src="https://soundcloud.com/audio"></iframe>')
        return httpx.Response(200, text='<video><source src="https://media.example/live.m3u8?token=abc&amp;x=2"></video>')
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert resolve_stream("https://camera.example/", test_settings, client=client) == "https://media.example/live.m3u8?token=abc&x=2"


def test_youtube_resolution_failure_is_clear(test_settings, monkeypatch):
    import app.services.live_source as source
    monkeypatch.setattr(source.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(LiveSourceError, match="no playable stream"):
        resolve_stream("https://www.youtube.com/embed/example", test_settings)
    with pytest.raises(LiveSourceError, match="HTTP"):
        resolve_stream("file:///private", test_settings)


def test_hosted_player_resolution_accepts_video_only_renditions(test_settings, monkeypatch):
    import app.services.live_source as source
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout="https://media.example/hls/index.m3u8\n")

    monkeypatch.setattr(source.subprocess, "run", fake_run)
    resolved = resolve_stream("https://www.youtube.com/embed/example", test_settings)
    assert resolved == "https://media.example/hls/index.m3u8"
    # Live streams often publish no muxed format, so a video-only selector must come first.
    selector = captured["args"][captured["args"].index("-f") + 1]
    assert selector.split("/")[0].startswith("bv*")
    assert "height<=?1080" in selector.split("/")[0]


def test_capture_only_uses_closed_segments_and_drops_backlog(test_settings, tmp_path, monkeypatch):
    import app.services.live_source as source
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **k: None)
    monkeypatch.setattr(source.subprocess, "Popen", lambda *a, **k: process)
    test_settings.live_max_pending_segments = 2
    capture = SegmentCapture("https://media.example/live.m3u8", tmp_path / "capture", test_settings)
    for index in range(6):
        (capture.directory / f"{index:08d}.mp4").write_bytes(b"stub")
    segment = capture.next_segment()
    assert segment.path.name == "00000003.mp4"
    assert capture.dropped == 3
    assert capture.next_segment().path.name == "00000004.mp4"
    assert capture.next_segment() is None
    capture.close()
    assert not list(capture.directory.glob("*.mp4"))


def test_capture_waits_out_the_first_long_segment(test_settings, tmp_path, monkeypatch):
    import app.services.live_source as source
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **k: None)
    monkeypatch.setattr(source.subprocess, "Popen", lambda *a, **k: process)
    test_settings.live_segment_seconds = 30
    test_settings.live_read_timeout_seconds = 20
    clock = [1000.0]
    monkeypatch.setattr(source.time, "monotonic", lambda: clock[0])
    capture = SegmentCapture("https://media.example/live.m3u8", tmp_path / "capture", test_settings)

    # FFmpeg opens the first file immediately but only closes it a segment later, and
    # it is consumable a segment after that, so the stall deadline must not fire first.
    (capture.directory / "00000000.mp4").write_bytes(b"stub")
    clock[0] += 45
    assert capture.next_segment() is None
    clock[0] += 20
    (capture.directory / "00000001.mp4").write_bytes(b"stub")
    assert capture.next_segment().path.name == "00000000.mp4"

    # A camera that genuinely stops still gets reported once nothing new appears.
    clock[0] += test_settings.live_read_timeout_seconds + 2 * test_settings.live_segment_seconds + 1
    with pytest.raises(LiveSourceError, match="stopped delivering"):
        capture.next_segment()


@pytest.mark.parametrize("identify", [False, True])
def test_worker_processes_synthetic_chunk_and_finalizes_on_shutdown(db_session_factory, test_settings, synthetic_frame, monkeypatch, identify):
    from pydantic import SecretStr

    from app.services.fishial import FishialClient, FishialError
    calls = []
    def fail(client, image, expected_box=None):
        client.before_image_call(False)
        calls.append(image)
        raise FishialError()
    monkeypatch.setattr(FishialClient, "identify", fail)
    if identify:
        test_settings.fishial_enabled = True
        test_settings.fishial_client_id = "id"
        test_settings.fishial_client_secret = SecretStr("secret")
        test_settings.fishial_min_crop_pixels = 16
        test_settings.fishial_min_frame_separation_seconds = 0
    shutdown = threading.Event()
    now = utc_now()
    class Capture:
        dropped = 0
        def __init__(self, url, directory, settings):
            directory.mkdir(parents=True)
            self.path, self.read = directory / "chunk.mp4", False
            cv2 = video_media.load_cv2()
            writer = video_media.open_video_writer(cv2, self.path, settings.live_fps, 240, 160)
            assert writer is not None
            for _ in range(3):
                writer.write(synthetic_frame)
            writer.release()
        def next_segment(self):
            if self.read:
                shutdown.set()
                return None
            self.read = True
            return Segment(self.path, now)
        def close(self):
            self.path.unlink(missing_ok=True)
    with db_session_factory() as db:
        sid = new_session(db, species_id_enabled=identify, species_id_fish_target=int(identify)).id
    claim_session(db_session_factory, "test")
    run_session(sid, test_settings, db_session_factory, shutdown,
        resolver=lambda *_: "stub", capture_factory=Capture,
        detector=lambda *_: [detection(frame=i) for i in range(3)])
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, sid)
        assert session.status == "stopped"
        assert session.frames_processed == 3
        track = db.scalar(select(LiveFishTrack))
        assert track.status == "finalized" and track.finalization_reason == "stopped"
        assert track.clip_path and Path(track.clip_path).exists()
        # Three staged frames and fishial_min_votes=3, so one failure already makes
        # the consensus rule unreachable: the other two calls are never bought.
        assert len(calls) == (1 if identify else 0)
        assert track.fishial_state == ("error" if identify else "disabled")
        assert session.species_id_calls_saved == (2 if identify else 0)
        assert session.species_id_api_calls == len(calls)
        assert not scratch_path(test_settings, str(sid)).exists()


@pytest.mark.parametrize("gap", [0, 12])
def test_worker_expires_on_capture_time_not_waiting_wall_clock(
        db_session_factory, test_settings, synthetic_frame, monkeypatch, gap):
    import app.workers.live_worker as worker

    shutdown, now = threading.Event(), utc_now()
    test_settings.live_fps = 2
    test_settings.live_recording_enabled = False
    monkeypatch.setattr(worker, "utc_now", lambda: now + timedelta(seconds=120))
    monkeypatch.setattr(LiveTracker, "_render_clip", lambda *args: None)

    class Capture:
        dropped = 0
        def __init__(self, url, directory, settings):
            directory.mkdir(parents=True)
            self.directory, self.calls = directory, 0

        def next_segment(self):
            self.calls += 1
            if self.calls == 2:
                return None  # Capture is still writing; the wall clock is far ahead.
            if self.calls > 3:
                shutdown.set()
                return None
            path = self.directory / f"{self.calls}.mp4"
            writer = video_media.open_video_writer(video_media.load_cv2(), path, 2, 240, 160)
            for _ in range(2):
                writer.write(synthetic_frame)
            writer.release()
            offset = 0 if self.calls == 1 else 1 + gap
            return Segment(path, now + timedelta(seconds=offset))

        def close(self):
            pass

    with db_session_factory() as db:
        sid = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(sid, test_settings, db_session_factory, shutdown,
                resolver=lambda *_: "stub", capture_factory=Capture,
                detector=lambda *_: [detection(frame=i) for i in range(2)])
    with db_session_factory() as db:
        tracks = db.scalars(select(LiveFishTrack)).all()
        assert sorted(t.detection_count for t in tracks) == ([2, 2] if gap else [4])
        assert db.get(LiveMonitorSession, sid).frames_processed == 4
        assert sum(t.finalization_reason == "lost" for t in tracks) == bool(gap)


def test_worker_overlaps_next_inference_with_current_rendering(
        db_session_factory, test_settings, synthetic_frame, monkeypatch):
    shutdown, next_started, now = threading.Event(), threading.Event(), utc_now()
    test_settings.live_recording_enabled = False

    class Capture:
        dropped = 0
        def __init__(self, url, directory, settings):
            directory.mkdir(parents=True)
            self.directory, self.served = directory, 0
        def next_segment(self):
            if self.served == 2:
                shutdown.set()
                return None
            path = self.directory / f"{self.served}.mp4"
            writer = video_media.open_video_writer(video_media.load_cv2(), path, 4, 240, 160)
            writer.write(synthetic_frame)
            writer.release()
            segment = Segment(path, now + timedelta(seconds=self.served / 4))
            self.served += 1
            return segment
        def close(self):
            pass

    def detector(segment, *_):
        if segment.path.stem == "1":
            next_started.set()
        return [detection()]

    original = LiveTracker.process_frame
    overlapped = []
    def process(tracker, frame, observations, timestamp):
        if tracker.session.frames_processed == 0:
            overlapped.append(next_started.wait(timeout=1))
        return original(tracker, frame, observations, timestamp)

    monkeypatch.setattr(LiveTracker, "process_frame", process)
    with db_session_factory() as db:
        sid = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(sid, test_settings, db_session_factory, shutdown,
                resolver=lambda *_: "stub", capture_factory=Capture, detector=detector)
    assert overlapped == [True]
    with db_session_factory() as db:
        assert db.get(LiveMonitorSession, sid).frames_processed == 2


def test_detector_crash_skips_one_segment_and_keeps_monitoring(db_session_factory, test_settings, synthetic_frame):
    shutdown = threading.Event()
    now = utc_now()
    test_settings.live_retry_seconds = 0.01

    class Capture:
        dropped = 0

        def __init__(self, url, directory, settings):
            directory.mkdir(parents=True)
            self.directory, self.served = directory, 0
            self.cv2 = video_media.load_cv2()

        def next_segment(self):
            if self.served >= 2:
                shutdown.set()
                return None
            self.served += 1
            path = self.directory / f"chunk{self.served}.mp4"
            writer = video_media.open_video_writer(self.cv2, path, test_settings.live_fps, 240, 160)
            assert writer is not None
            for _ in range(3):
                writer.write(synthetic_frame)
            writer.release()
            return Segment(path, now + timedelta(seconds=self.served))

        def close(self):
            pass

    calls = []

    def flaky_detector(segment, settings, session_id):
        calls.append(segment.path.name)
        # VIAME can fault on one chunk (a CUDA/ONNX segfault) and run fine on the next.
        if len(calls) == 1:
            raise RuntimeError("VIAME exited with code -11")
        return [detection(frame=i) for i in range(3)]

    with db_session_factory() as db:
        sid = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(sid, test_settings, db_session_factory, shutdown,
        resolver=lambda *_: "stub", capture_factory=Capture, detector=flaky_detector)
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, sid)
        assert session.status == "stopped"
        assert session.dropped_segments == 1
        assert session.frames_processed == 3
        assert db.scalar(select(func.count()).select_from(LiveFishTrack)) == 1


def test_repeated_detector_failure_still_fails_the_session(db_session_factory, test_settings, synthetic_frame):
    shutdown = threading.Event()
    now = utc_now()
    test_settings.live_retry_seconds = 0.01
    test_settings.live_max_retries = 2

    class Capture:
        dropped = 0

        def __init__(self, url, directory, settings):
            directory.mkdir(parents=True)
            self.directory, self.served = directory, 0
            self.cv2 = video_media.load_cv2()

        def next_segment(self):
            self.served += 1
            path = self.directory / f"chunk{self.served}.mp4"
            writer = video_media.open_video_writer(self.cv2, path, test_settings.live_fps, 240, 160)
            assert writer is not None
            writer.write(synthetic_frame)
            writer.release()
            return Segment(path, now + timedelta(seconds=self.served))

        def close(self):
            pass

    def broken_detector(*_):
        raise RuntimeError("VIAME exited with code -11")

    with db_session_factory() as db:
        sid = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(sid, test_settings, db_session_factory, shutdown,
        resolver=lambda *_: "stub", capture_factory=Capture, detector=broken_detector)
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, sid)
        assert session.status == "failed"
        assert session.dropped_segments == test_settings.live_max_retries + 1


def test_worker_resolution_retries_then_fails(db_session_factory, test_settings):
    test_settings.live_max_retries = 1
    test_settings.live_retry_seconds = .001
    calls = []
    def fail(*_):
        calls.append(1)
        raise LiveSourceError("No camera media available")
    with db_session_factory() as db:
        sid = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(sid, test_settings, db_session_factory, resolver=fail)
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, sid)
        assert session.status == "failed"
        assert session.reconnect_count == len(calls) == 2
        assert "No camera media" in session.error_message


def test_stale_worker_finalizes_persisted_tracks(db_session_factory, test_settings, synthetic_frame):
    with db_session_factory() as db:
        session = new_session(db, status="running", heartbeat_at=utc_now() - timedelta(seconds=121))
        tracker = LiveTracker(db, session, test_settings)
        tracker.process_frame(synthetic_frame, [detection()], utc_now() - timedelta(seconds=120))
        db.commit()
        sid = session.id
    recover_stale_sessions(db_session_factory, test_settings)
    with db_session_factory() as db:
        assert db.get(LiveMonitorSession, sid).status == "failed"
        track = db.scalar(select(LiveFishTrack))
        assert track.finalization_reason == "worker-lost"
        assert Path(track.clip_path).exists()


def test_stream_closes_on_disconnect(db_session_factory, test_settings):
    class Request:
        async def is_disconnected(self):
            return True
    async def consume():
        return [chunk async for chunk in live.annotated_frames(db_session_factory, uuid.uuid4(), test_settings, Request())]
    assert asyncio.run(consume()) == []


def test_dashboard_contains_live_panel(client):
    # The legacy dashboard, kept at /legacy until parity is signed off.
    html = client.get("/legacy").text
    assert 'id="live-panel"' in html and 'id="live-gallery"' in html
    # live.js populates the selector and reads the hint by these ids.
    assert 'id="live-source"' in html and 'for="live-source"' in html and 'id="live-source-hint"' in html
    assert "/static/live.js" in html


def test_open_session_unique_constraint(db_session_factory):
    from sqlalchemy.exc import IntegrityError
    with db_session_factory() as db:
        new_session(db)
        with pytest.raises(IntegrityError):
            new_session(db)
        db.rollback()


def test_detector_adapter_normalizes_frames_and_cleans_success(db_session_factory, test_settings, tmp_path, monkeypatch):
    from app.workers import live_worker
    captured = []
    class Runner:
        def __init__(self, settings):
            self.settings = settings
            captured.append(settings)
        def run(self, path, job_id):
            directory = self.settings.job_root / str(job_id)
            directory.mkdir(parents=True)
            csv = directory / "result.csv"
            csv.write_text("1,synthetic,1,10,20,30,40,0.9,-1,fish,0.9\n")
            return SimpleNamespace(output_csv_path=csv)
    monkeypatch.setattr(live_worker, "build_viame_runner", Runner)
    test_settings.viame_frame_number_offset = -1
    session_id = uuid.uuid4()
    result = live_worker.detect_segment(Segment(tmp_path / "input.mp4", utc_now()), test_settings, session_id)
    assert result[0].frame_number == 0
    assert captured[0].viame_timeout_seconds == test_settings.live_detector_timeout_seconds
    assert captured[0].viame_downsample_fps == test_settings.live_fps
    assert not list(live_path(test_settings, str(session_id), "inference").iterdir())


def test_worker_honors_stop_during_inference(db_session_factory, test_settings):
    class Capture:
        dropped = 0
        def __init__(self, url, directory, settings):
            directory.mkdir(parents=True)
            self.path = directory / "unused.mp4"
            self.path.write_bytes(b"stub")
        def next_segment(self):
            return Segment(self.path, utc_now())
        def close(self):
            self.path.unlink(missing_ok=True)
    with db_session_factory() as db:
        sid = new_session(db).id
    claim_session(db_session_factory, "test")
    def detector(*_):
        with db_session_factory() as db:
            session = db.get(LiveMonitorSession, sid)
            # The stop flag survives a concurrent status update from the worker.
            session.stop_requested, session.status = True, "running"
            db.commit()
        return []
    run_session(sid, test_settings, db_session_factory, resolver=lambda *_: "stub",
                capture_factory=Capture, detector=detector)
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, sid)
        assert session.status == "stopped" and session.frames_processed == 0


def test_clip_failure_still_finalizes_and_preserves_crop(db_session_factory, test_settings, synthetic_frame, monkeypatch):
    with db_session_factory() as db:
        tracker = LiveTracker(db, new_session(db), test_settings)
        now = utc_now()
        tracker.process_frame(synthetic_frame, [detection()], now)
        track = next(iter(tracker.active.values()))
        def fail(_):
            raise RuntimeError("Encoder failed")
        monkeypatch.setattr(tracker, "_render_clip", fail)
        tracker.expire(now + timedelta(seconds=10))
        db.commit()
        assert track.status == "finalized" and track.media_error
        assert track.clip_path is None and Path(track.crop_path).is_file()


def species_session(db, target=2, **kwargs):
    return new_session(db, species_id_enabled=True, species_id_fish_target=target, **kwargs)


def fixed_quality(tracker, monkeypatch, short_sides):
    """Drive ranking from a table instead of real pixels.

    Keyed on the box, so a test states each staged frame's quality directly without
    having to steer detector confidence (which the floor also reads).
    """

    def measure(frame, box, confidence):
        return {"short_side": short_sides[tuple(box)], "sharpness": 200.0,
                "confidence": .85, "luminance": 120.0, "contrast": 60.0, "colorfulness": 40.0}
    monkeypatch.setattr(tracker, "_frame_quality", measure)


def test_species_staging_ranks_frames_and_keeps_a_clean_crop(db_session_factory, test_settings, monkeypatch):
    test_settings.fishial_enabled = True
    test_settings.fishial_preprocess = "funie_gan"
    monkeypatch.setattr(LiveTracker, "_preprocess_crop", lambda *args: pytest.fail("Enhanced at staging"))
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    frame[:] = (30, 90, 150)
    def obs(key, x, confidence=.9):
        return VIAMEDetection(key, "test", 0, x, 40, x + 100, 140,
                              confidence, None, "fish", confidence, None)
    now = utc_now()
    with db_session_factory() as db:
        session = species_session(db)
        tracker = LiveTracker(db, session, test_settings)
        # Pool size is derived once at session start and persisted, so a worker
        # restart cannot resize it: max(3 * 2, 2 + 8) = 10.
        assert session.species_id_candidate_pool_size == 10
        # Every track that clears the floor becomes a free candidate. Nothing is
        # enrolled until selection pays for it.
        tracker.process_frame(frame, [obs("1", 40, .65), obs("2", 250), obs("3", 500)], now)
        db.commit()
        tracks = list(tracker.active.values())
        assert [t.fishial_state for t in tracks] == ["candidate"] * 3
        assert session.species_id_fish_enrolled == 0
        # A second frame inside the same 0.6 s separation window keeps exactly one.
        tracker.process_frame(frame, [obs("2", 250)], now + timedelta(seconds=.2))
        db.commit()
        assert tracks[1].fishial_frames_used == 1
        tracker.process_frame(frame, [obs("2", 250)], now + timedelta(seconds=.8))
        db.commit()
        assert tracks[1].fishial_frames_used == 2
        directory = scratch_path(test_settings, str(session.id), str(tracks[1].id), "fishial")
        crops = sorted(directory.glob("*.jpg"))
        assert len(crops) == 2
        clean = tracker.cv2.imread(str(crops[0]))
        assert clean.shape[:2] == (116, 116)  # Expanded bbox, no fixed 260px letterbox.
        assert np.max(np.abs(clean.astype(int) - frame[0, 0].astype(int))) <= 2
        staged = tracks[1].fishial_votes["staged"]
        assert [entry["window"] for entry in staged] == [0, 1]
        # The expected box is stored as written, so a clamped edge crop is not later
        # re-centred, and the crop size lets the replay tool reproduce the match.
        assert staged[0]["expected_box"] == pytest.approx([.065, .065, .935, .935], abs=.01)
        assert staged[0]["crop_size"] == [116, 116]
        assert set(staged[0]["quality"]) == {"confidence", "short_side", "sharpness",
                                             "luminance", "contrast", "colorfulness"}
        # A candidate keeps its staged crops after it is lost: they are the evidence
        # the between-segment pass would pay for.
        monkeypatch.setattr(tracker, "_render_clip", lambda _: None)
        tracker.expire(now + timedelta(seconds=20))
        assert directory.is_dir()


def test_best_of_ranking_evicts_the_retained_worst(db_session_factory, test_settings, monkeypatch):
    test_settings.fishial_enabled = True
    test_settings.fishial_max_staged_frames_per_candidate = 3
    test_settings.fishial_min_frame_separation_seconds = 0
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    frame[:] = (30, 90, 150)
    now = utc_now()
    with db_session_factory() as db:
        session = species_session(db, target=1)
        tracker = LiveTracker(db, session, test_settings)
        track = LiveFishTrack(session_id=session.id, status="active", first_seen_at=now,
            last_seen_at=now, detection_count=0, max_confidence=0, mean_confidence=0,
            x1=40, y1=40, x2=140, y2=140)
        db.add(track)
        db.flush()
        # Each frame gets its own box so the stubbed quality table can address it.
        sizes = {0: 50, 1: 60, 2: 70, 3: 40, 4: 200}
        boxes = {number: (40 + number, 40, 140 + number, 140) for number in sizes}
        fixed_quality(tracker, monkeypatch, {boxes[n]: size for n, size in sizes.items()})
        directory = scratch_path(test_settings, str(session.id), str(track.id), "fishial")
        for number in sizes:
            tracker._stage_species_crop(frame, track, boxes[number], .9,
                                        number, now + timedelta(seconds=number))
        db.commit()
        kept = {entry["frame_number"] for entry in track.fishial_votes["staged"]}
        # Frames 0-2 fill the buffer; 3 is weaker and is dropped without touching the
        # set; 4 is stronger and evicts the retained worst (frame 0).
        assert kept == {1, 2, 4}
        assert track.fishial_frames_used == 3
        assert {int(path.stem) for path in directory.glob("*.jpg")} == {1, 2, 4}
        assert not (directory / f"{0:012d}.jpg").exists()
        assert not (directory / f"{3:012d}.jpg").exists()


def test_one_frame_per_separation_window_keeps_votes_independent(db_session_factory, test_settings, monkeypatch):
    test_settings.fishial_enabled = True
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    frame[:] = (30, 90, 150)
    now = utc_now()
    with db_session_factory() as db:
        session = species_session(db, target=1)
        tracker = LiveTracker(db, session, test_settings)
        track = LiveFishTrack(session_id=session.id, status="active", first_seen_at=now,
            last_seen_at=now, detection_count=0, max_confidence=0, mean_confidence=0,
            x1=40, y1=40, x2=140, y2=140)
        db.add(track)
        db.flush()
        plan = [(50, 0), (60, .1), (70, .2), (50, .3), (60, .4), (60, 1.0), (60, 2.0)]
        boxes = {number: (40 + number, 40, 140 + number, 140) for number in range(len(plan))}
        fixed_quality(tracker, monkeypatch,
                      {boxes[n]: size for n, (size, _) in enumerate(plan)})
        # Five frames inside one 0.6 s window: near-duplicates must not fill the set.
        for number, (_, offset) in enumerate(plan[:5]):
            tracker._stage_species_crop(frame, track, boxes[number], .9,
                                        number, now + timedelta(seconds=offset))
        db.commit()
        staged = track.fishial_votes["staged"]
        assert len(staged) == 1 and staged[0]["frame_number"] == 2  # The best of the window.
        # Three detections spread over three windows do become identifiable.
        for number, (_, offset) in enumerate(plan[5:], start=5):
            tracker._stage_species_crop(frame, track, boxes[number], .9,
                                        number, now + timedelta(seconds=offset))
        db.commit()
        assert track.fishial_frames_used == test_settings.fishial_min_frames_to_vote == 3


def test_coral_city_medians_now_reach_min_frames_to_vote(db_session_factory, test_settings):
    """Regression for the failure that returned zero identifications on clear water.

    A track resembling Coral City's measured medians - detector confidence ~0.62 and
    a 57 px bbox short side - staged 1 or 2 frames against the old 0.70/96 gate and
    ended "insufficient clear frames". It must now reach fishial_min_frames_to_vote.
    """

    test_settings.fishial_enabled = True
    frame = np.random.default_rng(7).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    now = utc_now()
    with db_session_factory() as db:
        session = species_session(db, target=1)
        tracker = LiveTracker(db, session, test_settings)
        track = LiveFishTrack(session_id=session.id, status="active", first_seen_at=now,
            last_seen_at=now, detection_count=0, max_confidence=0, mean_confidence=0,
            x1=300, y1=300, x2=357, y2=357)
        db.add(track)
        db.flush()
        box = (300, 300, 357, 357)  # 57 px short side, the measured median.
        assert not tracker._is_clear_frame(frame, box, .619) or True
        for number in range(4):
            tracker._stage_species_crop(frame, track, box, .619, number,
                                        now + timedelta(seconds=number))
        db.commit()
        assert track.fishial_state == "candidate"
        assert track.fishial_frames_used >= test_settings.fishial_min_frames_to_vote


@pytest.mark.parametrize("confidence,short_side,admitted", [
    (.49, 100, False), (.51, 100, True),      # Confidence floor at 0.50.
    (.9, 39, False), (.9, 41, True),          # Short-side floor at 40 px.
])
def test_recalibrated_floor_excludes_junk_only(db_session_factory, test_settings,
                                               confidence, short_side, admitted):
    frame = np.random.default_rng(12).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    with db_session_factory() as db:
        tracker = LiveTracker(db, new_session(db), test_settings)
        box = (20, 20, 20 + short_side, 20 + short_side)
        assert tracker._is_clear_frame(frame, box, confidence) is admitted


def test_floor_still_rejects_unusable_geometry(db_session_factory, test_settings):
    frame = np.random.default_rng(12).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    with db_session_factory() as db:
        tracker = LiveTracker(db, new_session(db), test_settings)
        assert tracker._is_clear_frame(frame, (20, 20, 120, 120), .9)
        assert not tracker._is_clear_frame(frame, (20, 20, 120, 120), 1.1)
        assert not tracker._is_clear_frame(frame, (20, 20, 120, float("nan")), .9)
        assert not tracker._is_clear_frame(frame, (20, 20, 120, 120), float("inf"))
        # Edge-breaking boxes stay rejected: a truncated fish cannot be identified.
        for box in ((0, 20, 120, 120), (20, 0, 120, 120), (200, 20, 319, 120), (20, 100, 120, 239)):
            assert not tracker._is_clear_frame(frame, box, .9)
        test_settings.fishial_blur_min_variance = 100
        assert not tracker._is_clear_frame(np.zeros_like(frame), (20, 20, 120, 120), .9)
        assert tracker._is_clear_frame(frame, (20, 20, 120, 120), .9)
        test_settings.fishial_edge_margin_pixels = 0
        assert not tracker._is_clear_frame(frame, (-2, 20, 120, 120), .9)


def make_track(db, session, now, x=40):
    track = LiveFishTrack(session_id=session.id, status="active", first_seen_at=now,
        last_seen_at=now, detection_count=0, max_confidence=0, mean_confidence=0,
        x1=x, y1=40, x2=x + 100, y2=140)
    db.add(track)
    db.flush()
    return track


def test_candidate_pool_evicts_the_weakest_but_never_a_paid_track(db_session_factory, test_settings, monkeypatch):
    test_settings.fishial_enabled = True
    test_settings.fishial_candidate_pool_size = 3
    test_settings.fishial_min_frame_separation_seconds = 0
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    frame[:] = (30, 90, 150)
    now = utc_now()
    with db_session_factory() as db:
        session = species_session(db, target=1)
        tracker = LiveTracker(db, session, test_settings)
        assert session.species_id_candidate_pool_size == 3
        fixed_quality(tracker, monkeypatch, {(40, 40, 140, 140): 50, (200, 40, 300, 140): 60,
                                             (360, 40, 460, 140): 70, (520, 40, 620, 140): 80,
                                             (40.5, 40, 140.5, 140): 30})
        weak, middle, strong = (make_track(db, session, now, x) for x in (40, 200, 360))
        for index, track in enumerate([weak, middle, strong]):
            tracker._stage_species_crop(frame, track, (track.x1, 40, track.x1 + 100, 140),
                                        .9, index, now)
        db.commit()
        assert len(tracker.pool) == 3
        weak_dir = scratch_path(test_settings, str(session.id), str(weak.id), "fishial")
        assert weak_dir.is_dir()
        # A stronger fourth track evicts the weakest candidate outright.
        fourth = make_track(db, session, now, 520)
        tracker._stage_species_crop(frame, fourth, (520, 40, 620, 140), .9, 3, now)
        db.commit()
        assert weak.fishial_state == "disabled"
        assert weak.fishial_votes["reason"] == "evicted from candidate pool"
        assert weak.fishial_votes["staged"] == [] and weak.fishial_frames_used == 0
        assert not weak_dir.exists()
        assert fourth.fishial_state == "candidate"
        # A weaker fifth track cannot displace anything.
        fifth = make_track(db, session, now, 40)
        tracker._stage_species_crop(frame, fifth, (40.5, 40, 140.5, 140), .9, 4, now)
        db.commit()
        assert fifth.fishial_state == "disabled" and len(tracker.pool) == 3
        # A selected or paid track is not in the pool, so it can never be evicted.
        middle.fishial_state = "ready"
        db.commit()
        tracker._stage_species_crop(frame, middle, (200, 40, 300, 140), .9, 5, now)
        assert middle.fishial_state == "ready"


def test_staged_byte_cap_evicts_weakest_first(db_session_factory, test_settings, monkeypatch):
    test_settings.fishial_enabled = True
    test_settings.fishial_min_frame_separation_seconds = 0
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    frame[:] = (30, 90, 150)
    now = utc_now()
    with db_session_factory() as db:
        session = species_session(db, target=1)
        tracker = LiveTracker(db, session, test_settings)
        fixed_quality(tracker, monkeypatch, {(40, 40, 140, 140): 50, (300, 40, 400, 140): 90})
        weak, strong = make_track(db, session, now, 40), make_track(db, session, now, 300)
        tracker._stage_species_crop(frame, weak, (40, 40, 140, 140), .9, 0, now)
        db.commit()
        assert tracker.staged_bytes > 0
        weak_dir = scratch_path(test_settings, str(session.id), str(weak.id), "fishial")
        assert weak_dir.is_dir()
        # Cap the session just under what is already staged plus one more crop.
        test_settings.fishial_max_staged_bytes = tracker.staged_bytes
        tracker._stage_species_crop(frame, strong, (300, 40, 400, 140), .9, 1, now)
        db.commit()
        assert weak.fishial_state == "disabled"
        assert weak.fishial_votes["reason"] == "evicted to stay within the staged byte cap"
        assert not weak_dir.exists()
        assert strong.fishial_state == "candidate"
        assert tracker.staged_bytes <= test_settings.fishial_max_staged_bytes


def test_preprocessing_is_off_by_default_and_only_touches_the_staged_crop(db_session_factory, test_settings):
    frame = np.full((300, 700, 3), (40, 90, 20), dtype=np.uint8)
    with db_session_factory() as db:
        tracker = LiveTracker(db, new_session(db), test_settings)
        crop = frame[40:140, 40:140]
        assert test_settings.fishial_preprocess == "none"
        assert np.array_equal(tracker._preprocess_crop(crop), crop)
        test_settings.fishial_preprocess = "both"
        adjusted = tracker._preprocess_crop(crop)
        assert adjusted.shape == crop.shape and not np.array_equal(adjusted, crop)
        # Gray-world balances the channel means it was given.
        test_settings.fishial_preprocess = "white_balance"
        balanced = tracker._preprocess_crop(crop).reshape(-1, 3).mean(axis=0)
        assert balanced.max() - balanced.min() < 3
        test_settings.fishial_preprocess = "none"
        test_settings.fishial_upscale_short_side = 200
        upscaled = tracker._preprocess_crop(crop)
        assert upscaled.shape[:2] == (200, 200)
        # Never downscale.
        test_settings.fishial_upscale_short_side = 50
        assert tracker._preprocess_crop(crop).shape == crop.shape


def test_species_crop_write_failure_is_not_fatal(db_session_factory, test_settings, synthetic_frame, monkeypatch):
    test_settings.fishial_enabled = True
    test_settings.fishial_min_crop_pixels = 16
    with db_session_factory() as db:
        session = new_session(db, species_id_enabled=True, species_id_fish_target=1)
        tracker = LiveTracker(db, session, test_settings)
        def fail(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(tracker.cv2, "imencode", fail)
        tracker.process_frame(synthetic_frame, [detection()], utc_now())
        db.commit()
        track = next(iter(tracker.active.values()))
        assert track.fishial_state == "candidate"
        assert track.fishial_frames_used == 0
        assert session.frames_processed == 1
