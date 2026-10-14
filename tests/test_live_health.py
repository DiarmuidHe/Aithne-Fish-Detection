"""The camera registry, and reporting which of its cameras are broadcasting.

Most of the registry is duty-cycled - the thirteen Ocean Networks Canada seafloor
cameras typically publish five minutes at the top of the hour - so an operator who
cannot see which camera is live has to start a session and watch it fail. The probe
that answers that must stay cheap: registry URLs only, no subprocess, one bounded
concurrent request each, cached.
"""
from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import urlparse

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.api import live as live_api
from app.config import (
    LIVE_SOURCE_KEY_PATTERN,
    ONC_ATTRIBUTION,
    ONC_INFO_URL,
    SMARTBAY_ATTRIBUTION,
    SMARTBAY_INFO_URL,
    Settings,
)
from app.db.models import LiveMonitorSession
from app.services import live_source
from app.services.live_source import probe_source, probe_sources, reset_health_cache

LIVE_PLAYLIST = ("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION=1280x720\n"
                 "chunklist_w1234.m3u8\n")
# What Wowza serves while a camera is reachable but publishing nothing.
IDLE_PLAYLIST = "#EXTM3U\n"


@pytest.fixture(autouse=True)
def clear_health_cache():
    """The probe caches across calls, which must not leak between tests."""

    reset_health_cache()
    yield
    reset_health_cache()


@pytest.fixture
def no_subprocess(monkeypatch):
    """Fail the test if anything on the probe path shells out.

    Resolving a page source costs seconds through yt-dlp. Doing it per camera on a
    dashboard poll would fork seventeen processes every thirty seconds, so a page
    source is reported unprobed instead. This is the constraint most likely to be
    violated silently.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("The liveness probe must never invoke a subprocess")

    monkeypatch.setattr(live_source.subprocess, "run", forbidden)
    monkeypatch.setattr(live_source.subprocess, "Popen", forbidden)


@pytest.fixture
def probe_transport(monkeypatch):
    """Route the endpoint's probe through a mock transport, recording every request.

    The real probe logic - states, concurrency and the cache - still runs; only the
    socket is replaced, so no test reaches ONC.
    """

    def install(handler):
        requested: list[str] = []

        def handle(request):
            requested.append(str(request.url))
            return handler(request)

        def probe(sources, settings):
            with httpx.Client(transport=httpx.MockTransport(handle)) as client:
                return live_source.probe_sources(sources, settings, client)

        monkeypatch.setattr(live_api, "probe_sources", probe)
        return requested

    return install


def probe_one(settings, key, handler):
    """One camera's state, and the URLs the probe actually asked for."""

    requested: list[str] = []

    def handle(request):
        requested.append(str(request.url))
        return handler(request)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        health = probe_source(settings.live_source(key), settings, client)
    return health, requested


# --- The registry ------------------------------------------------------------


def test_registry_ships_seventeen_cameras_with_watchable_links(test_settings):
    sources = list(test_settings.live_source_registry)
    # Asserted so a careless edit to DEFAULT_LIVE_SOURCES is caught rather than
    # quietly changing what the dropdown offers.
    assert len(sources) == 17
    assert len({source.key for source in sources}) == 17
    for source in sources:
        assert LIVE_SOURCE_KEY_PATTERN.match(source.key), source.key
        assert source.label.strip()
        assert urlparse(source.url).scheme in {"http", "https"}
        if source.info_url is not None:
            assert urlparse(source.info_url).scheme in {"http", "https"}
            # A link an operator clicks must never be the playlist: Chrome plays
            # no HLS and shows a black frame stuck at 0:00 for a healthy camera.
            assert source.info_url != source.url
            assert not source.info_url.endswith(".m3u8")

    onc = [source for source in sources if source.key.startswith("onc-")]
    assert len(onc) == 13
    for source in onc:
        # No Northeast Pacific species list exists; the filter is opt-in, so unset
        # accepts every name rather than silently discarding identifications.
        assert source.region is None
        assert source.attribution == ONC_ATTRIBUTION
        assert source.info_url == ONC_INFO_URL
        assert source.url.startswith("https://media.oceannetworks.ca:4935/rtplive/")
        assert source.url.endswith("/playlist.m3u8")
    # Non-commercial, and more restrictive than the SmartBay cameras beside them.
    assert "CC BY-NC-SA 3.0" in ONC_ATTRIBUTION

    for key in ("smartbay-cam1", "smartbay-cam2", "smartbay-cam3"):
        source = test_settings.live_source(key)
        assert (source.attribution, source.info_url) == (SMARTBAY_ATTRIBUTION, SMARTBAY_INFO_URL)
    # Changing which camera is preselected is a separate decision.
    assert test_settings.live_default_source_key == "coral-city"


def test_an_info_url_gets_the_same_scheme_check_as_a_stream_url():
    entry = {"key": "bay", "label": "Bay", "url": "https://a.example/live.m3u8"}
    assert Settings(_env_file=None, live_default_source_key="bay",
                    live_source_registry=[entry]).live_source("bay").info_url is None
    with pytest.raises(ValidationError):
        Settings(_env_file=None, live_default_source_key="bay",
                 live_source_registry=[{**entry, "info_url": "file:///etc/passwd"}])


# --- One camera --------------------------------------------------------------


def test_a_broadcasting_playlist_reads_as_live(test_settings, no_subprocess):
    url = test_settings.live_source("onc-folger-pinnacle").url
    health, requested = probe_one(test_settings, "onc-folger-pinnacle",
                                  lambda request: httpx.Response(200, text=LIVE_PLAYLIST))
    assert (health.key, health.state, health.detail) == ("onc-folger-pinnacle", "live", None)
    assert health.checked_at is not None
    # The configured URL only; a redirect target is never recorded or reported.
    assert requested == [url]


def test_a_playlist_advertising_nothing_reads_as_idle(test_settings, no_subprocess):
    health, _ = probe_one(test_settings, "smartbay-cam2",
                          lambda request: httpx.Response(200, text=IDLE_PLAYLIST))
    assert health.state == "idle"
    # The existing resolver's wording, reused rather than a second HLS parser.
    assert "not currently broadcasting" in health.detail


@pytest.mark.parametrize("status", [404, 410])
def test_a_missing_publication_is_the_duty_cycle_not_a_fault(test_settings, status, no_subprocess):
    # Wowza's rtplive application 404s between scheduled windows. Reading that as
    # broken would make ten of the thirteen ONC cameras look faulty most of the hour.
    health, _ = probe_one(test_settings, "onc-tempo-mini",
                          lambda request: httpx.Response(status))
    assert (health.state, health.detail) == ("idle", "off schedule")


def test_a_timeout_or_refused_connection_reads_as_offline(test_settings, no_subprocess):
    def timeout(request):
        raise httpx.ConnectTimeout("too slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    assert probe_one(test_settings, "onc-cascadia-basin", timeout)[0].state == "offline"
    assert probe_one(test_settings, "onc-cascadia-basin", timeout)[0].detail == "timed out"
    assert probe_one(test_settings, "onc-cascadia-basin", refused)[0].state == "offline"


def test_a_server_error_reads_as_offline(test_settings, no_subprocess):
    health, _ = probe_one(test_settings, "onc-digby-island",
                          lambda request: httpx.Response(503))
    assert (health.state, health.detail) == ("offline", "HTTP 503")


def test_a_page_source_is_reported_without_being_touched(test_settings, no_subprocess):
    def unexpected(request):
        raise AssertionError("A page source must not be fetched by the probe")

    health, requested = probe_one(test_settings, "coral-city", unexpected)
    assert (health.state, health.detail) == ("unknown", "page source; not probed")
    assert requested == []


# --- The whole registry ------------------------------------------------------


def test_one_hanging_camera_does_not_cost_the_others(test_settings, no_subprocess):
    hanging = test_settings.live_source("onc-folger-pinnacle").url

    def handle(request):
        if str(request.url) == hanging:
            raise httpx.ConnectTimeout("too slow", request=request)
        return httpx.Response(200, text=LIVE_PLAYLIST)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        health = probe_sources(test_settings.live_source_registry, test_settings, client)

    assert set(health) == set(test_settings.live_sources())
    assert health["onc-folger-pinnacle"].state == "offline"
    assert health["onc-china-creek"].state == "live"
    assert health["smartbay-cam1"].state == "live"
    assert health["coral-city"].state == "unknown"


def test_the_result_is_cached_for_its_window(test_settings, no_subprocess, monkeypatch):
    requested: list[str] = []

    def handle(request):
        requested.append(str(request.url))
        return httpx.Response(200, text=LIVE_PLAYLIST)

    clock = [1_000.0]
    # Only this module's reference to time is replaced, so nothing else in the
    # process sees a frozen clock.
    monkeypatch.setattr(live_source, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    # Sixteen of the seventeen are playlists; the page source is never fetched.
    per_round = 16
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        first = probe_sources(test_settings.live_source_registry, test_settings, client)
        assert len(requested) == per_round
        # However many browser tabs poll, ONC sees one probe per camera per window.
        assert probe_sources(test_settings.live_source_registry, test_settings, client) is first
        clock[0] += test_settings.live_health_cache_seconds - 1
        probe_sources(test_settings.live_source_registry, test_settings, client)
        assert len(requested) == per_round

        clock[0] += 2
        probe_sources(test_settings.live_source_registry, test_settings, client)
        assert len(requested) == 2 * per_round


def test_a_changed_registry_is_not_answered_from_the_old_cache(test_settings, no_subprocess):
    def handle(request):
        return httpx.Response(200, text=LIVE_PLAYLIST)

    trimmed = [test_settings.live_source("smartbay-cam1")]
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        probe_sources(trimmed, test_settings, client)
        health = probe_sources(test_settings.live_source_registry, test_settings, client)
    assert len(health) == 17


# --- The endpoints -----------------------------------------------------------


def test_sources_carries_the_link_and_the_licence(client, test_settings):
    body = client.get("/live/sources").json()
    keyed = {source["key"]: source for source in body["sources"]}
    assert len(keyed) == 17
    onc = keyed["onc-holyrood-bay"]
    assert onc["info_url"] == ONC_INFO_URL and onc["attribution"] == ONC_ATTRIBUTION
    assert onc["location"] == "Conception Bay, NL"
    # Existing consumers still get url unchanged, and the page source still has none
    # of the new fields rather than a fabricated one.
    for key, source in keyed.items():
        assert source["url"] == test_settings.live_source(key).url
    assert keyed["coral-city"]["info_url"] is None
    assert keyed["smartbay-cam3"]["attribution"] == SMARTBAY_ATTRIBUTION


def test_health_reports_every_camera_without_shelling_out_or_opening_a_session(
        client, test_settings, db_session_factory, probe_transport, no_subprocess):
    off_schedule = test_settings.live_source("onc-clayoquot-bullseye").url

    def handle(request):
        if str(request.url) == off_schedule:
            return httpx.Response(404)
        return httpx.Response(200, text=LIVE_PLAYLIST)

    requested = probe_transport(handle)
    body = client.get("/live/sources/health").json()

    assert body["enabled"] is True and body["checked_at"]
    assert set(body["sources"]) == set(test_settings.live_sources())
    assert body["sources"]["onc-clayoquot-bullseye"] == {
        "state": "idle", "detail": "off schedule",
        "checked_at": body["sources"]["onc-clayoquot-bullseye"]["checked_at"]}
    assert body["sources"]["onc-cascadia-basin"]["state"] == "live"
    assert body["sources"]["coral-city"]["state"] == "unknown"
    # No resolved or redirected URL is ever reported back.
    assert not any("url" in entry for entry in body["sources"].values())
    assert test_settings.live_source("coral-city").url not in requested
    assert len(requested) == 16

    with db_session_factory() as db:
        assert db.scalar(select(func.count()).select_from(LiveMonitorSession)) == 0


def test_a_disabled_probe_degrades_to_no_badges(client, test_settings, probe_transport,
                                                no_subprocess):
    def unexpected(request):
        raise AssertionError("A disabled probe must not reach a camera")

    probe_transport(unexpected)
    test_settings.live_health_enabled = False
    response = client.get("/live/sources/health")
    # 200 with an empty map, not a 404: the camera list works without the badges.
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False and body["sources"] == {} and body["checked_at"]
