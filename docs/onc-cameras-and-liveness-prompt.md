# Implementation prompt: add the ONC cameras and show which are broadcasting

Paste the section below into a fresh Claude Code session in this repo.

---

Two related changes to live monitoring:

1. Add Ocean Networks Canada's thirteen seafloor cameras to the source registry.
2. Show, per camera, whether it is broadcasting **right now**, with a link to a page
   an operator can actually watch it on.

Part 2 exists because most of these cameras are duty-cycled — typically five minutes
at the top of each hour — so "the camera is in the dropdown" says nothing about
whether starting a session will get frames. Today the only way to find out is to
start one and watch it fail.

## Cameras

All URLs below are the configured source; they are direct HLS master playlists, so
they hit the `.m3u8` short-circuit in `resolve_stream()` and never invoke yt-dlp.
Verified reachable 2026-09-13 15:34 UTC; the three marked 404 were between scheduled
windows, which is the normal idle state, not a fault.

| Key | Label | Location | Stream URL | Schedule |
| --- | --- | --- | --- | --- |
| `onc-folger-pinnacle` | ONC Folger Pinnacle | Barkley Sound, BC | `https://media.oceannetworks.ca:4935/rtplive/AXISQ6074EPTZACCC8EACA579/playlist.m3u8` | 5 min hourly |
| `onc-china-creek` | ONC China Creek | Alberni Inlet, BC | `https://media.oceannetworks.ca:4935/rtplive/AXISCAMB8A44FB27F40/playlist.m3u8` | 5 min hourly |
| `onc-campbell-river` | ONC Campbell River | Campbell River, BC | `https://media.oceannetworks.ca:4935/rtplive/AXISCAMB8A44F271FBC/playlist.m3u8` | 5 min hourly |
| `onc-digby-island` | ONC Digby Island | Prince Rupert, BC | `https://media.oceannetworks.ca:4935/rtplive/AXISCAMB8A44F271FE9/playlist.m3u8` | 5 min hourly |
| `onc-kitamaat-village` | ONC Kitamaat Village | Douglas Channel, BC | `https://media.oceannetworks.ca:4935/rtplive/AXISCAMACCC8ED95D9B/playlist.m3u8` | 5 min hourly |
| `onc-hartley-bay` | ONC Hartley Bay | Hartley Bay, BC | `https://media.oceannetworks.ca:4935/rtplive/AXISCAMB8A44F78A6A8/playlist.m3u8` | 5 min hourly |
| `onc-holyrood-bay` | ONC Holyrood Bay | Conception Bay, NL | `https://media.oceannetworks.ca:4935/rtplive/SUBCRAYFIN21017/playlist.m3u8` | 5 min hourly |
| `onc-barkley-axis` | ONC Barkley Canyon Axis | Barkley Canyon, BC | `https://media.oceannetworks.ca:4935/rtplive/AXISCAMB8A44F720E0D/playlist.m3u8` | 5 min hourly |
| `onc-barkley-upper-slope` | ONC Barkley Canyon Upper Slope | Barkley Canyon, BC | `https://media.oceannetworks.ca:4935/rtplive/Z3CAM50009901224910005/playlist.m3u8` | 5 min, start of UTC hour |
| `onc-cascadia-basin` | ONC Cascadia Basin | Cascadia Basin, BC | `https://media.oceannetworks.ca:4935/rtplive/Z3CAM50016501240910002/playlist.m3u8` | 5 min hourly |
| `onc-clayoquot-bullseye` | ONC Clayoquot Bullseye | Clayoquot Slope, BC | `https://media.oceannetworks.ca:4935/rtplive/SUBCCAMMK6HDC_17690/playlist.m3u8` | every 4 h from 00:00 UTC |
| `onc-endeavour-mothra` | ONC Endeavour Mothra | Endeavour Ridge | `https://media.oceannetworks.ca:4935/rtplive/DRAGONFISHSUBC13113/playlist.m3u8` | 4 min every 4 h |
| `onc-tempo-mini` | ONC Tempo Mini — Main Endeavour | Endeavour Ridge | `https://media.oceannetworks.ca:4935/rtplive/SMOOVECAMTEMPOMINISMO002/playlist.m3u8` | 4 min every 4 h |

The human-viewable page for all thirteen is <https://www.oceannetworks.ca/multimedia/live-cameras/>.

**Licence.** ONC live camera video is **CC BY-NC-SA 3.0** — attribution, non-commercial,
share-alike. This is *not* the CC BY 4.0 that covers ONC's data portal, and it is
more restrictive than the CC BY 4.0 on the existing SmartBay cameras. Carry the
attribution string "Ocean Networks Canada (CC BY-NC-SA 3.0)" through to the UI, and
say plainly in the README that these thirteen are unsuitable for a commercial demo.

**Do not set `region` on any ONC entry.** `app/services/species_region.py` has no
Northeast Pacific list, and the existing comment on the Coral City entry records why
unset is the safe default: the filter is opt-in, and unset accepts every name. Adding
a `north_east_pacific` region is a separate job — do not invent a species list here.

## What already supports this (do not rebuild it)

- `LiveCameraSource` and `live_source_registry` in `app/config.py` already model and
  validate the registry. Adding entries needs no migration and no API change.
- `check_playlist_body()` in `app/services/live_source.py` already decides whether a
  playlist body means "broadcasting" or "online but publishing nothing". The liveness
  probe must reuse it rather than re-parsing HLS.
- `GET /live/sources` already returns `url` per source and the dropdown already
  renders `label · location`. You are adding fields, not a new endpoint shape.
- The worker's reconnect/backoff already handles an idle camera. Nothing about the
  capture path changes.

## Design constraints

1. **The health check must never accept or resolve a client-supplied URL.** It probes
   only URLs already in `live_source_registry`. Same SSRF invariant as
   `resolve_stream()`.
2. **The health check must never invoke yt-dlp or a subprocess.** A page-type source
   (Coral City, or any non-`.m3u8` URL) is reported as `unknown`, not probed. yt-dlp
   takes seconds and would turn a dashboard poll into a fork bomb across 17 cameras.
3. **Do not log or return a resolved stream URL.** The probe touches the *configured*
   URL only, and reports a state, never a redirect target.
4. **Link operators to a viewable page, never to the raw `.m3u8`.** Chrome has no
   native HLS support: pasting one of these playlist URLs into it yields a black
   `<video>` frozen at 0:00 even when the camera is broadcasting normally. A link
   that looks broken is worse than no link. Hence the new `info_url` field.
5. **Probing must be cheap and bounded.** One short-timeout GET per m3u8 source, run
   concurrently, with the whole result cached briefly. A camera that is down must not
   make the endpoint hang.

## Changes

### `app/config.py`

- Add to `LiveCameraSource`: `info_url: str | None = None` (a page a human can watch
  the camera on) and `attribution: str | None = None` (licence/credit line).
- Add the thirteen entries above to `DEFAULT_LIVE_SOURCES`, each with `info_url` set
  to the ONC live-cameras page and `attribution` set to
  `"Ocean Networks Canada (CC BY-NC-SA 3.0)"`. Backfill `attribution` on the existing
  SmartBay entries as `"Marine Institute SmartBay Observatory (CC BY 4.0)"` and their
  `info_url` as `https://smartbay.marine.ie/`.
- Validate `info_url` with the same http(s) scheme check the existing `url` gets.
- New settings: `live_health_enabled: bool = True`,
  `live_health_timeout_seconds: float = 2.0`, `live_health_cache_seconds: int = 20`.
- The registry goes from 4 to 17 entries. Leave `live_default_source_key` on
  `coral-city` — changing the default is a separate decision.

### `app/services/live_source.py`

Add a probe alongside the existing resolver:

```python
@dataclass(frozen=True)
class SourceHealth:
    key: str
    state: str          # "live" | "idle" | "offline" | "unknown"
    detail: str | None
    checked_at: datetime
```

- `probe_source(source, settings, client) -> SourceHealth`, for one camera:
  - URL is not `.m3u8` -> `unknown`, detail "page source; not probed" (constraint 2).
  - 2xx with a body that survives `check_playlist_body()` -> `live`.
  - 2xx whose body raises `LiveSourceError` -> `idle`, using the exception's message
    as detail. This is the existing SmartBay cam2 case.
  - 404 or 410 -> `idle`, detail "off schedule". Wowza's `rtplive` application
    returns 404 when nothing is published, which for these cameras is the duty
    cycle, not a fault. Getting this wrong makes ten of thirteen look broken.
  - Timeout, connection error, or 5xx -> `offline` with a short reason.
- `probe_sources(sources, settings) -> dict[str, SourceHealth]`, probing concurrently
  (`httpx.AsyncClient` + `asyncio.gather`, or a small thread pool — match whatever
  the codebase already does for concurrent I/O) with
  `live_health_timeout_seconds` per request. One camera hanging must not delay the
  rest.
- Cache the whole result for `live_health_cache_seconds` in a module-level
  timestamped value. The dashboard polls; ONC should see one probe per camera per
  cache window regardless of how many browser tabs are open.

### `app/api/live.py`

- `GET /live/sources`: add `info_url` and `attribution` to each entry in the
  `sources` list. Nothing else changes.
- New `GET /live/sources/health` ->
  `{"enabled": bool, "checked_at": <iso>, "sources": {"<key>": {"state": ..., "detail": ..., "checked_at": ...}}}`.
  With `live_health_enabled` false, return `enabled: false` and an empty map rather
  than a 404, so the UI degrades to no badges instead of showing an error.
- The endpoint creates no session rows and touches no DB table.

### `frontend/src/api/types.ts` and the API client

- `LiveSource` gains `info_url: string | null` and `attribution: string | null`.
- Add `LiveSourceHealth` / `LiveSourcesHealth` mirroring the payload, and
  `fetchLiveSourceHealth()` next to the existing `fetchLiveSources()`.

### `frontend/src/features/live/live-route.tsx`

- Query `['live-source-health']` with `refetchInterval: 30_000`, enabled only when
  `sources.data?.enabled`. A failed health query must not surface as a route error —
  the camera list still works without it.
- In the `SingleSelect` options, append the state to the existing label, e.g.
  `ONC Folger Pinnacle · Barkley Sound, BC · live`. Keep the existing
  `· monitoring` suffix, which takes precedence when a session is open.
- Add a camera list below the toolbar, inside the console slab: one row per camera
  with a status dot (`live` green, `idle` amber, `offline` red, `unknown` muted), the
  label and location, the attribution line, and — when `info_url` is set — an
  external link ("Watch on ONC ↗") with `target="_blank"` and
  `rel="noopener noreferrer"`. Sort `live` first so a demo-ready camera is at the top.
- The dot needs a text equivalent for screen readers, not colour alone.
- With 17 cameras this list is long. Collapse it behind a disclosure that remembers
  its state in `localStorage`, matching the `SOURCE_KEY_STORAGE` pattern already in
  this file.

### `.env.example` and `README.md`

- Document `LIVE_HEALTH_ENABLED`, `LIVE_HEALTH_TIMEOUT_SECONDS`,
  `LIVE_HEALTH_CACHE_SECONDS` in the live-monitoring block.
- In the README's live-monitoring section: the registry is now 17 cameras; the ONC
  thirteen are CC BY-NC-SA 3.0 and **non-commercial**; cameras are duty-cycled and an
  `idle` badge is expected rather than a fault; and one session still runs at a time.
- Note the Chrome/HLS trap from constraint 4 next to the camera table, so nobody
  "fixes" a camera that was only ever unplayable in their browser.

## Tests

Run them the way this repo needs (a plain `pytest` fails here):

```
FISHIAL_PREPROCESS=none pytest --basetemp=/tmp/pt tests/test_live_monitor.py
```

In `tests/test_live_monitor.py`, or a new `tests/test_live_health.py` if that file is
getting long:

- Settings load with the full 17-camera registry: keys unique, all match
  `^[a-z0-9-]{1,64}$`, every `url` and `info_url` http(s), no ONC entry has `region`
  set. Assert the count so a careless edit to `DEFAULT_LIVE_SOURCES` is caught.
- `probe_source` state table, driven by a mocked transport: master playlist with an
  `EXT-X-STREAM-INF` -> `live`; `#EXTM3U` with no variant and no segment -> `idle`;
  404 -> `idle` with "off schedule"; connect timeout -> `offline`; a non-m3u8 page
  URL -> `unknown` **and no request is made at all**.
- `probe_sources` with one camera timing out still returns states for the others.
- The cache: two calls inside `live_health_cache_seconds` issue one round of
  requests; a call after it expires issues a second round.
- `GET /live/sources/health` never invokes a subprocess — monkeypatch
  `subprocess.run` to fail the test if called. This is constraint 2, and it is the
  one most likely to be silently violated.
- `GET /live/sources` includes `info_url` and `attribution`, and still returns `url`
  unchanged for existing consumers.
- `live_health_enabled=false` -> `enabled: false`, empty map, 200.
- A health request creates no `LiveMonitorSession` row.

## Acceptance

- The dropdown lists 17 cameras; each shows `live`, `idle`, `offline` or `unknown`,
  refreshed every 30 s without a page reload.
- Every ONC row links to a page where the camera is actually watchable, and no UI
  path hands a user a bare `.m3u8`.
- Starting a session on a `live` ONC camera produces frames; starting on an `idle`
  one fails with the existing "online but not currently broadcasting" message rather
  than an opaque FFmpeg error.
- With the live worker stopped, the health endpoint still answers — it is independent
  of the worker.
