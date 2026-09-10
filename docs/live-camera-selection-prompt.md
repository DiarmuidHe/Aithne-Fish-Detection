# Implementation prompt: choose the live camera source

Paste the section below into a fresh Claude Code session in this repo.

---

Add a camera selector to live monitoring so an operator can choose which of four
cameras the live pipeline analyses, instead of the single hard-coded Coral City URL.

## Cameras (all verified reachable on 2026-09-10)

| Key | Label | Source URL |
| --- | --- | --- |
| `coral-city` | Coral City Camera (Miami) | `https://www.coralcitycamera.com/` |
| `smartbay-cam1` | SmartBay Cam 1 (Galway Bay) | `https://live.heanet.ie/261512497d1945a09c73de7f05421dd9/ngrp:_all/playlist.m3u8` |
| `smartbay-cam2` | SmartBay Cam 2 (Galway Bay) | `https://live.heanet.ie/6c3ea7d05d754f26ba324d2a59c5d26b/ngrp:_all/playlist.m3u8` |
| `smartbay-cam3` | SmartBay Cam 3 — ANERIS EMUAS | `https://live.heanet.ie/7b451dbe046149c4bba3d0e93792fae6/ngrp:_all/playlist.m3u8` |

The three SmartBay cameras are HEAnet Wowza live streams embedded on
<https://smartbay.marine.ie/>. Their master playlists advertise 1280x720, 854x480
and 426x240 H.264 renditions and no audio track, so they hit the fast path in
`resolve_stream()` (the `.m3u8` short-circuit) and never invoke yt-dlp. Coral City
still needs the existing page-scrape plus yt-dlp fallback.

Cam 2 currently returns a master playlist with **no `EXT-X-STREAM-INF` variants** —
the camera is online but publishing nothing. Treat "camera reachable but idle" as a
first-class failure case, not a crash.

## What already supports this (do not rebuild it)

- `LiveMonitorSession.source_url` and `LiveMonitorSession.source_key` already exist
  in `app/db/models.py` and in migration `0005_live_monitor.py`.
- `uq_live_monitor_open_source` is a partial unique index on `source_key`, so the
  "one open session" rule is already **per camera**, not global.
- `app/workers/live_worker.py:173` already resolves `session.source_url`, so the
  worker is source-agnostic. It should need no changes beyond logging the source key.

No new migration should be required. If you believe one is, say why before writing it.

## Design constraints

1. **The API must never accept a URL from the client.** `resolve_stream()` documents
   the invariant that source URLs are administrator configuration; passing an
   operator-supplied URL to FFmpeg/yt-dlp would be an SSRF sink. The API accepts a
   `source` *key* only, and rejects an unknown key with 404. URLs stay server-side.
2. **Do not log or return a resolved stream URL** — same reason as the existing note
   in `live_source.py`: resolved URLs can carry signed credentials. Returning the
   configured *page* URL for display is fine.
3. **One session runs at a time.** `run_worker_forever()` calls `run_session()`
   synchronously, and there is one GPU. So the dropdown selects the source for the
   *next* session; starting a second camera while one is running must fail cleanly
   (409 naming the running camera) rather than queue indefinitely. Switching cameras
   = stop, then start on the new key.

## Changes

### `app/config.py`

Replace the single `coral_city_url` setting with a registry. Keep it overridable by
environment so a deployment can retarget or trim the list, and keep `CORAL_CITY_URL`
working as an override for that one entry so existing `.env` files don't break.

Model each entry as `key`, `label`, `url`, and an optional `location`. Expose a
helper (e.g. `live_sources()` returning an ordered mapping, and `live_source(key)`
returning `None` for unknown keys). Add `live_default_source_key` defaulting to
`coral-city`. Validate at settings load: keys unique, non-empty, matching
`^[a-z0-9-]{1,64}$` so a key is safe in a URL path and fits `String(64)`.

### `app/api/live.py`

- `GET /live/sources` -> `{default_key, enabled, available, sources: [{key, label,
  location, url, active_session_id | null}]}`. Drives the dropdown in one request.
- `GET /live/latest?source=<key>` -> newest session **for that key** (unknown key
  404, omitted `source` uses the default). Replaces `GET /live/coral-city/latest`.
- `POST /live/start` with body `{"source": "<key>"}` -> replaces
  `POST /live/coral-city/start`. Sets both `source_url` and `source_key` on the new
  session. Keep the existing `IntegrityError` retry, which now correctly resolves a
  race per camera. Add the 409 for "another camera is already running".
- Add `source_key` and `source_label` to the `session_data()` payload so the UI can
  show which camera a session belongs to.
- Remove the old `coral-city` routes rather than aliasing them; nothing outside this
  repo consumes them (single commit, live feature unreleased).

### `app/static/live.js` and `app/templates/dashboard.html`

- Add a `<select id="live-source">` in `.live-controls`, labelled for screen readers,
  populated from `GET /live/sources`. Match the existing style: no framework, no
  build step, the `el("...")` id convention, DOM built with `createElement`, never
  `innerHTML`.
- Persist the selection in `localStorage` so a reload keeps the operator's camera.
- Disable the select while a session is open (`open.has(session.status)`) — changing
  it mid-session is meaningless and would misleadingly relabel the frames on screen.
  Show a hint that stopping is required to switch.
- When a session is running, force the select to that session's `source_key` so a
  reload or a second browser tab shows the truth.
- Change the heading `Live Coral City` and the `Ready to monitor Coral City Camera.`
  status string to use the selected camera's label. Poll
  `/live/latest?source=<key>` for the selected key; reset player, gallery and
  `imageVersion` when the key changes, reusing the existing `gallerySession` reset
  path.

### `app/services/live_source.py`

- Generalise the error strings that name `CORAL_CITY_URL` — they now mislead for a
  SmartBay camera. Take the camera label/key through to the message, e.g.
  "SmartBay Cam 2: no playable stream found; the camera may be offline."
- Add the idle-master-playlist case: a fetched `.m3u8` whose body contains no
  `EXT-X-STREAM-INF` and no segment lines should raise `LiveSourceError` with a
  "camera is online but not currently broadcasting" message, so the worker's
  existing retry/backoff handles it instead of FFmpeg failing opaquely.

### `.env.example` and `README.md`

Document the registry env var, the default key, and that `CORAL_CITY_URL` still
overrides the Coral City entry. In the README's live-monitoring section, note the
four cameras, that only one runs at a time, and the SmartBay attribution (Marine
Institute SmartBay Observatory, streams delivered via HEAnet).

## Tests (`tests/test_live_monitor.py`, plus `tests/test_live_source.py` if it fits better)

- `GET /live/sources` lists all four keys in configured order with the default flagged.
- `POST /live/start` with each key stores the matching `source_url` and `source_key`.
- Unknown/malformed key -> 404, and no session row is created.
- A URL in the `source` field is rejected (404, not a fetch) — guards constraint 1.
- Two different keys can each hold an open session at the DB level (the partial index
  permits it) while the API returns 409 for the second start.
- Starting the same key twice returns the existing session id (existing behaviour,
  keep that assertion).
- `resolve_stream` returns a `live.heanet.ie` playlist URL unchanged without invoking
  yt-dlp — assert the subprocess is not called.
- An empty master playlist raises `LiveSourceError`.
- Existing tests referencing `/live/coral-city/*` are updated, not deleted.

Do not add a test that hits the real cameras; stub `httpx` and `subprocess` the way
the existing live tests do.

## Verification

Run the suite with a short basetemp — plain `pytest` fails in this environment:

```
pytest tests -q --basetemp=C:\t\pt
```

Then report: the endpoints added and removed, whether a migration was needed, and any
place where you had to widen scope beyond the list above.
