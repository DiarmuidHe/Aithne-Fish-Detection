# Implementation prompt: optional Fishial AI species identification for live fish

Paste the section below into a fresh Claude Code session in this repo.

---

Add an **optional, budgeted** species-identification pass to live monitoring. When an
operator opts in before starting a session, the system picks the first N fish that
appear on the live feed and collect M genuinely clear frames each, sends those crops
to the **Fishial AI recognition API**, has the per-frame predictions vote on a
species, and shows the result. Fish that do not reach a confident consensus are
marked **review required**, never guessed. The Fishial API is expensive, so cost
control is a hard requirement, not a nice-to-have.

VIAME already gives us a bounding box and a generic `fish` class per detection. This
feature adds a real species name on top, for a small, operator-chosen number of fish
per session.

## Operator-facing behaviour

- In the live panel, before starting a session, the operator sets **"Identify species
  for the first ___ fish"** (`0` = off, default off) and optionally **"clear frames
  per fish"** (default from config, e.g. 5). These controls sit next to the camera
  selector, are disabled while a session is open (same rule as the camera `<select>`),
  and the fish count persists in `localStorage`.
- Once monitoring starts, the worker enrolls tracks in appearance order. A track is
  enrolled the first time it produces a *clear* frame (defined below), until N tracks
  have been enrolled.
- Each enrolled track collects up to M clear, well-separated crops. When it has M
  (or when it is lost with at least `fishial_min_frames_to_vote`), the crops go to
  Fishial one image at a time. Each frame's top prediction is a vote.
- Consensus rule (all must hold): the winning species has `>= fishial_min_votes`
  votes **and** `winning_votes / submitted_frames >= fishial_vote_ratio` **and** the
  mean Fishial score across the winning frames is `>= fishial_min_species_score`.
  Consensus -> store the species and a confidence (mean winning score). Otherwise ->
  `review_required`, species left as VIAME's generic value, with the full vote tally
  kept for a human.
- The dashboard shows, per finalized fish: the identified species (labelled
  "Fishial AI"), the confidence, or a "Review required" badge with the tally on
  hover; and while in progress, an "Identifying…" state. It also shows a
  **session species breakdown**: count and mean confidence per identified species,
  plus how many fish need review and how many Fishial image calls the session has
  used.

## What already exists — do not rebuild it

- `LiveTracker` (`app/services/live_monitor.py`) already runs per frame, assigns
  detections to `LiveFishTrack` rows, writes a letterboxed annotated crop per frame
  to the scratch spool (`scratch_path(...)/<session>/<track>/frames/<frame>.jpg`),
  keeps a `crop.jpg` "best" frame, and renders `clip.mp4` on expiry. Reuse the
  per-frame hook in `process_frame`; do not add a second decode loop.
- `run_session` (`app/workers/live_worker.py`) already has a clean "between
  segments" point after the `reader.read()` loop and a `finally:` shutdown path.
  Run the Fishial pass there, **off the frame hot path** — never call the API from
  inside `process_frame`.
- The live worker only runs with real VIAME (`VIAME_MOCK=false`) and one session at
  a time. Keep that; do not touch session claiming or the heartbeat.
- `app/services/live_source.py` shows the house pattern for an outbound `httpx`
  client and for testing it with `httpx.MockTransport`. Follow it.
- `app/api/live.py` already has `session_data()` and `track_data()` payload builders,
  a `GET /live/sources` bootstrap endpoint, and `POST /live/start` taking a JSON
  body. Extend these; do not add a parallel API surface.
- `app/static/live.js` already disables the camera `<select>` while a session is
  open, persists a choice to `localStorage`, rebuilds option lists only when their
  signature changes, and builds DOM with `createElement` (never `innerHTML`, no
  framework, no build step). Match all of that.

## Design constraints

1. **Opt-in twice.** The feature does nothing unless (a) the deployment sets
   `FISHIAL_ENABLED=true` with valid credentials, **and** (b) the operator sets a
   non-zero fish count for that session. Default off. With the feature unconfigured,
   the UI control is hidden/disabled and `POST /live/start` rejects a non-zero count
   with 409.
2. **Hard cost ceiling per session.** Total Fishial *image* calls for a session must
   never exceed `fish_target * frames_per_fish + fishial_max_api_retries`. Check the
   budget immediately before every call and stop when it is reached. Enrolling a
   track that later vanishes may waste a slot, but must not let call volume exceed
   this ceiling. Persist the running count on the session so it survives a worker
   restart and is visible in the API.
3. **Send crops, nothing else.** Only tight crops of already-detected fish leave the
   deployment. Never send full frames, never send anything derived from a
   client-supplied value, and never put image bytes or the API secret in logs. The
   secret comes from the environment only; redact it in any `repr`/error.
4. **Accuracy over coverage.** A wrong species label is worse than "review
   required". Gate frame quality hard, require genuine agreement, and abstain
   otherwise. Details in "Accuracy strategy" below.
5. **A Fishial failure costs one frame, not the session.** Timeouts, HTTP errors,
   auth failures, and unparseable responses are caught: that frame abstains, the
   worker continues. If every frame for a fish fails, mark it `error` (treated as
   review required in the UI).
6. **Live scope only.** Write `FishialClient` source-agnostic so the uploaded-video
   pipeline (`FishTrack.species`) could reuse it later, but do not wire that up now.

## Fishial AI API integration

Fishial's recognition API is a multi-step, authenticated flow (roughly: obtain an
OAuth2 access token from client credentials; request a signed upload target; upload
the image; call the recognition endpoint; read back a ranked list of species with
scores). **Verify the current contract against Fishial's own API documentation before
implementing** — endpoint paths, auth shape, and response JSON have changed over
time. Isolate all request/response shaping in one module so it is cheap to adjust.

Build `app/services/fishial.py`:

- `FishialClient(settings, client=None)` — accepts an injected `httpx.Client` for
  tests, like `resolve_stream`.
- Caches the access token in memory with its expiry; refreshes on expiry and once on
  a 401.
- `identify(image: bytes) -> FishialPrediction` where `FishialPrediction` carries an
  ordered list of `(species_name, score)` plus the raw response for storage. Runs
  the full upload+recognize sequence.
- Raises `FishialError` (with a redacted message) on any failure.
- Honors `fishial_request_timeout_seconds`; makes calls sequentially.
- No secret, token, or image bytes in log output or exception text.

Tests must use `httpx.MockTransport`; never hit the real API or the internet.

## Changes

### `app/config.py`

Add, with validation in the existing `@model_validator`:

- `fishial_enabled: bool = False`
- `fishial_api_base_url: str = "https://api.fishial.ai/v1"` (or whatever current docs
  specify; keep overridable)
- `fishial_client_id: str | None = None`
- `fishial_client_secret: str | None = None`
- `fishial_request_timeout_seconds: float = Field(default=30.0, gt=0)`
- `fishial_default_frames_per_fish: int = Field(default=5, ge=1, le=20)`
- `fishial_max_fish_per_session: int = Field(default=20, ge=1, le=200)` — ceiling on
  the operator's N
- `fishial_min_frame_confidence: float = Field(default=0.70, ge=0, le=1)` — VIAME
  detection confidence required for a frame to be "clear"
- `fishial_min_crop_pixels: int = Field(default=96, ge=16)` — min bbox short side
- `fishial_edge_margin_pixels: int = Field(default=4, ge=0)` — bbox must sit at least
  this far inside every frame edge (clipped fish are rejected)
- `fishial_crop_margin: float = Field(default=1.15, ge=1.0)` — bbox expansion before
  cropping (clean crop, no annotation, no letterbox)
- `fishial_blur_min_variance: float = Field(default=0.0, ge=0)` — variance-of-Laplacian
  floor; `0` disables the blur gate. Pick a sane non-zero default if you can validate
  one against the sample media, else leave it `0` and document it.
- `fishial_min_frame_separation_seconds: float = Field(default=0.6, ge=0)` — reject a
  candidate crop too close in time to one already kept, so the M frames are
  independent views
- `fishial_min_frames_to_vote: int = Field(default=3, ge=1)`
- `fishial_min_votes: int = Field(default=3, ge=1)`
- `fishial_vote_ratio: float = Field(default=0.6, ge=0, le=1)`
- `fishial_min_species_score: float = Field(default=0.5, ge=0, le=1)`
- `fishial_min_frame_margin: float = Field(default=0.0, ge=0)` — if `> 0`, a frame's
  vote only counts when its top score exceeds its second score by this margin;
  ambiguous frames abstain
- `fishial_max_api_retries: int = Field(default=2, ge=0)` — extra calls allowed above
  the ceiling for transient retries, across the whole session

Validation: if `fishial_enabled` is true, `fishial_client_id` and
`fishial_client_secret` must be non-empty (raise `ValueError` otherwise);
`fishial_min_votes <= fishial_default_frames_per_fish`;
`fishial_min_frames_to_vote <= fishial_default_frames_per_fish`.

### `app/db/migrations/versions/0006_live_species_id.py`

New migration, `down_revision = "0005_live_monitor"`. Add columns; provide a real
`downgrade()`. Extend `tests/test_live_migration.py` to run `0006` upgrade/downgrade
and assert the model/column parity check the existing test already does for `0005`.

`live_monitor_sessions` — add:

- `species_id_enabled BOOLEAN NOT NULL DEFAULT false`
- `species_id_fish_target INTEGER NOT NULL DEFAULT 0`
- `species_id_frames_per_fish INTEGER NOT NULL DEFAULT 5`
- `species_id_fish_enrolled INTEGER NOT NULL DEFAULT 0`
- `species_id_api_calls INTEGER NOT NULL DEFAULT 0`

`live_fish_tracks` — add:

- `fishial_state VARCHAR(16) NOT NULL DEFAULT 'disabled'`
  (`disabled | pending | ready | submitted | identified | review_required | error`)
- `fishial_species VARCHAR(256)` (nullable)
- `fishial_species_confidence FLOAT` (nullable)
- `fishial_frames_used INTEGER NOT NULL DEFAULT 0`
- `fishial_votes_json TEXT` (nullable) — tally `{species: votes}` plus per-frame
  `[{species, score, margin, frame_number}]` and a `reason` when review is required
- `fishial_completed_at DATETIME` (nullable)

Add a `CheckConstraint` for `fishial_state` values, matching the style of
`ck_fish_tracks_review_state`.

Prefer the JSON column over a new predictions table; add a `live_fishial_predictions`
table only if you find the JSON blob genuinely can't drive the UI — say why first.

### `app/db/models.py`

Mirror the migration on `LiveMonitorSession` and `LiveFishTrack`. Add a
`fishial_votes` `@property` that parses `fishial_votes_json` defensively (return `{}`
on error), like `ProcessingJob.configuration`.

### `app/services/fishial.py` (new)

`FishialClient` + `FishialPrediction` + `FishialError`, per "Fishial AI API
integration" above.

### `app/services/live_species.py` (new)

`SpeciesIdentifier(db, session, settings, client=None)`:

- `should_run(session)` — feature enabled for the session and any track is
  `pending`/`ready`.
- `run()` — called between segments and once at shutdown. For each `ready` track,
  and each `pending` track that is now `finalized` with
  `>= fishial_min_frames_to_vote` staged crops:
  - load staged clean crops, call `client.identify` per crop while
    `session.species_id_api_calls < budget`, incrementing `species_id_api_calls`
    after each call (including failures, except pure transient retries within
    `fishial_max_api_retries`);
  - tally votes (respecting `fishial_min_frame_margin`), apply the consensus rule,
    set `fishial_state` + `fishial_species` + `fishial_species_confidence` +
    `fishial_votes_json` + `fishial_completed_at`;
  - a `pending` track that finalized with too few clear frames ->
    `review_required`, `reason: "insufficient clear frames"`, **no API calls**;
  - delete that track's staged crops afterwards (success or failure).
- All DB writes committed here; a crash mid-pass must not double-spend budget
  (re-derive from `fishial_frames_used` / `fishial_state`, don't trust in-memory
  counters alone).

Keep staged crops under the scratch root (they are large and short-lived), e.g.
`scratch_path(settings, session_id, track_id, "fishial")`. The `finally:` in
`run_session` already wipes `scratch_path(settings, session_id)` on shutdown — make
sure a normal shutdown still runs `SpeciesIdentifier.run()` *before* that wipe.

### `app/services/live_monitor.py`

In `LiveTracker`:

- Constructor: when `session.species_id_enabled`, load the enrollment counter and a
  per-track set of "kept crop timestamps" for separation checks.
- New `_is_clear_frame(frame, box, confidence) -> bool`: confidence
  `>= fishial_min_frame_confidence`; short side `>= fishial_min_crop_pixels`; bbox at
  least `fishial_edge_margin_pixels` inside every edge; blur variance
  `>= fishial_blur_min_variance` when that is non-zero.
- In `process_frame`, for each assigned track when the feature is on:
  - if the track is not enrolled, the frame is clear, and
    `session.species_id_fish_enrolled < species_id_fish_target`: set
    `fishial_state = "pending"`, `session.species_id_fish_enrolled += 1`.
  - if `fishial_state == "pending"` and `fishial_frames_used < frames_per_fish` and
    the frame is clear and it is `>= fishial_min_frame_separation_seconds` after the
    last kept crop for this track: write a **clean** crop (expand box by
    `fishial_crop_margin`, clamp to frame, no annotation, no letterbox, high-quality
    JPEG) to the track's Fishial staging dir; `fishial_frames_used += 1`.
  - when `fishial_frames_used == frames_per_fish`: set `fishial_state = "ready"`.
- Do not call Fishial here. Do not block. Losing a staged crop to `OSError` is a
  warning, like the existing crop path.

### `app/workers/live_worker.py`

- In `run_session`, after the per-segment `reader.read()` loop and `db.commit()`,
  call `SpeciesIdentifier(db, session, settings).run()` when `should_run` — inside
  the same broad `try` so a failure is logged and downgraded, never fatal.
- In the `finally:` shutdown block, before the `scratch_path(...)` wipe, run one
  final `SpeciesIdentifier.run()` so `ready` and freshly-finalized enrolled tracks
  get a result (subject to remaining budget).
- Log the species-id config once at session start (`fish_target`,
  `frames_per_fish`) — never the credentials.

### `app/api/live.py`

- `GET /live/sources` -> add `"fishial": {"enabled": settings.fishial_enabled,
  "max_fish_per_session": settings.fishial_max_fish_per_session,
  "default_frames_per_fish": settings.fishial_default_frames_per_fish}`.
- `StartRequest` -> add `species_id_fish_target: int | None = None` and
  `species_id_frames_per_fish: int | None = None`. In `start()`: clamp/validate
  (`0 <= target <= fishial_max_fish_per_session`; frames in `[1, 20]`); if
  `target and target > 0` require `settings.fishial_enabled` else
  `raise HTTPException(409, "Species identification is not configured on this deployment")`;
  store `species_id_enabled`, `species_id_fish_target`, `species_id_frames_per_fish`
  on the new `LiveMonitorSession`. Keep the existing `IntegrityError` retry.
- `session_data()` -> add `species_id`: `{enabled, fish_target, frames_per_fish,
  fish_enrolled, fish_identified, fish_review_required, api_calls}`. Derive the
  identified/review counts with a grouped count query on `LiveFishTrack.fishial_state`
  for the session (like the `activity` endpoint's status counts).
- `track_data()` -> add `fishial_state`, `fishial_species`,
  `fishial_species_confidence`, `fishial_frames_used`, and `fishial_votes` (the
  parsed property).
- New `GET /live/{session_id}/species` -> `{session_id, as_of, api_calls,
  review_required, species: [{species, count, mean_confidence}]}` aggregating
  `LiveFishTrack` where `fishial_state == "identified"`, ordered by count desc.

### `app/static/live.js`, `app/templates/dashboard.html`, `app/static/dashboard.css`

- `dashboard.html`: in `.live-controls`, add a labelled number input
  `#live-species-count` (min 0, max from `/live/sources`) and, behind a small
  "options" affordance or always-visible, `#live-species-frames`. Add a hidden hint
  `#live-species-hint` ("Stop monitoring to change species identification.").
- `live.js`:
  - store the fishial block from `/live/sources`; hide/disable the species controls
    when `!fishial.enabled`; set `max` on the count input; default the frames input
    to `default_frames_per_fish`; persist the count under a new `localStorage` key.
  - disable both species inputs while a session is open, same as `#live-source`.
  - send `species_id_fish_target` / `species_id_frames_per_fish` in the
    `POST /live/start` body.
  - in `showStatus`, render a species-id progress line when
    `session.species_id.enabled` (`enrolled/target enrolled · X identified · Y need
    review · Z Fishial calls`).
  - in `showGallery`, when `track.fishial_state === "identified"` show the species
    name + `(NN% · Fishial AI)`; when `"review_required"` or `"error"` show a
    "Review required" badge with `title` set to the tally from `track.fishial_votes`;
    when `"pending"`/`"ready"`/`"submitted"` show "Identifying species…". Fall back to
    the existing `track.species || "Fish"` title otherwise.
  - add a small **species breakdown** list under the gallery fed by
    `GET /live/{session_id}/species` (fetch it in `refresh()` alongside
    `activity`/`clips` when a session exists).
- `dashboard.css`: styles for the badge and the breakdown list, matching existing
  `.chart-note` / `.count-pill` / `.clip-card` conventions. No new colors if an
  existing token fits.

### `.env.example` and `README.md`

- `.env.example`: add the `FISHIAL_*` vars, all commented except
  `FISHIAL_ENABLED=false`, with a one-line note that credentials come from Fishial
  and that each identified fish costs `frames_per_fish` API calls.
- `README.md`: in the live-monitoring section, document the opt-in flow, the
  per-session cost ceiling, the voting/consensus behaviour, the "review required"
  outcome, and that fish crops are sent to a third-party API (Fishial AI) only when
  enabled.

## Accuracy strategy (make these choices explicit in code + comments)

- **Frame quality gate** before a crop is ever staged: detector confidence floor,
  minimum pixels, not touching frame edges, optional blur floor.
- **Independent views**: enforce `fishial_min_frame_separation_seconds` between kept
  crops so M frames are spread across the track, not M near-duplicates.
- **Clean crops**: expanded bbox, no annotation overlay, no letterbox, high JPEG
  quality — Fishial should see the fish, not our HUD.
- **Per-frame abstention**: with `fishial_min_frame_margin > 0`, a frame whose top-1
  and top-2 scores are close does not vote.
- **Consensus, not first answer**: require `fishial_min_votes`, `fishial_vote_ratio`,
  and `fishial_min_species_score` together; otherwise `review_required`.
- **Never fabricate**: on too few clear frames or all-failed calls, `review_required`
  / `error` — do not fall back to a single frame's guess.
- **Auditability**: persist the full tally and per-frame scores in
  `fishial_votes_json` so a human reviewer sees exactly why.
- Keep every threshold a config value with a conservative default; do not hard-code.

## Tests (`tests/test_fishial.py` new; extend `tests/test_live_monitor.py`,
`tests/test_live_migration.py`, `tests/test_api.py` / `tests/test_dashboard.py` as
they fit)

- **Config**: `fishial_enabled=true` without id/secret -> `ValidationError`;
  bounds on `fishial_min_votes` vs frames; param clamping in `start()`.
- **FishialClient** (all via `httpx.MockTransport`): token fetched once and reused;
  refreshed after expiry; retried once on 401; full upload+recognize happy path
  parses the ranked list; HTTP error / bad JSON -> `FishialError`; `repr(client)`
  and `FishialError` text contain neither the secret nor image bytes.
- **Enrollment order & count**: with `fish_target=2`, only the first two tracks to
  produce a clear frame become `pending`; a third stays `disabled`.
- **`_is_clear_frame`**: rejects low confidence, sub-`fishial_min_crop_pixels`,
  edge-clipped, and (when enabled) blurry frames; accepts a good one.
- **Frame separation**: two clear detections `< fishial_min_frame_separation_seconds`
  apart stage only one crop.
- **Budget ceiling**: drive N tracks past M frames with a stubbed client that counts
  calls; assert total `identify` calls `<= N*M + fishial_max_api_retries` and that
  `session.species_id_api_calls` matches; once the ceiling is hit, no further calls
  and remaining `ready` tracks end `review_required`/`error`.
- **Voting**: 5 frames, 4×`"Lutjanus griseus"` + 1×other -> `identified`,
  `fishial_species == "Lutjanus griseus"`, confidence == mean of the 4 winning
  scores. 2/2/1 split -> `review_required` with a tally in `fishial_votes_json`.
  Low winning score -> `review_required`. `< fishial_min_frames_to_vote` clear
  frames at finalization -> `review_required`, reason recorded, **zero** client
  calls.
- **Failure isolation**: client raises for 1 of 5 frames -> still `identified` from
  the other 4; client raises for all -> `fishial_state == "error"`; the session
  keeps running (assert `run_session` completes and the session ends `stopped`).
- **Feature off**: `species_id_fish_target=0` (or `fishial_enabled=false`) -> no
  client construction, every track stays `fishial_state == "disabled"`.
- **API payloads**: `/live/sources` carries the `fishial` block; `/live/start`
  stores the params and rejects a non-zero target when disabled (409);
  `session_data` carries `species_id` counts; `track_data` carries the fishial
  fields; `GET /live/{id}/species` aggregates counts and mean confidence and lists
  the review-required count.
- **Migration**: `0006` upgrade adds exactly the listed columns with the right
  nullability/defaults; downgrade removes them and leaves `0005` intact; re-upgrade
  to head works. (Extend the existing parity loop.)

Do not call the real Fishial API, the internet, or the real cameras in any test.
Stub `httpx`/`subprocess` the way the existing live tests do. Use a fake/stub
`FishialClient` (injected) for the worker/tracker tests and reserve
`httpx.MockTransport` for the `FishialClient`'s own tests.

## Verification

Plain `pytest` fails in this environment; pass a short basetemp:

```
pytest tests -q --basetemp=C:\t\pt
```

Run any linter/formatter the repo already configures (check `pyproject.toml`).

Then report:

- the columns and endpoints added, and whether the predictions table was needed;
- exactly how the per-session call ceiling is enforced and where it is checked;
- the enrollment rule you implemented for "first N fish" and how a vanished
  enrolled fish is handled;
- every default threshold and one line on why it is safe;
- anywhere the Fishial API contract differed from the sketch above and what you
  verified it against;
- any place you had to widen scope beyond this list.
