# Implementation prompt: rebuild the dashboard on Base UI

Paste everything below the rule into a fresh Claude Code session in this repo.
Written 2026-09-10 against commit `1b6673f`.

---

Rebuild the operator dashboard as a React application built on **Base UI**
(`@base-ui/react`), replacing `app/templates/dashboard.html`,
`app/static/dashboard.js`, `app/static/product.js`, `app/static/live.js` and
`app/static/dashboard.css`.

Three things are wrong with the dashboard today, and all three must be fixed:

1. **Nothing is findable.** `GET /videos` returns every video ordered by
   `created_at DESC` and the client renders all of them. There is no search, no
   filter, no sort, no pagination. An operator with 300 uploads cannot answer
   "which videos still need review?" without opening each one.
2. **The review workflow is buried.** Track review is the point of this product
   — `is_accepted_track()` in `app/services/fish_counter.py` lets a human
   override the detector — but review lives in a `<select>` inside a table
   inside a panel that only appears after you click a video. There is no way to
   see outstanding review work across the library.
3. **It looks generated.** Gradient header, radial-gradient page wash, every
   block a white rounded card with a 50px shadow, an uppercase letterspaced
   eyebrow over every heading, a `◎` glyph as an icon. See the visual direction
   section — this is treated as a hard requirement, not a preference.

Everything that works today must still work. This is a re-skin plus a
findability layer, not a re-scope.

## Read before you start

Base UI is unstyled and composition-first; do not guess its API.

- <https://base-ui.com/llms.txt> — index of every docs page.
- Every docs page has a `.md` twin (`…/components/select.md`), which is the
  form to read.
- Quick start: <https://base-ui.com/react/overview/quick-start.md>. Two setup
  requirements that are easy to miss:
  - the app root needs `isolation: isolate` so portalled popups stack above the
    page regardless of local `z-index`;
  - `body { position: relative }` for iOS 26+ Safari backdrops.
- Handbook pages worth reading in full before writing components: **Styling**
  (data attributes and CSS variables exposed by each part), **Composition**
  (the `render` prop — use it instead of wrapper divs), **Forms** and **Field**
  (validation and labelling), **Animation** (`data-starting-style` /
  `data-ending-style`, which is how you animate exits without a motion library).
- Current line is `@base-ui/react` v1.8.x. The package was renamed from
  `@base-ui-components/react`; if you find an example using the old name it
  predates 1.0 and its API has probably moved.

## Stack (decided — implement it, do not re-open)

| Choice | Value | Why |
| --- | --- | --- |
| Build | Vite + React 19 + TypeScript, in `frontend/` | Static output, no Node at runtime |
| Components | `@base-ui/react` only | No second UI kit, no shadcn copy-paste |
| Styling | CSS Modules + one token file, `frontend/src/styles/tokens.css` | Tailwind's defaults are a large part of the generated-looking house style this task exists to remove |
| Server state | TanStack Query | Replaces the hand-rolled polling in `dashboard.js` |
| Routing | React Router v7, filter state in the URL query string | Deep links and back-button behaviour |
| Charts | Hand-rolled SVG, extending what `product.js` already does | Four small bar charts do not justify a charting dependency |
| Icons | One `frontend/src/icons.tsx` of inline 16px single-stroke SVGs | No icon font, no emoji |

No CDN `<script>` or `<link>` tags — the deployment assumes offline-capable,
self-hosted assets, and the existing code says so in a comment at the top of
`product.js`.

**README correction.** `README.md:4` and `README.md:208` claim "no Node.js
server or frontend build step is required". After this change there is still no
Node **server** — FastAPI serves the built bundle — but there is a build step.
Update both lines and the Local Development section rather than leaving the
README lying.

## Preserve every one of these

Work through this list at the end and tick it off explicitly in your summary.
If you drop one, say so and why; do not drop one silently.

| Capability | Endpoint(s) today |
| --- | --- |
| System/mode banner, worker + queue health, compose command | `GET /system/status` |
| Upload with optional camera ID | `POST /videos` (multipart, `camera_id`) |
| Video library list with status | `GET /videos` |
| Start processing; queued/processing/failed states with job error text | `POST /videos/{id}/process`, `GET /jobs/{id}` |
| Fish summary with pipeline / model / VIAME provenance | `GET /videos/{id}/summary` |
| Track table with review decisions | `GET /videos/{id}/track-summaries`, `PATCH /tracks/{id}/review` |
| Show-all-tracks toggle (low confidence + excluded) | `accepted_only=false` |
| Track inspector: source video, per-observation rows, seek to frame | `GET /videos/{id}/source-video`, `GET /tracks/{id}` |
| Cropped per-track clip in the inspector | `POST` / `GET /tracks/{id}/clip` |
| Annotated video generation + playback + download | `POST /videos/{id}/annotate`, `GET /videos/{id}/annotated-video` |
| Fish clips generation + grid | `POST` / `GET /videos/{id}/fish-clips` |
| Batch select, batch process, batch annotate, progress + per-video results | `POST /batch/{process,annotate}` |
| CSV exports, 4 kinds, per-video and batch | `GET /videos/{id}/exports/{kind}.csv`, `GET /exports/batch.csv` |
| Library analytics: accepted fish and detections per video | `GET /analytics/videos` |
| Per-video analytics: timeline, confidence histogram, duration histogram | `GET /analytics/videos/{id}` |
| Live: source picker, start/stop, status, lag, annotated MJPEG stream | `GET /live/sources`, `/live/latest`, `POST /live/start`, `POST /live/{id}/stop`, `GET /live/{id}/status`, `GET /live/{id}/annotated-stream` |
| Live: activity chart, finalized clip gallery, per-track clip and crop | `GET /live/{id}/activity`, `/live/{id}/clips`, `/live/tracks/{id}/clip`, `/live/tracks/{id}/crop` |
| Live: Fishial species controls, progress, session breakdown | `GET /live/{id}/species`, `species_*` fields on `POST /live/start` |
| "Stop monitoring to switch camera" / "to change species ID" guards | client-side rules in `live.js` |

Security invariants that survive the rewrite unchanged:

- the live API accepts a source **key**, never a URL (SSRF sink — see the note
  in `app/services/live_source.py`);
- resolved stream URLs are never returned or logged;
- `VideoRead.storage_path` is basename-only on purpose (`hide_storage_directory`);
- `JobRead.error_message` strips log paths (`hide_legacy_log_paths`).

## Backend work

The filters below cannot be done honestly client-side: the review state that
matters lives on `fish_tracks`, and the list endpoint returns none of it.
Fetching every video's tracks to filter in the browser is an N+1 against a
table that grows with every detection. So the API grows first. All changes are
**additive** — no migration, no removed field, no changed default response
shape.

### 1. `GET /videos` — aggregates, filters, sorting, paging

Add to the response model (call it `VideoListRead`, extending `VideoRead`):

```
track_count            int    all tracks, any state
accepted_track_count   int    is_accepted_track() against the run threshold
detection_count        int    accepted detections
unreviewed_count       int    review_state == "unreviewed"
flagged_count          int    review_state == "needs-review"
disputed_count         int    human decision disagrees with the threshold
review_status          str    "n/a" | "awaiting" | "in-progress" | "complete"
has_annotation         bool   annotated_at is not None
clip_count             int    generated track clips
species                list   distinct non-null species on accepted tracks
```

Compute these with grouped aggregate queries over `fish_tracks`, one round trip
for all listed videos — not a Python loop over `video.tracks` per row, which is
what `GET /analytics/videos` does today and is already the slowest endpoint in
the app.

`review_status` is derived per video:

- `n/a` — `processing_status != "completed"`;
- `awaiting` — no track has been touched (`reviewed_at is None` for all);
- `complete` — every track has `review_state != "unreviewed"` and none is
  `needs-review`;
- `in-progress` — anything else.

Query parameters:

| Param | Type | Notes |
| --- | --- | --- |
| `q` | str | case-insensitive substring of `original_filename` or `camera_id` |
| `status` | repeatable | `pending` / `queued` / `processing` / `completed` / `failed` |
| `camera_id` | repeatable | exact match; `__none__` selects videos with no camera |
| `review` | repeatable | `flagged`, `unreviewed`, `borderline`, `disputed`, `done` — see below |
| `annotated` | bool | |
| `has_clips` | bool | |
| `species` | repeatable | matches any accepted track's species |
| `created_after` / `created_before` | datetime | ISO 8601 |
| `min_fish` / `max_fish` | int | accepted track count |
| `min_duration` / `max_duration` | float | seconds |
| `band` | float | borderline half-width, default `0.10`, `ge=0, le=1` |
| `sort` | enum | `created_at`, `filename`, `camera_id`, `status`, `duration`, `size`, `accepted_fish`, `detections`, `unreviewed`, `flagged`, `review_status`, `annotated_at` |
| `order` | enum | `asc` or `desc`, default `desc` |
| `limit` / `offset` | int | default `limit=50`, `le=200` |

Keep returning a bare JSON array (`tests/test_api.py` asserts on that) and put
the unfiltered/filtered totals in `X-Total-Count` and `X-Filtered-Count`
headers. Unknown enum values are 422, not silently ignored.

### 2. Review vocabulary — define it once, server-side

Put this in `app/services/reporting.py` next to `track_accepted()` so exports,
the API and the UI agree. A track is:

- **flagged** — `review_state == "needs-review"`;
- **unreviewed** — `review_state == "unreviewed"`;
- **borderline** — unreviewed and `abs(max_confidence - threshold) <= band`,
  i.e. the detector was near its own cut-off and a human decision would change
  the count;
- **disputed** — `review_state == "accepted"` but `max_confidence < threshold`,
  or `review_state == "rejected"` but `max_confidence >= threshold`;
- **done** — `review_state` in `{reviewed, accepted, rejected}`.

`threshold` is the per-run snapshot, via the existing `threshold(video, job)`
helper — not `video.confidence_threshold` — so an old run keeps the threshold
it actually ran with.

*Assumption to confirm with the user before you commit to it:* `band = 0.10`.
It is the only invented number in this document. If they want a different
default, it is one constant.

### 3. `GET /videos/{id}/track-summaries` — filters and sorting

Add `review_state` (repeatable), `review` (the vocabulary above), `species`,
`min_confidence`, `max_confidence`, `min_detections`, `time_from`, `time_to`,
`sort` (`first_frame`, `duration`, `detection_count`, `mean_confidence`,
`max_confidence`, `species`, `review_state`), `order`, `band`.

Keep `accepted_only` working with its current default so nothing that calls it
today breaks; document it as superseded by `review`.

### 4. `GET /videos/facets`

Returns the distinct `camera_id` values, distinct species, and a count per
`status` and per `review_status`, so filter menus can show counts without the
client loading the whole library. One endpoint, one query per facet.

### 5. `POST /tracks/review` (bulk)

`{"track_ids": [...], "review_state": "..."}`, capped at 200 ids, returning the
updated summaries. The review queue needs "accept all visible" and undo; 200
sequential `PATCH`es is not an acceptable implementation of either.

### 6. Serving the SPA

- `app/main.py`: mount the build at `/static/app`, serve `index.html` for `/`,
  `/dashboard` and any non-API client route, keep `/health` and `/docs` intact,
  and return 404 (not `index.html`) for unmatched `/static/*`.
- `Dockerfile.api`: multi-stage — `node:22-alpine` builds `frontend/`, the
  Python stage copies `dist/` in. `docker compose` behaviour is otherwise
  unchanged.
- Dev loop: `npm run dev` in `frontend/` with a Vite proxy to
  `http://localhost:8000` for `/videos`, `/jobs`, `/tracks`, `/system`,
  `/batch`, `/exports`, `/analytics`, `/live`.
- Keep the old dashboard at `/legacy` while you build. Delete the five legacy
  files and the `/legacy` route in the same commit, once parity is signed off —
  not before, and not "later".

## Information architecture

The current dashboard is one 270-line scroll where live monitoring, upload,
library analytics, batch tools, the library and the detail panel all compete.
Split it:

```
/                     Overview — system health, queue, recent activity, review debt
/library              Video library (list + filters), detail in the right pane
/library/:videoId     Same, detail pane populated; deep-linkable
/review               Cross-video review queue
/live                 Live monitoring, full width
```

- `/library` is a two-pane layout: a resizable list rail (min 320px) and a
  detail pane. Selecting a video changes the route, so browser back moves
  between videos.
- The detail pane uses Base UI **Tabs**: Summary · Tracks · Analytics ·
  Annotated · Clips. Today all five stack vertically and each one re-renders
  the whole panel.
- `/review` is the new surface and the reason for the backend work: every
  flagged, borderline and disputed track across the library, one at a time,
  clip on the left, decision keys on the right, video context in a header.
- Live monitoring moves off the library page entirely. It is a different job,
  done at a different time, and putting a live MJPEG stream above an upload
  form is why the current page feels arbitrary.

## Filters and sorting — required UI

**Library filter bar** (one row, wraps; the whole state serialises to the URL):

- search field (debounced 250ms) over filename and camera;
- **Status** — multi-select, with counts from `/videos/facets`;
- **Review** — multi-select: Needs review (flagged), Unreviewed, Borderline,
  Disputed, Fully reviewed;
- **Camera** — multi-select, includes "No camera";
- **Species** — multi-select;
- **Date** — Any / Today / 7 days / 30 days / Custom range;
- **Has** — Annotated video, Fish clips, Detections > 0;
- **Fish count** — numeric range;
- **Sort** — the enum above, with an asc/desc toggle beside it.

Applied filters render as removable chips under the bar with a "Clear all" when
more than one is set. The result count line is factual: `48 of 312 videos`.
Saved views (name + current query string) persist in `localStorage`, seeded
with three built-ins: **Needs review**, **Failed runs**, **Ready to export**.

**Track table filters** (in the detail Tracks tab): review state, species,
confidence range slider, minimum detections, time-in-video range, plus sorting
on any column header. Keep the existing "show excluded tracks" toggle as a
one-click preset for `review=disputed,unreviewed`, and keep its explanatory
sentence.

**Live**: filter the finalized clip gallery by species and by whether a species
was identified at all; sort by recency, duration, or track confidence.

## UX requirements

- **The URL is the state.** Filters, sort, selected video, open tab, open
  inspector. A shared link reproduces the screen.
- **Selection survives filtering.** The bulk bar reads `12 selected` and, when
  relevant, `(3 hidden by filters)`. Selecting across pages is allowed;
  "select all" means all filtered results, and says so.
- **Bulk actions are a sticky bottom bar**, not four disabled buttons parked
  above the list. Process · Annotate · Export ▾ · Mark reviewed · Clear.
- **Review is optimistic with undo.** Apply immediately, show a Base UI Toast
  with an Undo action for 8 seconds, roll back and toast the error on failure.
- **Review queue keyboard map**, shown in a `?` dialog: `j`/`k` next/previous,
  `a` accept, `r` reject, `f` flag, `u` unreview, `Space` play/pause clip,
  `Enter` open source, `Esc` back to the library. Nothing is keyboard-only.
- **Polling is conditional.** Poll `/system/status` and any active job only
  while a job is `queued`/`processing`, back off to 15s after two minutes,
  pause when the tab is hidden, refetch on focus. `dashboard.js` currently
  polls on a flat interval forever.
- **Loading is a skeleton of the real layout**, never a centered spinner.
  Charts keep their axes while loading.
- **Long tables virtualize past 200 rows**, sticky header, sticky first column.
- **Empty states name the next action.** "No videos match these filters — clear
  the review filter to see the other 264." Not "No data available".
- **Errors are recoverable**: a route-level error boundary with a retry that
  refetches instead of reloading; per-panel errors stay inside the panel.
- **Accessibility is not optional.** Base UI gives correct roles and focus
  management for free, so the burden is on the parts you write: visible focus
  rings on every interactive element, `aria-live` on job status and live
  metrics (already right in the current markup — keep it), real `<label>`s,
  charts with a text summary alongside, colour never the sole carrier of
  status. Honour `prefers-reduced-motion`.
- **Light and dark**, driven by tokens and `color-scheme`, following the system
  by default with a manual override. Operators run this in dim rooms.
- **Mobile**: single column, filter bar collapses into a Base UI **Drawer**,
  the video list becomes cards, tables scroll horizontally inside a **Scroll
  Area**. The current layout only survives above ~1100px.

## Visual direction

The brief is "make it not look AI-generated while staying professional". That
is achievable but it is specific work, so here it is as rules.

### Remove these — every one is in the current CSS

- `linear-gradient(112deg, …)` header bar and the `radial-gradient` page wash.
- `--shadow: 0 18px 50px rgba(21,54,58,0.09)` applied to every block. Elevation
  belongs to things that actually float: menus, popovers, dialogs, toasts.
- The universal white rounded card. Most sections are regions of one page, not
  floating objects, and should be separated by a hairline rule and space.
- 16px border radius on everything. Pick 4px for controls, 8px for popovers,
  0 for table and page regions.
- The `.eyebrow` pattern — uppercase, 0.12em tracking, 800 weight — repeated
  above every heading. Use it at most once per screen, or delete it.
- Glyph icons (`◎`) and any emoji.
- Pills for things that are not statuses.
- Equal-width three-up card grids of summary numbers.
- Centered display copy over full-width sections.
- Vague microcopy. There is very little in this repo — the existing copy is
  unusually good ("Tracks are observations, not unique biological fish";
  "0 = off. Sends fish crops to Fishial AI; each frame uses an API call") —
  keep that voice and extend it. Never introduce "seamless", "powerful",
  "effortless", "at a glance", "insights", or an exclamation mark.
- Default indigo/violet/teal-gradient palettes, glassmorphism, blurred colour
  blobs, and full-width hero sections.

### Build this instead

- **Reference class**: an instrument console — Linear, Sentry, Datadog,
  Observable — not a SaaS marketing page. The screen's job is to show a lot of
  true numbers, densely, without shouting.
- **Colour**: ink on paper. One near-black text colour, one off-white surface,
  a hairline border token, and **one** accent used only for interactive
  affordance and selection. Status colours (queued / processing / completed /
  failed / flagged) are a separate reserved set, used only for status, never
  decoratively. The existing teal `--sea: #0b5960` is a defensible accent and
  is domain-appropriate — keep the hue, drop the gradients.
- **Type**: two families. A text face for prose and labels, and a monospace for
  every number, ID, timecode, frame, confidence and hash. Scale:
  12 / 13 / 15 / 18 / 24 / 32, with real line-height decisions per step, not
  1.5 throughout. `font-variant-numeric: tabular-nums` on every numeric cell so
  columns of confidences align.
- **Layout**: an 8px spacing scale used with rhythm — related things at 8,
  groups at 24, sections at 48 — not one gap everywhere. The chrome is
  asymmetric: a persistent left nav rail of 240–280px, content fluid to a
  1600px max, tables full-bleed within the content column. Charts are wide and
  short (about 3:1), not square cards in a grid.
- **Density**: table rows 34–36px, controls 32px high, 13px body in dense
  regions. An operator should see 20 tracks without scrolling.
- **Borders over shadows.** A 1px hairline and a background shift is enough
  separation for a region. Reserve shadow for portals.
- **Icons**: one 16px single-stroke set in `icons.tsx`, drawn to a consistent
  grid, used sparingly — status, actions, disclosure. No icon beside every
  label.
- **The brand mark**: keep the existing fish SVG path from `dashboard.html` —
  it is yours and it is specific. Render it flat, one colour, ~20px, in the nav
  rail. Drop the rounded translucent tile and the gold fill.
- **Motion**: 120–160ms, ease-out, on state changes only. No entrance
  animations on page load.

The test to apply to every screen: *would a marine scientist working through
400 tracks want this pixel?* Decoration that fails that test comes out.

## Base UI component map

| UI element | Component |
| --- | --- |
| Status / camera / species / review multi-selects | `Select` (multiple) or `Combobox` when the list can grow |
| Search with suggestions | `Autocomplete` |
| Sort control | `Select`, or `Menu` with a checked group |
| Filter chips | `Toggle` / `ToggleGroup` |
| Date range preset | `ToggleGroup` + `Popover` for custom |
| Confidence / fish-count range | `Slider` (range) |
| Detail tabs | `Tabs` |
| Row and bulk selection | `Checkbox`, `CheckboxGroup` |
| Track review decision | `ToggleGroup` — not a `<select>`; it is a 5-way single choice made hundreds of times |
| Row actions | `Menu`, `ContextMenu` on the row |
| Track inspector | `Dialog` |
| Destructive confirms (regenerate annotation, bulk reject) | `AlertDialog` |
| Undo / errors / batch results | `Toast` |
| Upload and live-config forms | `Form`, `Field`, `Fieldset`, `Input`, `NumberField` |
| Batch progress, job progress | `Progress` |
| Confidence bar in a table cell | `Meter` |
| Live species toggle | `Switch` |
| Column visibility, density | `Menu` with checkbox items |
| Table and gallery scroll containers | `ScrollArea` |
| Mobile filter panel | `Drawer` |
| Truncated filename, provenance hashes | `Tooltip` |
| Section separators | `Separator` |
| Bulk action bar | `Toolbar` |
| Advanced filters block | `Collapsible` |
| Clip hover preview | `PreviewCard` |

## Suggested file layout

```
frontend/
  index.html
  vite.config.ts
  src/
    main.tsx  app.tsx  routes.tsx
    api/            client.ts  videos.ts  tracks.ts  live.ts  system.ts  types.ts
    hooks/          use-filters.ts  use-selection.ts  use-poll.ts
    components/     base/ (styled Base UI wrappers: button, select, dialog, table…)
                    charts/  filters/  layout/
    features/
      overview/  library/  video-detail/  review-queue/  live/
    styles/         tokens.css  reset.css  typography.css
    icons.tsx
```

`components/base/` is the only place that imports from `@base-ui/react`
directly. Feature code imports the wrappers, so restyling is one directory.

## Verification

- Backend: extend `tests/test_api.py` with coverage for every new query
  parameter, the aggregate counts, the review vocabulary (especially
  *borderline* and *disputed* against a per-run threshold snapshot), the
  `X-Total-Count` headers, and bulk review including the 200-id cap.
- `tests/test_dashboard.py::test_dashboard_and_static_assets_are_served`
  asserts on Jinja strings ("Upload an underwater video", "Show low-confidence
  tracks"). Rewrite it to assert the SPA shell is served, that a client route
  falls back to `index.html`, and that a missing static asset still 404s. Do
  not delete the test.
- Run the suite as `python -m pytest --basetemp=<short path>`; a bare `pytest`
  fails in this environment on temp-directory paths.
- `ruff check .` clean.
- Frontend: Vitest + Testing Library over the filter-state reducer, URL
  serialisation round-trip, selection-across-filters, and the review queue
  keyboard map. Do not chase coverage on presentational components.
- Manual: mock mode via `docker compose up`, walk the preserved-capabilities
  table end to end, then Lighthouse/axe with zero critical accessibility
  violations.

## Phasing — five reviewable commits

1. `frontend/` scaffold, tokens, Base UI setup, FastAPI serving + SPA fallback,
   Dockerfile, README correction. Old dashboard still at `/legacy`.
2. Backend: aggregates, review vocabulary, filters, sorting, facets, bulk
   review, with tests. No UI change.
3. Library route: list, filter bar, saved views, bulk bar, detail tabs at
   parity with today's panel.
4. Review queue and Overview.
5. Live route, then delete the five legacy files and the `/legacy` route.

## Do not

- Do not change any existing endpoint's response shape or default parameters.
- Do not add a migration; every new field is computed.
- Do not introduce a second component library, a charting library, a CSS
  framework, or a state manager beyond TanStack Query.
- Do not move review, export, or acceptance logic into the client.
  `is_accepted_track()` is the single source of truth and exports must keep
  matching the screen.
- Do not accept a stream URL from the client, log a resolved stream URL, or
  return a server-side storage path.
- Do not "improve" the domain copy into marketing copy.
- Do not delete the legacy dashboard before parity is confirmed.

## Confirm before coding

1. Borderline band default of `0.10` — right number?
2. Is the `/review` cross-video queue in scope now, or a follow-up?
3. Library page size: 50 with paging, or infinite scroll?
4. Does dark mode ship in v1?
