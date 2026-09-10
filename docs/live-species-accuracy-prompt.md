# Implementation prompt: raise Fishial identification yield at lower API cost

Paste the section below into a fresh Claude Code session in this repo.

---

The Fishial species-identification pass built from
`docs/live-species-identification-prompt.md` works end to end and has identified
nothing on any camera. On clear tropical footage it did not even reach the API. Fix
**which frames we keep**, **which fish we pay for**, **when we stop paying**, and
**how we read the answer**, so that identification yield goes up while per-session
API calls go *down*. Accuracy still beats coverage: a wrong species label remains
worse than "review required".

Do not rebuild the feature. Every file below already exists.

## Evidence — read this before designing anything

Two reference sessions, both `species_id_fish_target=5`, `frames_per_fish=5`, both
completed cleanly with no errors and no stuck rows, and both identified **zero** fish:

- `ab69aaca-7ddc-46db-b3af-70b11bf6ba30` — SmartBay Cam 3 (Galway Bay), 4m47s, 360
  frames, 78 tracks, 11 of 27 allowed API calls spent.
- `3c04e487-6077-4549-9f19-ba96d38d1d02` — Coral City (Miami), 3m17s, 240 frames, 132
  tracks with 3+ detections, **0 API calls spent**.

The Miami session is the important one. Clear tropical water, Fishial's own training
domain, five fish enrolled — and the pipeline never contacted the API at all. All
five enrolled tracks staged 1 or 2 crops against `fishial_min_frames_to_vote=3` and
ended "insufficient clear frames".

What the stored audits and detection rows show, and what each fact implies:

1. **The frame-quality gate is calibrated above the data it filters.** Measured over
   the detection rows of both sessions:

   | | conf p50 | conf p90 | short side p50 | short side p90 | pass both gates |
   |---|---|---|---|---|---|
   | Coral City | 0.619 | 0.694 | 57 px | 95 px | **3.2%** |
   | SmartBay 3 | 0.592 | 0.719 | 88 px | 223 px | 14.1% |

   On Coral City, `fishial_min_frame_confidence=0.70` sits *above* the 90th
   percentile of detector confidence, and `fishial_min_crop_pixels=96` sits *above*
   the 90th percentile of bbox short side. Both thresholds independently reject more
   than 90% of detections, and `_is_clear_frame` requires them **conjunctively**, so
   3.2% of frames survive. *These defaults were never calibrated against what VIAME
   actually emits on these streams.*
2. **That gate leaves almost no track identifiable, and first-come enrollment then
   picks at random from the wrong pool.** Of Coral City's 132 tracks with 3+
   detections, only **7** can produce 3 gate-passing frames. Enrollment commits a
   slot on a track's first clear frame and never frees it, so it drew 5 tracks
   effectively at random from 132 and hit none of the lucky 7. On SmartBay the same
   interaction cost 2 of 5 slots. *A 5%-pass gate multiplied by arrival-order
   selection is why both sessions returned nothing.*
3. **Ranking beats thresholding, for free.** Taking the best 3 frames of each track
   by measured quality — instead of the first 3 that clear an absolute bar — makes
   **132 of 132** Coral City tracks identifiable, versus 7 today. No extra API calls;
   the frames are already decoded and staged locally.
4. **Enrollment is first-come, so even within the passable pool the budget goes to
   mediocre fish.** SmartBay's five enrolled tracks had 26–77 detections while later
   tracks had 151, 113 and 110. *Arrival time is uncorrelated with identifiability.*
5. **9 of 11 SmartBay API calls returned a segmented fish with an empty species list**
   (`objects[0]` carried a `bbox` and a full `shape` outline, `species: []`,
   `definitions: {}`). The detector saw the fish; the classifier declined to name it.
   Consecutive frames from the same track all declined. *Once a track starts coming
   back empty, the remaining frames are near-certain to come back empty too — and we
   paid for every one of them.*
6. **2 of 11 calls did return a candidate, and we threw both away.** Those responses
   carried two objects, so the `len(objects) != 1` guard in
   `FishialClient._parse` abstained rather than risk a neighbour's identity. *We paid
   for an answer, then discarded it — even though we know exactly where our fish sits
   in the crop, because we centred it there.*
7. **Both discarded candidates were geographically impossible.** `Pomatomus
   saltatrix` (bluefish) at 0.46 and `Sparisoma aurofrenatum` (redband parrotfish, a
   Caribbean species) at 0.48 — on a Galway Bay camera. Both also sat below
   `fishial_min_species_score=0.5`. *A regional prior would have rejected these for
   free, and would reject the same class of error in future when a score does clear
   the threshold.*
8. **Every raw response is already persisted** in `fishial_votes_json`, averaging
   12 KB per track and peaking at 24 KB, overwhelmingly `shape` polygon arrays. *We
   can re-run scoring logic against past sessions at zero API cost — and we are
   storing a lot of bytes we never read.*

**Root cause, in order.** The primary failure is ours, not the model's: an
uncalibrated conjunctive frame gate starves every track of frames, and arrival-order
enrollment then spends the budget on tracks drawn from the wrong pool. Fix that first
— it is the only change that turns 7 identifiable tracks into 132, and it costs
nothing.

A secondary failure is real but **not yet measured**: on SmartBay, 9 of 11 calls came
back with an empty species list, consistent with domain mismatch (Fishial is trained
largely on bright, high-contrast angler photography; the ANERIS feed is dim, green
and low-contrast). Do not treat this as settled. The Coral City session was run
specifically to test it and never made a call, so **we still have no evidence of how
Fishial performs on clear-water footage from this pipeline.** Once the gate is fixed,
re-run Coral City before drawing any conclusion about the model, and before spending
effort on preprocessing (step 7).

## What already exists — do not rebuild it

- `app/services/fishial.py` — `FishialClient` (token cache, 401 refresh, retry on
  429/5xx, `before_image_call` reservation hook), `FishialPrediction`,
  `FishialError` with its fixed reason codes. Keep the redaction guarantees exactly
  as they are: no secret, token, URL or image byte ever reaches a log or exception.
- `app/services/live_species.py` — `SpeciesIdentifier` with its durable reservation
  journal (`_reserve` commits the budget increment *before* the request, so a crash
  can waste a call but never buy an extra one), per-frame journalling that refuses to
  replay an uncertain send, `run_safely()`, and `finish()`. **Preserve every one of
  these properties.** They are the reason the last run had no stuck rows.
- `app/services/live_monitor.py` — `LiveTracker._is_clear_frame` and
  `_stage_species_crop`, which write tight, original-resolution, un-annotated JPEGs
  to `scratch_path(settings, session, track, "fishial")`. Staging is local and free.
- `app/workers/live_worker.py` — `run_session` already calls the identifier between
  segments and again in `finally:` before the scratch wipe. Do not add a second
  decode loop or a second wipe.
- `app/api/live.py`, `app/static/live.js` — existing payload builders and DOM
  conventions (`createElement` only, no framework, no build step).

## The core idea

**Everything before the API call is free. Only the call costs money.** Today the
pipeline throws away most of its free information — 97% of frames on Coral City — by
gating on absolute thresholds, and then commits paid budget by arrival order. Invert
both. Keep and rank generously across many frames and many tracks, because that costs
nothing, then spend the budget on the best of what you have, and stop spending the
moment an answer is decided or doomed.

Work the changes in numbered order. Step 1 is a precondition for judging any of the
others: until tracks can actually reach `fishial_min_frames_to_vote`, no measurement
of selection, stopping, or preprocessing means anything.

## Design constraints

1. **The existing hard ceiling stands.** Total image calls per session must still
   never exceed `fish_target * frames_per_fish + fishial_max_api_retries`, still
   enforced by the pre-request `UPDATE ... WHERE species_id_api_calls < budget`
   reservation. Every change below may only spend *less*, never more.
2. **No new call sites.** All identification stays in `SpeciesIdentifier`, off the
   frame hot path, between segments or at shutdown.
3. **Abstention stays the default.** Every new inference path must have an explicit
   abstain branch. When a rule cannot be applied confidently, fall through to
   `review_required` with a recorded reason.
4. **Every threshold is config, with a conservative default.** Anything that could
   loosen accuracy ships **off** (neutral value) until the replay harness in step 6
   demonstrates it helps on stored data.
5. **Staging must stay bounded.** Generous staging is free in API terms but not in
   disk terms. Cap the candidate pool and the per-session staged bytes, and evict
   the weakest candidates first.
6. **No new third-party egress.** The regional filter is local data. No new outbound
   host beyond Fishial.

## Changes

### 1. `app/services/live_monitor.py` — rank frames instead of gating them

**This is the change that matters most. Do it first, and verify it before moving on.**

Today `_is_clear_frame` is a conjunctive pass/fail gate whose two main thresholds
both sit near or above the 90th percentile of the data they filter, so 3–14% of
frames survive and most tracks never reach `fishial_min_frames_to_vote`. Replace
absolute gating with relative ranking.

- Demote `_is_clear_frame` to a **safety floor only**: reject frames that are
  genuinely unusable — non-finite boxes, short side below `fishial_min_crop_pixels`,
  bbox breaking the `fishial_edge_margin_pixels` inset, confidence below
  `fishial_min_frame_confidence`. Recalibrate those two defaults *down* so the floor
  excludes junk rather than the bulk of the distribution: set
  `fishial_min_frame_confidence` to `0.50` and `fishial_min_crop_pixels` to `48`.
  Both are now floors, not selection criteria — selection is the ranking below.
- Score every surviving frame with `_frame_quality` (defined below) and keep a
  **bounded best-of ranking per track** rather than the first N that pass. Retain the
  top `fishial_max_staged_frames_per_candidate` frames by score; when a better frame
  arrives and the buffer is full, evict the current worst — delete its JPEG, drop its
  `staged` entry, and stage the new one. `fishial_frames_used` becomes "frames
  currently retained", not "frames ever seen".
- Keep `fishial_min_frame_separation_seconds` as a **diversity** rule, not a
  first-come rule: a new frame may displace a retained frame within the same
  separation window if it scores higher, but the retained set must always span at
  least `fishial_min_frames_to_vote` distinct windows so the votes stay independent.
  State in a comment why near-duplicate frames must not be allowed to fill the set.

Report the measured pass rate of the recalibrated floor on both reference sessions.
If a floor still rejects more than ~25% of detections on either camera, it is still
doing selection's job — say so and lower it further rather than leaving it.

Then replace slot-committing enrollment with an unpaid candidate pool.

- `_stage_species_crop` no longer touches `species_id_fish_enrolled` and no longer
  compares against `species_id_fish_target`. A track that passes `_is_clear_frame`
  becomes `fishial_state = "candidate"` and stages crops.
- Admit candidates while `len(pool) < fishial_candidate_pool_size` (new config,
  default `max(3 * fish_target, fish_target + 8)` computed at session start and
  stored on the session, so it survives a worker restart).
- Stage up to `fishial_max_staged_frames_per_candidate` (default
  `frames_per_fish + 2`) so the selector has spare frames to choose from — the extra
  frames cost nothing and give the early-stop logic room to skip weak ones.
- When the pool is full, admit a stronger track by evicting the weakest candidate:
  delete its staged directory, set it back to `fishial_state = "disabled"`, and
  record `reason: "evicted from candidate pool"`. Never evict a track that is already
  `ready`/`submitted`/terminal — those have either been selected or paid for.
- Enforce `fishial_max_staged_bytes` (default 512 MB) across the session's fishial
  scratch; on breach, evict weakest-first before staging more.

Add `_frame_quality(frame, box, confidence) -> dict` returning the per-frame
measurements the selector needs, computed once while the frame is in hand:
`confidence`, `short_side` (px), `sharpness` (variance of Laplacian), `luminance`
(mean L), `contrast` (RMS of L), `colorfulness` (Hasler–Süsstrunk metric). Persist
each staged frame's dict into the existing `staged` audit entry alongside
`frame_number`/`timestamp`. These are cheap, already-decoded-pixel operations — keep
them out of any per-detection path that runs for non-candidates.

`_frame_quality` now serves two purposes — ranking frames within a track, and feeding
`track_quality` in step 2 — so compute it once at staging time and persist it. Keep it
out of any path that runs for non-candidate detections.

### 2. `app/services/live_species.py` — select the best candidates, spend late

Add a selection step that runs before identification in each pass.

`_select(candidates) -> list[LiveFishTrack]`:

- Consider only tracks that are `candidate`, `status == "finalized"`, and hold
  `>= fishial_min_frames_to_vote` staged frames. An unfinalized track may still
  improve, so never pay for it mid-life.
- Score each with `track_quality(track, settings) -> float` in a new
  `app/services/species_quality.py`: a weighted blend of the median staged
  `short_side`, median `sharpness`, median `confidence`, median `contrast`, median
  `colorfulness`, and the count of staged frames. Normalise each term against a
  configured reference value so the blend is scale-free, and keep the weights in
  config (`fishial_quality_weights`, a dict with documented defaults). Write the
  resulting score to a new `fishial_quality_score` column so it is inspectable.
- Reject anything below `fishial_quality_floor` outright: mark `review_required`,
  `reason: "below quality floor"`, **zero API calls**, and free its staged frames.
- Order the survivors by score descending.

Budget pacing — the session cannot know its best fish until it ends, so do not spend
everything early:

- Hold back `fishial_late_reserve_fish` slots (default 2, clamped to
  `fish_target - 1`) until either `fishial_reserve_after_seconds` (default 180) have
  elapsed since `started_at`, or the identifier is running from `finish()`.
- In `finish()`, release the reserve and spend whatever budget remains on the best
  remaining candidates before the scratch wipe. This is the last chance, so the
  quality floor still applies but the pacing rule does not.

`species_id_fish_enrolled` now means "fish actually selected for identification".
Increment it at selection, not at first sighting. This makes the existing budget
formula continue to hold, and finally makes the dashboard counter mean something.

### 3. `app/services/live_species.py` — stop paying for decided or doomed fish

`_identify_track` already sends frames one at a time in a loop. Add a stop check
evaluated after each response, before the next `_reserve`:

- **Decisive.** If the current leader already satisfies `fishial_min_votes`,
  `fishial_vote_ratio` and `fishial_min_species_score`, and no distribution of the
  remaining staged frames could unseat it, stop and finalise `identified`.
- **Hopeless.** If, assuming *every* remaining staged frame votes for the current
  leader, the consensus rule still cannot be satisfied, stop and finalise
  `review_required` with `reason: "consensus unreachable"`.
- **Declining classifier.** If `fishial_max_empty_responses` (new, default 2)
  consecutive responses came back with zero species candidates, stop and finalise
  `review_required` with `reason: "classifier returned no candidates"`.

Record `stopped_early` and the triggering rule in the audit, and add the count of
calls not made to a new `species_id_calls_saved` session column.

Implement the two counting rules as one pure helper in `species_quality.py`
(`consensus_outlook(frames, remaining, settings) -> "decided" | "possible" |
"unreachable"`) so they are unit-testable without a DB or a client. Getting this
arithmetic wrong silently costs accuracy, so test it directly and thoroughly.

On the evidence above, rule 3 alone would have taken the reference session from 11
calls to about 6 with no change in outcome.

### 4. `app/services/fishial.py` — resolve multi-object responses instead of abstaining

`_parse` currently discards any response whose `objects` length is not exactly 1.
Replace the blanket guard with a targeted match, keeping the safety property.

- `identify(image, expected_box=None)`: when `expected_box` is given (normalised
  `(x1, y1, x2, y2)` within the crop), and more than one object is returned, compute
  IoU between `expected_box` and each object's `bbox`.
- Accept the best-matching object only when `iou >= fishial_object_match_min_iou`
  (default 0.5) **and** it beats the runner-up object's IoU by
  `fishial_object_match_min_margin` (default 0.2). Otherwise return an empty
  prediction exactly as today.
- With no `expected_box`, or a single object, behaviour is unchanged.
- Record `object_index`, `object_iou` and `object_count` in the frame audit.

`_stage_species_crop` knows the expected box precisely: the crop is the detection box
expanded by `fishial_crop_margin` and clamped to the frame. Persist that normalised
box in the `staged` entry at staging time — do not recompute it later from the
detection row, because clamping at a frame edge shifts it off-centre. Pass it through
`SpeciesIdentifier._identify_track`.

This recovers answers already paid for. It must not become a way to attribute a
neighbour's identity: when the match is ambiguous, abstain.

### 5. `app/services/species_region.py` (new) — reject impossible species for free

- A module-level `REGIONS: dict[str, frozenset[str]]` mapping a region key to
  accepted scientific names. Seed `north_east_atlantic` with the species plausible
  for Irish coastal water, and leave a comment naming the source you used. Keep the
  data in one literal; no network, no new dependency.
- Add an optional `"region"` key to `DEFAULT_LIVE_SOURCES` entries in
  `app/config.py`. Set the three SmartBay cameras to `north_east_atlantic`. Leave
  `coral-city` unset, and treat unset as "no filtering" so nothing regresses for
  cameras we have no list for.
- Persist the region on the session at start, so a later edit to the source table
  cannot retroactively reinterpret a completed session.
- In `SpeciesIdentifier`, filter the ranked candidate list before voting: drop names
  outside the region set, and record each dropped name plus
  `reason: "implausible_for_region"` in the frame audit. If filtering empties the
  list, that frame abstains.
- Gate the whole behaviour on `fishial_region_filter_enabled` (default `true`, since
  an unset region already means no filtering).

Matching must be on `scientificName`, normalised for case and whitespace only. Do not
attempt fuzzy or genus-level matching — a near-miss should abstain, not guess.

### 6. `scripts/fishial_replay.py` (new) — tune thresholds at zero API cost

This is the tool that keeps future tuning free. Two modes, no network in the first.

- `--dry-run --session <id>`: load every `fishial_votes_json` for the session, re-run
  parsing, the multi-object match, the regional filter, voting, consensus and the
  early-stop rules against the **already stored** `raw` responses under the current
  (or an overridden) config. Report per track: old outcome vs new outcome, and per
  session: identified / review_required / error counts, calls actually spent, and
  calls the stop rules *would* have spent. Makes **zero** API calls — assert that in
  a test.
- `--replay <directory> --max-calls <n>`: send stored crops through `FishialClient`
  with an explicit hard ceiling, for A/B-ing preprocessing (step 7). Refuse to run
  without `--max-calls`. Print a running call count.
- Accept `--set key=value` overrides for any `fishial_*` setting so thresholds can be
  swept without editing `.env`.

Use this to calibrate `fishial_quality_floor`, the quality weights and
`fishial_max_empty_responses` against the reference session before choosing defaults.
Report the numbers you land on and the data you derived them from.

### 7. `app/services/live_monitor.py` — optional underwater preprocessing

Dim green low-contrast input is the most likely reason the classifier declines. Add
preprocessing applied **only** to the staged crop, never to the annotated view or the
stored `crop.jpg`:

- `fishial_preprocess: str = "none"` — `"none" | "white_balance" | "clahe" | "both"`.
- White balance: gray-world channel scaling. CLAHE: on the L channel of LAB only,
  `clipLimit` from `fishial_clahe_clip` (default 2.0).
- `fishial_upscale_short_side: int = 0` — when non-zero and the crop's short side is
  below it, upscale with `INTER_CUBIC` to that short side. Never downscale, never
  change aspect ratio.

**Ship `fishial_preprocess="none"`.** Then use `--replay` on stored crops from the
reference session to compare variants under a fixed small call budget, and report
which variant produced more non-empty species lists. Change the default only if the
data supports it, and say so explicitly in the report. If you cannot get stored crops
(the scratch wipe removes them), add `fishial_keep_staged_crops: bool = False` that
copies staged crops to the output volume for later replay, document that it retains
fish imagery on disk, and default it off.

### 8. Trim what we store

`raw` averages 12 KB per track, almost entirely `shape` polygon arrays we never read.
Before persisting, strip each object's `shape`, keeping `bbox`, `species` and the
`definitions` entries actually referenced. Keep everything the replay harness needs —
verify by running `--dry-run` against a trimmed audit. Note in the report how much
this saves.

### 9. `app/db/migrations/versions/0007_live_species_quality.py`

`down_revision = "0006_live_species_id"`, with a real `downgrade()`.

`live_monitor_sessions` — add:
- `species_id_candidate_pool_size INTEGER NOT NULL DEFAULT 0`
- `species_id_calls_saved INTEGER NOT NULL DEFAULT 0`
- `species_id_region VARCHAR(64)` (nullable)

`live_fish_tracks` — add:
- `fishial_quality_score FLOAT` (nullable)

Extend the `ck_live_fish_tracks_fishial_state` constraint to include `candidate`.
Postgres cannot alter a check constraint in place — drop and recreate it, in both
directions. `downgrade()` must first move any `candidate` rows to `disabled`, or the
recreated narrower constraint fails on real data.

Mirror all of it in `app/db/models.py` and extend the parity loop in
`tests/test_live_migration.py` the way `0006` already is.

### 10. `app/config.py`

Add every setting named above with the stated default, validated in the existing
`@model_validator`:

- `fishial_candidate_pool_size` (`0` = auto), `fishial_max_staged_frames_per_candidate`,
  `fishial_max_staged_bytes`
- `fishial_quality_floor` (`0.0` = admit everything), `fishial_quality_weights`
- `fishial_late_reserve_fish`, `fishial_reserve_after_seconds`
- `fishial_max_empty_responses` (`0` = disabled)
- `fishial_object_match_min_iou`, `fishial_object_match_min_margin`
- `fishial_region_filter_enabled`
- `fishial_preprocess`, `fishial_clahe_clip`, `fishial_upscale_short_side`,
  `fishial_keep_staged_crops`

Validation: `fishial_late_reserve_fish < fishial_max_fish_per_session`;
`fishial_max_staged_frames_per_candidate >= fishial_min_frames_to_vote`;
`fishial_preprocess` in the allowed set; weights non-negative and not all zero.

### 11. `app/api/live.py`, `app/static/live.js`, `app/static/dashboard.css`

- `session_data()` `species_id` block gains `candidates`, `calls_saved`, `region`.
- `track_data()` gains `fishial_quality_score`.
- `GET /live/{id}/species` gains `calls_saved` and a `declined` count (tracks whose
  reason was `classifier returned no candidates`), so an operator can tell "the model
  won't name these fish" apart from "the fish disagreed with each other".
- `live.js`: extend the existing status line to `enrolled/target · X identified · Y
  need review · Z calls (N saved)`. In the review-required tooltip, surface the
  `reason` verbatim — the four reasons are now genuinely distinct and diagnostic.
  Keep `createElement`, keep the `localStorage` keys, add no dependency.
- `dashboard.css`: reuse existing tokens; add no new colors.

### 12. `.env.example` and `README.md`

Document every new var, and add a short "tuning identification yield" subsection to
the live-monitoring docs covering: the candidate pool, the quality floor, the three
stop rules, the regional filter, and how to run `fishial_replay.py --dry-run` to
evaluate a config change for free. State plainly that preprocessing ships off and why.

## Accuracy strategy — make these explicit in code comments

- **Rank, do not gate.** Absolute thresholds calibrated above the data reject
  everything. Floors exclude unusable frames; ranking picks the best of what remains.
  Any future threshold must be justified against a measured percentile, in a comment.
- **Pay for the best fish, not the first fish.** Selection is deferred to
  finalization, ordered by measured crop quality, floored, and paced so late fish can
  still win a slot.
- **Stop early in both directions.** Decided, unreachable, and repeatedly-declined
  all terminate a track without further spend.
- **Read the answer we already bought.** Multi-object responses resolve by IoU
  against the known target box, and abstain when ambiguous.
- **Reject the impossible for free.** Regional filtering removes whole classes of
  wrong label before voting, at no API cost.
- **Preprocess only on evidence.** Ships off; enabled only if replay proves it.
- **Distinguishable abstentions.** `insufficient clear frames`, `below quality
  floor`, `consensus unreachable`, `classifier returned no candidates` and `budget
  exhausted` are five different operational problems and must never collapse into one
  message.
- **The reservation journal is load-bearing.** Selection, stop rules and preprocessing
  all run around it, never through it. A crash must still be unable to buy a call.

## Tests

Extend `tests/test_fishial.py`, `tests/test_live_species.py`,
`tests/test_live_monitor.py`, `tests/test_live_migration.py`, and the API tests.

- **Frame ranking**: a track fed 10 frames with known scores retains exactly the top
  `max_staged_frames_per_candidate`; a later high-scoring frame evicts the retained
  worst and deletes its JPEG; a later low-scoring frame is dropped without touching
  the set. A track fed frames that all fall in one separation window retains no more
  than one of them, and a track with 3 detections spread over 3 windows becomes
  identifiable. Add a regression test asserting that a track resembling the Coral City
  medians (confidence ~0.62, short side ~57 px) **does** reach
  `fishial_min_frames_to_vote` — today it does not.
- **Recalibrated floor**: confidence `0.49` and short side `47 px` are rejected;
  `0.51`/`49 px` are admitted; non-finite boxes and edge-breaking boxes still rejected.
- **Candidate pool**: with `pool_size=3`, a fourth stronger track evicts the weakest
  and deletes its staged directory; a `ready`/`submitted` track is never evicted; the
  byte cap triggers eviction.
- **Selection**: given candidates with known quality inputs, the highest scorers are
  chosen; sub-floor candidates end `review_required` with **zero** client calls; an
  unfinalized track is never selected; `species_id_fish_enrolled` counts selections.
- **Pacing**: before `reserve_after_seconds`, at most `fish_target -
  late_reserve_fish` are identified; `finish()` releases the reserve.
- **`consensus_outlook`** as a pure function: decided / possible / unreachable across
  a table of vote distributions, including the exact boundary at `min_votes` and at
  `vote_ratio`.
- **Stop rules**: a stubbed client counting calls shows a decisive track stops before
  exhausting its frames; a hopeless track stops; `max_empty_responses=2` with an
  all-empty client makes exactly 2 calls; `species_id_calls_saved` matches.
- **Multi-object**: two objects, target IoU 0.8 vs 0.1 → identified from the matched
  object; 0.55 vs 0.5 (inside the margin) → abstains; one object → unchanged; no
  `expected_box` → unchanged. A clamped edge crop passes the *stored* box, not a
  recomputed centred one.
- **Region filter**: a Caribbean species is dropped for `north_east_atlantic` and
  recorded; an unset region filters nothing; filtering to empty abstains. Assert
  directly that the reference session's `Sparisoma aurofrenatum` would be rejected.
- **Replay**: `--dry-run` over a fixture audit makes **zero** `identify` calls
  (assert on an injected client) and reproduces the recorded outcome under the
  original config; `--replay` without `--max-calls` exits non-zero.
- **Trimming**: `shape` is stripped, referenced `definitions` survive, and
  `--dry-run` still reproduces the same verdict from a trimmed audit.
- **Budget invariant, unchanged**: across all new paths, total `identify` calls never
  exceed `fish_target * frames_per_fish + fishial_max_api_retries`. Include a case
  where selection, eviction and early stopping all fire in one session.
- **Migration**: `0007` up/down, the check-constraint swap in both directions, and a
  `downgrade()` run against a table containing `candidate` rows.
- **Feature off**: `fish_target=0` or `fishial_enabled=false` → no candidates staged,
  no client constructed, every track `disabled`.

Never call the real Fishial API, the internet, or a real camera. Keep
`httpx.MockTransport` for `FishialClient`'s own tests and injected stubs everywhere
else, as the existing tests do.

## Verification

Plain `pytest` fails in this environment; pass a short basetemp:

```
pytest tests -q --basetemp=C:\t\pt
```

Run whatever linter/formatter `pyproject.toml` configures.

Then report:

- **first**, the recalibrated floor's measured pass rate on both reference sessions,
  and how many tracks in each become able to reach `fishial_min_frames_to_vote` —
  against the baselines of 7/132 (Coral City) and 27/74 (SmartBay);
- the selection rule and quality weights you settled on, and the numbers from
  `--dry-run` against session `ab69aaca-7ddc-46db-b3af-70b11bf6ba30` that justify
  them;
- for that same session, what the new pipeline would have produced: outcomes per
  track and calls spent vs the 11 actually spent;
- how many of the 11 recorded responses the multi-object matcher recovers, and what
  the regional filter then does with them;
- whether preprocessing helped, the evidence, and what you defaulted it to — and
  explicitly, whether a post-fix Coral City run produced non-empty species lists,
  since that is the first real test of the model this project will have run;
- where the budget ceiling is now checked, and proof the reservation journal's
  crash-safety is intact;
- how much the `shape` trim saves per track;
- anything you had to widen scope beyond this list, and any place the evidence above
  turned out to be wrong.
