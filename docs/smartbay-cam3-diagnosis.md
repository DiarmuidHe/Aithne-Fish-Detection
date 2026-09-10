# SmartBay Cam 3 diagnosis — 2026-09-10

Diagnosis used stored database audits, pure response parsing and `--dry-run` only.
No Fishial API calls, database writes, configuration changes or container rebuilds.

## Evidence and root cause

Latest enabled session: `d3090141-7d08-4de6-b3f6-5a1c65b30a21`, created
2026-09-10 13:57:57 UTC. Five fish selected, nine image calls, zero identified,
32 review tracks and 143 disabled tracks (including six candidate-pool evictions).

| Stored review reason | Tracks | Explanation |
| --- | ---: | --- |
| insufficient clear frames | 17 | Too few staged frames to submit for consensus; zero calls |
| below quality floor | 0 | Configured floor is 0.0 |
| classifier returned no candidates | 3 | Repeated empty species lists for the matched fish |
| consensus unreachable | 2 | One empty matched response; one region-rejected response, each with only three staged frames |
| budget exhausted | 10 | Five-fish selection limit reached; these tracks were never submitted |
| parser errors | 0 | All nine raw responses parse successfully |

The diagnostic view separates the last two consensus failures into their underlying
causes: four tracks with classifier declines and one `implausible_for_region` track.
The original stopping reason remains available separately. UI/reporting mismatch
affected all review cards: their visible text was only “Review required”, with the
reason hidden in a tooltip. “Budget exhausted” also concealed the fish-selection
limit despite only nine of 27 permitted image calls having been used.

Of nine responses, **seven matched objects had empty species lists**. The two named
matched objects were:

- `Pollachius virens`, 0.51: retained by the region filter and voted successfully.
  Track `f3df237b` then received two empty matched answers, so stopped without the
  required three votes. The first multi-object match had IoU 0.8744; subsequent
  matches had IoU 0.8794 and 0.8785. Other objects' names were not this fish's votes.
- `Lepisosteus oculatus`, 0.46: filtered on track `bbf26db0`, correctly abstaining.

The regional list already includes saithe, an Irish coastal species documented by
the [Irish Specimen Fish Committee](https://specimenfish.ie/irish-record-fish/marine-species/coalfish-pollachius-virens/).
The spotted gar's documented range and freshwater habitat do not support a Galway
Bay label ([Texas Parks & Wildlife](https://tpwd.texas.gov/huntwild/wild/species/spottedgar/)).
No plausible matched SmartBay candidate was found being discarded by the current
regional list. No names, fuzzy matching or genus matching were added.

Earlier enabled session: `ab69aaca-7ddc-46db-b3af-70b11bf6ba30`, created
12:12:55 UTC. Five review tracks: two insufficient-frame tracks and three without
consensus; 11 recorded calls. Re-parsing every raw response produces nine empty
matched lists and just two weak candidates: `Pomatomus saltatrix` at 0.46 and
`Sparisoma aurofrenatum` at 0.48. This session stored region `None`, so regional
filtering did not cause its failures. The current matcher recovers these old
multi-object answers, but neither supplies enough evidence to identify a track.
The still older `9fc86c97-c735-476f-aa89-c1f2137264aa` had species identification off.

**Conclusion:** inadequate classifier evidence on SmartBay imagery is the immediate
cause, compounded by too few eligible frames on many tracks. Image/domain mismatch
is a plausible contributor, not a measured causal result. There is no stored evidence
justifying weaker consensus or a parser/matcher change.

## Image quality and preprocessing

Latest-session staged quality measurements have medians: sharpness 13.31,
LAB luminance 131.26, contrast 31.87 and colorfulness 7.95; short side 171.5 px.
Sharpness ranges 4.44–82.81 and luminance 59.45–231.27: brightness varies, so
uniform darkness alone does not explain the failures. The inspected gallery crop
for `f3df237b` is dim and washed out, with strong glare and a dark foreground fish.

Neither enabled session retained clean `fishial/*.jpg` inputs. Only annotated
gallery crops/clips remain, and `fishial_keep_staged_crops=False`. Consequently no
controlled comparison of `none`, `white_balance`, `clahe` or `both` was possible on
the actual submitted images. No paid replay was run. `fishial_preprocess="none"`
and every quality/voting threshold remain unchanged.

## Changes and before/after dry-run

- `scripts/fishial_replay.py`: use staged frame capacity, capped by the session's
  frames-per-fish, when evaluating early stopping. Previously, already-skipped
  frames disappeared from the possible future evidence and caused fictitious savings.
  Count recorded retry reservations accurately, count explicit unreserved records
  as zero calls, and flag missing responses when an offline continuation is unknown.
- `app/services/species_quality.py`: a read-only diagnostic helper derives frame
  abstention counts and an explanatory reason while preserving the stop reason.
- `app/api/live.py`, `app/static/live.js`: expose diagnostics and aggregate reason
  counts; display the reason directly on review cards, including selection-limit
  context. Stored decisions and audits are not rewritten.
- Focused regressions in `tests/test_fishial_replay.py`,
  `tests/test_live_species.py` and `tests/test_species_region.py` cover the observed
  early-stopped patterns, call accounting, API diagnostics and exact species names.

Both sessions were run before and after changes using:

```powershell
.venv\Scripts\python.exe scripts/fishial_replay.py --dry-run --session d3090141-7d08-4de6-b3f6-5a1c65b30a21
.venv\Scripts\python.exe scripts/fishial_replay.py --dry-run --session ab69aaca-7ddc-46db-b3af-70b11bf6ba30
```

| Session | Identified / review, before → after | Dry-run calls, before → after | Claimed additional savings, before → after |
| --- | --- | --- | --- |
| d3090141… | 0 / 32 → 0 / 32 | 6 → 9 | 3 → 0 |
| ab69aaca… | 0 / 5 → 0 / 5 | 6 → 6 | 5 → 5 |

The latest dry-run now reproduces all nine reserved calls and reports 17 insufficient
frames, 10 selection-limit tracks, four classifier declines and one regional rejection.
The older session's five-call saving is hypothetical under existing stop rules;
it is not a refund or new saving caused by this patch. The latest live audit already
records 11 skipped staged calls from its existing early-stop behavior (20 available,
nine submitted). **New API calls spent: zero. Additional live savings from this patch: zero.**

## Validation and budget

The requested `.venv\Scripts\python.exe -m pytest tests -q --basetemp=C:\t\pt`
passed, including existing budget ceiling, retry/401, persisted restart and interrupted
reservation tests. `node --check app/static/live.js` also passed. Pytest emitted
dependency deprecation and cache-write warnings, with no test failures.

`SpeciesIdentifier._reserve`, the before-image-call callback and the live decision
path are unchanged. Reservations still commit atomically with the frame journal
before each image request, including retries. The invariant remains:

`image calls <= fish_target * frames_per_fish + fishial_max_api_retries`

For each enabled SmartBay session the ceiling is `5 * 5 + 2 = 27`.
Changes are local; the running containers have not been rebuilt.

Subsequent authorized work automated and ran a six-call preprocessing screen on
fresh clean crops. See [the automated experiment](smartbay-automated-experiment.md)
for its separate call accounting and results; the zero-call figures above refer to
the initial stored-session diagnosis only.

## Tentative species display — 2026-09-10

Rechecked the database: `d3090141…` remains the latest Cam 3 session. Its nine
stored responses still contain seven empty matched lists, saithe (`Pollachius
virens`) at 51%, and spotted gar (`Lepisosteus oculatus`) at 46%. Neither track is
confirmed; the gar suggestion was rejected for region.

Review cards now display tentative common/scientific names, the best individual
Fishial score, and “Review required”, with an explicit regional-rejection label
where applicable. Names come only from the stored matched object or an existing
frame vote; names belonging to other fish in the image are excluded. The API
derives this information from existing audits, including historical records,
without changing decisions, thresholds, confirmed species counts or call budgets.
No new Fishial requests were made. All 61 focused Python tests and JavaScript
syntax/rendering checks passed.
