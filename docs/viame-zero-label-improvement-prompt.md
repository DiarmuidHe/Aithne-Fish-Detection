# Prompt: improve VIAME fish detection without manual labels

Use this prompt from the repository root with an implementation-capable coding agent.

```text
You are improving the VIAME detection and tracking path in the
`Aithne-Fish-Detection` repository. Work autonomously from diagnosis through a
verified implementation and final report. Do not claim that an unlabeled experiment
measures real accuracy.

Goal
----
Find and implement the best defensible, cost-efficient improvement to fish detection
and tracking that can be obtained from the existing model, pipelines, videos and
runtime without asking me to draw boxes or label frames. Prefer a small, reproducible
improvement over a large speculative rewrite.

Important current facts to verify rather than blindly assume
-------------------------------------------------------------
- Batch inference uses the checked-in DEIMv2 + ByteTrack pipeline at
  `app/viame_pipelines/tracker_fish_no_motion.pipe`.
- The checked-in defaults are detector score 0.10, ByteTrack high/low/new-track
  thresholds 0.45/0.10/0.50, track buffer 30 sampled frames, and batch sampling at
  10 FPS.
- The repository records a previous change from the stock 5 FPS setup to the current
  pipeline: 3 tracks/19 observations became 9 tracks/182 observations, with 7 tracks
  above the then-current 0.60 acceptance threshold. The stock motion-fusion pipeline
  produced 7 accepted tracks/124 observations on that one sample. These are output
  counts, not ground truth and not proof of better accuracy.
- The local `.env` currently uses `LIVE_FPS=2` and `MIN_FISH_CONFIDENCE=0.50`. Do not
  print, copy, commit or modify credentials from `.env`.
- `VIAME_TRACKER_BUFFER_FRAMES=30` is frame-based. At 10 FPS it represents about
  3 seconds; at 2 FPS it represents about 15 seconds. Establish whether that mismatch
  causes false joins or track suppression, and make retention semantics consistent in
  seconds if the evidence supports it.
- Fishial is a separate species-identification service, not the VIAME fish detector.
  It is out of scope. Experimental runs must disable Fishial and must make zero paid
  API calls.

Truthfulness boundary
---------------------
With no trusted labels, do not report precision, recall, mAP, F1, false-positive rate
or false-negative rate. Do not call extra detections an improvement by themselves.
Temporal consistency and agreement between models are proxy metrics only. Do not
fine-tune on raw self-generated boxes: this can amplify the current detector's errors.
If model-weight training is not justified, say so and optimize the inference/tracking
system instead.

Hard cost and safety limits
---------------------------
1. Spend zero money and make zero Fishial or other paid API requests.
2. Reuse existing local videos first. Do not download a dataset or model, pull a new
   container image, or make a long live-camera recording without explicit approval.
3. Limit the experiment to at most 10 new VIAME inference passes and 90 minutes of
   aggregate GPU inference, whichever occurs first. Stop earlier when a candidate is
   clearly dominated. Offline rescoring of already-produced CSV files does not count
   as an inference pass.
4. Never run experiments against the production database. Use an isolated output
   directory under `data/outputs/viame-zero-label-benchmark/` and a manifest/journal
   so interrupted work resumes without repeating completed inference.
5. Do not overwrite existing job outputs, videos, `.env`, review decisions or current
   pipelines. Create candidate files separately. Preserve unrelated working-tree
   changes.
6. Do not silently deploy a candidate. Promotion is allowed only after it meets the
   conservative criteria below, tests pass, and the report explains the limitation of
   unlabeled evidence. Otherwise keep the baseline and report the best next option.

Phase 1: establish provenance and a deterministic baseline
----------------------------------------------------------
1. Inspect `README.md`, `app/config.py`, `app/services/viame_runner.py`,
   `app/services/viame_parser.py`, `app/workers/live_worker.py`, both Compose files,
   the checked-in pipeline, existing job CSVs/logs and relevant tests.
2. Inventory the exact installed VIAME version, available fish detector/tracker
   pipelines, model filenames, file hashes, ONNX input resolution and available
   ByteTrack parameters. Use local files/container introspection only. If the runtime
   is unavailable, implement the harness and give an exact resume command; do not
   invent results.
3. Explain any mismatch between README defaults, Compose defaults and `.env` values.
   Treat batch and live profiles separately where their sampling rates differ.
4. Select a small fixed benchmark from existing local videos:
   - Prefer three 30-60 second clips representing different existing scenes or quality
     levels.
   - If only one scene exists, select early/middle/late non-overlapping clips.
   - Record source SHA-256, clip start/duration, resolution, source FPS and a basic
     brightness/contrast/motion summary in a manifest.
   - Never choose clips because a candidate looks good on them.
5. Run the current baseline once per benchmark clip. Repeat one clip once to detect
   nondeterminism. Save commands, safe configuration, runtime, exit status, logs,
   detection CSV and track CSV.

Phase 2: build an unlabeled evaluation harness before tuning
------------------------------------------------------------
Create a focused script, preferably `scripts/viame_zero_label_benchmark.py`, plus unit
tests. It must parse candidate outputs using the application's existing VIAME parser
where practical and emit machine-readable `report.json` plus a concise `report.md`.
Avoid adding a large dependency.

Report at least these per-clip and aggregate proxy metrics:
- total detections and tracks, but never use either alone as the objective;
- accepted-track counts over an offline `MIN_FISH_CONFIDENCE` sweep of
  0.45, 0.50, 0.55, 0.60 and 0.65;
- track-length distribution and the fractions with 1, 2, 3-4 and 5+ observations;
- temporal coverage: fraction of detections assigned to tracks with at least 3 and
  at least 5 observations;
- gap distribution in seconds, using actual sampled/source timestamps;
- normalized center-velocity/acceleration and log-area-change statistics, with robust
  medians/percentiles so one long track cannot dominate;
- likely duplicate-track overlap: same-frame pairs with high IoU sustained across
  multiple sampled frames;
- likely fragmentation: a track ending and another starting soon afterward with
  spatially compatible boxes and motion, reported as a proxy rather than a fact;
- invalid/out-of-bounds/tiny boxes and tracks that exist only at a frame edge;
- run time, effective analyzed FPS and peak GPU memory when it can be collected
  cheaply;
- run-to-run determinism on the repeated baseline clip.

Add paired-run comparison using frame timestamps rather than assuming identical frame
numbers. Match boxes with IoU and report agreement/unique detections between the
DEIM-only baseline and the stock motion-fusion pipeline. Model agreement is useful
weak evidence, but correlated models can agree and still be wrong, so retain the raw
disagreement galleries/metadata for inspection and do not label unique detections as
false.

Use either a Pareto table or a clearly documented composite ranking. If using a
composite score, normalize each term against the repeated-baseline noise and publish
the formula and all component values. Never give positive credit merely for producing
more tracks or lowering a confidence threshold.

Phase 3: successive-halving experiment, cheapest tests first
------------------------------------------------------------
Do not grid-search every knob. First use the existing CSV files for free offline
threshold and quality analysis. Then use one representative clip to eliminate poor
candidates. Only the baseline and at most two survivors may run across all benchmark
clips.

Test no more than these targeted hypotheses, one conceptual change at a time:

A. Time-normalized ByteTrack retention. Compare the current 30-frame buffer against a
buffer representing approximately 3-5 seconds at the actual inference FPS. Batch and
live may need distinct effective frame counts. Verify how the installed ByteTrack
implementation defines `track_buffer` before changing it.

B. Association gates. The checked-in pipeline hard-codes `match_thresh`,
`second_match_thresh` and `unconfirmed_match_thresh` at 0.98. Inspect the installed
implementation/config documentation to confirm parameter direction and valid range.
Test at most two evidence-based alternatives. Do not assume that a numerically lower
value is looser or better.

C. Sampling rate. Compare only rates that fit the real-time budget. For live video,
test 2 FPS versus one survivor such as 5 FPS on the same saved source clip; do not
recapture the camera for each run. Include VIAME's fixed startup overhead separately
from per-video processing time. Do not promote 10 FPS live if it cannot keep up.

D. Pipeline/model alternative. Re-evaluate the locally installed stock motion-fusion
pipeline on the fixed benchmark, not just the historical single sample. If a newer
locally installed default-fish, RF-DETR or text-prompted fish pipeline exists, record
it as an optional candidate only when it can be run without downloading assets and
within the pass budget. Compare its speed, temporal behavior and agreement; never call
it more accurate without labels.

E. Input enhancement. Only test a reversible, modest underwater enhancement if the
installed pipeline accepts it cleanly and there is remaining pass budget. Apply the
same transform to all benchmark clips. Reject it if it increases unstable one-frame
detections, worsens consensus, clips highlights, or materially slows inference.

Do not test arbitrary large threshold grids, train a new neural network, add a second
paid classifier, or build a complex ensemble during this bounded pass.

Conservative promotion rule
---------------------------
A candidate may replace the current setting only if all of the following hold:
1. It is deterministic within the measured baseline tolerance.
2. It improves at least two independent temporal-quality proxies (for example lower
   fragmentation and smoother trajectories, or better persistence and fewer duplicate
   overlaps) on every benchmark clip, not only in aggregate.
3. No other temporal-quality proxy regresses by more than 5%, unless the report gives
   a strong, concrete reason the proxy is misleading for that clip.
4. Agreement with the independent motion-fusion run does not materially decrease.
5. Accepted counts remain stable across nearby acceptance thresholds; a gain that
   exists only at one cutoff is not sufficient.
6. Batch processing stays within 1.25x baseline runtime. A live candidate must process
   faster than wall-clock duration with at least 20% headroom after startup is
   amortized, and must not increase dropped segments.
7. The change is small, configurable, recorded in processing provenance, documented,
   and covered by tests.

If no candidate meets every condition, make no production configuration change. The
correct result may be that the current configuration is the best defensible unlabeled
configuration tested, not that the model itself is optimal.

Implementation expectations
---------------------------
- If supported by the result, replace ambiguous frame-only retention with an explicit
  seconds-level setting that derives the correct ByteTrack frame buffer from effective
  inference FPS. Preserve backward compatibility and snapshot the derived value into
  job provenance. Test 2 FPS and 10 FPS calculations.
- Make association thresholds configurable only if the installed pipeline proves they
  are valid runtime overrides. Validate their ranges and ordering in `Settings`.
- Keep batch and live configurations reproducible. Persist every effective detector,
  tracker and sampling parameter in the job/experiment report.
- Add focused tests for metric arithmetic, timestamp alignment, resume/journaling,
  budget enforcement, malformed CSV handling and any new configuration behavior.
- Run the focused tests first, then the full Python test suite. Run frontend tests or
  builds only if frontend code changes.
- Do not weaken existing security boundaries or expose filesystem paths, commands,
  source URLs, credentials or raw logs through public APIs.

Required deliverables
---------------------
1. The benchmark script, tests and deterministic manifest/journal behavior.
2. `data/outputs/viame-zero-label-benchmark/report.json` and `report.md` when the GPU
   runtime is available; otherwise a checked-in example schema and exact resume
   command without fabricated measurements.
3. A small code/configuration change only if the conservative promotion rule passes.
4. README documentation covering how to reproduce the benchmark and how to interpret
   proxy metrics.
5. A final summary that states:
   - what was tested and the exact compute/API budget used;
   - baseline versus candidate values for every promotion criterion;
   - what changed, or why nothing changed;
   - known uncertainties and why this is not a true accuracy measurement;
   - the smallest future labeling effort that would unlock real precision/recall
     measurement, as an optional recommendation only (do not require it for this task).

Finish the work rather than only proposing it. If the VIAME runtime is unavailable,
finish all safe implementation and tests that do not require it, then stop with the
single exact command needed to resume the bounded GPU benchmark.
```

## Expected interpretation

This prompt can improve the deployed detector/tracker's consistency and efficiency
without manual annotation. It deliberately cannot certify a higher biological-fish
detection accuracy, because that requires an independent truth set.
