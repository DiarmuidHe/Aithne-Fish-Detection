# VIAME zero-label benchmark — 2026-09-11

**Keep the production baseline.** Neither motion fusion nor a three-second live
retention buffer meets the conservative promotion rule. This experiment measures
temporal proxies and cost, not fish-detection accuracy. No model training,
production configuration, `.env`, application jobs or review decisions changed.

## Budget and reproducibility

- 10 new VIAME passes; **562.448 seconds** aggregate inference-process wall time
  (9 minutes 22.448 seconds), including startup and monitoring overhead. Limits:
  10 passes and 5,400 seconds. CPU clip preparation and offline rescoring are separate.
- Zero Fishial/paid API calls, downloads, image pulls, or live-camera captures.
- Four baseline passes (three clips plus a repeat), three motion-fusion reference
  passes, two live-retention profiles on one saved clip, and one candidate repeat.
- 15 focused tests and **293 full-suite tests passed**. Existing deprecation warnings
  remain. Windows tests need a short temporary path; an initial long-path run failed
  on media-file access. No frontend changes were made for this task.

Run/resume from the repository root:

```powershell
.venv\Scripts\python.exe scripts/viame_zero_label_benchmark.py --docker
```

The existing journal has exhausted the pass budget; this command reuses its outputs.
Append `--report-only` for explicit offline rescoring. Full local evidence is in
`data/outputs/viame-zero-label-benchmark/`: `manifest.json`, `inventory.json`,
`offline-history.json`, `journal.json`, `report.json`, `report.md`, run CSVs/logs,
and `disagreements-*.json`. These generated artifacts remain gitignored.

## Inputs and runtime

SHA-256 deduplication found three unique camera recordings of approximately 3.2,
9.1 and 50.8 seconds, plus a one-second smoke clip. The benchmark uses consecutive
early/middle/late 16-second sections of the longest recording, selected before any
candidate inference. Source SHA-256:

```text
0de3bcdd8cf3663d3adceda651ad46fdee1ef4b5fe4b56b671ec0b8ee19476f8
```

This is one camera, not three independent scenes; the requested 30–60 seconds per
clip could not be obtained non-overlapping from the existing longer source. Batch
clips retain 1920×1080 resolution and approximately 29.92 source FPS, sampled at
10 FPS. Live uses the middle clip, saved at 1280×720 and 2 FPS.

The GPU is an NVIDIA GeForce RTX 5060 Laptop GPU. KWIVER reports **2.4.1**; no
VIAME release identifier or build commit was available. Installed pipeline/model
hashes identify this runtime. DEIM's actual ONNX inputs are `N×3×1024×1024` images
and `N×2` original target sizes. ONNX SHA-256:

```text
6973de8fd61b215156977dd4c98b0053bd5bbad3dca9c3f373b33347599cc4c2
```

The fusion graph adds the locally installed motion RF-DETR weights at 960×1728.
It retains the current application confidence/buffer overrides so the conceptual
change is the detector graph. It is partly correlated with the DEIM baseline.

Observed numeric settings match the prompt: detector .10, tracker high/low/new
.45/.10/.50, buffer 30 and association-distance gates .98. The installed code
accepts matches when `1 − IoU <= gate`; lowering a gate is stricter. Its internal
frame counter increments on each processed frame, and expired tracks are removed
**after association** when age is greater than the buffer. This explains why the
six-frame candidate can still contain a 3.5-second gap at 2 FPS.

Compose's acceptance default is .60 and live default is 5 FPS; the numeric local
overrides are .50 and 2 FPS. Bare Settings defaults also differ from GPU Compose
in pipeline path and frame offset. No credentials were copied into provenance.

## Observations

Coverage below means detector observations assigned to tracks of at least three
observations. Counts are context, not an improvement objective.

| Profile | Early coverage | Middle coverage | Late coverage | Total tracks / observations | Accepted at .45/.50/.55/.60/.65 |
|---|---:|---:|---:|---:|---|
| Batch baseline | 35.51% | 43.79% | 42.10% | 41 / 2,201 | 41/41/38/36/31 |
| Motion fusion | 26.19% | 37.53% | 30.94% | 32 / 1,781 | 32/32/30/30/29 |

Fusion/baseline detection IoU agreement is **85.62%, 83.79%, 77.30%** across the
three clips. Unique boxes are retained as disagreements, never labeled false.
Both batch profiles have zero sustained-duplicate and fragmentation pairs under
the documented proxy definitions; those zeros cannot establish perfect tracking.

| Live representative | Buffer 30 | Buffer 6 |
|---|---:|---:|
| Nominal retention at 2 FPS | 15 s | 3 s |
| Detections / tracks / observations | 401 / 14 / 157 | 401 / 14 / 157 |
| Coverage, ≥3 observations | 37.66% | 37.16% |
| Coverage, ≥5 observations | 36.66% | 36.16% |
| Per-track median acceleration | .034046 | .036049 |
| Per-track median absolute log-area change/s | .404033 | .404033 |
| Fragmentation proxy pairs | 1 | 1 |
| Sustained duplicate proxy pairs | 0 | 0 |
| Maximum observation gap | 5.0 s | 3.5 s |
| Accepted tracks, .45/.50/.55/.60/.65 | 14/14/14/14/13 | 14/14/14/14/13 |
| Total runtime for 16 s footage | 20.327 s | 18.239 s |

Shorter retention alters associations and the maximum gap, but there is no trusted
evidence identifying which joins are correct. It did not improve track suppression
or counts on this clip, and its acceleration proxy regressed **5.88%**. The live
candidate repeat took 19.229 seconds and reproduced identical detections and track
partitions. Longer live segments might amortize startup differently; dropped
segments and cross-segment association were not measured.

## Every promotion criterion

| Criterion | Baseline versus candidate evidence | Outcome |
|---|---|---|
| Determinism within baseline noise | Middle baseline repeat: exact detections and track partition; temporal-component noise zero. Runtime 56.203 → 53.202 s. Live candidate repeat also exact; fusion not repeated. | Live representative passes; fusion unproven. |
| Two independent temporal improvements on every clip | Fusion coverage falls on all three clips. Shorter live retention reduces ≥3 coverage 37.66 → 37.16%, with unchanged duplicate/fragmentation rates. Live tested on one clip only. | Neither passes. |
| No unexplained regression above 5% | Fusion early acceleration .28857 → .37116 (+28.6%); middle log-area change .78425 → .84329 (+7.53%). Live acceleration .03405 → .03605 (+5.88%). | Both fail. |
| Maintained independent fusion agreement | Baseline/fusion agreement 85.62/83.79/77.30%. Fusion shares DEIM and cannot validate itself. Live detector output is identical between buffers, but no same-FPS fusion reference was run. | Independent evidence unproven. |
| Stability across nearby acceptance cutoffs | Exact sweeps above. Fusion has no larger adjacent relative acceptance drop than baseline +5 percentage points. Live sweeps are identical on the representative. | Fusion passes this proxy; live all-clip requirement unproven. |
| Runtime ≤1.25× batch; live ≥20% headroom and no more drops | Fusion ratios .748/1.352/1.402×. First baseline pays a cold-start/cache cost (105.027 s), so .748× is not a reliable speed gain. Live 18.239 s exceeds the 12.8 s total-time headroom target for 16 s footage; isolated startup/drop behavior unavailable. | Neither demonstrated required runtime. |
| Small configurable, tested and provenanced production change | Experiment configs, hashes, CSVs and tests provided. Other criteria fail, so no production knobs or pipeline changes were introduced. | No promotion. |

Peak observed device memory was about 2,434–2,436 MiB for DEIM and 3,144 MiB for
fusion. Fixed startup is not isolated reliably by the installed logs: the writer's
`exec_time` and first CSV write are saved as diagnostics, not claimed to be pure
per-video processing time. All ten outputs parsed with zero malformed, invalid,
conflicting or missing-timestamp observations after normalization.

## Interpretation and next evidence

The evaluator handles the different detection/track writer frame conventions and
nonzero MP4 timestamp origins. These timing corrections were tested and applied
by rescoring saved CSVs; they spent no additional inference passes. No production
parser behavior was changed based on this experiment.

Association-gate alternatives, 5 FPS live, enhancement, other models and training
were deferred within the ten-pass budget. Fusion was retained across the fixed
clips as a comparison reference even after its representative failed promotion;
the live candidate did not advance beyond one clip plus its repeat.

The current baseline is the defensible configuration **among those tested**, not
an optimal model or a certified accuracy result. A future optional truth set could
start with exhaustive fish boxes on about 30 stratified held-out frames, including
empty frames, independently checked with a fixed IoU rule. That enables preliminary
precision/recall with substantial sampling uncertainty; a few fully tracked short
sequences are also needed to assess identity errors. No labeling is required for
the delivered benchmark.
