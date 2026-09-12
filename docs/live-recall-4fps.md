# Live detection at 60% and 4 FPS — 2026-09-12

The deployed configuration is being updated from 2 FPS / 50% to **4 FPS / 60%**.
The saved `.env` already specified 60%, but the old containers still used 50%.
The acceptance cutoff is applied to original detector scores. No scores are raised,
no labels are requested, no weights are retrained and no Fishial API calls are used.

## No-label inference improvement

Capture previously reduced all footage to 720p before the 1024×1024 DEIM detector,
and hosted-player resolution selection preferred 720p. Capture now preserves up to
1920×1080, and resolution selection prefers a matching rendition. Smaller feeds
are not enlarged. `LIVE_CAPTURE_WIDTH` and `LIVE_CAPTURE_HEIGHT` allow reverting to
1280×720 on slower hardware. FPS defaults and Compose settings now agree on 4.

This improves the input evidence for the existing model; it is not a new trained model.
The alternative tested was modest CLAHE contrast enhancement, with no automatic
white balance or saturation change. It helped less and was not enabled.

## Saved-footage comparison

The same early/middle/late 16-second sections from the previous benchmark were used,
all from one 50.8-second recording. At the same 4 FPS, the higher-detail input gave:

| Clip | 720p observations ≥60% | 1080p observations ≥60% | 720p sustained tracks | 1080p sustained tracks |
| --- | ---: | ---: | ---: | ---: |
| Early | 94 | 96 | 3 | 3 |
| Middle | 143 | 165 | 8 | 10 |
| Late | 184 | 187 | 9 | 11 |
| Sum | 421 | 448 | 20 | 24 |

“Sustained” here requires at least three original VIAME observations scoring ≥0.60.
Track counts across clips are not distinct biological fish counts. At 2 FPS / 720p,
the middle clip gave 58 accepted observations and 7 sustained tracks; doubling FPS
alone naturally increases observation counts. The table isolates resolution at equal FPS.

The resolution change yielded 6.4% more accepted observations and 20% more sustained
track instances in these clips. This is **unlabeled proxy evidence, not measured
precision or recall**. There were no sustained duplicate pairs in either profile.
Late-clip fragmentation proxy fell from 0.12 to 0.048, but singleton tracks with just
one ≥60% observation rose from 1 to 4. Early temporal coverage fell from about 0.30
to 0.28; other motion/area proxies were mixed. Higher resolution therefore does not
dominate every measure or establish better accuracy on other cameras. A 720p source
cannot gain detail from the new capture bound.

## Keeping up at 4 FPS

Doubling the sampling rate doubles both the detector's work and the per-frame CPU
work that follows it. The worker now analyzes the next closed segment on the GPU
while the current one's tracking, media and database writes run on the CPU, so a
segment costs the larger of the two rather than their sum.

Replaying the complete saved recording (203 frames, 50.8 seconds at 4 FPS/1080p)
with its stored detections isolates the CPU half:

| Stage | Before | After |
| --- | ---: | ---: |
| Annotated view published | 9.6 s (203 writes) | 2.3 s (102 writes) |
| Per-detection crop and spool | 16.4 s | 2.7 s |
| Finalized clip rendering | 6.7 s | 5.4 s |
| **Whole replay** | **30.3 s / 31.6 s** | **11.2 s** |

Stage figures carry instrumentation overhead; the replay totals do not. Scaled to a
60-second segment, CPU rendering falls from about 37 seconds to about 13, against
roughly 54 seconds of GPU inference — so inference, not rendering, now sets the pace.

The cause was the output volume, not computation. On this deployment's bind-mounted
`./data` one 1080p JPEG takes 142 ms to publish against 10.6 ms on the container's own
filesystem, a 260-pixel crop 13.2 ms against 0.42 ms, and `mkdir(exist_ok=True)` on a
directory that already exists 4.1 ms against 0.01 ms. The tracker created each track's
output directory once per detection and published a full-size annotated frame on every
frame. It now creates each directory once and publishes the annotated view at
`LIVE_SNAPSHOT_FPS` (2) and at most `LIVE_SNAPSHOT_MAX_WIDTH` (1280) pixels wide — the
dashboard polls that file every two seconds, and at 1080p capture it was previously a
720p view anyway. Raise both on a fast output volume.

**Nothing about detection changed.** Replaying the same recording under the deployed
package and the new code produced the identical detection digest: 513 detections and
75 tracks, same frames, boxes, species and confidences. One deployed run lost a single
clip to a transient rename failure on the bind mount and a repeat run did not, so the
track digests then matched too.

This is a throughput measurement on this machine's storage. On a fast output volume the
rendering cost was never the constraint and the saving would be smaller.

## Cost and verification

Nine isolated GPU passes used 273.28 seconds of aggregate inference-process time.
Existing footage, model assets and containers were reused; no model/dataset download,
live-camera recording, production job or paid API request was used for the experiment.
The first 2-FPS run took 64.6 seconds because of a cold start and must not be used to
claim a speedup at 4 FPS. Warm 16-second runs took 21.6–22.5 seconds at 720p and
22.5–24.7 seconds at 1080p, including per-segment startup.

The complete saved recording took 47.46 seconds to infer at 4 FPS / 1080p.
60-second live segments and the 180-second detector timeout are retained. GPU load,
rendering and storage cost increase; busy scenes can reduce the remaining headroom.
Full cold starts and other GPU jobs may still cause delayed or dropped segments.

Full suite: **445 passed, 2 skipped** (local FFmpeg unavailable), including three new
tests covering the annotated view's publish rate, its width cap and an unscaled small
feed. Container FFmpeg checks verify 4 FPS, 640×360 remaining 640×360, 1080p remaining
1080p and 1440p reducing to 1080p. The rebuilt live worker carries the motion/identity
association fix, the GPU/CPU segment overlap and the rendering changes above; the running
containers were confirmed to report 4 FPS, a 0.60 floor, 1920×1080 capture and a 2 FPS
annotated view after the rebuild.

Evidence is under `data/outputs/live-recall-4fps/` (untracked): input and pipeline hashes
in `manifest.json`, commands/timing in `journal.json`, per-run CSVs, `report.json`, and
the isolated runtime replay. The benchmark script is `scripts/live_recall_benchmark.py`.
Completed runs are cached and its journal enforces the existing ten-pass limit.

Throughput evidence sits beside it: `validate_runtime.py` (capture check and replay),
`profile_runtime.py` (per-stage timings, `profile.json`), `profile_io.py` (bind mount
versus container filesystem, `profile-io.json`), `profile_crop.py` (`profile-crop.json`,
showing the image work is under 7 ms) and `replay_digest.py`, whose `digest-*.json`
record the detection digests that must match across code versions. Run any of them
against the working tree with `-e PYTHONPATH=/app` and the source mounted at
`/app/app`; without it a script run by path imports the image's installed package.
