# Murky and dark water from public data — 2026-10-11

**Keep production unchanged.** No production configuration, `.env`, job output, upload,
review decision or the existing SmartBay CVAT task changed.

**Part 1, tracker.** No setting passed the promotion gate at either frame rate. The only
lead is a looser 4 FPS setting under live mode's 2 s segments, where it found 2.3×
production's true boxes. It still failed the gate on MOTA, false positives and ID
switches.

**Part 2, detector.** On real Brackish test water, a YOLO11s fine-tuned on Brackish
(run A) found far more fish than production VIAME on the same frames: recall 0.53 vs
0.31, precision 0.81 vs 0.71. Adding synthetic murk (run B) made real-water results
worse. Run A **did not transfer** to the SmartBay clips: zero agreement with VIAME, and
the contact sheet shows it boxing seagrass while VIAME boxes the visible fish.

**Part 3, Megalodon.** On the ONC cameras it boxed a static object in every lit frame,
and nothing in the black clip. It adds nothing useful for fish.

Every accuracy number below was measured on **public** labelled data: BrackishMOT and
Brackish, filmed in Limfjorden in Denmark. Those numbers describe that water, not
SmartBay's. On SmartBay and ONC footage, which have no labels here, this report gives
only agreement, disagreement and temporal proxies.

## Budget actually used

- **Paid APIs: zero calls.** Fishial stayed disabled (`FISHIAL_ENABLED=false` in every
  container run; no host script calls it).
- **Downloads.** All but the last were on the approved list:
  - BrackishMOT Kaggle version 4: 1,962,308,947 bytes, SHA256
    `e717dc1438aab65eaccb59b135b58f50ca7af09043ef7effbf9fd2825410c3ff`. No
    authentication was needed.
  - FathomNet Megalodon `mbari-megalodon-yolov8x.pt` at commit
    `f312604d2f59dcccde74499cf0e0ae0647331f01`: 136,714,110 bytes, SHA256
    `24fe6d9808fdd44b64d2fbfc23bbe6455520eb7b392d3664c92cbcdf88a09196`.
  - `yolo11s.pt`: 19,313,732 bytes, SHA256
    `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
  - CUDA PyTorch 2.11.0+cu128 (lists `sm_120`) and the `.[train]` extra into `.venv`.
  - `lap` 0.5.13 (BSD-2-Clause, wheel `cp312-win_amd64`; `_lapjv` SHA256
    `9c67b31ebe61e0f5c42e125e92e1864b164c429d1cfa1cb4281721f192019ebd`) into `.venv`.
    You approved this one mid-task. Ultralytics tracking (`fish_model.py track`)
    imports it, `.[train]` does not include it, and offline mode blocks Ultralytics'
    own auto-install.

  The Brackish v4 archive was already on disk. Ultralytics' AMP self-check tries to
  fetch `yolo26n.pt`, which is not approved. Every Ultralytics process ran with
  `YOLO_OFFLINE=true` and an unreachable proxy, so nothing was fetched: the check was
  skipped and AMP stayed on.
- **VIAME GPU: 7,701 s of the 10,800 s limit** (Part 1 6,736 s, Part 2 695 s, Part 3
  270 s). These are pass wall times from the three resumable journals. An interrupted
  pass is charged its elapsed time.
- **Training: 12,296 s of the 14,400 s limit.** Run A took 2,156 s and run B 2,606 s.
  Another 7,533 s went on aborted starts:
  - 50 epochs projected far past the budget;
  - host memory ran out next to the CPU sweep;
  - dataloader workers thrashed the page file;
  - the laptop was thermally throttled.

  Each abort is recorded with its reason in `part2/training-journal.json`. The
  two-run comparison therefore uses **10 epochs**, not the requested ~50. Both runs used
  the same seed (0), image size (640) and batch size (16). With 10 epochs, Ultralytics'
  default `close_mosaic=10` keeps mosaic off for the whole run, identically for A and B.
  Run B saw twice as many images per epoch, so it trained for about twice as many steps.
- CVAT's containers were paused, at your request, while training ran, and restarted
  afterwards. No other container was stopped, rebuilt or restarted.

All generated data is under `data/outputs/murky-water-20261011/` (gitignored), with
`RESUME.md` there giving the resume order. Datasets and models are under
`data/training/` and `data/models/`.

## Facts verified before any scoring

- **Production batch profile** (taken from the running worker environment,
  `docker-compose.gpu.yml`, `app/config.py` and the `.pipe` block):
  - DEIMv2 1024 plus ByteTrack through `tracker_fish_no_motion.pipe`, at 10 FPS;
  - detector score 0.10;
  - ByteTrack high/low/new-track 0.45/0.10/0.50, `track_buffer` 30;
  - `VIAME_FRAME_NUMBER_OFFSET=-1`.

  **Live mode** uses 4 FPS, 1080p capture and `MIN_FISH_CONFIDENCE` 0.60. VIAME runs
  **once per 2 s camera segment** (`app/workers/live_worker.py`), so the tracker
  restarts every 8 frames and `live_monitor` joins the segments.
- **Installed ByteTrack** (`viame/core/bytetrack_tracker.py`):
  - Detections at or above `high_thresh` go to the first association pass.
    Detections with `low_thresh <= score < high_thresh` go to a second pass that can
    only extend tracks that are currently tracked, not lost ones.
  - Only first-pass detections at or above `new_track_thresh` start tracks.
  - A new track must be matched again on the next frame or it is discarded, except on
    a run's first frame.
  - `match_thresh`, `second_match_thresh` and `unconfirmed_match_thresh` are IoU
    **distance** bounds (1 − IoU). Production's 0.98 already accepts IoU ≥ 0.02, and a
    higher number is looser, so they were not swept.
  - `track_buffer` counts sampled tracker calls: 30 is 3 s at 10 FPS and 7.5 s at
    4 FPS.
  - All of these are read in `set_configuration`, so they are valid runtime overrides.
  - The detector's own 0.10 score is the floor for `low_thresh`.
- **Tracker replay is exact.** VIAME's `detected_object_input` reader replays saved
  detections by image file name. On three BrackishMOT sequences at both profiles, the
  production pipeline (image list, downsampler, detector, tracker) matched the
  detector-only pass followed by `scripts/viame_pipelines/bytetrack_replay.pipe`. Track
  IDs, boxes, scores and averaged class scores were identical. The sweep therefore
  replays VIAME's own tracker; no other ByteTrack was used.
  - A repeated detector pass gave identical output: 444 frames, 38,401 rows.
  - The 0.05-score detection pass, filtered at 0.10, equals the production 0.10 pass.
- **Analysed frames** come from VIAME's own downsampler
  (`scripts/viame_pipelines/downsampler_probe.pipe`). At 15 FPS input it passes positions
  0,1,3,4,6,… for 10 FPS and 0,3,7,11,15,… for 4 FPS. Only those frames are scored, so
  frames the detector never saw do not count as misses.
- **Frame numbering.** Image-list track CSVs number rows by one-based list position and
  name the image in column 2. Production video CSVs use one-based source frames, which
  `VIAME_FRAME_NUMBER_OFFSET=-1` makes zero-based. `viame_tracks_to_mot` handles both and
  rejects any row whose position and image name disagree. Across all 98 sequences,
  matched boxes peak at shift 0: 4,113 at IoU 0.7, against 2,983 at −1 and 2,876 at +1.
- **BrackishMOT package.** It has 98 real sequences (78 in `train`, 20 in `test`), all
  1920×1080 at 15 FPS, with no licence or readme inside the package. 64 sequences
  contain fish (27,141 fish boxes) and 34 contain none.
  - `brackishMOT-54` ships 200 images although `seqinfo.ini` says 610.
  - Two rows in `brackishMOT-92` have identity −1. They are excluded and listed in the
    import report.
- **Brackish v4 empty frames are not verified empty.** The package's own script writes an
  empty label file for every frame without detections. 4,089 of 15,084 frames are such
  placeholders: 3,425 train, 135 validation and 529 test. They are excluded from mAP
  and false positives, and predictions on them are reported separately.
- **Laptop.** The GPU is an RTX 5060 Laptop (8 GB) and the host has 31.7 GB of RAM, of
  which the Docker VM holds about 6–7 GB.
  - The GPU runs hot (85–93 °C). Once, the CPU was throttled to about 25% of base speed
    and training ran 6× slower until the power mode and cooling were changed.
  - One VIAME pass hit ONNX Runtime `CUDA failure 999` under host memory pressure. It
    was retried.
  - On Windows, each Ultralytics dataloader worker holds about 1 GB of host memory, and
    validation gets twice the training count by default. `fish_model.py train` now
    takes `--val-workers` (this run used 4 training and 2 validation workers).

## Part 1 — VIAME ByteTrack on BrackishMOT (detector unchanged)

**Split.** The 98 sequences were split by stable hash: 49 for tuning and 49 held out,
with 32 fish-bearing sequences on each side. Round one used 25 tuning sequences.

**Configurations.** Round one tried twelve configurations: four threshold sets times
three track lifetimes. The lifetimes were 1.5 s, 3 s and 7.5 s, converted to tracker
calls per profile.

| Set | high | new track | low | detector score |
|---|---|---|---|---|
| A (production thresholds) | 0.45 | 0.50 | 0.10 | 0.10 |
| B | 0.35 | 0.40 | 0.10 | 0.10 |
| C | 0.25 | 0.30 | 0.10 | 0.10 |
| D | 0.35 | 0.40 | 0.05 | 0.05 |

**Selection rule.** The ranking first keeps candidates that do not lower MOTA and do not
raise false positives by more than 10%, then orders them by IDF1. The top three went to
the full tuning set, and the best of those to the held-out set.

**Metrics.** IDF1 is TrackEval-style, using a global minimum-cost identity assignment.
MOTA and ID switches use CLEAR MOT with per-frame Hungarian matching that prefers
continued pairs. IoU is 0.5, and only analysed frames are scored. HOTA was not computed
because scipy and TrackEval are not installed. The earlier greedy counts remain in each
report as a diagnostic.

### What limits tracking is the detector

On held-out sequences at 10 FPS, production found **1,022 of 9,134** fish boxes on
analysed frames, a recall of about 11%, with 817 false positives. A tracker can only
link the boxes it is given, so absolute IDF1 values here are low for every setting.

### 10 FPS (batch profile)

| Stage (sequences) | Setting | IDF1 | MOTA | FP | TP | ID sw. per GT track |
|---|---|---|---|---|---|---|
| Round 1 (25) | production | 0.464 | 0.167 | 444 | 786 | 0.082 |
| Round 1 (25) | B, 1.5 s (best IDF1) | 0.465 | 0.073 | 736 | 891 | 0.131 |
| Round 2 (49) | production | 0.239 | 0.068 | 1,188 | 1,868 | 0.344 |
| Round 2 (49) | A, 1.5 s | 0.241 | 0.067 | 1,187 | 1,856 | 0.325 |
| Round 2 (49) | B, 1.5 s | 0.334 | 0.157 | 2,173 | 3,783 | 0.943 |
| Round 2 (49) | **A, 7.5 s (selected)** | 0.239 | 0.068 | 1,188 | 1,868 | 0.344 |
| **Held-out (49)** | production | 0.138 | 0.017 | 817 | 1,022 | 0.178 |
| **Held-out (49)** | A, 7.5 s | 0.138 | 0.017 | 817 | 1,022 | 0.178 |

Neither round-two alternative met the constraints. A at 1.5 s lowered MOTA very slightly,
and B at 1.5 s nearly doubled false positives. The only compliant finalist, A at 7.5 s,
behaves **identically** to production on held-out data, with a gain on 0 of 32
fish-bearing sequences. It also produced the same tracks on SmartBay.

### 4 FPS (live profile)

| Stage (sequences) | Setting | IDF1 | MOTA | FP | TP | ID sw. per GT track |
|---|---|---|---|---|---|---|
| Round 1 (25) | production | 0.430 | 0.113 | 184 | 277 | 0.033 |
| Round 1 (25) | B, 3 s (highest IDF1) | 0.485 | 0.144 | 235 | 355 | 0.067 |
| Round 2 (49) | production | 0.188 | 0.030 | 400 | 536 | 0.129 |
| Round 2 (49) | **B, 3 s (selected)** | 0.302 | 0.107 | 768 | 1,317 | 0.771 |
| **Held-out (49)** | production | 0.116 | 0.025 | 215 | 316 | 0.035 |
| **Held-out (49)** | B, 3 s | 0.172 | −0.020 | 625 | 630 | 0.304 |

All three round-two finalists broke a constraint. The two A variants lowered MOTA by
0.0006, and B raised false positives by 92%. The protocol therefore advanced the
highest-IDF1 finalist. 4 FPS lifetimes are 6, 12 and 30 tracker calls, so "B, 3 s" is
`track_buffer` 12.

### Promotion gate (held-out; SmartBay proxies on the three fixed clips)

| Criterion | 10 FPS: A 7.5 s | 4 FPS: B 3 s |
|---|---|---|
| IDF1 +2 points | **fail** (+0.0) | pass (+5.6) |
| MOTA does not fall | pass (equal) | **fail** (0.025 → −0.020) |
| FP rise ≤ 10% | pass (0%) | **fail** (+191%) |
| ID switches per GT track do not rise | pass (equal) | **fail** (0.035 → 0.304) |
| Gain on a majority of sequences | **fail** (0/32) | **fail** (11/32) |
| SmartBay proxies within 5% | pass (identical) | pass (fragmentation −18.6%, others equal) |
| SmartBay acceptance stable at 0.55/0.60/0.65 | pass | pass (38/37/29 vs 35/34/28 accepted) |
| Runtime ≤ 1.1× | pass on paper (0.51×; see note) | pass on paper (0.68×; see note) |
| **Promote** | **no** | **no** |

On SmartBay at 4 FPS, candidate B made 43 tracks against production's 35 from the
same 2,070 detections. That is more tracks, not more detections; with no labels, it
cannot be called better or worse.

Runtime could not be judged at the 10% level on this laptop. Re-running the same
production pass on the same clip took 181 s the first time and 80 s the second, with
identical track rows (the files differ only in VIAME's export-time header). The gate
result is recorded in `part1/journal.json` under `decisions.gate`, with `promote: false`. The candidate
runs were faster than baseline (ratios 0.51 and 0.68), and the sweep changes only tracker
thresholds, which cost nothing measurable next to the detector.

**Best candidates if a future check wants them.** At 4 FPS, B at 3 s (high 0.35, new
track 0.40, low 0.10, buffer 12 calls) gave +5.6 IDF1 points on held-out data. It cost
2.9× the false positives and 8.7× the ID switches per track, and it helped only 11 of 32
sequences. At 10 FPS, the same thresholds with a 1.5 s buffer, run outside the protocol,
gave held-out IDF1 0.196 vs 0.138. MOTA went from 0.017 to −0.005, FP from 817 to 1,702,
and ID switches per track from 0.178 to 0.367, with a gain on 11 of 32 sequences. Neither
is a safe default: each trades many more false and switched tracks for more found fish.

### Supplementary: live mode's 2 s segments (boxes only, not a gate criterion)

Live mode runs VIAME once per 2 s camera segment, so the tracker restarts every 8 frames
at 4 FPS. Each held-out sequence was replayed as independent 2 s segments and scored on
boxes only (CLEAR TP/FP/FN at IoU 0.5, analysed frames only). Track identities are not
comparable across segments, so this check gives no IDF1.

| Mode | Setting | TP | FP | FN | Recall | Precision |
|---|---|---|---|---|---|---|
| Continuous | production | 316 | 215 | 3,339 | 0.086 | 0.595 |
| Continuous | B, 3 s | 630 | 625 | 3,025 | 0.172 | 0.502 |
| 2 s segments (as live runs) | production | 228 | 132 | 3,427 | 0.062 | 0.633 |
| 2 s segments (as live runs) | B, 3 s | 528 | 352 | 3,127 | 0.144 | 0.600 |

The restarts cost production 28% of its true positives: a new track needs a match on the
next frame, and short segments leave little room for that. Under segmenting, candidate B
finds 2.3× production's true boxes, with precision falling only from 0.63 to 0.60. False
boxes still rise 2.7×.

This is the most promising tracker lead for live mode. It still failed the held-out gate
on MOTA, false positives and ID switches, so it was not promoted. A labelled SmartBay
check set would decide whether the extra boxes are fish.

## Part 2 — YOLO fine-tuned on Brackish

**Data.** The full Brackish v4 import has 15,084 frames from 89 recordings (75 train, 4
validation and 10 test recordings, kept whole by stable hash) and 13,814 fish boxes.
Labels sit in a 960×540 space; the AVIs are 1920×1080.

**Murk copies.** Run B added one degraded copy of each of the 13,015 train frames, from
`scripts/murk_augmentation.py`, seed 20261011. Each copy got:

- brightness ×0.45–0.90 and contrast ×0.60–0.95;
- a tint of B ×0.90–1.10, G ×0.95–1.15 and R ×0.60–0.90;
- backscatter `I = J·t + B·(1 − t)`, with transmission 0.55–0.90 plus a ±0.10 vertical
  gradient and a veil colour of B 60–130, G 90–160, R 30–90;
- Gaussian blur σ 0–1.2 and noise σ 2–6.

Only the train split was copied, which `combined_data_yaml` enforces. Average brightness
fell from about 118 to about 88 on a 0–255 grey scale. The contact sheet is
`data/training/brackish-fish-v4-murk/contact-sheet.jpg`.

**Synthetic dark test.** Seed 7 produced a darkened copy of the 1,500 test frames, using
ranges disjoint from training:

- brightness ×0.25–0.45 and contrast ×0.95–1.00;
- a tint of B ×1.10–1.25, G ×0.80–0.95 and R ×0.35–0.60;
- transmission 0.90–0.98 and a veil of B 20–60, G 40–90, R 5–30;
- blur σ 1.2–2.2 and noise σ 6–12.

That gives an average brightness of about 37 against about 130 in the real test frames:
**a harsh synthetic stress test, not real water.**

**Runs.** Both runs started from COCO `yolo11s.pt`, at 640 px, seed 0, batch 16, 10
epochs, with deterministic mode and AMP on. They used 4 training and 2 validation
dataloader workers. Best weights were chosen by Ultralytics' validation fitness on the 569
validation frames.

| Run | Train images | Wall time | Best validation mAP50 / mAP50-95 | `best.pt` SHA256 |
|---|---|---|---|---|
| A (default augmentation) | 13,015 | 2,156 s | 0.639 / 0.376 | `ddbcbc28…e7b4` |
| B (+ murk copies) | 26,030 | 2,606 s | 0.572 / 0.310 | `8f78637f…21a1` |

**Choosing the better model.** The rule was set before any test scoring: the higher
mAP50-95 on the **real** Brackish test split wins. That is run A.

### VIAME vs A vs B on the Brackish test split (real labels)

**Alignment proof first.** VIAME ran its production batch profile (10 FPS) on the 10 test
recordings' original 1920×1080 AVIs, and its boxes were scaled ×0.5 into the label space.
Matched boxes at IoU 0.5 peak at a frame shift of 0: 205, against 192 at −1, 193 at +1,
143 at −2 and 151 at +2. Neighbouring frames match nearly as well because the fish move
slowly. The overlays in `part2/viame-test-alignment.jpg` show truth and VIAME boxes on the
same fish.

**Same frames, same truth.** These are VIAME's 643 analysed test frames with verified
labels: 669 fish boxes, at IoU 0.5. The YOLO counts use confidence ≥ 0.25. VIAME's
"tracked" boxes are every box on a track; "accepted" keeps tracks whose best box reaches
production's 0.60.

| Model | Recall | Precision | TP | FP | FN | Boxes on unverified-empty frames |
|---|---|---|---|---|---|---|
| VIAME production, tracked | 0.306 | 0.709 | 205 | 84 | 464 | 40 |
| VIAME production, accepted (≥ 0.60) | 0.272 | 0.700 | 182 | 78 | 487 | 40 |
| **YOLO A** | **0.532** | **0.805** | 356 | 86 | 313 | 11 |
| YOLO B | 0.297 | 0.751 | 199 | 66 | 470 | 10 |

There were 356 unverified-empty frames among VIAME's analysed frames. They are left out
of these counts.

**All test frames** (`fish_model.py evaluate`; 971 frames with verified labels, 1,012
fish boxes; the 529 unverified-empty frames are excluded):

| Model | mAP50 | mAP50-95 | Recall @0.25 | Precision @0.25 | Dark test mAP50 | Dark test mAP50-95 | Dark test recall / precision |
|---|---|---|---|---|---|---|---|
| A | **0.615** | **0.363** | **0.525** | **0.809** | 0.078 | 0.032 | 0.002 / 0.667 |
| B | 0.454 | 0.280 | 0.292 | 0.762 | **0.296** | **0.139** | **0.153 / 0.838** |

**What this says (about Brackish water only):**

- VIAME leaves a lot of headroom here. On the same frames, run A finds 1.7× as many
  fish (recall 0.53 vs 0.31) at higher precision (0.81 vs 0.71).
- The murk copies **hurt** on real Brackish water: −16 mAP50 points, and recall fell
  from 0.53 to 0.30.
- The murk copies **help** on the synthetic dark copy, where A collapses (recall 0.002)
  and B keeps a little (recall 0.15).
- In this budget, simulated murk is not a substitute for real murky footage. One
  likely reason is that B spent half its steps on synthetic frames that look unlike the
  real test water. With 10 epochs instead of ~50, both runs are under-trained; the
  comparison is fair between them, but the absolute numbers would probably rise with
  longer training.

### Run A on the fixed SmartBay clips (no labels: agreement and proxies only)

Run A tracked the three fixed SmartBay benchmark clips with `fish_model.py track`, with
stride 3 (about 10 FPS from the 29.9 FPS clips), confidence 0.25 and Ultralytics
ByteTrack. Its boxes were compared with production VIAME's batch-profile tracks from
Part 1. Boxes count as agreeing when they share a timestamp within 0.02 s and overlap at
IoU ≥ 0.5. Frames that only one model analysed count as unaligned, not as disagreement.

| Clip | VIAME boxes (aligned) | YOLO A boxes (aligned) | Matched | VIAME tracks | YOLO tracks | YOLO singleton tracks | YOLO fragmentation |
|---|---|---|---|---|---|---|---|
| 0 | 456 (290) | 304 (294) | **0** | 7 | 71 | 21% | 1.4% |
| 1 | 860 (663) | 683 (495) | **0** | 13 | 126 | 14% | 7.1% |
| 2 | 885 (6) | 790 (22) | **0** | 21 | 148 | 16% | 7.4% |

VIAME showed no singleton tracks and no fragmentation on any clip. Agreement was **zero
on all three clips**.

Both models use the same 1920×1080 pixel space, and the agreement code had already been
validated in Part 1. The disagreement contact sheet, `part2/smartbay/only-one-model.jpg`,
shows what each model is boxing:

- VIAME's boxes sit on fish visibly swimming above the seagrass, including a pipefish.
- Run A's boxes sit on patches of seagrass near the bottom of the frame, and it misses
  those fish.

Run A also makes 5–10× as many tracks, with many single-frame tracks. That is a visual
observation of six frames plus temporal proxies, not a measured precision or recall.

On these clips, the Brackish-trained detector **does not transfer**: its strong showing
on Brackish's own water says nothing about this scene. **Do not swap it in.** Clip 2
aligned on few frames because stride 3 drifts against VIAME's 10 FPS downsampler; the
other clips had 290 and 663 aligned VIAME boxes, none matched.

If you want reviewers to see run A's boxes as suggestions, create a **separate** CVAT
candidate task; do not run it as part of this experiment. CVAT dev mode is configured. First
confirm the upload's video ID in the app. The clips are cut from upload
`3a4a98f7-55db-4355-882a-d1e6d140b4d5.mov`, so the candidates must be made on the whole
upload to keep its frame numbers:

```powershell
$env:YOLO_OFFLINE='true'
.venv\Scripts\python.exe scripts/fish_model.py track data\models\murky-water-20261011\A\fish-20261011T210326Z\weights\best.pt data\uploads\3a4a98f7-55db-4355-882a-d1e6d140b4d5.mov data\outputs\murky-water-20261011\part2\cvat-candidate-A --conf .25
docker compose exec api python scripts/external_fish.py cvat-suggestions 3a4a98f7-55db-4355-882a-d1e6d140b4d5 brackish-yolo-A /data/outputs/murky-water-20261011/part2/cvat-candidate-A/tracks.csv
```

### Tracking on BrackishMOT (leakage-guarded, real labels)

BrackishMOT was cut from the same 2019 recordings as the detection archive, so A and B
were scored only on the **17 sequences the leakage guard allows**:

- 10 cut from Brackish **test** recordings;
- 7 that match no archive recording.

Excluded were 54 sequences cut from training or validation recordings and 27 of
undetermined origin. Each system saw VIAME's analysed 10 FPS frames in order, and scoring
is at IoU 0.5 on those frames only. The YOLO rows use Ultralytics' default ByteTrack, so
this compares detector-plus-tracker systems, not detectors alone.

| System | IDF1 | IDP | IDR | MOTA | TP | FP | FN | ID sw. per GT track |
|---|---|---|---|---|---|---|---|---|
| VIAME production | 0.209 | 0.480 | 0.134 | 0.110 | 1,098 | 427 | 4,370 | 0.36 |
| YOLO A + ByteTrack | **0.346** | 0.487 | **0.268** | **0.308** | **2,511** | 498 | 2,957 | 1.64 |
| YOLO B + ByteTrack | 0.266 | **0.593** | 0.172 | 0.194 | 1,385 | **198** | 4,083 | 0.63 |

**On Brackish water, run A plus a stock tracker beats production:**

- 2.3× the true boxes, for 17% more false ones;
- IDF1 +13.7 points and MOTA +19.8 points;
- but 4.6× the ID switches per track, a tracker-tuning matter.

As with the detection table, this is evidence about **Brackish water only**, and it does
not carry over to the SmartBay clips (see above).

## Part 3 — FathomNet Megalodon on ONC footage

A read-only query (`SET default_transaction_read_only = on`) found two library
recordings from ONC live sessions. Both were cut into identical lossless FFV1 clips, and
Megalodon and production VIAME (live profile, 4 FPS, every frame analysed) ran on the
same frames.

| Clip | Frames | Median grey (0–255) | VIAME tracked boxes | Megalodon boxes ≥ 0.25 | Matched |
|---|---|---|---|---|---|
| `onc-folger-pinnacle` (lit, first 120 s) | 480 | 95 | 0 | 488 (on all 480 frames) | 0 |
| `onc-tempo-mini` (black, whole 30 s) | 120 | 1.6 | 0 | 0 | 0 |

On Folger Pinnacle, Megalodon's boxes are almost all one stationary sponge-like object,
boxed on every frame; 476 boxes remain at confidence ≥ 0.50. Megalodon has a single
generic `object` class, so these are **not fish detections**. The output is labelled
`object` and never fish. On the black Tempo-mini clip, neither model finds anything,
and no detector can recover content from a frame with a median grey level of 1.6.
Megalodon therefore adds nothing useful for fish on these cameras as it stands. Its
candidates could at most become review suggestions in a **separate** CVAT candidate
task.

The disagreement contact sheet is `part3/disagreements.jpg`, and the full comparison is
`part3/comparison.json`. Megalodon ran at its checkpoint's 640 px training size, after
its SHA256 was verified and with the pickle loaded only from that verified path. The
pickle's one non-standard reference (`dill._dill._load_type('set')`) resolves through
an allow-listed stand-in, because `dill` is not installed.

## What changed in production

Nothing changed in production. The Part 1 gate failed at both FPS profiles, so
`app/config.py`, `.env.example`, `docker-compose.gpu.yml`, the `.pipe` block and the
README are untouched. Parts 2 and 3 were evaluation-only by design.

Code changes are tooling and tests only:

- the BrackishMOT and Megalodon commands in `scripts/external_fish.py`;
- IDF1 and CLEAR MOT, the VIAME→MOT converter and the leakage guard in
  `app/services/external_datasets.py`;
- the replay pipes and the three experiment drivers;
- murk augmentation;
- `fish_model.py` options:
  - train: `--seed`, `--workers`, `--val-workers`, `--extra-train`,
    `--extra-train-fraction`;
  - evaluate: `--save-predictions`, `--workers`;
  - track: `--stride`;
- a `set +u` fix in `scripts/run_viame.sh`. Sourcing VIAME's setup script failed under
  `set -u`, which broke the README's manual command and the plain-video mode of
  `viame_review_experiment.py`. The application does not use this script.

`.venv` gained `lap` (see Budget). `fish_model.py track` cannot run offline without it.
Adding `lap>=0.5.12` to the `train` extra in `pyproject.toml` would make that explicit;
this task did not change the dependency list.

## Tests

- **Focused:** `tests/test_murky_water.py` and `tests/test_murky_water_detectors.py`, 33
  tests, all passing. They cover:
  - IDF1 on hand-built cases, including an ID swap and a fragmented track;
  - CLEAR MOT;
  - VIAME CSV → MOT alignment at native and downsampled FPS, with scoring of analysed
    frames only;
  - the sequence → video frame-count assertion;
  - seeded, train-only murk copies and the disjoint dark-test ranges;
  - the leakage guard and temporal overlap alignment;
  - `fetch-megalodon` hash verification and cleanup, with a mocked download;
  - the refusal to load unverified Megalodon weights;
  - Megalodon rows to `object` candidates;
  - the `dill` allow-list;
  - inference of analysed frames;
  - the validation-worker cap.
- **Full suite:** 528 passed, 2 skipped and 1 failed. The failure,
  `tests/test_external_fish.py::test_cvat_suggestions_use_separate_candidate_task`, was
  already failing before this work. Its fake `Gateway.import_tracks` does not accept the
  `break_long_gaps` argument that `scripts/import_model_suggestions.py` passes.

## Licences

- **BrackishMOT** and **Brackish v4**: CC BY-SA 4.0 per Kaggle metadata. The BrackishMOT
  package contains no licence file of its own. ShareAlike applies to adapted datasets,
  including the murk copies.
- **Ultralytics** code and `yolo11s.pt`: AGPL-3.0, or an Ultralytics Enterprise licence.
  Runs A and B are AGPL-encumbered. For a commercial deployment, retrain an Apache-2.0
  detector instead, such as **RF-DETR** or **D-FINE**.
- **FathomNet Megalodon** weights: CC BY 4.0 (model card metadata at the pinned commit).
  Running them uses Ultralytics (AGPL-3.0).
- **Ocean Networks Canada** video: CC BY-NC-SA 3.0, non-commercial.
- **VIAME** (BSD-3-Clause) and its installed DEIMv2 fish model are unchanged.

## Reproduce

PowerShell, from the repository root, with Docker Desktop running:

```powershell
$env:FISHIAL_PREPROCESS='none'
$run = 'docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps -T --volume "${PWD}:/app:ro" -e PYTHONPATH=/app -e FISHIAL_ENABLED=false --entrypoint python3 worker'
# Part 1 (VIAME worker image)
foreach ($stage in 'detect','validate','sweep','supplementary','segments','smartbay','gate') {
  Invoke-Expression "$run /app/scripts/murky_water_tracker.py $stage"
}
# Part 2
Invoke-Expression "$run /app/scripts/murky_water_detector.py viame-test"
.venv\Scripts\python.exe -m scripts.murky_water_detector alignment
.venv\Scripts\python.exe -m scripts.murky_water_detector train --epochs 10 --runs A,B
.venv\Scripts\python.exe -m scripts.murky_water_detector evaluate
.venv\Scripts\python.exe -m scripts.murky_water_detector smartbay --model BEST_PT --name RUN
.venv\Scripts\python.exe -m scripts.murky_water_detector brackishmot
# Part 3
.venv\Scripts\python.exe -m scripts.murky_water_onc find
.venv\Scripts\python.exe -m scripts.murky_water_onc clips
Invoke-Expression "$run /app/scripts/murky_water_onc.py viame"
.venv\Scripts\python.exe -m scripts.murky_water_onc megalodon
.venv\Scripts\python.exe -m scripts.murky_water_onc compare
```

Every stage resumes from its journal and skips completed passes.
