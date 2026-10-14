# SAM 3 auto-labelling, BoT-SORT and static-object suppression — 2026-10-14

**Status: stopped at the review gate (end of Part 2).** SAM 3's labels and the
side-by-side video are ready for your verdict. Part 3 (the student detector) waits for
your go.

**Production is unchanged.** No production setting, `.env`, job output, upload, review
decision or CVAT task changed. Nothing was committed, rebuilt or restarted.

**Where the real numbers come from.** Real labelled numbers here come from BrackishMOT
only (Limfjorden, Denmark). For SmartBay there are no labels: this report gives
agreement, disagreement and temporal proxies, never precision or recall.

## Results so far

- **SAM 3 as the labeller.** It boxes most of what VIAME tracks and much more:
  - SAM 3's labels match **81% of VIAME's stored boxes** (IoU ≥ 0.5, frames both
    analysed).
  - They add **12,769 boxes VIAME does not have**. In a random sample most of those are
    faint fish in the murk; a noticeable minority are doubtful (empty water, a dark
    corner, a box between two fish).
  - The labels are 700 tracks and 23,515 boxes, with no duplicate or single-frame tracks.
- **SAM 3 runs as a per-frame detector, with BoT-SORT for identities.** SAM 3's own video
  tracker ran at 0.29 FPS on this GPU; a pass would have used 83 of the 120 budgeted
  minutes. The detector alone runs at 2.5 FPS in FP16, flat at 2 GB of GPU memory.
- **BoT-SORT beats ByteTrack on SAM 3's boxes** (SmartBay, proxies): fragmentation 0.061
  against 0.157 at the same thresholds, with longer tracks.
- **Static suppression removed what never moves:**
  - the post between the "2.0" and "1.0" signs;
  - the base of the pillar;
  - a speck;
  - mesh patches;
  - in VIAME's raw boxes, the lamp glare.

  Swimming and resting fish stay. Switching on the background-match rule was needed for
  short fixture fragments (decision recorded before any comparison).
- **BoT-SORT on BrackishMOT (real labels):**
  - **4 FPS:** IDF1 0.173 against VIAME production's 0.116, with about half the false
    boxes and switches of VIAME's own best 4 FPS setting.
  - **Live 2 s segments:** 2.0× VIAME's true boxes at higher precision.
  - **10 FPS:** a trade, not a win.

## Budgets used

| Budget | Limit | Used |
|---|---|---|
| SAM 3 GPU | 2 h (7,200 s) | **1,585 s (26 min)** |
| Training | 2 h | 0 s (Part 3 waits for your go) |
| New VIAME passes | 30 min | 0 s (VIAME's stored result is used) |
| CPU work (not budgeted) | — | 504 s |

The SAM 3 GPU time breaks down as:

- the video predictor, stopped after 264 s;
- the detector probe, 123 s;
- the "fish" pass, 588 s;
- the "small fish" pass, 596 s.

**Paid APIs: zero calls.** Fishial was never imported.

**Downloads, all approved:**

- **The SAM 3 checkpoint**, placed by you. It landed in `C:\Users\dhess\data\models\sam3`
  because the command ran from your home folder, so I moved it and Meta's `LICENSE`
  (SAM License, 19 November 2025) into the repository. It is 3,450,062,241 bytes, SHA256
  `9999e234…8c9e`, and is verified before every load. It loads with
  `torch.load(weights_only=True)`, which confirms it holds plain tensors only.
- **Packages** (approved in chat), installed with `--no-deps` from
  `scripts/sam3-requirements.txt`:
  - `timm` 1.0.30, `safetensors` 0.8.0, `ftfy` 6.3.1, `wcwidth` 0.9.2 and `regex`
    2026.9.29, all hash-checked;
  - Ultralytics' CLIP fork at commit `7ffa84b3…dd40` (AGPL-3.0).

  Logs are in `installs/`.

The journal is `data/outputs/sam3-botsort-20261014/journal.json`. It records every step
(inputs, settings, hashes, timings, every attempt) and every decision.

## Test footage

| | |
|---|---|
| Video ID | `85ea19d4-db7b-4ed7-90a0-a8acccba2b0c` (`SmartBayCam2_2025-07-17_0001.mp4`) |
| File | `data/uploads/936e0581-d6de-42d7-8a1b-fadd9c517e01.mp4`, 29,551,911 bytes |
| SHA256 | `06af200811cf8e3ab2d2d2b34c95d5f1c5a4ae5b96399b64eaf035a6d1c500de` |
| Decoded | 1,440 frames, 12.5 FPS, 1920×1080, H.264, 115.2 s |

**The read-only database check is still pending.** Docker Desktop is off; you chose to
carry on without it. VIAME's result comes from the job's own output,
`data/jobs/10c7ddbd-…/3da0cd7e-…/viame_tracks.csv` (SHA256 `0f1ddcf8…3726`, frame
offset −1). It has the 213 tracks and 7,445 boxes you gave for the database tables. With
Docker running, `viame --viame-source db` rereads them through a session forced
read-only and records whether they equal the CSV.

## Part 0 — facts

**VIAME on this video.**

- VIAME analysed **1,152 of 1,440 frames** (10 of 12.5 FPS): 0, 2, 3, 4, 5, 7, ….
- The set is exact: the detection writer numbers and timestamps every frame it receives.
- All 7,445 track boxes lie on analysed frames after the −1 offset. Unanalysed frames
  never count as misses.
- VIAME's `exec_time` is **290 s, 2.5× slower than real time** (4.0 analysed frames per
  second).

**SAM 3 on the RTX 5060 Laptop GPU (8 GB).**

| Loader and setting | Speed | GPU memory | Verdict |
|---|---|---|---|
| `SAM3VideoSemanticPredictor` (detector + SAM 3 mask tracker), FP16, 1008 px | 0.29 FPS (3.5 s per frame) | 7.85 GB in use, flat over 70 frames | fits, far too slow: ~83 min per pass of this video |
| `SAM3SemanticPredictor` detector, boxes only, **FP16, 1008 px** | **2.7 FPS** (probe), **2.5 FPS** over the whole video | 2.0 GB allocated, 3.2 GB peak, **flat from frame 0 to 1,439** | **used** |
| same detector, FP32, 1008 px | 0.89 FPS | 6.0 GB peak | fits, 3× slower |

- **Loader.** Ultralytics 8.4.175's own SAM 3 code works once `timm` and the CLIP fork
  are installed; `facebookresearch/sam3` was not needed.
- **Input and precision.** Native input is 1008 px (14-px patches). Half precision is
  `quantize=16`. Loading takes 16–18 s.
- **Mask upscaling is skipped.** The detector path, `scripts/sam3_autolabel.py
  detect_video`, applies the same score, threshold (0.25) and NMS as Ultralytics' own
  post-processing, without building a full-size mask per box.
- **Memory does not grow with video length.** The detector keeps no state between frames:
  GPU allocation stayed at 2,014 MB, and process memory near 5.3 GB, for all 1,440
  frames. No chunking is needed. The video predictor, which does keep state, held a flat
  7.85 GB over the 70 frames it ran.
- **Thermals.** The GPU ran at 85–88 °C with software thermal slowdown; the CPU was not
  throttled (158–168% of base speed).
- **Pixel and frame alignment holds.**
  - All 5,171 probe boxes lie inside the 1920×1080 frame; the overlay is
    `sam3/probe/overlay.jpg`.
  - Boxes matching VIAME peak at frame shift 0: 447, against 334 at −1 and 383 at +1 on
    the probe frames.
  - Over the whole video the labels match VIAME 6,032 times at shift 0, against 4,310 at
    −1 and 5,055 at +1.
- **Scores** at the 0.25 floor: median 0.51, 10th percentile 0.28, 90th percentile 0.88.
  That is about 34 boxes per frame.
- **No exemplar-box variant.** In Ultralytics' SAM 3 a box prompt pools features from the
  frame it is drawn on and applies to that frame only (the video predictor's first frame).
  It cannot stand for "fish like this" across a video.

**BoT-SORT in Ultralytics 8.4.175 (AGPL-3.0).**

- Config keys:
  - `track_high_thresh` .25, `track_low_thresh` .10, `new_track_thresh` .25;
  - `track_buffer` 30, `match_thresh` .8, `fuse_score` true;
  - `gmc_method` sparseOptFlow;
  - `proximity_thresh` .5, `appearance_thresh` .8;
  - `with_reid` false, `model` auto.
- `track_buffer` counts tracker calls, and a new track must match on the next call, both
  as in VIAME.
- `match_thresh` 0.8 needs IoU ≥ 0.2 between frames; VIAME production's 0.98 needs 0.02.
- **ReID is off.** `model: auto` needs a YOLO backbone's features, and any other ReID
  model is an unapproved download.
- **GMC is off:** a fixed camera (see BrackishMOT).
- **External boxes** go in through `app/services/box_tracking.py`. It reports each track's
  own detector box and emits a late track's first box once the track is confirmed, as
  VIAME's CSV does.

## Part 1 — static-object suppression

`app/services/static_filter.py` works on any detector's tracks or boxes. A track is
removed if any of these holds:

- **Still track:** it spans at least 20 s, has at most 2 observations more than half a
  median-box diagonal from its median centre, and its boxes have a median IoU of at
  least 0.6 with its own median box.
- **Static zone:** a box shape recurs (IoU ≥ 0.6 with the zone's box) on at least 25% of
  analysed frames, for at least 20 s, in at least 6 of 10 time bins. A track is removed
  when at least 80% of its boxes are in zones and it never moves. An untracked box in a
  zone is removed.
- **Background match:** a still track of 3 s or more whose crops correlate at 0.9 or more
  with the per-pixel median of 60 frames. Fish drop out of that median image.

**Why fish survive.**

- Three excursions keep a whole track, so a fish that rests and then swims away is kept.
- A few seconds' rest is far below 20 s.
- A resting fish is not in the median background, so it doesn't match it.

**One change, made before any comparison.** The background rule only flagged until the
static report on SAM 3's "fish" tracks; it is recorded as decision
`static_background_removes`.

- The still-track and zone rules had left fixture fragments shorter than 20 s, which
  recur on under 25% of frames.
- `static/sam3-fish/still-tracks-vs-background.jpg` shows every still track of 3 s or
  more, with its frame crop beside the background crop:
  - everything correlating 0.93–0.99 is a fixture (the speck, the post junction, mesh);
  - every visible fish, including two resting 6–7 s, correlates at most 0.80.
- The 0.9 threshold is unchanged.

| Detector | Boxes in | Removed | Tracks removed | Rules that fired | Contact sheet |
|---|---|---|---|---|---|
| VIAME stored tracks | 7,445 | 0 | 0 of 213 | none | `static/viame/contact-sheet.jpg` |
| VIAME raw detections (≥ 0.10, untracked) | 40,913 | 3,048 | — | 5 zones | `static/viame-raw/contact-sheet.jpg`, `zones-on-background.jpg` |
| SAM 3 "fish" (BoT-SORT tracks) | 26,051 | 914 | 10 of 911 | still 2, zone 3, background 10 | `static/sam3-fish/contact-sheet.jpg` |
| SAM 3 "small fish" | 28,054 | 1,309 | 10 of 982 | still 3, zone 3, background 10 | `static/sam3-small-fish/contact-sheet.jpg` |

Paths are under `data/outputs/sam3-botsort-20261014/videos/85ea19d4-…/`.

**What went, per detector.**

- **SAM 3:**
  - the post between the faint "2.0" sign and the top of the "1.0" sign (the longest
    track, 27 s);
  - the base of the pillar;
  - a speck right of centre;
  - mesh patches.
- **VIAME's raw boxes:** the lamp glare, the mesh just below it, and two specks.
- **VIAME's stored tracks:** nothing static at all.

**The "1.0" sign and the post under it** were never tracked by either detector, so there
was nothing there to remove.

## Part 2 — SAM 3 as the auto-labeller

**Runs.** These used SAM 3's detector on all 1,440 frames, then BoT-SORT with the
BrackishMOT-selected settings (thresholds .45/.50/.10, 7.5 s, no score fusion) for
identities.

| Prompt | Boxes ≥ 0.25 | BoT-SORT tracks | Tracked boxes | After static | Reliable-label tracks |
|---|---|---|---|---|---|
| "fish" | 48,937 | 911 | 26,051 | 25,137 | 700 |
| "small fish" | 51,002 | 982 | 28,054 | 26,745 | 744 |

**The prompt choice, by a rule written down before any SAM 3 output existed.** The rule
(decision `sam3_choice_rule`) takes the variant with the most reliable-label tracks,
unless its extra boxes are non-fish. VIAME agreement was not used.

- "Small fish" led, 744 to 700.
- But its 1,986 boxes that "fish" lacks are mostly empty water and mesh texture, plus
  the post junction (`sam3/small-fish-unique.jpg`).
- "Fish" has 382 such boxes, among them real fish both others miss and one box on the
  same junction (`sam3/fish-unique.jpg`).
- **Chosen: "fish"** (decision `sam3_prompt_choice`).

**Reliable labels.**

- These are tracks with at least 5 boxes and a median score of at least 0.5, after
  static suppression. Every box clears the 0.25 floor.
- 700 of 901 tracks were kept: 149 were too short and 52 scored too low. That is
  **23,515 boxes**.
- `labels/fish/candidates.csv` is the project's VIAME-style candidate CSV: class `fish`,
  zero-based source frames, unreviewed. SHA256 `6e82329e…e1f5`.
- These labels are suggestions, never ground truth. Meta's SAM License applies to them.

**Agreement with VIAME's stored result.** This is on the 1,152 frames both analysed, at
IoU ≥ 0.5. Boxes on frames only SAM 3 analysed are unaligned, not disagreement.

| | VIAME boxes | SAM 3 label boxes | Matched | VIAME only | SAM 3 only | Share of VIAME matched | Share of SAM 3 matched |
|---|---|---|---|---|---|---|---|
| VIAME all tracks | 7,445 | 18,801 (+4,714 unaligned) | 6,032 | 1,413 | 12,769 | **0.81** | 0.32 |
| VIAME tracks ≥ 0.60 (shown in production) | 6,295 | 18,801 | 5,246 | 1,049 | 13,555 | **0.83** | 0.28 |

**Temporal proxies.**

| | Tracks | Boxes | Fragmentation | Duplicates | Single-frame tracks | Median track length | Median acceleration |
|---|---|---|---|---|---|---|---|
| VIAME stored (10 FPS) | 213 | 7,445 | 0.000 | 0 | 0 | 25 boxes = 2.5 s | 0.91 |
| **SAM 3 labels, "fish"** (12.5 FPS) | 700 | 23,515 | 0.047 | 0 | 0 | 22 boxes = 1.8 s | 0.83 |
| SAM 3 "fish", all tracks after static | 901 | 25,137 | 0.061 | 0 | 0.003 | 16 boxes = 1.3 s | 0.83 |
| SAM 3 "small fish", all tracks after static | 972 | 26,745 | 0.053 | 0 | 0.003 | 16 boxes = 1.3 s | 0.80 |

**For your review.**

- **The video:** `render/side-by-side-viame-labels-fish.mp4`.
  - Layout: VIAME on the left, SAM 3's labels on the right.
  - Colours: green where both have the box, magenta VIAME only, orange SAM 3 only, grey
    suppressed fixtures.
  - Format: 1,152 frames (every frame VIAME analysed) at 10 FPS, H.264 Baseline, 167 MB.
  - It was encoded on the host by Windows Media Foundation; Docker isn't needed.
- **The video without VIAME:** `render/labels-fish.mp4` (all 1,440 frames, 12.5 FPS).
- **Two crop sheets** of 48 random boxes each:
  - `compare/sam3-labels-only-sample.jpg`: boxes only SAM 3's labels have. Most look like
    faint fish in the murk; a noticeable minority are doubtful.
  - `compare/viame-only-sample.jpg`: boxes only VIAME has. Some are real fish that SAM 3
    missed or boxed differently (IoU < 0.5). Others are low-score VIAME boxes on the lamp
    glare, the mesh and a ring marking.

  These are visual impressions of unreviewed boxes, not measurements.

## Part 4.1 — BoT-SORT on BrackishMOT (real labels; Brackish water only)

**Setup.**

- BoT-SORT and Ultralytics' ByteTrack (as a control) replay the VIAME detections saved by
  the murky-water work (production detector, score ≥ 0.10).
- They read the same analysed frames as before.
- Scoring is `evaluate_mot` at IoU 0.5 on analysed frames only, with the tuning and
  held-out split in `part1/manifest.json`.

**Grid v1, fixed before scoring.**

- Three threshold sets: VIAME's .45/.50/.10, .35/.40/.10 and Ultralytics' .25/.25/.10.
- Three lifetimes: 1.5, 3 and 7.5 s.
- Score fusion on and off.
- Round one ran on 25 sequences; the top three per tracker went to the 49 tuning
  sequences, and the best of those to the 49 held-out sequences.
- The ranking rule is the murky-water one: MOTA ≥ VIAME's and false positives ≤ 1.10×
  rank first, then IDF1.

| Held-out (49 sequences) | IDF1 | IDP | IDR | MOTA | TP | FP | ID sw. | ID sw. per GT track |
|---|---|---|---|---|---|---|---|---|
| **10 FPS** VIAME ByteTrack, production | 0.138 | 0.411 | 0.083 | 0.017 | 1,022 | 817 | 46 | 0.178 |
| BoT-SORT, selected (.45/.50/.10, 7.5 s, no score fusion) | 0.139 | 0.558 | 0.079 | 0.044 | 859 | 442 | 18 | 0.069 |
| ByteTrack (Ultralytics), selected (same settings) | 0.137 | 0.586 | 0.077 | 0.043 | 805 | 399 | 15 | 0.058 |
| BoT-SORT at VIAME's settings (3 s, fusion on) | 0.135 | 0.588 | 0.076 | 0.043 | 800 | 385 | 25 | 0.097 |
| **4 FPS** VIAME ByteTrack, production | 0.116 | 0.458 | 0.066 | 0.025 | 316 | 215 | 9 | 0.035 |
| BoT-SORT, selected (.35/.40/.10, 1.5 s, no score fusion) | **0.173** | 0.471 | 0.106 | 0.033 | 488 | 332 | 36 | 0.140 |
| ByteTrack (Ultralytics), selected (same settings) | 0.163 | 0.452 | 0.099 | 0.028 | 471 | 332 | 38 | 0.148 |
| BoT-SORT at VIAME's settings (7.5 s, fusion on) | 0.085 | 0.764 | 0.045 | 0.034 | 173 | 43 | 4 | 0.016 |
| *murky-water VIAME candidate B (.35/.40, 3 s), for reference* | 0.172 | — | — | −0.020 | 630 | 625 | — | 0.304 |

Per-sequence comparison with VIAME (32 held-out sequences with fish):

| Selected tracker | Better | Worse |
|---|---|---|
| BoT-SORT, 10 FPS | 9 | 7 |
| BoT-SORT, 4 FPS | 13 | 4 |
| ByteTrack, 10 FPS | 10 | 7 |
| ByteTrack, 4 FPS | 10 | 7 |

**What this says, for Brackish water:**

- **The gain is mostly the thresholds, not BoT-SORT's extras;** Ultralytics' ByteTrack is
  within 0.01 IDF1.
- **At VIAME's own thresholds, Ultralytics' trackers lose fish at 4 FPS** (173 true boxes
  against 316): they need IoU ≥ 0.2 between frames, and VIAME accepts 0.02.
- **Exploratory grid v2** (`match_thresh` 0.95, added after seeing v1) was slightly better
  on tuning but worse on held-out at 10 FPS (IDF1 0.128). It is not used.
- **Camera-motion compensation:** the estimated shift is a median of 0.12 px per frame
  (95th percentile 0.53, maximum 1.0). Sparse optical flow cost 16 ms per frame and
  changed no metric.
- **Cost:** the tracker takes about 0.2 ms per frame here.

**Live mode's 2 s segments (4 FPS, held-out; boxes only).**

| Setting | Tracker state | TP | FP | Precision | Recall | IDF1 |
|---|---|---|---|---|---|---|
| VIAME production | restarts per segment (today) | 228 | 132 | 0.633 | 0.062 | — |
| VIAME candidate B | restarts per segment | 528 | 352 | 0.600 | 0.144 | — |
| BoT-SORT, selected | restarts per segment | 461 | 222 | **0.675** | 0.126 | — |
| BoT-SORT, selected | **carried across segments** | 488 | 332 | 0.595 | 0.134 | 0.173 |

**Can BoT-SORT's state carry across segments? Yes.** An Ultralytics tracker is a plain
Python object that a live worker can keep per camera and feed segment after segment;
the "carried" row does exactly that. It adds 6% more true boxes and keeps identities,
at lower precision. `live_monitor.py` and the live worker are unchanged, as asked.

## Part 4.2 — tracking on SmartBay (proxies only)

These are tracked after static suppression, at the settings chosen on BrackishMOT.

| Boxes from | Tracker | Tracks | Boxes | Fragmentation | Single-frame | Median length | Median acceleration | Cost |
|---|---|---|---|---|---|---|---|---|
| SAM 3 "fish" | **BoT-SORT** | 901 | 25,137 | **0.061** | 0.003 | 16 | 0.83 | 4.0 ms per frame |
| SAM 3 "fish" | ByteTrack | 1,086 | 24,232 | 0.157 | 0.003 | 12 | 0.83 | 2.3 ms per frame |
| SAM 3 "small fish" | **BoT-SORT** | 972 | 26,745 | **0.053** | 0.003 | 16 | 0.80 | 2.4 ms per frame |
| SAM 3 "small fish" | ByteTrack | 1,174 | 25,781 | 0.142 | 0.003 | 12 | 0.81 | 2.5 ms per frame |
| VIAME raw | BoT-SORT | 269 | 5,518 | 0.059 | 0 | 16 | 0.76 | 3.1 ms per frame |
| VIAME raw | ByteTrack | 285 | 5,052 | 0.077 | 0 | 13 | 0.72 | 1.1 ms per frame |
| VIAME stored | VIAME ByteTrack | 213 | 7,445 | 0.000 | 0 | 25 | 0.91 | — |

- **BoT-SORT splits SAM 3's fish into fewer pieces than ByteTrack** (about 2.5× less
  fragmentation) at the same thresholds.
- With GMC and ReID off, the remaining difference is BoT-SORT's width/height Kalman
  filter, which suits fish that turn. Without labels this is a proxy.
- On BrackishMOT's sparser VIAME detections the two trackers were within 0.01 IDF1.

## Does VIAME stay?

**Not decided yet;** it depends on Part 3.

- **VIAME is optional in the pipeline already.** `run --no-viame` goes from SAM 3 through
  static suppression, labels and tracking to a video, and was checked end to end on this
  upload.
- **SAM 3 is a labeller, not a live detector.**
  - It took 572 s for this video's 1,440 frames, about twice VIAME's 290 s.
  - At 2.5 FPS it cannot keep up with live mode's 4 FPS.
- **What VIAME adds that SAM 3's labels lack is small and mixed:** 1,413 boxes, some real
  fish and some glare and mesh.
- **A student detector trained on SAM 3's labels decides it.** If it keeps up at 4 FPS
  and agrees with SAM 3, VIAME would be the bottleneck and could be dropped. Until then
  it is the only fast detector here.

## Licences

- **SAM 3 weights:** Meta's SAM License. It is free and allows commercial use, and it
  forbids military, weapons, nuclear and espionage uses. Its text
  (`data/models/sam3/LICENSE`) must ship with anything derived from SAM 3, including
  these labels and any student trained on them.
- **Ultralytics** (trackers, SAM 3 code) and its CLIP fork: AGPL-3.0, or an Ultralytics
  Enterprise licence. They are used here for experiments.
- **The other new packages:**
  - `timm`, `safetensors` and `ftfy`: Apache-2.0;
  - `wcwidth`: MIT;
  - `regex`: Apache-2.0 AND CNRI-Python.
- **Student (Part 3):** RF-DETR, Apache-2.0 variants only.
- **BrackishMOT:** CC BY-SA 4.0 per Kaggle metadata.
- **VIAME:** BSD-3-Clause.

## Code and tests

- `app/services/static_filter.py`: the static filter.
- `app/services/box_tracking.py`: the BoT-SORT/ByteTrack adapter.
- `scripts/sam3_autolabel.py`: checkpoint verification, SAM 3 detector and video
  inference, reliable labels, and the candidate CSV.
- `scripts/sam3_botsort.py`: the journal, VIAME, static, comparison, H.264 rendering,
  tracking and BrackishMOT.
- `scripts/sam3_stages.py`: the SAM 3 probe, inference, variants, labels and `run`.
- `scripts/sam3-requirements.txt`: the pinned SAM 3 packages.
- `tests/test_sam3_botsort.py`: 17 tests, no network or GPU. They cover:
  - the static rules (still box, rest-then-move, short rest, zone, untracked boxes,
    background match with a resting fish kept);
  - mocked SAM 3 output → candidate CSV on zero-based source frames, as
    `cvat-suggestions` reads it;
  - detector-only frames given IDs by BoT-SORT;
  - the SHA256 refusal before any load;
  - an ID kept through a one-frame gap, and a late track's first box;
  - the time split with gaps;
  - journal resume, interruption charging, budget refusal and cross-process merging;
  - VIAME analysed frames;
  - agreement on shared frames only.

## Commands (PowerShell, repository root)

**The whole pipeline on any new upload, by video ID.** It needs Docker for the read-only
lookup; completed steps are skipped.

```powershell
$env:FISHIAL_PREPROCESS='none'; $env:YOLO_OFFLINE='true'
.venv\Scripts\python.exe -m scripts.sam3_botsort run --video-id VIDEO_UUID
```

The same without VIAME:

```powershell
.venv\Scripts\python.exe -m scripts.sam3_botsort run --video-id VIDEO_UUID --no-viame
```

`--root` picks the experiment folder. By default it is the one already holding that
video, otherwise `data/outputs/sam3-botsort-<UTC date>`. A new upload uses the recorded
prompt choice ("fish"); add `--prompts fish,small-fish` to rerun the comparison.

**SAM 3's labels into a separate CVAT candidate task.** I have not run this. It creates
`Aithne candidate <video UUID> sam3-fish` and never touches the existing SmartBay task:

```powershell
docker compose exec api python scripts/external_fish.py cvat-suggestions 85ea19d4-db7b-4ed7-90a0-a8acccba2b0c sam3-fish /data/outputs/sam3-botsort-20261014/videos/85ea19d4-db7b-4ed7-90a0-a8acccba2b0c/labels/fish/candidates.csv
```

The file holds 700 tracks and 23,515 boxes, a large review. As with the other candidate
commands, the API container needs the current source (rebuild it or mount it).
