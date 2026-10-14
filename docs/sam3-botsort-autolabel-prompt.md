# Implementation prompt: SAM 3 auto-labelling, BoT-SORT tracking and static-object suppression for SmartBay

**Goal:** improve fish detection and tracking on SmartBay footage **without manual labelling**. The approach has three pieces:

- **SAM 3** (text prompt "fish") as an automatic labeller;
- **BoT-SORT** as the tracker;
- **ignoring things that never move.**

VIAME is the current production detector. Keep it only where it earns its place. Go from diagnosis to a verified result and a written report.

Read `docs/murky-water-results.md` and `docs/external-fish-datasets.md` first. They record what was already tried and why it didn't transfer to SmartBay:

- public datasets;
- synthetic murk;
- tracker thresholds;
- Megalodon.

## Test footage

**Use only the latest SmartBay upload for now:**

| | |
|---|---|
| File name | `SmartBayCam2_2025-07-17_0001.mp4` |
| Video ID | `85ea19d4-db7b-4ed7-90a0-a8acccba2b0c` |
| Path | `data/uploads/936e0581-d6de-42d7-8a1b-fadd9c517e01.mp4` |
| Video | 115.2 s, 1920×1080, 12.5 FPS, 1,440 frames |

Before anything else, confirm the video with a read-only query and record its SHA256. I'll upload more footage later, so make every command take a video ID or path. The same pipeline must rerun on new uploads without code changes.

**VIAME's production result for this video is already stored.** It is job `10c7ddbd-9b64-40e7-855c-163557326fd2`: 213 tracks and 7,445 detections in `fish_tracks` and `fish_detections`. Read it from the database read-only; don't re-run production VIAME to get it.

**No labels exist for this footage.** On it, report only agreement, disagreement and temporal proxies. Never state SmartBay precision or recall.

## Hard limits

1. **Zero paid API calls.** Keep Fishial disabled.
2. **Approved downloads. Download nothing else without asking me:**
   - **SAM 3 weights.** The download is gated behind Meta's SAM License; I will accept it and place the checkpoint at `data/models/sam3/` myself.
     - Never ask for, store or print a Hugging Face token.
     - If the file is missing, stop and tell me the exact page and target path.
     - The checkpoint is a pickle. Record its SHA256, verify it before every load, and load only from that verified path.
   - **The `rfdetr` Python package and a COCO-pretrained RF-DETR checkpoint.**
     - Pin the version and record the SHA256.
     - Use a variant whose licence is Apache-2.0, and check it, because some larger RF-DETR variants are not.
   - **Nothing for BoT-SORT.** Use the Ultralytics implementation already in `.venv`. If ReID needs a separate model, use the detector's own features or ask me first.
3. **Budgets:**
   - at most 2 h of SAM 3 GPU time;
   - at most 2 h of training;
   - at most 30 min of new VIAME passes.

   Keep a resumable journal (inputs, settings, hashes, timings, status) and never redo completed work.
4. **Treat the production database as read-only.**
   - Don't change job outputs, uploads, review decisions, the existing SmartBay CVAT task or my `.env`.
   - Don't change production defaults: `app/config.py`, `.env.example`, the docker-compose files and the `.pipe` block.
5. **Where things go:**
   - Generated data goes under `data/outputs/sam3-botsort-<UTC date>/` (gitignored).
   - Code, tests and docs go in the repo.
   - Preserve unrelated working-tree changes.
   - No commits, image rebuilds or container restarts unless I ask. Give me the exact PowerShell commands instead.
6. **Pseudo-labels are unreviewed suggestions, never ground truth.** Never mark them reviewed.
7. **Licences:**
   - **SAM 3:** Meta's SAM License. It's free and allows commercial use. It forbids military, weapons, nuclear and espionage uses. Its text must ship with anything derived from SAM 3.
   - **The student model:** prefer Apache-2.0 (RF-DETR).
   - **Ultralytics:** AGPL-3.0. It's fine for the tracker experiment, but say so wherever it's used.

## Machine limits (learned the hard way)

- **Hardware:** a Windows laptop with an RTX 5060 Laptop GPU (8 GB) and 31.7 GB of RAM. Docker takes about 6–7 GB of the RAM.
- **Dataloader workers are memory-hungry.** On Windows, each spawned PyTorch dataloader worker holds about 1 GB of RAM, and Ultralytics doubles the count for validation.
  - Use `fish_model.py train --workers 4 --val-workers 2` or fewer.
  - Check free RAM before and during heavy runs.
- **The laptop throttles thermally.** Watch `\Processor Information(_Total)\% Processor Performance` and the GPU temperature. If the CPU sits far below 100%, stop and ask me to switch Windows to Best performance and clear the vents.
- **CVAT's containers use several GB.** Ask me before pausing them, and restart them afterwards.
- **Keep Ultralytics offline.** Run it with `YOLO_OFFLINE=true` and a dead proxy (see `OFFLINE` in `scripts/murky_water_detector.py`), so it can't auto-download anything. `lap` is already installed.
- **pytest** needs `FISHIAL_PREPROCESS=none` and a short `--basetemp`.
- **Container scripts** need `-e PYTHONPATH=/app` with the source mounted.
- **Documented commands** should be in PowerShell form.

## Part 0: facts to verify first

- **SAM 3 on this GPU:**
  - which loader works: Ultralytics `SAM3VideoSemanticPredictor` or `facebookresearch/sam3`;
  - which precision and input size fit in 8 GB;
  - frames per second;
  - how memory grows with video length (chunk the video if needed).

  Prove with an overlay that its boxes line up with the video's 1920×1080 frames and frame numbers.
- **BoT-SORT in the installed Ultralytics:**
  - its config keys;
  - whether ReID and camera-motion compensation are on (the camera is fixed, so compensation may be wasted time);
  - how to feed it external detections, such as VIAME's or SAM 3's, instead of a YOLO model's.
- **VIAME's stored detections:**
  - frames are zero-based after `VIAME_FRAME_NUMBER_OFFSET`;
  - VIAME analysed 10 of the video's 12.5 FPS, so unanalysed frames are not misses.

## Part 1: ignore things that never move

1. Build a reusable static-object filter for a fixed camera that works on any detector's tracks or boxes. Flag:
   - a track that stays essentially in place for a long time: tiny centroid movement and a high IoU with its own median box;
   - boxes that recur in the same spot across a large share of the video, using a static-zone map;
   - optionally, boxes whose pixels match a median background image.
2. Be conservative: a fish resting for a few seconds must survive. Fix the thresholds before looking at the final comparison, and document them.
3. Report what was removed, per detector, with counts and a contact sheet. On the test video, expect the post under the "1.0" sign, the sign and the mesh to go, and swimming fish to stay.

## Part 2: SAM 3 as the auto-labeller

1. Run SAM 3 on the test video with the text prompt "fish". Try one or two prompt variants (for example "fish" and "small fish") and, if supported, one exemplar box. Pick one before the final comparison.
2. Keep only reliable labels: tracked over several frames, above a confidence floor and not static. Write them in the project's VIAME-style candidate CSV (class `fish`, unreviewed), so the existing `cvat-suggestions` command can load them.
3. Make a side-by-side video of VIAME's stored result vs SAM 3, after static suppression:
   - distinct colours with a legend;
   - H.264, so it plays in a browser (ffmpeg is in the worker image).

   Report:
   - agreement;
   - boxes unique to each model;
   - track counts;
   - the temporal proxies: fragmentation, single-frame tracks, duplicates and smoothness.
4. **Stop here, show me the video and ask whether SAM 3's labels look good enough to train on.** Don't continue to Part 3 without my answer.

## Part 3: a small student detector (only after I say go)

1. Split the test video by time with a gap, for example the first ~75% for training, the last ~20% for testing and a 5 s gap between them, so neighbouring frames can't leak across.
   - With one video this is a pipeline smoke test; say so.
   - It will be rerun when I upload more footage.
2. Train RF-DETR from its COCO weights on SAM 3's filtered labels, using the smallest variant that fits and is Apache-2.0.
   - On the held-out segment, compare it with SAM 3 (agreement) and with VIAME.
   - Measure its speed: it should keep up with 4 FPS live at 1080p.

## Part 4: BoT-SORT tracking

1. **Start with a labelled, measured check on BrackishMOT.**
   - Replay BoT-SORT over the VIAME detections already saved by the murky-water work, in `data/outputs/murky-water-20261011/part1/detections`.
   - Use the tuning and held-out split in `part1/manifest.json`.
   - Score with `evaluate_mot`: IDF1, MOTA and ID switches, on analysed frames only.
   - Compare it with VIAME's own ByteTrack at production settings, at 10 and 4 FPS, on the held-out sequences.

   This needs no new GPU time and gives real measurements, valid for Brackish water.
2. On the test video, track SAM 3's detections, and the student's if one was trained, with both BoT-SORT and ByteTrack, after static suppression. Report proxies only.
3. **Live mode** runs VIAME on each 2 s segment, which cost 28% of the true boxes in the earlier replay.
   - Note whether BoT-SORT's state could carry across segments.
   - Don't change `live_monitor.py` or the live worker in this task.

## VIAME stays only if it earns it

VIAME's stored result is the free comparison baseline. In the new pipeline, VIAME is optional:

- **Drop it** if it is the bottleneck: much slower per video than the alternatives, or adding nothing they don't already find. Back the decision with timings.
- **Make sure the pipeline runs end to end with VIAME switched off.**

## Viewing

Leave me:

- the side-by-side video or videos;
- the exact `cvat-suggestions` command for loading SAM 3's labels (or the student's) into a **separate** CVAT candidate task. Don't run it, and leave the existing SmartBay task untouched;
- one PowerShell command that runs the whole pipeline on any new upload, by video ID.

## Existing code to reuse

- `scripts/fish_model.py`: `train`, `evaluate`, and `track --tracker botsort.yaml`.
- `scripts/murky_water_tracker.py` with `app/services/external_datasets.py`: the replay harness, the VIAME→MOT conversion and IDF1/CLEAR scoring.
- `scripts/fuse_fish_suggestions.py`: the persistence-filtered fusion pattern.
- `scripts/external_fish.py cvat-suggestions`: candidate CSV → a separate CVAT task.
- `scripts/viame_pipelines/fish_detections_only.pipe`: a detector-only VIAME pass, if raw VIAME detections are needed.

## Tests

Add focused tests that need no network and no GPU:

- the static filter: a still box is removed, a fish that rests and then moves is kept, and a recurring static zone is found;
- mocked SAM 3 output → candidate CSV rows and frame numbers;
- the BoT-SORT adapter on synthetic detections, keeping an ID through a one-frame gap;
- the time split with its gap;
- resuming from the journal;
- the SHA256 check refusing an unverified checkpoint before loading it.

Then run the full suite. Report pre-existing failures separately; `tests/test_external_fish.py::test_cvat_suggestions_use_separate_candidate_task` already fails.

## Deliverables

1. Code and tests.
2. `docs/sam3-botsort-results.md`, covering:
   - the budgets actually used;
   - the Part 0 facts;
   - static suppression;
   - SAM 3 vs VIAME agreement;
   - BoT-SORT vs ByteTrack numbers on BrackishMOT;
   - the student model, if trained;
   - whether VIAME stays, and why;
   - licences;
   - the commands for rerunning on new footage.
3. A short final message in plain language covering:
   - what improved;
   - which numbers are real labelled measurements (BrackishMOT only) and which are proxies;
   - what my next upload should be used for.
