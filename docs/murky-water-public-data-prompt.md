# Implementation prompt: better fish detection and tracking in murky and dark water, from public data

Paste the section below into a fresh Claude Code session in this repo.

---

Improve fish detection and tracking for murky (SmartBay, Galway Bay) and dark (Ocean
Networks Canada deep-sea) footage **without me labelling any of my own frames**. Use
public, already-labelled datasets and public pretrained weights instead. Work from
diagnosis through a verified result and a written report. There are three parts, in
priority order. Finish each part before starting the next, and stop at a part's
promotion gate if it fails.

1. **Tune the VIAME tracker on BrackishMOT.** This uses the current detector unchanged.
2. **Fine-tune a fish detector on the full Brackish archive** with darkness/turbidity
   augmentation, and compare it against VIAME on the same test footage.
3. **Add FathomNet Megalodon as an offline candidate detector** for the dark ONC cameras.

## Why this approach

- In murky water the detector's confidence scores drop. The tracker only starts a
  track above 0.50 (`app/config.py:156`, `app/viame_pipelines/tracker_fish_no_motion.pipe:52`),
  so a dim fish can be detected but never tracked, or tracked in fragments.
  BrackishMOT is turbid Limfjord (Denmark) footage with ground-truth track IDs, so
  tracker settings can be measured against real identities rather than proxies.
- Brackish is the closest public match to Galway Bay: temperate, turbid, greenish water
  under LED lighting. `data/external/brackish-v4.zip` is already downloaded and
  hash-checked. Only a 60-frame smoke import has been run.
- FathomNet Megalodon is trained on MBARI deep-sea imagery under artificial light,
  which is closer to the ONC cameras than anything VIAME or Fishial was trained on.

Explicitly **out of scope**: more image enhancement in front of the detector (CLAHE
helped less than 1080p capture, see `docs/live-recall-4fps.md`; FUnIE-GAN is for species
crops only), retraining VIAME's DEIM model, MegaFishDetector, OpenAqua (non-commercial,
already excluded), Roboflow Universe datasets, Fishial species identification, the live
cross-segment association in `app/services/live_monitor.py`, and frontend changes.

## Existing code to reuse (do not rebuild it)

- `scripts/external_fish.py`: `fetch-brackish`, `import-brackish-archive`,
  `import-brackish-mot`, `evaluate-mot`, `fetch-fishial`, `infer-fishial`,
  `cvat-suggestions`. Read `docs/external-fish-datasets.md` first. It records the
  archive's class list, pixel-space 960×540 "YOLO" boxes, one-based frames, and the
  per-recording split rule.
- `app/services/external_datasets.py`: `import_brackish_archive` (75/4/10 recording
  split by stable hash), `parse_mot`, `import_brackish_mot`, and `evaluate_mot`.
  `evaluate_mot` uses **greedy per-frame IoU matching**, and its ID-switch count and
  MOTA are diagnostic only.
- `scripts/viame_review_experiment.py run VIDEO OUTPUT`: runs VIAME inside the worker
  image with `--fps`, `--score`, `--high`, `--low`, `--new-track`, `--buffer` and
  `--pipeline` overrides, and never touches application settings. Use it for every
  VIAME pass.
- `scripts/fish_model.py`: `train`, `evaluate`, `track` (Ultralytics). `train` refuses
  a dataset whose val/test rows are not `reviewed`. The Brackish importer sets
  `reviewed: true` for the publisher's annotation scope and `reviewed_for_smartbay: false`.
  Keep that distinction in every report.
- `app/services/detection_metrics.py`: `iou`, `count_matches`.
- `scripts/viame_zero_label_benchmark.py` and its manifest at
  `data/outputs/viame-zero-label-benchmark/manifest.json`: the fixed SmartBay
  early/middle/late clips and the temporal-proxy metrics. Its own pass journal is
  exhausted. Reuse its clips and metric code with a **new** output directory rather
  than its journal.

## Facts to verify rather than assume

- Production batch profile: DEIMv2 + ByteTrack via `tracker_fish_no_motion.pipe`,
  10 FPS, detector score 0.10, ByteTrack high/low/new-track 0.45/0.10/0.50, buffer 30
  sampled frames, `VIAME_FRAME_NUMBER_OFFSET=-1` in `docker-compose.gpu.yml`.
- Live profile: 4 FPS, 60% acceptance, 1080p capture. A 30-frame buffer is 3 s at
  10 FPS but 7.5 s at 4 FPS, so tune and report **both** profiles.
- Check how the installed VIAME ByteTrack defines `track_buffer`, `match_thresh` and
  `new_track_thresh` (direction, valid range, required ordering) before sweeping
  them. Don't assume a lower number is looser.
- Check whether the installed VIAME can run the tracker from a **saved detection CSV**
  (a detection-reader → tracker pipeline). If so, run the detector once per sequence
  and FPS profile and sweep tracker settings from the saved detections. That is far
  cheaper, and it still uses VIAME's own tracker, so results transfer. If not, budget
  full passes. Do not substitute a different ByteTrack implementation for the sweep.
- Check the BrackishMOT source frame rate (`seqinfo.ini`), resolution, sequence count,
  and licence in the downloaded package. Kaggle metadata said CC BY-SA 4.0, but a
  secondary inventory says CC BY-NC-SA 4.0. Record what the package itself states.
- Check whether BrackishMOT sequences come from the same recordings as the Brackish
  detection archive (recording names, dates, or frame hashes). This decides leakage
  handling in Part 2.
- The GPU is an RTX 5060 Laptop (8 GB, Blackwell). Local PyTorch needs a CUDA 12.8+
  build that lists `sm_120`. The existing VIAME worker image's PyTorch already runs on
  this GPU (the Fishial smoke used it) and is an acceptable fallback environment.

## Hard limits

1. Zero paid API calls. Keep Fishial disabled in every experiment.
2. Approved downloads, and nothing else without asking me:
   - BrackishMOT Kaggle release (about 2.2 GB) into `data/external/`. Use the same
     Kaggle API URL pattern as `fetch-brackish`. If Kaggle demands authentication,
     stop and ask me to download it by hand. Do not ask for, store, or print Kaggle
     credentials. Record the archive SHA256 once obtained.
   - FathomNet Megalodon weights from the `FathomNet/megalodon` Hugging Face repo only,
     pinned to a specific commit, with the SHA256 recorded.
   - COCO-pretrained `yolo11n.pt` / `yolo11s.pt` through Ultralytics.
   - A CUDA PyTorch build and the project's `.[train]` extra into `.venv`.
3. The Megalodon `.pt` file is a pickle, and loading it executes code. Verify its
   SHA256 before the first load, and load it only from that verified path.
4. GPU budget: at most 3 hours of aggregate VIAME inference and 4 hours of aggregate
   training across all parts. Keep a resumable journal (inputs, settings, hashes,
   timings, status) so an interrupted run never repeats completed work. Stop early
   when a candidate is clearly dominated.
5. Never use the production database for experiments. Read-only queries are fine for
   finding which local recordings came from which camera. Do not change job outputs,
   uploads, review decisions, the existing SmartBay CVAT task, or my `.env`.
6. All generated data goes under `data/outputs/murky-water-<UTC date>/` (gitignored).
   Code, tests and docs go in the repo. Preserve unrelated working-tree changes. Do
   not commit, rebuild, or restart containers unless I ask. Give me the exact
   commands instead.
7. A public-dataset score is evidence about **that dataset's water**, not SmartBay
   accuracy. Never state SmartBay precision or recall. On SmartBay clips, report only
   agreement, disagreement and temporal proxies.

## Part 1: tracker tuning on BrackishMOT

1. Download and import BrackishMOT with the existing commands. Before seeing any
   results, split sequences into **tuning** and **held-out** sets by stable hash
   (about half each). Record the split in the manifest.
2. Convert each sequence into VIAME input: either an image-list input, if the
   installed reader supports it, or a near-lossless video encoded at the sequence's
   native FPS. Assert that the decoded frame count equals the sequence's frame count.
3. Write a VIAME CSV → MOT converter. Get frame numbering exactly right: MOT is
   one-based, and production VIAME CSVs need `frame_offset -1` to become zero-based.
   Prove alignment with an overlay preview of ground truth and VIAME boxes on a few
   frames before scoring anything.
4. When VIAME samples below native FPS (10 FPS batch, 4 FPS live), score **only the
   analysed frames**. Unanalysed frames must not count as misses.
5. Extend `evaluate_mot` with **IDF1** (global min-cost ID assignment). Add HOTA if
   TrackEval can be used as an optional extra without bloating the main dependencies.
   Keep the existing greedy counts for continuity, and label which numbers are
   official-style and which are diagnostic. Use scipy's `linear_sum_assignment` only
   if scipy is already installed. Otherwise use a small tested assignment routine.
6. Run the current production settings as the baseline at both 10 FPS and 4 FPS on
   tuning sequences. Repeat one run to measure nondeterminism.
7. Successive-halving sweep on **tuning sequences only**. Sweep `new_track_thresh`,
   `high_thresh`, `low_thresh`, and `track_buffer` expressed in seconds (converted to
   frames per FPS profile). Sweep match thresholds only if the checks in "Facts to
   verify" proved them valid runtime overrides. Use at most 12 configurations in round one, then the top
   3 at full tuning size, then the single best on held-out sequences.
8. **Promotion gate.** Change production defaults only if all of these hold on
   **held-out** sequences, at **both** FPS profiles:
   - IDF1 improves by at least 2 points absolute;
   - MOTA does not fall, and false positives rise by no more than 10% relative;
   - ID switches per ground-truth track do not rise;
   - the gain appears on a majority of held-out sequences, not just in aggregate;
   - on the fixed SmartBay clips, no temporal proxy (fragmentation, duplicates,
     singleton tracks, trajectory smoothness) regresses by more than 5%, and accepted
     counts stay stable across the 0.55/0.60/0.65 acceptance cutoffs;
   - runtime stays within 1.1× baseline.

   If it passes, update the defaults consistently in `app/config.py`,
   `.env.example`, `docker-compose.gpu.yml`, the `.pipe` block and the README, and
   record the effective values in job provenance. If the best setting differs between
   the 10 FPS and 4 FPS profiles, add a separate live setting rather than compromising
   both. If the gate fails, change nothing and report the best candidate with its
   numbers.

## Part 2: fine-tune a detector on full Brackish

1. Set up `.venv` with CUDA PyTorch and `.[train]`. Confirm
   `torch.cuda.is_available()` and the `sm_120` arch before training.
2. Run the full `import-brackish-archive` into a new `data/training/brackish-fish-v4`.
   Inspect `report.json` and the previews. Find out whether frames without boxes are
   verified empty. Count false positives on empty frames only if they are.
3. **Baseline first.** Run the production VIAME batch profile on the 10 **test**
   recordings' AVIs and score it against the same test frames and truth as the
   trained model will be scored on. Handle the 960×540 label space and the AVI's
   native resolution explicitly, and prove alignment with overlays. Report recall,
   precision and box counts at IoU 0.5 for VIAME. If VIAME already scores well here,
   say so plainly: there may be little headroom.
4. **Murk augmentation.** Generate deterministic, seeded, degraded copies of
   **training** images only: lower brightness and contrast, a green/blue water tint,
   and backscatter (`I = J·t + B·(1 − t)` with a varied transmission `t` and veil
   colour `B`), plus mild blur and sensor noise. Keep each copy in its source
   recording's split, cap copies to about 1× the train set, document the parameter
   ranges, and save a contact sheet. Never augment val or test.
5. Train two runs with the same seed, epochs and image size: **A** default
   augmentation, **B** with the murk copies. Start from `yolo11s.pt` (fall back to
   `yolo11n.pt` if 8 GB is not enough), 640 px, about 50 epochs.
6. Evaluate A and B against the Brackish test split with `fish_model.py evaluate`. As
   a secondary, clearly-labelled synthetic robustness check, also score a darkened copy
   of the test split. Use **different** degradation ranges from training so it isn't
   circular.
7. Run the better model on the fixed SmartBay clips with `fish_model.py track`. Report
   agreement and disagreement with VIAME, temporal proxies, and a contact sheet of
   boxes found by only one model. Unique boxes are **suggestions, not misses or false
   positives**. If CVAT dev mode is configured, leave the exact `cvat-suggestions`
   command for a **separate** candidate task, but don't run it.
8. If the Part 2 model is ever scored on BrackishMOT, exclude sequences that overlap
   Brackish training recordings. If overlap can't be determined, don't score it there.
9. No production integration in this task. The report must state the licence
   position: Ultralytics code and weights are AGPL-3.0 (Enterprise licence otherwise),
   and Brackish is CC BY-SA 4.0. Name RF-DETR or D-FINE (Apache-2.0) as the route if a
   commercial deployment is wanted later.

## Part 3: FathomNet Megalodon for the ONC cameras

1. Add `fetch-megalodon` to `scripts/external_fish.py`, modelled on `fetch-fishial`:
   pinned URL and commit, SHA256 verification, a `provenance.json` with the licence
   exactly as the model card states it (expected CC BY 4.0; verify), and cleanup on
   failure.
2. Add `infer-megalodon`, writing the same VIAME-style `candidates.csv` +
   `report.json` as `infer-fishial`. Share code with `infer_fishial` where it's
   natural, without changing Fishial's behaviour or output. Megalodon has one generic
   "object" class, so it will box crabs and jellyfish too. Label the output that way,
   and never as "fish".
3. Find local recordings from ONC cameras via a read-only database query (live session
   source keys). If there are any, run Megalodon and VIAME on the same bounded clips
   and report counts, agreement, and a contact sheet of disagreements. If there are
   none, don't record new footage. Leave the exact commands to run once a recording
   exists.
4. ONC video is CC BY-NC-SA 3.0. Note that in the report. No production change.

## Practical notes for this machine

- Plain `pytest` fails here on long temp paths. Use a short base temp, for example
  `--basetemp=C:\t\pt`.
- The repo `.env` can break `Settings()` at import. Set `$env:FISHIAL_PREPROCESS='none'`
  for local test and script runs.
- A script run by path inside a container imports the image's **installed** package,
  not the mounted source. Pass `-e PYTHONPATH=/app` with the source mounted, as in
  `docs/external-fish-datasets.md`.
- Write documented commands in PowerShell form, like the other docs.

## Tests

Add focused tests for the new code:

- IDF1 arithmetic on hand-built cases, including an ID swap and a fragmented track.
- VIAME CSV → MOT frame alignment at native and downsampled FPS, including
  analysed-frames-only scoring.
- Sequence → video frame-count assertion.
- Seeded augmentation: same output for the same seed, and train split only.
- The leakage guard.
- `fetch-megalodon` hash verification and cleanup, using a mocked download (no
  network in tests).
- Megalodon output parsing to candidate rows.

Run the focused tests, then the full suite. Fix failures you cause. Report any
pre-existing failures separately.

## Deliverables

1. Code and tests for the above, and updated `docs/external-fish-datasets.md`.
2. `docs/murky-water-results.md` covering:
   - the compute, download and API budget actually used;
   - baseline vs candidate on every Part 1 gate criterion, per FPS profile;
   - VIAME vs runs A and B on the Brackish test split;
   - the SmartBay agreement findings;
   - Megalodon findings or the pending commands;
   - what changed in production, or why nothing did;
   - licence positions.
3. A final message that states plainly which numbers are real labelled measurements
   (on public data) and which are proxies. End it with the one thing that would turn
   the SmartBay result from "suggestive" into "measured": a small reviewed SmartBay
   check set, using the existing CVAT export flow. Keep that as a recommendation, not
   a requirement of this task.

If the GPU runtime or a download is unavailable, finish everything that doesn't need
it, including the tests, and stop with the exact command to resume.
