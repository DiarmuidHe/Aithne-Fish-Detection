# SmartBay rolling training: results

The task is `docs/smartbay-rolling-training-prompt.md`.

**These labels are SAM 3 pseudo-labels, never ground truth.** No one reviewed them. Every number
below is *agreement with SAM 3*, never precision, recall or accuracy. Expert days were never
downloaded, labelled, trained on or evaluated on.

Round 1's outputs are under `data/outputs/smartbay-rounds/round-1/` (gitignored). The instance's
copy came back into `round-1/cloud/`, checked file by file against the instance's SHA256s.

## Round 1 (2026-10-30 to 2026-10-31)

**Status: done and at Gate 3.**
- 212 windows labelled; 194 kept after the automatic exclusions.
- RF-DETR Nano was trained for 15 epochs and evaluated on the test days.
- The instance `fish-sam3-labelling` (`i-0d5319c21ded85a92`) was due to stop itself at
  12:35:50 UTC on 2026-10-31. Round 2 waits for your yes.

### Timeline (UTC)

| When | What |
|---|---|
| 10-30 20:40 | Instance launched; identity checked; setup (Part 2) in 150 s |
| 20:58–21:05 | Probe (Part 3): SAM 3 speed with one and two processes |
| 21:34 | Labelling (Part 4) started with 2 SAM 3 workers |
| 10-31 04:35 | **The backstop guard stopped the instance** with 198 of 212 windows labelled, and `sam3_gpu` nearly used up (44,168 of 45,000 s) |
| 08:09 | You started the instance again (new IP); `sam3_gpu` raised to 50,000 s with your approval |
| 09:17 | Labelling finished: 212 of 212 |
| 09:18–09:26 | Checks (Part 5) and dataset build (Part 6) |
| 09:26–11:55 | One-epoch probe, then 15 training epochs (Part 7) |
| 11:55–12:05 | Evaluation on test days |
| 12:13 | Final copy to the PC: 696 files, 1.4 GB, SHA256-checked |
| 12:35:50 | Guard stops the instance |

**Why the first night stopped early.**
- Two SAM 3 workers managed about 27 windows an hour, not the roughly 36 extrapolated from the
  first 40 minutes. The guard was set from that faster estimate.
- The journal charges each process's own wall time, so two workers drain `sam3_gpu` twice as fast
  as the clock.
- Nothing was lost: `run_windows` skipped every finished window on restart and redid the two that
  were cut off.

### Selection and day roles

Selected with `--keep` on round 0 (132 windows, 60 days), so round 0's roles and windows are kept
as they are. Round 0's last screened day is 2025-08-17.

**New days in round 1: 41** (2025-08-19 to 2025-10-01):
- 24 training;
- 6 test: 08-20, 08-29, 08-31, 09-06, 09-20, 09-30;
- 4 expert: 08-26, 09-01, 09-18, 09-28;
- 7 unused: 09-09 to 09-15, which have no PNG coverage, so the scan can't screen them.

**New windows: 80 of 84 slots** (70 training, 10 test).

| Quota | Got | Target |
|---|---|---|
| Strong evidence | 42 | 46 |
| Weak / none / murky | 13 / 13 / 12 | per day's mix |
| Light low / mid / high | 34 / 29 / 17 | at least 17 each |
| Hour bins 00–06 / 06–12 / 12–18 / 18–24 | 15 / 29 / 21 / 15 | at least 13 each |
| Reserve window | 79 of 80 | one per window |

The 4 unfilled slots are on two short days that have murky footage only: 08-31 (test, 0 of 2) and
09-08 (training, 1 of 3). One window (07-03) was replaced by its reserve.

**Overall: 212 windows** (184 training, 28 test). By evidence: strong 114, weak 33, none 33,
murky 32.

### SAM 3 speed on the A10G

| Setup | Per process | Per window |
|---|---|---|
| One process (probe) | 4.8 FPS; 14 s load; 3.2 GB peak GPU | about 115 s |
| Two processes (probe) | 2.69 FPS each, 5.38 combined | about 92 s effective |
| Two processes (actual run) | 2.45 FPS each | about 230 s each, including about 30 s load and 14 s checkpoint check; about 27 windows an hour combined |

Cutting a 30 s window took 27–42 s of CPU (libx264 `-preset veryfast`, CRF 14, frame-exact).
The static-object filter took up to 11 s. Labels took under 1 s.

### Cost: estimated against actual

Instance time at the g5.xlarge on-demand rate in eu-west-1 (about $1.12 an hour). The 100 GB disk
adds about $0.30 a day.

| | Hours | Cost |
|---|---|---|
| Estimate at Gate 1 (4 FPS case) | 12.8 | about $14.40 |
| Estimate at Gate 2 (after the probe) | about 10.5–11.7 | $11.80–13.10 |
| Re-estimate after the overnight stop | about 13.6 | about $15.40 |
| **Actual: first night (20:40–04:35)** | 7.9 | about $8.90 |
| **Actual: second start (08:09–12:36)** | 4.45 | about $5.00 |
| **Actual total** | **about 12.35** | **about $13.90, 56% of the $25 cap** |

The actual figures come from the instance's clock. Check them against the bill in the console.

Journal budgets used:
- `sam3_gpu`: 46,818 of 50,000 s. Two workers each charge their own time.
- `training`: 8,948 of 10,800 s, including the probe epoch.

### Automatic exclusions (Part 5)

The `auto_exclusions` rules are frozen in the journal and apply to every round:
- a frame with more than 150 boxes;
- more than half of SAM 3's boxes removed as static;
- a *strong* window with no labels left;
- a failed stage.

| Result | Windows |
|---|---|
| Labelled | 212 of 212 (100%) |
| Excluded: more than half the boxes static | 16 |
| Excluded: strong window with no labels | 2 |
| **Kept** | **194** (169 training, 25 test) |
| Kept but flagged for a look | 30 |

The kept windows split as follows:
- by evidence: strong 110, none 31, weak 27, murky 26;
- by light: high 56, mid 68, low 70;
- by hour bin: 00–06 55, 06–12 56, 12–18 47, 18–24 36.

The per-window reasons are in `round-1/cloud/checks/windows.csv` and `quality_report.json`.

**Optional spot check** (about 30 minutes): `round-1/cloud/checks/spot_check_1.jpg` to `_5.jpg`
show 100 random test-day frames with SAM 3's boxes drawn.

### Dataset counts (Part 6)

- Every 5th frame, so 90 images per window. COCO layout.
- Frames without labels are kept as negatives.
- Validation is 6 seeded training days (seed 17, 10% of the new training days): 06-30, 07-17,
  07-31, 08-16, 08-30 and 09-19. They are frozen for later rounds.

| Split | Windows | Images | Negative images | Boxes |
|---|---|---|---|---|
| Training | 154 | 13,860 | 6,495 | 103,455 |
| Validation | 15 | 1,350 | 883 | 1,219 |
| Test | 25 | 2,250 | 790 | 19,621 |

| Split | Strong | Weak | None | Murky |
|---|---|---|---|---|
| Training: images / boxes | 7,740 / 98,844 | 2,070 / 590 | 2,160 / 217 | 1,890 / 3,804 |
| Validation: images / boxes | 810 / 1,133 | 180 / 65 | 270 / 21 | 90 / 0 |
| Test: images / boxes | 1,350 / 19,191 | 180 / 178 | 360 / 124 | 360 / 128 |

Strong windows hold 96% of the training boxes. Dense schools dominate the labels.

The annotation files are in `round-1/cloud/dataset/*/_annotations.coco.json`. The images stay on
the instance's disk.

### Training (Part 7)

- RF-DETR Nano at 640 px, from the verified COCO checkpoint. The head was re-initialised for one
  class, *fish*.
- A one-epoch probe took about 10 minutes, so 15 epochs fitted the 3-hour budget. Each epoch took
  about 9 minutes 15 s.
- The best checkpoint is from epoch 11 of 15.
  - It scores 0.739 EMA mAP50:95 against SAM 3's labels on the validation days.
  - At the last epoch: mAP50 0.870, mAP50:95 0.738, recall 0.84, precision 0.75.
  - These scores are agreement with SAM 3 on the validation days, not accuracy.
- **Weights:** `round-1/cloud/models/rfdetr-nano-round-1/train/checkpoint_best_total.pth`.
  - 122 MB, SHA256 `fcbbc1d49e7a252b4192d44cdd3d47e1fee30d4e6fb3cf24d3a794a1c2c6d889`.
  - This matches the journal's `student_model` decision.
- **Threshold 0.5.** It was chosen on the validation days before any test-day number was
  computed: it gave the best agreement there, 0.658.

### Evaluation on test days (Part 7)

The student's boxes at threshold 0.5 were compared with SAM 3's labels: a match needs IoU ≥ 0.5.
*Agreement* is matched / (student boxes + SAM 3 boxes − matched). The test covers 25 windows on
13 days, none of them trained on.

| Evidence | Windows | Agreement | SAM 3 boxes the student also found | Student boxes SAM 3 also has |
|---|---|---|---|---|
| **All** | 25 | **0.75** | 82% | 89% |
| Strong | 15 | 0.76 | 82% | 91% |
| Weak | 2 | 0.43 | 72% | 51% |
| None | 4 | 0.65 | 72% | 88% |
| Murky | 4 | 0.35 | 95% | 36% |

**Empty-water proxy** (boxes per frame on *none* and *murky* windows):

| Evidence | Student boxes per frame | SAM 3 boxes per frame | Frames with any student box |
|---|---|---|---|
| None | 0.28 | 0.34 | 24% |
| Murky | **0.95** | 0.36 | 30% |

**Speed at 1080p on the A10G:** 32.7 FPS median, 30 ms per frame at p50, 35 ms at worst p95, over
40 windows. That keeps up with the 15 FPS live stream on every window. The model ran unoptimised
in FP32, and this was not measured on the PC's GPU.

**What this says:**
- On clear footage with fish, the student finds most of what SAM 3 finds and adds few boxes of
  its own.
- **In murky water it boxes about 2.6 times as much as SAM 3.** One murky test window,
  07-08 11:55, has 1,621 student boxes against SAM 3's 648. This is the main weakness to fix in
  round 2.
- Four test windows show agreement 0.0 for a mundane reason:
  - two have no boxes from either model (08-29 22:55 and 09-06 23:22);
  - two have a few student boxes where SAM 3 had none (07-04 04:46 with 96, 07-08 19:06 with 6).
  - The per-window file reports 0/0 as 0.0, not as agreement.
- None of this is accuracy. A small reviewed check set from the expert days would show how often
  SAM 3 and the student are actually right.

### Licences

- **RF-DETR** 1.11.2 and its COCO Nano weights: Apache-2.0.
- **SAM 3:** Meta SAM License.
  - Free, and commercial use is allowed.
  - It forbids military, weapons, nuclear and espionage uses.
  - Its text must ship with anything derived from SAM 3, including this student, which was trained
    on SAM 3's labels.

### Where the outputs are

All of these are under `data/outputs/smartbay-rounds/round-1/`.

| What | Path |
|---|---|
| Selection, day roles, contact sheets | `selection/` |
| Probe | `probe/` |
| Per-clip SAM 3 output, static filter, labels, contact sheet | `cloud/videos/<window>/` |
| Checks and spot-check sheets | `cloud/checks/` |
| Dataset report, evaluation | `cloud/student/round_dataset.json`, `cloud/student/round_eval.json` |
| Trained weights and training metrics | `cloud/models/rfdetr-nano-round-1/train/` |
| Step logs | `cloud/after/*.log`, `cloud/after-labelling.log`, `cloud/run_windows/` |
| Budgets and frozen decisions | `cloud/journal.json` |
| PC-side copy log and 10-minute status snapshots | `pull.log`, `status.log` |

Left on the instance: the cut windows (`*.mp4`), the dataset JPEGs, and the resumable
`checkpoint*.pth` files. They aren't needed to use the student, but they would let round 2 avoid
recutting.

### Commands

**On the PC (PowerShell, from the repository root):**

```powershell
# Part 1: round 1's selection, keeping round 0's roles and windows
.venv\Scripts\python.exe -m scripts.smartbay_label_selection `
  --keep data/outputs/sam3-cloud-labels-20261021/selection
```

Part 2 uploads the code bundle, `to_label.csv`, `rf-detr-nano.pth` and its LICENSE with scp. Each
is checked by SHA256 on the instance.

The 30-minute copy and the final copy run from Git Bash. Without `MSYS_NO_PATHCONV=1`, `--remote`
is rewritten as a Windows path.

```bash
MSYS_NO_PATHCONV=1 ./.venv/Scripts/python.exe -m scripts.smartbay_archive_pull --host ubuntu@<public IP> \
  --remote /home/ubuntu/Aithne-Fish-Detection/data/outputs/smartbay-rounds/round-1 \
  --local data/outputs/smartbay-rounds/round-1/cloud --every 30 \
  --exclude "journal.json" --exclude "*.mp4" --exclude "*.part" --exclude "*.tmp" --exclude "*.lock" \
  --exclude "logs/*" --exclude "run_windows/*" --exclude "after/*" --exclude "dataset/*" \
  --exclude "models/*" --exclude "student/*" --exclude "checks/*"
```

The final copy uses `--once` and, in place of those excludes:
`--exclude "*.mp4" --exclude "*.part" --exclude "*.tmp" --exclude "*.lock" --exclude "dataset/*.jpg" --exclude "*/probe/*.pth" --exclude "checkpoint.pth" --exclude "checkpoint[0-9]*.pth"`

Windows OpenSSH's `ssh-keyscan` can't negotiate sntrup761. Use Git's `/usr/bin/ssh-keyscan` to
compare host keys after a restart.

**On the instance (bash, in `/home/ubuntu/Aithne-Fish-Detection`):**

```bash
# Part 2: install Ubuntu's ffmpeg and the 50 hash-pinned packages; then check imports, CUDA and checkpoints
scripts/cloud_labelling/bootstrap_labelling.sh install
scripts/cloud_labelling/bootstrap_labelling.sh --check

# Parts 4-7: labelling with 2 workers, then checks, dataset, training and evaluation once labelling ends
sudo shutdown -h +330
tmux new-session -d -s labelling "AITHNE_GUARD_WHEN_DONE=30 scripts/cloud_labelling/run_labelling.sh 2>&1 | tee -a data/outputs/smartbay-rounds/round-1/run_windows/labelling.log"
tmux new-session -d -s after-labelling "scripts/cloud_labelling/run_after_labelling.sh 2>&1 | tee -a data/outputs/smartbay-rounds/round-1/after-labelling.log"
```

`run_after_labelling.sh` runs these steps in order, each logging to `round-1/after/<step>.log`:

```bash
/opt/pytorch/bin/python -m scripts.cloud_labelling.round_checks --root data/outputs/smartbay-rounds/round-1 --min-labelled 0.9
/opt/pytorch/bin/python -m scripts.student_detector round-data  --root data/outputs/smartbay-rounds/round-1
/opt/pytorch/bin/python -m scripts.student_detector round-train --root data/outputs/smartbay-rounds/round-1
/opt/pytorch/bin/python -m scripts.student_detector round-eval  --root data/outputs/smartbay-rounds/round-1
```

These run with `PYTHONPATH=.labelling-python-packages` and `YOLO_OFFLINE=true`.

After the overnight stop, the `sam3_gpu` budget was raised under the journal lock. The script
recorded the decision `sam3_gpu_budget_raised` with your approval quoted. Then the same two tmux
sessions were started again.

### Lessons for round 2

- **Size `sam3_gpu` for the number of workers.** With N workers, allow about N × the wall hours.
  Plan for about 250 s of SAM 3 per window per worker when two share the A10G.
- **Set the backstop guard from the slower measured rate**, so it covers labelling and Parts 5–7.
- **Check the PC's public IP before a restart.** It changed overnight, so the `fish-sam3-ssh`
  rule had to be updated.
- **Murky windows produce the most extra boxes.** Round 2 could add more murky training windows,
  or more negatives from them.

## Cleanup checklist (for you to do in the console, after your yes)

1. **EC2 → Instances → `fish-sam3-labelling` (`i-0d5319c21ded85a92`):**
   - confirm its state is **Stopped** (its guard stopped it at 12:35:50 UTC on 2026-10-31);
   - if it is still running, choose **Instance state → Stop**;
   - keep it stopped until you decide about round 2. Its disk (about 100 GB) keeps the cut
     windows and the dataset, and costs about $9 a month while it is stopped.
2. **If you ever terminate an instance,** first make sure it holds no SAM 3 or RF-DETR checkpoint
   you still need:
   - SAM 3: `sam3.pt`;
   - RF-DETR: `rf-detr-nano.pth` and `round-1/models/`.
   The trained student is already on the PC. Terminating also deletes the root volume, as long as
   "Delete on termination" is set.
3. **EC2 → Instances, all regions you use:** check that no instance is left running that you don't
   expect. Instance A, `fish-sam3-gpu-backup`, belongs to the archive scan and is yours to manage.
4. **EC2 → Volumes and Snapshots:** check that nothing is left over from instances already
   terminated.
