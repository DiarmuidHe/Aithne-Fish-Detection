# SmartBay hard negatives: runbook for a new instance

The student keeps boxing things that never move, such as debris, muck on the lens and background
objects. `scripts/hard_negatives.py` finds those spots in round 1, adds them to training as hard
negatives, retrains and compares. Every number is agreement with SAM 3's unreviewed labels, never
accuracy.

## What was measured before building this (2026-11-06)

Round 1's student at its 0.5 threshold, using results already on the PC:

| | Student boxes | No SAM 3 label matches | **No SAM 3 box near at all** | ...of those, on a static spot |
|---|---|---|---|---|
| The local clip 2025-08-01 00:00 | 27,648 | 3,967 | **380** | **339 (89%)**, on one stuck piece of debris |
| 40 held-out windows | 97,105 | 11,231 | **1,383** | **120 (9%)**, on 4 spots |

- **Static false positives are real, but they vary by clip.** On the clip above they are nearly all of
  the unsupported boxes. Across the 40 held-out windows they are about 9%.
- **Most boxes SAM 3's labels don't match aren't wrong.** About 88% of them have a SAM 3 box near them:
  the same fish, drawn differently.
- **This won't fix the other 88% of unsupported boxes (1,211).** They don't stay in one place, and 677
  of them come from one murky window, 2025-07-08 11:55. Nobody has looked at them yet: they could be
  drifting muck or real fish SAM 3 missed. `review/no_zone_boxes.jpg` shows a sample.
- The 120 boxes are an upper limit. Those 4 spots passed the SAM 3 test, but their videos aren't on
  the PC, so the stillness test hasn't run on them yet.

## The rule

A **zone** is a student box that recurs in one place (IoU >= 0.5) on at least 5% of a window's
frames, across at least half of its time. It is a **hard negative** when both hold:

1. SAM 3's final labels overlap it on at most 20% of the frames where the student fired there.
2. Its crops stay the same object: median NCC with their own median crop >= 0.9.

| Example from the local clip | SAM 3 raw | SAM 3 labels | Stillness | Result |
|---|---|---|---|---|
| Stuck debris | 39% | 0% | 0.98 | hard negative |
| 68 spots dense schools kept crossing (searched at a looser 3%) | 90-100% | above 70% on 67 of them | 0.67 median, 0.94 at most | not a hard negative |

The test uses SAM 3's final labels because raw SAM 3 did fire on that debris, weakly or briefly, and
it never became a label.

**How this changes training.** RF-DETR has no "hard negative" label: a region with no box already
means "not fish". So each zone adds up to 45 more frames with no box on that spot, and fish elsewhere
in those frames keep their labels. Each frame lists its `hard_negative_boxes` in the COCO file. Test
and validation windows are only counted and never reach training, so both students stay comparable.

## Cost estimate

About 5.5 h on a g5.xlarge, so about **$6-7** at $1.12 an hour, plus about $0.30 a day for the disk.
These are estimates from round 1's speeds, not measurements.

| Stage | Time | Work |
|---|---|---|
| Setup | 30 min | Upload 517 MB, install packages |
| `mine` | 2 h | Re-cut 206 windows, run the student on 166 of them |
| Review | Your time | The instance is stopped |
| `train` | 3 h | Dataset, 15 epochs as in round 1, evaluation |

## 1. Launch the instance (you, in the console)

- Name `fish-sam3-hard-negatives`, type g5.xlarge in eu-west-1.
- AMI: search "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04)" and take the newest
  x86_64 one owned by Amazon. Not 2.11 or 2.12, and not Amazon Linux: the package lock is pinned to
  this AMI's `/opt/pytorch` (Python 3.13, torch 2.10). The old custom image `ami-0f6af2756fb3f5f14`
  was deleted on 2026-11-07; the bundle carries everything else, so nothing from it is needed.
- Key pair `fish-sam3-key`, security group `fish-sam3-ssh` (check it allows your current IP).
- 100 GB gp3 root, delete on termination. Shutdown behaviour **Stop**.
- Two g5.xlarge running at once need 8 vCPUs of quota, so check nothing else is running.

## 2. Upload (PC, PowerShell, repository root)

The bundle holds an allowlist only: no `.env`, keys, videos or SAM 3 checkpoint. Rebuild it after any
code change.

```powershell
$env:FISHIAL_PREPROCESS='none'; .venv\Scripts\python.exe -m scripts.cloud_labelling.package_hard_negatives
$key = "$env:USERPROFILE\.ssh\fish-sam3-key.pem"
scp -o IdentitiesOnly=yes -i $key data\outputs\smartbay-rounds\round-1-hn-upload\aithne-hard-negatives.tgz ubuntu@<IP>:/home/ubuntu/
```

## 3. Set up (instance)

Compare the first command's result with `aithne-hard-negatives.tgz.sha256` on the PC.

```bash
sha256sum ~/aithne-hard-negatives.tgz
mkdir ~/aithne-hn && cd ~/aithne-hn && tar xzf ~/aithne-hard-negatives.tgz
sha256sum --quiet -c SHA256SUMS && echo "all files match"
AITHNE_NO_SAM3=1 scripts/cloud_labelling/bootstrap_labelling.sh install
```

## 4. Mine (instance)

Try 2 windows first. It takes a few minutes and catches a broken setup cheaply.

```bash
AITHNE_LIMIT=2 scripts/cloud_labelling/run_hard_negatives.sh mine
tmux new-session -d -s hn "scripts/cloud_labelling/run_hard_negatives.sh mine 2>&1 | tee -a hn-mine.log"
```

The run arms a 4 h shutdown guard, and re-arms it for 30 minutes when it ends or fails.

## 5. Copy back and review (PC, Git Bash, repository root)

```bash
MSYS_NO_PATHCONV=1 ./.venv/Scripts/python.exe -m scripts.smartbay_archive_pull --host ubuntu@<IP> --remote /home/ubuntu/aithne-hn/data/outputs/smartbay-rounds/round-1-hn --local data/outputs/smartbay-rounds/round-1-hn --once --exclude "*.mp4" --exclude "*.part" --exclude "*.tmp" --exclude "*.lock" --exclude "dataset/*.jpg" --exclude "*.ckpt" --exclude "checkpoint.pth" --exclude "checkpoint[0-9]*.pth"
```

Then look in `data/outputs/smartbay-rounds/round-1-hn/`:

- `review/hard_negative_*.jpg`: **is every one of these really not a fish?** A fish wrongly listed here
  teaches the student to ignore fish.
- `review/moving_content_*.jpg`: spots the stillness test turned down. Did it miss any debris?
- `review/no_zone_boxes.jpg`: the boxes this approach can't reach.
- `hard_negatives.json`: `by_use.train` counts what training would get. Windows marked `report` never
  reach training.

**Decide whether to train.** The student already practised on these windows, so it may fire less on
them than on new days. With only a handful of `train` zones, retraining won't change much, and the
money may be better spent on `no_zone_boxes`. To also use zones from windows the frozen exclusions
left out, which the student has never seen, set `AITHNE_USE_EXCLUDED=1` for the `train` run.

## 6. Train (instance, after your yes)

Start the instance again. Its IP changes, so check the `fish-sam3-ssh` rule. To drop a zone, put its ID
on a line of `data/outputs/smartbay-rounds/round-1-hn/review/rejected_zones.txt` on the instance.

```bash
cd ~/aithne-hn
tmux new-session -d -s hn "scripts/cloud_labelling/run_hard_negatives.sh train '<your approval, in your words>' 2>&1 | tee -a hn-train.log"
```

Copy back with the same command as step 5. `comparison.json` shows both students on the same days:

- `agreement` shouldn't fall.
- `static_zones_mined_from_each_student` should fall.
- `empty_water_proxy` shows the boxes on empty and murky water.

Mining each student separately keeps the comparison fair: counting boxes only on round 1's spots
would flatter the new student, because those spots were picked for being where round 1 went wrong.

To watch the new student on your own clip (PC, PowerShell):

```powershell
$env:FISHIAL_PREPROCESS='none'; $env:YOLO_OFFLINE='true'; .venv\Scripts\python.exe -m scripts.student_detector local-eval --render --root data/outputs/smartbay-rounds/round-1-hn --video-dir data/outputs/sam3-botsort-20261014/videos/SmartBayCam1_2025-08-01_0000
```

## 7. Clean up (you, in the console)

- Confirm `fish-sam3-hard-negatives` is **Stopped**, then terminate it once the results are on the PC.
  The bundle and SAM 3's output can be rebuilt from the PC.
- `fish-sam3-labelling` is still stopped and costs about $9 a month for its disk. Its cut windows are
  no longer needed, as `restore` can rebuild them.
- Check that no instance is left running that you don't expect, and that no volume is left over.

## If `restore` stops

- **A source's SHA256 changed:** the archive replaced that clip, so its frames may not match SAM 3's
  labels. The stage refuses it, and `data` won't run without every window, as the base dataset must
  match round 1's. That needs a decision: the clip can't simply be swapped.
- **A download failed:** run the same command again. Finished windows are kept.
- **Windows not byte-identical:** this is fine if the source SHA256 matched. The encoder may differ,
  and the frames are the same. `restored.json` records it per window. Four round 0 windows have no
  source hash on record, so they appear under `source_not_verifiable`.

## Licences

RF-DETR and its COCO weights are Apache-2.0. This student is also trained on SAM 3's labels, so Meta's
SAM License ships with it (`data/models/sam3/LICENSE`, in the bundle). Video: Marine Institute SmartBay
Observatory, CC BY 4.0.
