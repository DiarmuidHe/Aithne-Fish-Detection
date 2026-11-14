# Bigger or gentler students: runbook for a new instance

Round 1's student boxes background objects on cameras it never saw. This run trains other students
on **exactly round 1's data, for round 1's 15 epochs**, so the only difference is the model, then
scores them all beside round 1's. Code: `scripts/student_variants.py`. Every number is agreement
with SAM 3's unreviewed labels, never accuracy.

## What "bigger" means in RF-DETR

RF-DETR Nano, Small, Medium and Large all share **one backbone**: DINOv2 ViT-S, the part that learned
what images look like before it ever saw SmartBay. A bigger one of those has a deeper decoder (the
part that turns features into boxes), not more general knowledge. The one size with a bigger
backbone is RF-DETR's first Large (2025, DINOv2 ViT-B), still in rfdetr 1.11.2 but deprecated.

Round 1 trained its backbone at RF-DETR's default rate, 1.5e-4, faster than the rest of the model.
A **gentler** rate, a tenth of that, keeps more of what the backbone already knew.

## The arms

Fixed before any arm ran.

| Arm | Size | Backbone | Decoder layers | Backbone rate | Question | Budget |
|---|---|---|---|---|---|---|
| round 1 (baseline) | Nano | ViT-S | 2 | 1.5e-4 | | already trained |
| `nano-gentle` | Nano | ViT-S | 2 | **1.5e-5** | Does keeping more of the backbone help? | 3 h |
| `medium` | **Medium** | ViT-S | **4** | 1.5e-4 | Does a bigger RF-DETR help? | 3.5 h |
| `vitb-gentle` (optional) | **Large 2025** at 672 px | **ViT-B** | 3 | 1.5e-5 | Does a bigger backbone help? Compare with `nano-gentle` | 9 h |
| `nano-frozen` (optional) | Nano | ViT-S, **frozen** | 2 | none | Does keeping all of it help? | 2.5 h |

The default run is `nano-gentle` and `medium`. After the first epoch each arm checks whether its 15
epochs fit its budget, and stops there if not. That way no arm is cut short on the clock and then
compared unfairly.

## What is compared

`comparison.json` (instance), on round 1's 25 test windows from 13 Cam 1 days no student trained on:

- **Agreement with SAM 3**, overall and per stratum (strong, weak, none, murky). This shouldn't fall.
- **Boxes per frame on empty and murky water.** These should fall.
- **Boxes with no SAM 3 box near**: the likeliest background boxes, counted as `hard_negatives`
  counts them. Most boxes SAM 3's labels don't match are the same fish drawn differently, so those
  are left out.
- **Speed** on the A10G.

`local_comparison.json` (PC) does the same on clips SAM 3 labelled on the PC, plus speed on your
RTX 5060 against the 15 FPS live stream.

**The limit of this test.** The PC has **one** SAM 3-labelled clip from a camera round 1 never saw:
your upload at the gate (`85ea19d4…`, 1,440 frames, about 2 minutes). On that clip, a filter for
things that never move removed 89% of round 1's boxes with no SAM 3 box near, so most of its mistakes
there are fixtures. One clip is a signal, not proof. More cameras need SAM 3 labels first, which
means a cloud labelling run.

## Decision rule (fixed before any results)

An arm is worth adopting for new cameras if all of these hold:

1. On the gate clip, its share of boxes with no SAM 3 box near is **at most half** of round 1's.
2. On the Cam 1 test days, its agreement is **no more than 0.02 below** round 1's 0.75.
3. On the PC it **keeps up with 15 FPS** (`keeps_up_with_live`).

If no arm passes, the answer is training data from the new cameras, not a different model.

## Cost estimate

Estimates from round 1's speeds, at $1.12 an hour for a g5.xlarge in eu-west-1, plus about $0.30 a
day for the disk.

| Stage | Time |
|---|---|
| Upload, install | 10 min |
| `weights`, `restore` (43 min last time), `data` | about 1 h |
| `nano-gentle`: 15 epochs, as round 1 (about 9 min each), then evaluation | about 2.5 h |
| `medium`: same backbone, two more decoder layers | about 2.5-3.2 h |
| **Default run** | **about 6.5-7 h, about $7-8** |
| Guard (worst case, then 30 more minutes) | 8 h 45 min, about $10 |
| `vitb-gentle`, if added: about 4 times the backbone's work | about 7-10 h more, about $8-11 |

The `weights` stage downloads two files from RF-DETR's own bucket (`storage.googleapis.com/rfdetr`):
the Medium COCO checkpoint, and the Large 2025 one if you add `vitb-gentle`. Their sizes aren't known
until fetched, roughly a few hundred MB each. Each is checked against rfdetr 1.11.2's published MD5
before anything loads it, and its SHA256 is recorded in `round-1-variants/weights.json`.

## 1. Launch the instance (you, in the console)

The same as the hard-negatives instance:

- Name `fish-sam3-variants`, type g5.xlarge in eu-west-1.
- AMI: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04)", the newest x86_64 one
  owned by Amazon.
- Key pair `fish-sam3-key`, security group `fish-sam3-ssh` (check it allows your current IP).
- 100 GB gp3 root, delete on termination. Shutdown behaviour **Stop**.
- Check nothing else is running: a second g5.xlarge needs more vCPU quota.

**If `fish-sam3-hard-negatives` still exists,** you can start it instead and unpack into `~/aithne-hn`
in step 3. Its 206 restored windows are already there, so `restore` skips most of its 43 minutes.

## 2. Upload (PC, PowerShell, repository root)

Rebuild the bundle after any code change.

```powershell
$env:FISHIAL_PREPROCESS='none'; .venv\Scripts\python.exe -m scripts.cloud_labelling.package_student_variants
$key = "$env:USERPROFILE\.ssh\fish-sam3-key.pem"
scp -o IdentitiesOnly=yes -i $key data\outputs\smartbay-rounds\round-1-variants-upload\aithne-student-variants.tgz ubuntu@<IP>:/home/ubuntu/
```

## 3. Set up (instance)

Compare the first command's result with `aithne-student-variants.tgz.sha256` on the PC.

```bash
sha256sum ~/aithne-student-variants.tgz
mkdir -p ~/aithne-variants && cd ~/aithne-variants && tar xzf ~/aithne-student-variants.tgz
sha256sum --quiet -c SHA256SUMS && echo "all files match"
AITHNE_NO_SAM3=1 scripts/cloud_labelling/bootstrap_labelling.sh install
```

## 4. Run (instance)

Default arms:

```bash
tmux new-session -d -s variants "scripts/cloud_labelling/run_student_variants.sh 2>&1 | tee -a variants.log"
```

With the bigger backbone too (the guard grows to cover it):

```bash
tmux new-session -d -s variants "scripts/cloud_labelling/run_student_variants.sh nano-gentle medium vitb-gentle 2>&1 | tee -a variants.log"
```

The run sets its shutdown guard first, and re-arms it for 30 minutes when it ends or fails. If an
arm fails, the next one still runs, and the comparison uses every arm that finished. Watch it with
`tail -f variants.log`. Per-stage logs are in `round-1-variants/logs/`.

## 5. Copy back (PC, Git Bash, repository root)

This keeps each arm's `checkpoint_best_total.pth`, which the PC step needs, and leaves the frames,
windows and the other checkpoints behind.

```bash
MSYS_NO_PATHCONV=1 ./.venv/Scripts/python.exe -m scripts.smartbay_archive_pull --host ubuntu@<IP> --remote /home/ubuntu/aithne-variants/data/outputs/smartbay-rounds/round-1-variants --local data/outputs/smartbay-rounds/round-1-variants --once --exclude "*.mp4" --exclude "*.part" --exclude "*.tmp" --exclude "*.lock" --exclude "dataset/*.jpg" --exclude "*.ckpt" --exclude "checkpoint.pth" --exclude "checkpoint[0-9]*.pth" --exclude "checkpoint_best_ema.pth" --exclude "last_ema.pth"
```

If you used `~/aithne-hn`, change `--remote` to match.

## 6. Score on the unseen camera (PC, PowerShell)

Close CVAT and set Windows to Best performance first, or the speed numbers will be low (see the
local GPU notes). `--render` writes an MP4 per student and clip, so you can watch what each one boxes.

```powershell
$env:FISHIAL_PREPROCESS='none'; $env:YOLO_OFFLINE='true'
.venv\Scripts\python.exe -m scripts.student_variants local --render `
  --video-dir data/outputs/sam3-botsort-20261014/videos/85ea19d4-db7b-4ed7-90a0-a8acccba2b0c `
  --video-dir data/outputs/sam3-botsort-20261014/videos/SmartBayCam1_2025-08-01_0000
```

The first clip is the gate, a camera no student trained on. The second is Cam 1 for reference: it's
from a round 1 training day, so it isn't a held-out day. Results:

- `data/outputs/smartbay-rounds/round-1-variants/local_comparison.json`;
- the renders, in each student's `local-eval/<clip>/` folder (round 1's are under
  `round-1/cloud/local-eval/`).

Then apply the decision rule above.

## 7. Clean up (you, in the console)

- Confirm the instance is **Stopped**, then terminate it once the results are on the PC.
- Check that no instance is left running that you don't expect, and that no volume is left over.

## Licences

RF-DETR's code and its Nano, Small, Medium and Large 2025 COCO weights are Apache-2.0 (rfdetr
1.11.2). XLarge and 2XLarge are under Roboflow's platform licence and aren't used. Every student is
trained on SAM 3's labels, so Meta's SAM License ships with it (`data/models/sam3/LICENSE`, in the
bundle). Video: Marine Institute SmartBay Observatory, CC BY 4.0.
