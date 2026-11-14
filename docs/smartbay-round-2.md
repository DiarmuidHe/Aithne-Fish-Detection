# SmartBay round 2: more months and Cam 2, on four instances

Round 1's student was trained on SmartBay Cam 1 from 19 June to 1 October 2025 only. It does well
there and makes mistakes on Cam 2. Round 2 adds:

- **Cam 1, 2 October to 30 November 2025**: days the Cam 1 archive scan already screened.
- **Cam 2**: its 13 sampled days plus **36 more**, scanned for round 2. Cam 2 days get their own
  training, test and expert days (keyed `<date>@SmartBayCam2`). Cam 2's 17 July is forced to *test*:
  the PC's one SAM 3-labelled Cam 2 clip (upload `85ea19d4`) is from that day.

Round 0 and 1's 212 windows **keep their labels**; only the new windows are labelled. Every number
is agreement with SAM 3's unreviewed labels, never accuracy.

## Decisions (2026-11-14)

| Setting | Value |
|---|---|
| Instances | 4 × g5.xlarge (the 16-vCPU quota), launched from one image; 1 trains |
| Hard cap | **€60** for round 2, of the €200 available |
| Cam 2 scan | 36 more days, one from each slice of Cam 2's 2025 season, at the first sample's settings (30 PNGs and 3 MP4s a day, seed 17) |
| Windows per day, role shares | As before: 3 per training day, 2 per test day; 10% expert and 15% test of each camera's new days |
| Training | RF-DETR Nano at 640 px, epochs sized to a 6 h budget (about 15, as round 1), round 1's validation days kept |

## Cost estimate

At about $1.12 an hour per g5.xlarge in eu-west-1. Estimates from round 1's measured speeds.

| Stage | Instance hours | Cost |
|---|---|---|
| Base instance set-up and Cam 2 scan | about 1.5 | about $2 |
| Labelling about 280 windows, about 27 an hour per instance | about 10.5 over 4 instances (2.6 h each) | about $12 |
| Waiting while the PC merges | about 0.5 | about $0.50 |
| Restore, dataset, 15 epochs, evaluation (about 2.3 times round 1's data) | about 7.5 | about $8.50 |
| Disks (100 GB each) for a day, and the image snapshot for a month | | about $5 |
| **Total** | | **about $28-33, about €26-30** |

Delete the image and its snapshot when round 2 is done: they cost about $3 a month.

## 0. The base instance (done 2026-11-14)

`i-078d7d00d8602bb37` (the hard-negatives instance, restarted). Set up in `~/aithne`:

- code bundle `aithne-base-code.tgz` (`scripts.cloud_labelling.package_base_image`), every file
  checked against `SHA256SUMS`;
- SAM 3 (`sam3.pt`, uploaded, size and SHA256 checked) and RF-DETR Nano (pinned SHA256);
- the labelling packages (copied from the 7 November install: the lock differs only in comments)
  and the archive scanner's packages, both checked by their bootstrap scripts, with a 30-frame SAM 3
  run on Cam 2 footage;
- round 1's 206 restored windows, hard-linked from `~/aithne-hn`, so training needn't fetch them.

**You:** create the image from it: Actions → Image and templates → Create image, name
`fish-sam3-round2-base`, **untick Reboot instance**. Wait for **Available**.

## 1. Cam 2 scan (base instance)

```bash
cd ~/aithne
tmux new-session -d -s cam2-scan "scripts/cloud_labelling/run_scan_days.sh 2 cam2-days.txt 2>&1 | tee -a cam2-scan.log"
```

`cam2-days.txt` came from (PC):

```powershell
.venv\Scripts\python.exe -m scripts.smartbay_label_selection --pick-scan-days 36 --camera SmartBayCam2
```

Some Cam 2 days have no PNGs in the archive (12 and 20 July 2025): those can't be screened and
become *unused*, as 14 September already was.

Copy the scan back (PC, Git Bash). No local file is ever deleted:

```bash
MSYS_NO_PATHCONV=1 ./.venv/Scripts/python.exe -m scripts.smartbay_archive_pull --host ubuntu@<base IP> --remote /home/ubuntu/aithne/data/archive_scan --local data/archive_scan --once --exclude ".cloud-scan.lock"
```

## 2. Choose the windows (PC)

```powershell
.venv\Scripts\python.exe -m scripts.smartbay_label_selection `
  --keep data/outputs/smartbay-rounds/round-1/selection --add-camera SmartBayCam2 --only-new
```

This writes `data/outputs/smartbay-rounds/round-2/selection/`. `summary.json` has Cam 1's new days
under `this_round`, Cam 2's under `other_cameras`, and `to_label.by_camera`. Look at the contact
sheets before going on; this is the point where the labelling cost is fixed.

## 3. Label on four instances

Build the one bundle every labeller uses (PC):

```powershell
$env:FISHIAL_PREPROCESS='none'; .venv\Scripts\python.exe -m scripts.cloud_labelling.round2 labelling-bundle
```

**You:** launch 3 more g5.xlarge from `fish-sam3-round2-base`: names `fish-sam3-label-1` to `-3`, key
`fish-sam3-key`, group `fish-sam3-ssh`, shutdown behaviour **Stop**. The base instance is shard 0.

Upload to each, check, unpack and start (`i` is 0 for the base, 1 to 3 for the others):

```bash
scp -o IdentitiesOnly=yes -i ~/.ssh/fish-sam3-key.pem data/outputs/smartbay-rounds/round-2/upload/aithne-round2-labelling.tgz ubuntu@<IP>:
ssh -o IdentitiesOnly=yes -i ~/.ssh/fish-sam3-key.pem ubuntu@<IP>
sha256sum aithne-round2-labelling.tgz      # compare with the .sha256 file on the PC
cd ~/aithne && tar xzf ~/aithne-round2-labelling.tgz && sha256sum --quiet -c SHA256SUMS && echo "all files match"
tmux new-session -d -s label "scripts/cloud_labelling/run_round2.sh label <i> 4 2>&1 | tee -a round2-label.log"
```

Unpack the labelling bundle **once** per instance: it carries a fresh journal. Each run sets a 4 h
guard and re-arms it for 20 minutes when labelling ends.

## 4. Copy back and merge (PC)

When an instance's `round2-label.log` says labelling finished (Git Bash):

```bash
MSYS_NO_PATHCONV=1 ./.venv/Scripts/python.exe -m scripts.smartbay_archive_pull --host ubuntu@<IP> --remote /home/ubuntu/aithne/data/outputs/smartbay-rounds/round-2 --local data/outputs/smartbay-rounds/round-2/shards/<i> --once --exclude "*.mp4" --exclude "*.part" --exclude "*.tmp" --exclude "*.lock" --exclude "*.jpg" --exclude "*.png" --exclude "logs/*"
```

Then stop the labellers 1-3 (you, console). Keep the base: it trains.

```powershell
$env:FISHIAL_PREPROCESS='none'
.venv\Scripts\python.exe -m scripts.cloud_labelling.round2 merge `
  --shard data/outputs/smartbay-rounds/round-2/shards/0 --shard data/outputs/smartbay-rounds/round-2/shards/1 `
  --shard data/outputs/smartbay-rounds/round-2/shards/2 --shard data/outputs/smartbay-rounds/round-2/shards/3
.venv\Scripts\python.exe -m scripts.cloud_labelling.round2 training-bundle
```

`merge` stops if two sources hold different bytes for one file, or two labellers both ran one step.
`round-2/cloud/merge_summary.json` lists any window no labeller finished; the round checks
(`checks/quality_report.json`) judge every window by round 1's exclusion rules.

## 5. Train (base instance)

The base has round 1's windows and its own shard's, and enough disk (about 65 GB free).

```bash
scp ... aithne-round2-training.tgz ubuntu@<base IP>:     # as in step 3, then check its SHA256
cd ~/aithne && tar xzf ~/aithne-round2-training.tgz && sha256sum --quiet -c SHA256SUMS && echo "all files match"
tmux new-session -d -s train "scripts/cloud_labelling/run_round2.sh train 2>&1 | tee -a round2-train.log"
```

`restore` fetches every window it lacks from the public archive and checks each source's SHA256;
then the dataset (round 1's validation days kept, 10% of the new training days added), about 15
epochs and the evaluation. 8 h guard.

Copy back as in step 4 into `data/outputs/smartbay-rounds/round-2/cloud`, adding
`--exclude "dataset/*.jpg" --exclude "*.ckpt" --exclude "checkpoint.pth" --exclude "checkpoint[0-9]*.pth"`.
`student/round_eval.json` has `by_camera`: Cam 2's test days on their own.

## One instance instead of four (what ran on 2026-11-14)

The user chose to label on the base instance alone. Same cost, about 7.5 h of labelling instead of
2.5, and no PC step between labelling and training:

```powershell
$env:FISHIAL_PREPROCESS='none'
.venv\Scripts\python.exe -m scripts.cloud_labelling.round2 labelling-bundle --shards 1   # sam3_gpu 73,200 s
.venv\Scripts\python.exe -m scripts.cloud_labelling.round2 round1-bundle                 # round 1's labels, 57 MB
```

On the instance, after unpacking both (the round 1 labels into a scratch folder, checked, then
`cp -an data ~/aithne/` so nothing is overwritten):

```bash
tmux new-session -d -s label "AITHNE_GUARD_MINUTES=660 scripts/cloud_labelling/run_round2.sh label 0 1 2>&1 | tee -a round2-label.log"
tmux new-session -d -s after "scripts/cloud_labelling/run_round2.sh after-label 2>&1 | tee -a round2-after.log"
```

`after-label` waits for the `label` session to end, merges round 1's labels with the instance's into
`round-2-train`, stops unless at least 90% of the new windows finished labelling, then restores the
missing windows and trains as in step 5, under an 8 h guard. Copy back
`data/outputs/smartbay-rounds/round-2-train` (with the step 5 excludes) and `round-2` (labels and logs).

Never overwrite a running shell script in place: bash reads it as it runs. Upload a new copy under
another name and `mv` it over.

### Adding three labellers mid-run (2026-11-14, 21:15 UTC)

The user later had time to launch three more g5.xlarge from `fish-sam3-round2-base`. The base kept
labelling; the end of its two workers' lists went to the new instances.

1. Bundle for the new instances (PC): `round2 labelling-bundle --shards 4 --output
   data/outputs/smartbay-rounds/round-2/upload/aithne-round2-labellers.tgz` (SAM 3 budget 18,300 s
   each). Upload, check and unpack it on each as in step 3, but start nothing yet.
2. Test the copy path on labeller 1 before it labels (PC, Git Bash), into a scratch folder on the
   base, then delete that folder:
   ```bash
   ssh ... ubuntu@<labeller 1> 'echo "labelling finished (relay test)" > ~/relay-test.log'
   AITHNE_RELAY_LOG=relay-test.log AITHNE_RELAY_TARGET=round-2-relay-test AITHNE_RELAY_HOURS=0 \
     scripts/cloud_labelling/relay_shards.sh <base IP> <labeller 1 IP>
   ```
3. Split (base). This writes `round-2/split/instance-<i>.csv` and marks those windows labelled in
   the base's journal, so its workers skip them. `--dry-run` first shows the counts:
   ```bash
   PYTHONPATH=.labelling-python-packages /opt/pytorch/bin/python -m scripts.cloud_labelling.round2 split --instances 3 --head-start 4
   ```
4. Copy `split/instance-<i>.csv` to labeller i's `round-2/split/`, then start it:
   ```bash
   tmux new-session -d -s label "AITHNE_TO_LABEL=data/outputs/smartbay-rounds/round-2/split/instance-<i>.csv scripts/cloud_labelling/run_round2.sh label 0 1 2>&1 | tee -a round2-label.log"
   ```
5. On the base, replace the waiting `after` session (it had set no guard yet):
   ```bash
   tmux kill-session -t after
   tmux new-session -d -s after "AITHNE_EXTRA_SHARDS='data/outputs/smartbay-rounds/round-2-shards/1 data/outputs/smartbay-rounds/round-2-shards/2 data/outputs/smartbay-rounds/round-2-shards/3' scripts/cloud_labelling/run_round2.sh after-label 2>&1 | tee -a round2-after.log"
   ```
6. On the PC, in the background, relay each labeller's results to the base when it finishes:
   `scripts/cloud_labelling/relay_shards.sh <base IP> <IP 1> <IP 2> <IP 3>`. The PC must stay awake
   until the last labeller is copied; the base waits up to 4 h for them, then merges what arrived
   (and trains only if at least 90% of the new windows are labelled).

Restore then fetches the new instances' windows from the public archive, as the base never had
their videos. The labellers stop 20 minutes after labelling ends; terminate them once their
archives are in `round-2/relay` on the PC.

## 6. Clean up (you, console)

- Terminate the labellers once their folders are on the PC, and the base once training's results are.
- Deregister `fish-sam3-round2-base` and delete its snapshot (EC2 → Snapshots).
- Check no instance or volume is left that you don't expect.

## Licences

Video: Marine Institute SmartBay Observatory, CC BY 4.0. SAM 3: Meta SAM License, which travels with
the student. RF-DETR and its COCO weights: Apache-2.0.
