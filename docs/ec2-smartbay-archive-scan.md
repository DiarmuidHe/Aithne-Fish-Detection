# Resume the SmartBay archive scan on the EC2 A10G

This procedure continues the existing Cam 1 2025 ledger rather than rescanning completed
days. At the 2026-10-29 handoff it contained 60 completed days, 60,150 screened PNGs and
116 pending days. Cumulative SAM inference was 37,841.627 seconds. The first pending day
was 2025-08-19; one PNG had timed out.

## Measured EC2 probe — 2026-10-29

The deployment and bounded probe completed on the user's `g5.xlarge` at
`108.130.125.132`:

- AMI Python 3.13.16, PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130;
- NVIDIA A10G, 23,028 MiB visible VRAM, driver 595.91.07;
- SAM 3 checkpoint size and SHA-256 matched the pinned values;
- 2025-08-19 resumed and 2025-08-20 completed, with zero current-day file errors;
- 743 new PNGs used 168.992 seconds of GPU inference: **4.40 PNG/s**;
- bounded wall time was 327 seconds, including downloads, model load, MP4 sampling and reports;
- peak observed model residency was about 4,821 MiB VRAM, with utilization reaching 99%;
- ledger after the probe: 62 completed days, 60,893 PNGs, 114 pending days;
- persistent root disk after deployment: 21 GiB used, 76 GiB free.

The scanner stopped itself after the probe. Using the fresh day's 742 PNGs gives about
84,600 remaining PNGs and 5.3 inference hours; using the completed-history average gives
about 114,300 and 7.2 inference hours. Network transfers and per-day reporting make the
more relevant instance-time range roughly **10–14 hours**, or approximately **$11–16** at
the previously recorded $1.123/hour price. Recheck the current AWS price before treating
that dollar estimate as authoritative.

The screening output is candidate evidence, not truth. A SAM 3 detection helps rank likely
fish footage; no detection does not prove that a frame is empty. Training selection should
retain hard negatives, murky footage and varied light/time conditions rather than using
only the highest-yield hours.

For a bounded comparison with the other archive camera, use a seeded, seasonally spread
Cam 2 sample rather than a second full-year scan:

```bash
scripts/cloud_labelling/run_archive_scan.sh --camera 2 --year 2025 --sample-days 12 \
  --max-pngs 30 --max-mp4s 3 --retry-incomplete 3 --retry-delay-seconds 10
```

This selects one day from each of 12 consecutive slices of Cam 2's available 2025 season,
then screens at most 30 PNGs and samples at most three MP4s on each selected day. The day
list is persisted in the year ledger, so every resume uses the same sessions.

The bounded Cam 2 sample completed on 2026-10-29 in 4 minutes 24 seconds:

- 141 available days discovered; 12 selected and completed, 129 intentionally unselected;
- selected dates: 2025-07-18, 07-28, 08-07, 08-25, 09-05, 09-14, 10-10, 10-29,
  11-07, 11-24, 11-30 and 12-16;
- 330 PNGs and 34 MP4 samples processed (some selected days held fewer than the caps);
- 108.232 seconds of GPU inference and 1,376,468,955 bytes transferred;
- zero index errors, zero current-day file errors and zero pending sampled days.

## 1. Put a stop guard on the running instance

The instance is billed while it is running. In the SSH session, choose a limit that fits
your budget; this example stops it after 12 hours and leaves the 100 GB EBS root intact:

```bash
sudo shutdown -h +720
```

Cancel or replace the timer with `sudo shutdown -c`. Confirm in EC2 that the instance's
shutdown behaviour is **Stop**, not Terminate.

## 2. Copy the working code, scan ledger and checkpoint

Run these from PowerShell in the repository root. The destination is the persistent root
volume (`/home/ubuntu`), not `/opt/dlami/nvme`, which is erased when the instance stops.

```powershell
$key = "$env:USERPROFILE\.ssh\fish-sam3-key.pem"
$hostName = "ubuntu@108.130.125.132"

ssh -o IdentitiesOnly=yes -i $key $hostName "mkdir -p /home/ubuntu/Aithne-Fish-Detection/data/models/sam3 /home/ubuntu/Aithne-Fish-Detection/data/archive_scan"

tar -czf data/outputs/aithne-archive-code.tgz `
  app/__init__.py app/services/__init__.py app/services/external_datasets.py `
  scripts/sam3_autolabel.py scripts/smartbay_archive_scan scripts/cloud_labelling `
  scripts/cloud-labelling-archive-requirements.txt
scp -o IdentitiesOnly=yes -i $key data/outputs/aithne-archive-code.tgz "${hostName}:/home/ubuntu/"
ssh -o IdentitiesOnly=yes -i $key $hostName "tar -xzf /home/ubuntu/aithne-archive-code.tgz -C /home/ubuntu/Aithne-Fish-Detection"

scp -o IdentitiesOnly=yes -i $key data/models/sam3/sam3.pt data/models/sam3/LICENSE `
  "${hostName}:/home/ubuntu/Aithne-Fish-Detection/data/models/sam3/"

tar -czf data/outputs/aithne-archive-state.tgz -C data archive_scan
scp -o IdentitiesOnly=yes -i $key data/outputs/aithne-archive-state.tgz "${hostName}:/home/ubuntu/"
ssh -o IdentitiesOnly=yes -i $key $hostName "tar -xzf /home/ubuntu/aithne-archive-state.tgz -C /home/ubuntu/Aithne-Fish-Detection/data"
```

The checkpoint is 3.45 GB and the existing scan state is about 1 GB, so these uploads can
take tens of minutes on a typical home uplink. Re-running `scp` overwrites the same remote
files; it does not duplicate them.

## 3. Bootstrap and verify

```bash
cd /home/ubuntu/Aithne-Fish-Detection
chmod +x scripts/cloud_labelling/*.sh
scripts/cloud_labelling/bootstrap_archive_scan.sh
scripts/cloud_labelling/bootstrap_archive_scan.sh --check
```

The script runs the AMI's matched PyTorch 2.10/CUDA 13 environment directly and layers
hash-locked scanner additions from `.archive-python-packages/` through `PYTHONPATH`. It
uses `--no-deps`, so pip cannot replace Torch or CUDA, then verifies CUDA and the
checkpoint's exact 3,450,062,241-byte size and SHA-256 before any model load.

## 4. Measure one new day first

```bash
cd /home/ubuntu/Aithne-Fish-Detection
tmux new-session -d -s smartbay-scan \
  'scripts/cloud_labelling/run_archive_scan.sh 2>&1 | tee -a data/archive_scan/ec2-scan.log'
tmux attach -t smartbay-scan
```

Detach without stopping the scan with `Ctrl+B`, then `D`. The runner retries an incomplete
day up to three times and uses a file lock, so accidentally starting a second copy fails
safely. With no arguments it resumes the nearly finished 2025-08-19 day, then processes
one untouched day, and stops after completing **two** ledger days. This bounds the first
A10G probe while still measuring a substantially fresh day.

Status from a later SSH session:

```bash
cd /home/ubuntu/Aithne-Fish-Detection
scripts/cloud_labelling/archive_scan_status.sh
tail -n 80 data/archive_scan/ec2-scan.log
```

Subtract the before/after `gpu_inference_seconds` and `processed_pngs` in
`year_summary.json`. Use that measured PNG/s rate and the pending-day counts to estimate
the remaining hours before committing more instance time. The old laptop run averaged
about 1.59 screened PNG/s; do not assume the A10G rate until this probe finishes.

If the estimate fits your budget, run the rest explicitly:

```bash
tmux new-session -d -s smartbay-scan \
  'scripts/cloud_labelling/run_archive_scan.sh --camera 1 --year 2025 --all-pngs --max-mp4s 3 --retry-incomplete 3 --retry-delay-seconds 10 2>&1 | tee -a data/archive_scan/ec2-scan.log'
```

To stop cleanly, attach to tmux and press `Ctrl+C`. State is saved after each image and
transfer checkpoint. Run the same tmux command later to resume.

## 5. Bring results back before terminating

Stop the scanner first, then from PowerShell:

```powershell
$key = "$env:USERPROFILE\.ssh\fish-sam3-key.pem"
$hostName = "ubuntu@108.130.125.132"
ssh -o IdentitiesOnly=yes -i $key $hostName "tar -czf /home/ubuntu/aithne-archive-results.tgz -C /home/ubuntu/Aithne-Fish-Detection/data archive_scan"
scp -o IdentitiesOnly=yes -i $key "${hostName}:/home/ubuntu/aithne-archive-results.tgz" data/outputs/
tar -xzf data/outputs/aithne-archive-results.tgz -C data
```

Keep the instance stopped—not terminated—until the local summary and contact sheets have
been checked. Then regenerate the balanced training-window manifest with
`scripts.smartbay_label_selection`; its selection includes fish-rich, weak, clear-negative
and murky strata across time/light/clarity instead of simply taking the most detections.
