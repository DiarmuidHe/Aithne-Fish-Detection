# Automated SmartBay preprocessing screen

The replay command can now collect clean crops, compare preprocessing and write a
report in one bounded run. It reuses live tracking, candidate ranking, quality gates,
target-box matching, the exact regional filter and the existing preprocessing modes.

Run from the repository in the existing GPU environment (PowerShell):

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps --volume "${PWD}:/app:ro" --entrypoint python3 live-worker scripts/fishial_replay.py --capture-source smartbay-cam3 --capture-seconds 120 --compare-preprocess --replay /data/outputs/fishial-experiments/smartbay-screen-20260910 --max-calls 6
```

The camera is captured for at most 120 seconds plus one bounded detector pass. The
collector uses its own in-memory database and never runs a species-identification
pass or changes a production session. It waits for two tracks with enough clear
frames, selects each track's best crop, and retains the original JPEG, quality
measurements and expected target box. If two suitable tracks are unavailable, it
exits without making Fishial calls.

The paid phase runs only through `scripts/fishial_replay.py --replay ... --max-calls 6`.
It sends the two originals, then the same two crops with CLAHE, then white balance.
Six is the total image-request ceiling, including retries and authentication refresh
retries. Retries can leave fewer than six completed comparisons. Each request is
reserved durably in `replay.sqlite` before sending. Restarting with the same directory
reuses the capture and finished results; uncertain requests are not resent. Changing
inputs, settings or the budget in an existing experiment is refused.

The output directory contains:

- `crop-1.jpg`, `crop-2.jpg`: clean originals, without overlays.
- `manifest.json`: capture source, region, track/frame identifiers and target boxes.
- `variants/`: the actual JPEG bytes used for each mode.
- `replay.sqlite`: persistent experiment specification, reservations and responses.
- `report.json`: reserved-call count, per-crop species results, rejected regional
  candidates, object matches, usable-answer counts by mode and a recommendation.

The report distinguishes a usable classifier answer from verified species accuracy.
Two crops cannot establish production accuracy, so this screen leaves
`fishial_preprocess=none` and the voting thresholds unchanged. It does not start a
recurring paid experiment. Rerunning the same command resumes the same six-call
experiment rather than creating a new allowance. Preserve its journal.

For standalone deployment, the same Python command runs wherever real VIAME,
FFmpeg and configured Fishial credentials are available. The Docker command mounts
the current code read-only and does not rebuild or replace the running services.

## Completed run — 2026-09-10

The automatic capture selected two eligible tracks after 20 decoded frames and
saved the originals at 14:37:45 UTC. Capture spent zero Fishial calls. Replay then
spent exactly six image calls, with no retries:

| Mode | Crop 1 | Crop 2 | Usable species answers |
| --- | --- | --- | ---: |
| none | One object, empty species | One object, empty species | 0/2 |
| clahe | No objects | One object, empty species | 0/2 |
| white_balance | No objects | One object, empty species | 0/2 |

No candidate was dropped by the regional filter: the classifier supplied none.
The saved clean images are hazy, dark and blurred, with strong glare; their measured
sharpness values are 6.56 and 6.96. This small experiment provides no justification
for changing preprocessing, so `none` remains configured.

The real command was rerun against the same experiment directory; it reused the
capture and report and retained exactly six total reservations with no new requests.
Results are in `data/outputs/fishial-experiments/smartbay-screen-20260910/report.json`.

All 213 tests passed using the requested full-suite command, including eight new
experiment tests covering capture without a client, actual mode transformations,
target matching, 401/server-error/timeout retries, restarts, uncertain reservations
and rejected nonpositive budgets. The live session reservation path is unchanged.
