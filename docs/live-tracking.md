# Live tracking diagnosis

Repository and running-worker inspection on 2026-09-12 established:

1. **Sampling:** the running worker and local `.env` use `LIVE_FPS=2`, or one frame
   every 0.5 seconds. `Settings`, Compose fallbacks and `.env.example` default to 5 FPS.
2. **Frames removed before inference:** `SegmentCapture` applies FFmpeg's `fps` filter
   before writing segments. `detect_segment` overrides VIAME's batch downsample target
   with the same live rate. The worker reads every decoded frame, including frames
   without detections. There is no additional every-other-frame loop. Capture backlog
   limits and failed inference can discard whole segments.
3. **VIAME IDs:** each segment starts a fresh runner, so IDs are segment-local. The
   running deployment uses 60-second segments; the code default is 2 seconds. Frequent
   two-second restarts were therefore not the explanation for this deployment.
4. **Previous stitching:** `begin_segment` cleared local ID mappings, but within a
   segment the ID bypassed all geometry, class and time checks. Other observations
   were greedily matched by last-box IoU or center proximity, without motion or
   ambiguity checks. A moving fish could lose its identity across a segment while
   an unrelated fish near its abandoned box inherited its track.
5. **Reassignment conditions:** a repeated VIAME ID could jump arbitrarily, change
   size or class, or reappear anywhere during the 10-second active lifetime. Without
   a repeated ID, nearby fish and crossings could still be misassigned by greedy
   last-box matching. No appearance evidence was required in either path.
6. **Causes:** unsafe association is the demonstrated identity-switch bug. A separate
   worker bug expired capture-time tracks against wall time while waiting or retrying;
   segment/inference latency could finalize tracks before continuous footage arrived.
   Sparse sampling increases motion between observations and makes both VIAME tracking
   and last-box matching harder. The configured ByteTrack pipeline has permissive
   matching thresholds (`0.98`) and a 30-processed-frame buffer, but this investigation
   does not establish a threshold change that improves identity accuracy. Those settings
   and detection confidence thresholds are unchanged.

## Fix and verification

All associations now pass capture-time, class, area/aspect and normalized displacement
gates. The previous observation and current box give a constant-velocity prediction,
held in memory across segment boundaries without a schema change. Misses require
tighter agreement with that prediction; without history, reacquisition requires strong
overlap. No association is allowed beyond two seconds, even while the track remains
active for media finalization. A local ID is only a small scoring hint. Mutual best
matches need a margin over alternatives in both directions, so greedy removal cannot
promote a doubtful second choice. Old ID aliases and expired motion entries are removed.

Worker expiry now advances only through analyzed capture timestamps. Empty frames
and skipped footage still age tracks; waiting for footage does not. Stop and terminal
failure paths still finalize retained tracks. Fishial spending, species identification,
recording finalization and frontend behavior are unchanged.

Focused regressions cover large 2-FPS steps with new segment IDs, implausible jumps,
size/aspect/class changes despite repeated IDs, crossings in either input order,
brief misses with valid motion recovery, stale-track rejection, ambiguity in either
matching direction, and valid existing cross-segment matching. Real encoded-segment
worker tests distinguish capture delay from actual missing footage. A worker-to-library
test checks that crossing trajectories remain separate in the published recording.
Runtime tests verify capture and VIAME use the same rate at 2 and 5 FPS.

The pre-fix tests reproduced 12 association failures and two worker timing failures.
Run the requested suites with:

```powershell
# Process-only override if the local .env enables enhancement without a model path/checksum.
$env:FISHIAL_PREPROCESS='none'
.venv\Scripts\python.exe -m pytest tests/test_live_monitor.py -q
.venv\Scripts\python.exe -m pytest tests/test_live_recording.py tests/test_live_video_runtime.py tests/test_live_species.py -q
```

Combined result: 126 passed, one FFmpeg-only test skipped because FFmpeg was not on
the local PATH. Recording integration used the existing OpenCV assembly fallback.
These are workspace changes; the running worker image must be rebuilt and restarted
to apply them. No live session was interrupted during this investigation.

## Sampling and limits

Leave the deployed 2 FPS and default 5 FPS unchanged. A 2-to-5 FPS trial would process
2.5 times as many frames; fixed startup costs mean runtime need not scale exactly with
that factor. Measure identity continuity as well as fish counts, inference time,
end-to-end latency and dropped segments before adopting a higher rate.

This is conservative geometry/motion association, not visual re-identification. Fully
occluded or visually indistinguishable fish can remain unresolvable; uncertainty is
allowed to fragment tracks rather than forcing a match. Sudden turns, large scale
changes, long gaps and worker reloads may therefore create new identities. No live
camera identity-accuracy benchmark is claimed by the synthetic regressions.
