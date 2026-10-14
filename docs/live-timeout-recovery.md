# Live detector timeout recovery — 2026-09-12

The failed Coral City session captured camera footage, but VIAME exceeded the
180-second deadline on 60-second batches. The worker discarded the detections
before rendering, leaving zero published frames and zero Fishial calls. The
generic skipped-segment message hid the timeout.

The original captured chunks were already deleted. Subsequent controlled runs
completed successfully: the same 20-second camera sample took 26.3 seconds with
shared output storage and 24.8 seconds with container-local output. These runs
do not establish why the original inference was unusually slow. Restarting the
live session restored annotated frames and identified all five requested fish
using 15 Fishial calls; 720 analyzed frames were saved to the library.

The live worker now recognizes detector timeouts separately from other failures.
After a timeout it reconnects with half the previous capture duration, down to
0.5 seconds, and reports the timeout and retry duration in the session error.
The reduced duration lasts for that session. FPS, input resolution, confidence
thresholds and species budgets remain unchanged. Consecutive failures still stop
at the configured retry limit. Successful inference logs elapsed time and
observation count to make future slowdowns diagnosable.

This deployment's `.env` now uses 30-second batches instead of 60 seconds to
reduce startup latency and the amount of work discarded on failure. Smaller
batches add model startup overhead; this is timeout recovery, not a guarantee
of real-time throughput under all GPU loads or scene conditions.

Verification: 135 focused runner, live runtime, monitor, recording and species
tests passed; one local FFmpeg-only test was skipped. Regressions exercise a
timeout followed by successful smaller-batch processing, ordinary detector
crashes, and bounded repeated timeout failures.

After deployment, session `6b51086d-040f-4d14-bb34-dc632e2711e0` reached
`running`, published 120 frames and identified three fish without a session
error. The first two 30-second batches took 77.4 and 45.9 seconds to infer.
The view was about 135 seconds behind capture at that check; processing is
restored, but reducing sustained latency remains a separate performance issue.
