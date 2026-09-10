# Fish Monitor

An operator-friendly underwater fish detection and tracking application built with FastAPI,
PostgreSQL, and VIAME. The FastAPI service includes a responsive web dashboard, so no Node.js
server or frontend build step is required.

The system treats VIAME as an isolated worker. The API stores uploaded or registered videos,
queues database-backed processing jobs, and a separate worker runs VIAME or mock mode, parses VIAME
CSV output, stores tracks and detections in PostgreSQL, and exposes JSON results through FastAPI.

```text
underwater video
-> VIAME fish detector/tracker
-> VIAME CSV output
-> Python parser
-> PostgreSQL
-> REST API
-> JSON results
```

## What Is Implemented

- Web dashboard at `http://localhost:8000` for upload, processing, results, and annotation.
- `POST /videos` upload for MP4/MOV/AVI/MKV using UUID storage paths and SHA-256 recording.
- `POST /videos/{video_id}/process` queueing without blocking the HTTP request.
- Database-backed worker with `queued -> processing -> completed/failed` states and heartbeats.
- VIAME subprocess boundary using `bash -lc` only for sourcing the VIAME environment.
- `VIAME_MOCK=true` mode that uses a realistic CSV fixture and needs no GPU or VIAME install.
- VIAME CSV parsing, track grouping, confidence filtering, timestamp calculation, and malformed-row tolerance.
- PostgreSQL-oriented SQLAlchemy models for videos, jobs, fish tracks, and detections.
- FastAPI endpoints for videos, jobs, tracks, and summaries.
- Synchronous annotated MP4 generation from stored database detections.
- Short cropped clips that follow one fish each, generated per video or per track and cached on disk.
- Docker Compose for API, worker, and PostgreSQL.
- Concurrency-safe queueing enforced by PostgreSQL: at most one queued/processing job per video.
- Alembic migrations applied automatically when the API container starts.
- Persistent per-track review decisions with original VIAME confidence and observations preserved.
- CSV exports for summaries, accepted/all tracks, observations, and selected/all-video batches.
- Batch queueing and annotation actions with per-video results and aggregate progress.
- Library counts, observation timelines, confidence histograms, and track-duration histograms.

## Quick Start

Choose **one** of these commands from the project directory.

### Mock mode (no NVIDIA GPU or VIAME required)

```bash
docker compose up -d --build
```

Mock mode exercises the complete upload, queue, database, results, and annotation workflow. It
uses the checked-in sample VIAME CSV instead of running a detector. The uploaded file must still be
a valid playable video if you want to generate an annotated MP4.

### Real VIAME mode (NVIDIA GPU required)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Always include both Compose files for real processing. Plain `docker compose up` starts the mock
worker and must not be used to validate real VIAME.

When the containers are healthy, open [http://localhost:8000](http://localhost:8000). The banner at
the top identifies the active mode, shows PostgreSQL and worker availability, and repeats the
correct startup command.

## Dashboard Workflow

1. Select **Choose a video**, pick an MP4, MOV, AVI, or MKV file, and select **Upload video**.
2. Select the video in the library and choose **Start processing**.
3. Leave the dashboard open. Queued and processing videos refresh automatically. If the worker is
   unavailable, the top banner explains which stack command to run.
4. When processing completes, review the accepted fish-track count, accepted detections, first and
   last fish times, pipeline, threshold, and track table. Enable **Show low-confidence tracks** to
   include stored tracks below the acceptance threshold.
5. Select **Generate annotated video**. The result appears in the page player; use **Download MP4**
   to save a copy.
6. Select **Generate fish clips** to cut one short cropped video per accepted fish track. Clips
   appear as a gallery under the annotated video, each following a single fish with its box and
   label drawn, and each downloadable on its own. Inspecting a track also offers **Show cropped
   clip of this fish**, which works for low-confidence tracks too.
6. If processing fails, the saved error appears beside the video. Select **Retry processing** to
   create a fresh job. Repeated clicks cannot create duplicate active jobs.

To stop either stack without deleting uploaded videos or the PostgreSQL volume:

```bash
docker compose down
```

If the GPU override is currently active, this equivalent command is also safe:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

## Terminology

`fish_tracks` means accepted VIAME track IDs, also described as fish passages. It is not a guaranteed
count of unique biological fish. A fish can leave the camera view and later receive a new VIAME track
ID, and two fish can be confused by the tracker.

For unreviewed or reviewed tracks, the counting rule is:

```text
fish passage count = number of VIAME tracks where max_confidence >= MIN_FISH_CONFIDENCE
```

Cross-track biological re-identification is a non-goal for v1.

## Reviewing tracks

Select a completed video and open **Review fish tracks**. Enable **Show low-confidence tracks
and other excluded tracks** to include every stored track. The VIAME column shows the original
threshold result separately from the effective accepted/excluded result. Click a track number to
inspect the original source video, exact stored frame numbers, confidence, and bounding boxes.
Click an observation frame to seek to that frame in the source player.

The review dropdown saves immediately:

| Review state | Included in accepted counts and annotations? |
| --- | --- |
| Unreviewed | Yes when original maximum confidence passes the run's threshold |
| Reviewed | Same threshold rule; records that the track was inspected |
| Accepted | Yes, including originally low-confidence tracks |
| Rejected | No; all observations and original scores remain stored |
| Needs-review | No, pending a decision |

`PATCH /tracks/{track_id}/review` accepts JSON such as `{"review_state":"rejected"}`.
Review state and UTC review time are persisted separately from detector evidence. Returning to
unreviewed clears the review timestamp. This is a current decision, not a multi-user audit log.
Review is allowed only for completed videos. Changes invalidate the existing annotated-video
reference; regenerate annotations to use the new decision. Boxes still come **only** from
FishTrack/FishDetection rows on their exact stored frames: no interpolation, prediction, or holding.
The parser's VIAME frame alignment and the GPU override's `VIAME_FRAME_NUMBER_OFFSET=-1` are unchanged.

**Process again** replaces the previous track results on successful completion, as before. The
new run receives new track IDs and starts unreviewed. Export results before reprocessing if you
need to retain a reviewed dataset. Failed retries leave the previous observations intact.

## Exporting CSVs

The completed-video summary has four CSV download links:

| Endpoint | Contents |
| --- | --- |
| `GET /videos/{id}/exports/summary.csv` | One row with accepted fish/observation totals and all-track count |
| `GET /videos/{id}/exports/accepted-tracks.csv` | Tracks accepted after applying review decisions |
| `GET /videos/{id}/exports/all-tracks.csv` | Every track, including low-confidence, rejected, and needs-review |
| `GET /videos/{id}/exports/detections.csv` | All original observations, including excluded tracks |
| `GET /exports/batch.csv` | All-video summaries; also includes uploaded/failed videos |

For selected videos, repeat `video_ids` and choose `kind`:
`/exports/batch.csv?kind=all-tracks&video_ids=UUID1&video_ids=UUID2`.
Valid kinds are `summary`, `accepted-tracks`, `all-tracks`, and `detections`.
Omitting `video_ids` exports all videos. Unknown IDs fail with 404; a selection supports up to
100 videos. Empty track/observation results still include a CSV header.

Exports include video ID, safe original filename, SHA-256, video FPS, source job ID, pipeline name,
model/version and VIAME version when known, confidence threshold, frame offset, downsample FPS,
processing mode, command digest, processing timestamps, and export time. Track/observation rows
also contain machine acceptance, effective acceptance, review state/time, and original data.
Pipeline/model path prefixes, storage paths, shell commands, and raw logs are not exported.
Potential spreadsheet formula strings are prefixed with an apostrophe. CSVs use UTF-8 with a BOM.

The source job is the job that **produced the results**, not a newer queued or failed retry.
Migration `0004_track_review` links legacy results to the latest successful job and defaults all
reviews to unreviewed. Missing historical configuration is left blank rather than invented.
Summaries without completed runs have blank job fields. Review and processing can change results,
so retain an exported CSV when you need a fixed report.

## Batch workflow

Use the checkboxes beside library videos (or **Select all videos**) independently of the video
currently open for inspection. **Process selected** queues each video using the same retry and
configuration-snapshot logic as individual processing. The database partial unique index remains
the final guard against concurrent requests creating more than one active job per video.

**Generate annotations for selected completed videos** renders eligible videos in sequence and
reports skips/failures separately. The dashboard shows the active filename, completed action count,
and a retained per-video result list. The progress bar then shows completed/failed videos out of
the selection, with queued/processing/uploaded counts alongside it. It reports video-level progress,
not a guessed percentage of frames processed. Select a CSV type and **Export selected** for combined
results; **Export all summaries** is available in the library analytics section.

API clients can send `{"video_ids":["UUID1","UUID2"]}` to `POST /batch/process` or
`POST /batch/annotate` (1–100 IDs; duplicates are deduplicated). Responses contain `results`,
`total`, `succeeded`, and `failed`; each result has `video_id`, `ok`, `status`, and a job ID,
annotation URL, or safe error. A failure does not undo successes. Annotation remains synchronous;
the UI submits one video per request so it can display progress. Keep the page open until annotation
actions finish. Processing jobs themselves remain durable in the database after closing the page.

## Analytics

**Fish activity** compares accepted track and accepted observation counts across the library.
Select a completed video for its accepted observations over time, confidence distribution across
all observations, and duration distribution across all tracks. Review changes update the accepted
totals and timeline; distributions keep rejected/low-confidence evidence for comparison.

`GET /analytics/videos` supplies library counts. `GET /analytics/videos/{id}?bins=20` returns a
bounded timeline (1–100 bins), ten confidence bins, and ten duration bins. Durations mean last minus
first observation time; a one-observation track has zero duration. Bins include their lower bound
and exclude their upper bound, except the final bin includes both. Missing timestamps/durations are
reported separately rather than inferred. Time windows with zero counts contain no accepted
observations; they do not imply that the detector examined every frame. Charts use native HTML/CSS
served by FastAPI, with visible numeric labels and responsive layouts; no Node server or CDN.

## Local Development

Create a Python environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy and edit configuration:

```bash
cp .env.example .env
```

For local PostgreSQL without Docker, set:

```text
DATABASE_URL=postgresql+psycopg://fish:fish@localhost:5432/fish_monitor
```

Run migrations:

```bash
alembic upgrade head
```

Start the API after applying migrations:

```bash
alembic upgrade head
uvicorn app.main:app --reload
```

Run the worker in another terminal:

```bash
VIAME_MOCK=true python -m app.workers.processing_worker
```

Run tests on Windows with the project environment used for this repository:

```bash
.venv\Scripts\python.exe -m pytest
```

The tests use SQLite and mock VIAME output, so they do not require PostgreSQL, VIAME, or an NVIDIA GPU.

For local video metadata and browser playback, put `ffprobe` and an `ffmpeg` build with libx264 on
PATH. Both are already installed in the API Docker image. Without them, local metadata may be
unknown and the OpenCV annotation fallback may not play in a browser.

For a separate GPU-free end-to-end validation database and two generated sample videos:

```powershell
.venv\Scripts\python.exe scripts/validate_product_mock.py
# Optionally serve the isolated validation dashboard (ensure port 8000 is free):
.venv\Scripts\python.exe scripts/validate_product_mock.py --serve
```

Each run uses a new directory below ignored `data/product-validation/`, migrates SQLite, queues
and processes two videos, generates both annotations, and checks exports/analytics. Stop its local
server with Ctrl+C. This helper does not access the main PostgreSQL database or validate real VIAME.

## Docker Services

Both startup commands run:

- `api` on `http://localhost:8000`
- `worker` in mock mode with the base file, or real GPU mode with the GPU override
- `postgres`

The API container runs `alembic upgrade head` before Uvicorn starts. The worker waits for both
PostgreSQL and the API health check, so it cannot claim a job against an outdated schema.

The API container does not require VIAME or GPU access. It does include FFmpeg plus
`opencv-python-headless` runtime support so it can probe uploaded videos and render annotated MP4
outputs.

The override builds `Dockerfile.worker.viame` from Kitware's `kitware/viame:gpu-algorithms-default`
image, sets `VIAME_MOCK=false`, and requests NVIDIA GPU access for the worker. The host must have
Docker GPU support configured. Plain `docker compose up` deliberately starts the mock worker and
must not be used to validate real VIAME.

The GPU override uses `app/viame_pipelines/tracker_fish_no_motion.pipe`, which combines VIAME's
standalone DEIMv2 fish detector with ByteTrack. The checked-in tracker thresholds are calibrated to
the standalone detector's score distribution instead of VIAME's much higher stock ByteTrack gates.
The override also raises sampling from VIAME's stock 5 FPS to 10 FPS. On the included real sample,
that changed the result from 3 tracks/19 observations to 9 tracks/182 observations, with 7 tracks
passing the existing `MIN_FISH_CONFIDENCE=0.60` rule. The official motion-fusion tracker was also
tested; it produced 7 accepted tracks/124 observations at 10 FPS, so the lighter required pipeline
remains the default.

The API container stays separate from VIAME and does not need GPU access.

## VIAME Setup

Default paths:

```text
VIAME_ROOT=/opt/noaa/viame
VIAME_SETUP_SCRIPT=/opt/noaa/viame/setup_viame.sh
VIAME_TRACKER_PIPELINE=/app/app/viame_pipelines/tracker_fish_no_motion.pipe
VIAME_DETECTOR_PIPELINE=/opt/noaa/viame/configs/pipelines/detector_default_fish_no_motion.pipe
```

The real-worker tuning knobs and their GPU Compose defaults are:

| Setting | Default | Effect |
| --- | ---: | --- |
| `VIAME_DOWNSAMPLE_FPS` | `10` | Frames per second sent to the detector; higher values improve temporal coverage at greater GPU cost. |
| `VIAME_DETECTOR_SCORE_THRESHOLD` | `0.10` | Lowest standalone detector score passed to ByteTrack. |
| `VIAME_TRACKER_HIGH_THRESHOLD` | `0.45` | Start of ByteTrack's high-confidence association pass. |
| `VIAME_TRACKER_LOW_THRESHOLD` | `0.10` | Lowest score considered in its recovery pass. |
| `VIAME_TRACKER_NEW_TRACK_THRESHOLD` | `0.50` | Minimum score that may initialize a track. |
| `VIAME_TRACKER_BUFFER_FRAMES` | `30` | Sampled frames for which a temporarily lost track is retained. |
| `MIN_FISH_CONFIDENCE` | `0.60` | Minimum track maximum used by accepted-track APIs and annotation. |
| `CLIP_PADDING_SECONDS` | `0.6` | Lead-in and lead-out around a track's observations in its cropped clip. |
| `CLIP_ZOOM_MARGIN` | `2.2` | Crop window size as a multiple of the track's largest bounding box. |
| `CLIP_MIN_CROP_PIXELS` | `128` | Smallest crop window, so small fish are not cut to a few pixels. |
| `CLIP_OUTPUT_SHORT_SIDE` | `260` | Target short side of a clip; small crops are scaled up to about this size. |

The threshold and FPS values can be placed in the project `.env` file because the Compose files
use environment interpolation for them. Rebuild/recreate the worker after changing a value. The
pipeline path and `VIAME_FRAME_NUMBER_OFFSET=-1` remain explicit in `docker-compose.gpu.yml`.

Every queued job records the mode, pipeline, model identifiers, downsample rate, frame offset,
detector and tracker thresholds, acceptance threshold, mock fixture name (when applicable), and a
digest of the administrator-controlled runner command. Workers execute the persisted numerical and
pipeline settings, so later environment changes do not silently alter an already queued job.

VIAME's `vidl_ffmpeg` track writer reports the first source frame as frame 1. The GPU worker keeps
`downsampler:renumber_frames=false` so downsampled frames retain source-video identifiers, then
applies `VIAME_FRAME_NUMBER_OFFSET=-1` while parsing to store zero-based frame numbers matching
OpenCV. Mock CSV fixtures remain zero-based and use the application default offset of `0`.

For an environment where motion fusion performs better, change `VIAME_TRACKER_PIPELINE` in the GPU
override to:

```text
/opt/noaa/viame/configs/pipelines/tracker_default_fish_fusion.pipe
```

That pipeline uses both DEIMv2 color detections and VIAME's motion-channel RF-DETR model and uses
more GPU memory. The same runner command and tracker knobs are compatible with it.

Check VIAME manually:

```bash
source /opt/noaa/viame/setup_viame.sh
viame -h
```

Or use the helper:

```bash
scripts/check_viame.sh
```

Run the default tracker pipeline manually:

```bash
scripts/run_viame.sh /path/to/underwater.mp4 /tmp/viame_tracks.csv
```

The default command used by the app is configured by `VIAME_RUN_COMMAND`:

```bash
source "$VIAME_SETUP_SCRIPT" && kwiver runner -I "$VIAME_ROOT/configs/pipelines" "$VIAME_PIPELINE" -s "input:video_filename=$VIAME_INPUT_VIDEO" -s "input:video_reader:type=vidl_ffmpeg" -s "downsampler:target_frame_rate=$VIAME_DOWNSAMPLE_FPS" -s "downsampler:renumber_frames=false" -s "detector:detector:onnx:score_thresh=$VIAME_DETECTOR_SCORE_THRESHOLD" -s "tracker:track_objects:bytetrack:high_thresh=$VIAME_TRACKER_HIGH_THRESHOLD" -s "tracker:track_objects:bytetrack:low_thresh=$VIAME_TRACKER_LOW_THRESHOLD" -s "tracker:track_objects:bytetrack:new_track_thresh=$VIAME_TRACKER_NEW_TRACK_THRESHOLD" -s "tracker:track_objects:bytetrack:track_buffer=$VIAME_TRACKER_BUFFER_FRAMES" -s "track_writer:file_name=$VIAME_OUTPUT_CSV"
```

Some VIAME installations use different pipeline process names or output writer settings. If that
varies, change `VIAME_RUN_COMMAND` in configuration rather than editing application code.

## API Usage

Upload a video:

```bash
curl -F "file=@underwater.mp4" http://localhost:8000/videos
```

Start processing:

```bash
curl -X POST http://localhost:8000/videos/{video_id}/process
```

Check job status:

```bash
curl http://localhost:8000/jobs/{job_id}
```

List accepted fish tracks:

```bash
curl http://localhost:8000/videos/{video_id}/tracks
```

List all stored tracks, including tracks below the confidence threshold:

```bash
curl "http://localhost:8000/videos/{video_id}/tracks?accepted_only=false"
```

Get summary:

```bash
curl http://localhost:8000/videos/{video_id}/summary
```

Generate an annotated MP4 after processing completes:

```bash
curl -X POST http://localhost:8000/videos/{video_id}/annotate
```

By default, labels include species/class text when available, for example
`track 12 | fish 0.94`. To draw only track IDs and detection confidence values:

```bash
curl -X POST "http://localhost:8000/videos/{video_id}/annotate?include_species=false"
```

Download or stream the generated annotated video:

```bash
curl -L http://localhost:8000/videos/{video_id}/annotated-video --output annotated.mp4
```

### Cropped Fish Clips

Cut one short clip per accepted fish, reusing any clip that already exists:

```bash
curl -X POST http://localhost:8000/videos/{video_id}/fish-clips
```

Add `?accepted_only=false` to include tracks below the threshold, or `?refresh=true` to re-render
clips that are already cached. A single track can be clipped on its own:

```bash
curl -X POST http://localhost:8000/tracks/{track_id}/clip
curl -L http://localhost:8000/tracks/{track_id}/clip?download=true --output fish.mp4
```

`GET /videos/{video_id}/fish-clips` lists the clips already on disk without rendering anything, and
`GET /tracks/{track_id}/clip` returns 404 until that track has been clipped.

Each clip covers the track's observed frames plus `CLIP_PADDING_SECONDS` of lead-in and lead-out.
The crop is a fixed-size window sized from the track's largest box times `CLIP_ZOOM_MARGIN`, floored
by `CLIP_MIN_CROP_PIXELS` and clamped to the source frame. Between reported frames the window
follows a straight line from one detection centre to the next, so the fish stays near the middle
without inventing boxes: rectangles and labels are still drawn only on frames with stored
detections. Small crops are scaled up to roughly `CLIP_OUTPUT_SHORT_SIDE` for legibility.

Clips for one video are rendered in a single pass over the source, so generating a gallery costs one
decode rather than one per fish. They are cached under `OUTPUT_ROOT/clips/{video_id}/{track_id}.mp4`
with a small JSON sidecar, are independent of review decisions, and are deleted when reprocessing
replaces the tracks they were rendered from. Review decisions change which clips the gallery lists,
not the clips themselves.

Annotation is synchronous in v1. It renders accepted `FishTrack` and `FishDetection` database rows,
not VIAME internals, so it works with real VIAME results and with `VIAME_MOCK=true`. Lower-confidence
tracks remain stored and available through `tracks?accepted_only=false`, but are not painted as fish.
Mock mode still requires the uploaded file to be a valid playable video for annotation, even though
mock processing itself only copies the CSV fixture.

Example summary:

```json
{
  "video_id": "uuid",
  "status": "completed",
  "fish_tracks": 37,
  "total_detections": 814,
  "mean_track_confidence": 0.91,
  "first_fish_timestamp_seconds": 14.2,
  "last_fish_timestamp_seconds": 532.8
}
```

## Architecture Notes

The v1 worker uses the database as the queue. It claims `queued` jobs with row locking where the
database supports it, marks the job `processing`, runs VIAME, parses CSV, replaces prior video
results, inserts tracks and detections, and marks the job `completed`. Errors are persisted on the
job and the video is marked `failed`.

A partial unique database index guarantees one active job per video even when process requests
arrive concurrently. The API also locks the video row while queueing. Workers publish a heartbeat
throughout long VIAME runs; the dashboard warns when no worker of the configured mode is available.
A processing job whose heartbeat has been stale beyond `JOB_STALE_AFTER_SECONDS` is marked failed
when retried, preventing a crashed worker from blocking that video indefinitely.

This is less operationally complex than Celery or RQ for a starter project, while preserving the
important production boundary: API and GPU worker are separate processes and containers.

Processing is retry-safe at the video-result level: before inserting parsed results for a completed
run, the worker deletes existing tracks/detections for that video. Each worker run writes to a unique
UUID directory under `JOB_ROOT`, so logs and CSV output are not overwritten.

## Live camera monitoring

Set `LIVE_MONITOR_ENABLED=true` in `.env`, then start the optional live worker with the GPU deployment:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile live up --build -d
```

Open the dashboard's **Live monitoring** panel, pick a camera, and select **Start monitoring**. The panel refreshes
every two seconds with an annotated camera snapshot, rolling activity, processing lag, and the latest
50 finalized fish histories. **Stop** finishes active histories and releases the camera connection.
The API queues work; the separate live worker must be running. Mock mode deliberately rejects live
monitoring so fixture detections are never presented as camera observations. Batch uploads and
processing continue through the existing worker.

For a local installation, install `pip install -e '.[live]'`, provide FFmpeg with libx264 on PATH and
a working VIAME installation, run `alembic upgrade head`, and run
`python -m app.workers.live_worker` alongside the API. Set `VIAME_MOCK=false` and
`LIVE_MONITOR_ENABLED=true` in both processes. The Docker API applies migrations through `0006_live_species_id`
on startup.

Four cameras ship in the registry, and the dashboard dropdown selects which one the pipeline
analyses:

| Key | Label | Source |
| --- | --- | --- |
| `coral-city` | Coral City Camera | Miami, Florida - page scrape plus yt-dlp |
| `smartbay-cam1` | SmartBay Cam 1 | Galway Bay, Ireland - direct HLS |
| `smartbay-cam2` | SmartBay Cam 2 | Galway Bay, Ireland - direct HLS |
| `smartbay-cam3` | SmartBay Cam 3 (ANERIS EMUAS) | Galway Bay, Ireland - direct HLS |

The SmartBay streams come from the Marine Institute SmartBay Observatory in Galway Bay and are
delivered via HEAnet; the observatory publishes them at <https://smartbay.marine.ie/>. A camera can
be reachable while publishing nothing, which is reported as "online but not currently broadcasting"
rather than an FFmpeg failure.

**Only one camera runs at a time.** The live worker runs one session to completion, and there is one
GPU, so the dropdown chooses the camera for the *next* session. Starting a second camera while one is
running returns 409 naming the running camera; switching means stopping first. The selector is
disabled while a session is open, and a running session pins the selector for every browser tab.

The registry is administrator configuration. `LIVE_SOURCE_REGISTRY` overrides it with a JSON array of
`{key, label, url, location}` objects (unset, or `[]`, keeps the cameras that ship in code),
`LIVE_DEFAULT_SOURCE_KEY` (default `coral-city`) picks the preselected camera, and `CORAL_CITY_URL`
still overrides the URL of the `coral-city` entry so existing `.env` files keep working. Keys must match `^[a-z0-9-]{1,64}$` and are unique; the settings
load fails otherwise. The API accepts a camera **key** only and answers 404 for an unknown key -
an operator-supplied URL would make FFmpeg and yt-dlp an SSRF sink.

The resolver returns a configured `.m3u8` URL unchanged (after checking the playlist advertises a
stream), and otherwise follows camera embeds and uses yt-dlp for hosted players such as YouTube,
refreshing expiring media URLs on reconnect. If a provider changes or restricts access, status
reports a resolution failure naming that camera. Update yt-dlp (and install its required JavaScript
runtime if requested by the provider), or configure an administrator-supplied direct HTTP(S) HLS URL.
Signed media URLs are never logged or returned by the API.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LIVE_SOURCE_REGISTRY` | four cameras | JSON array of selectable cameras (`key`, `label`, `url`, `location`) |
| `LIVE_DEFAULT_SOURCE_KEY` | `coral-city` | Camera preselected in the dashboard |
| `CORAL_CITY_URL` | `https://www.coralcitycamera.com/` | Overrides the `coral-city` entry's URL |
| `LIVE_LOST_TRACK_SECONDS` | `10` | Finalize after this interval without an observation |
| `LIVE_SEGMENT_SECONDS` | `2` | Continuous FFmpeg capture chunk duration |
| `LIVE_FPS` | `5` | Capture and VIAME sampling rate; capture is scaled to fit 1280×720 |
| `LIVE_ACTIVITY_WINDOW_SECONDS` | `60` | Dashboard rolling window |
| `LIVE_RETRY_SECONDS` / `LIVE_MAX_RETRY_SECONDS` | `3` / `30` | Exponential reconnect delay bounds |
| `LIVE_MAX_RETRIES` | `10` | Consecutive retries before failing the session |
| `LIVE_READ_TIMEOUT_SECONDS` | `20` | Source resolution/read timeout |
| `LIVE_DETECTOR_TIMEOUT_SECONDS` | `60` | Maximum VIAME time per chunk |
| `LIVE_WORKER_STALE_SECONDS` | `120` | Recover abandoned sessions on worker startup/poll |
| `LIVE_MAX_PENDING_SEGMENTS` | `3` | Backlog retained when inference falls behind |
| `LIVE_SCRATCH_ROOT` | system temp dir | Short-lived per-frame crop spool; keep it on fast local disk |

Size `LIVE_SEGMENT_SECONDS` against your VIAME startup cost. Every chunk starts a fresh pipeline, and
that startup dominates: on the reference GPU box it is about 12 seconds before any frame is analyzed.
A chunk shorter than the time it takes to analyze means the worker never keeps up and
`dropped_segments` climbs, so it sees only samples of the camera; a chunk comfortably longer gives
continuous coverage at the price of latency. Sampling rate is the other half of the budget: on a busy
reef `LIVE_FPS=5` cost about 30 seconds per 20-second chunk, while `LIVE_FPS=2` cost about 17 seconds
and found nearly the same distinct fish. Measure one chunk on your own footage, then set the segment
length above the measured time and `LIVE_DETECTOR_TIMEOUT_SECONDS` well above it (cold starts are
slower). This deployment uses 60-second chunks at 2 fps with a 180-second timeout: on its GPU a
60-second chunk costs about 37 seconds, so the worker stays ahead while the annotated view
trails the camera by a chunk plus its inference time.

Keep `LIVE_SCRATCH_ROOT` on local disk. Each detection spools an annotated crop frame, and small-file
writes to a bound volume are slow enough to decide whether inference keeps up: 420 crop writes take
about 13 seconds on a Windows bind mount against 0.3 seconds on container-local disk. Only finished
crops and clips are written under `OUTPUT_ROOT`; the spool is discarded once a track is finalized.

Lost-track timing uses capture timestamps. In-flight inference is applied before expiry, so visible
finalization is delayed by chunking/inference latency. The annotated view updates per analyzed chunk;
it is not a zero-latency broadcast. Keep an eye on lag and skipped segments: slow inference can miss
activity. VIAME IDs are local to a chunk; spatial/species association carries tracks between chunks.
These are tracking histories, not guaranteed identities of biological fish.

Annotated crops and cropped MP4 clips use the existing track colors, box labels, and crop settings.
Outputs live under `OUTPUT_ROOT/live/<session>/<track>/`; temporary source chunks are removed after
processing and temporary crop frames after successful clip rendering. Final media and database
observations are retained, so provision storage for long monitoring runs. A render failure preserves
the crop and detections and is reported on the track. Failed VIAME runs retain diagnostic logs under
the session's `inference` directory. A crashed worker's histories are finalized when
a live worker next recovers its stale session; start a new session to resume monitoring.

### Optional Fishial AI species identification

Set `FISHIAL_ENABLED=true`, `FISHIAL_CLIENT_ID`, and `FISHIAL_CLIENT_SECRET` in the
API and live worker environment (Compose forwards these from `.env`). Credentials come
from Fishial. The feature is off by default; enabling it without credentials fails configuration
validation. Before starting, set **Identify species for the first N fish** (0 = off) and
**Clear frames per fish** beside the camera selector. The browser remembers N. Controls lock
while a session is open, and unconfigured deployments hide them and reject nonzero N with 409.
The per-session frame override accepts 1–20; choosing fewer than the configured voting minimum
intentionally produces review required without image calls.

Only clean, expanded crops of already-detected fish go to the third-party **Fishial AI** API,
and only with both opt-ins. Full frames, annotated crops, letterboxing, camera URLs, and
operator-supplied images are never submitted. Crops stay in the scratch spool until voting
finishes, then are deleted. API secrets use redacted settings and do not enter logs/errors.
If the Fishial API key has a portal upload folder configured, Fishial may also retain submitted
crops in that collection; configure the third-party account accordingly.

Any track that clears the frame **floor** becomes an unpaid **candidate**: it stages crops
locally and costs nothing. Candidates are held in a bounded pool (`CANDIDATE_POOL_SIZE`,
by default several times the fish target), and when the pool is full a stronger track evicts
the weakest one, deleting its staged crops. Within a candidate, frames are **ranked, not gated**:
the best `MAX_STAGED_FRAMES_PER_CANDIDATE` are retained by measured crop quality, at most one
per `MIN_FRAME_SEPARATION_SECONDS` window so the votes stay independent rather than being cast
by near-duplicate frames of a single moment. A better frame evicts the retained worst.

Nothing is paid for until a track **finalizes**. Selection then ranks the finalized candidates
by `fishial_quality_score` — a weighted blend of median crop short side, sharpness, detector
confidence, contrast, colourfulness and retained frame count — rejects anything below
`QUALITY_FLOOR` with zero image calls, and spends the budget on the best. `species_id_fish_enrolled`
counts fish actually **selected**, not fish first sighted. Because the session cannot know its
best fish until it ends, `LATE_RESERVE_FISH` slots are held back until `RESERVE_AFTER_SECONDS`
have elapsed or shutdown releases them. Too few staged frames still means **Review required**
with no image calls. The same biological fish can become a new track after loss; the budget
applies to tracks, not individual animals.

This replaced first-come enrollment, which permanently committed a paid slot on a track's first
clear frame. Arrival time is uncorrelated with identifiability, and on the reference sessions it
spent the whole budget on fish that could not produce enough frames to vote.

Each attempted frame has one vote. Ambiguous predictions abstain; zero Fishial fish objects in
one crop abstains. A response carrying **several** objects is resolved against the target box
stored when the crop was written — the crop is deliberately centred on the detection, so the
fish we asked about is known — accepting the best match only when its IoU clears
`OBJECT_MATCH_MIN_IOU` and beats the runner-up by `OBJECT_MATCH_MIN_MARGIN`; otherwise it
abstains rather than risk attributing a neighbour's identity.

Ranked candidates are then filtered against the camera's **region** before voting. A camera with
no configured region filters nothing. Names are matched on `scientificName`, normalised for case
and whitespace only; there is no fuzzy or genus-level matching, so a near miss abstains. The
region is frozen on the session at start, so later edits to the camera registry cannot
retroactively reinterpret a finished session. A unique winning species must meet the minimum vote count,
the winning-votes / submitted-frames ratio, and the mean winning-score threshold together.
Failed, interrupted, and ambiguous image attempts remain in the ratio denominator; retries
never add votes. All failed frames produce `error`, shown as **Review required**.

A track also stops buying frames early, in both directions. **Decided**: the leader already
satisfies the consensus rule and no distribution of the remaining staged frames could unseat it.
**Unreachable**: even if every remaining frame voted for the leader, the rule still could not be
satisfied. **Declining classifier**: `MAX_EMPTY_RESPONSES` consecutive responses named no species
at all. Calls not made are counted in `species_id_calls_saved` and shown on the dashboard.

The five abstentions are five different operational problems and never collapse into one message:
`insufficient clear frames` (never staged enough views), `below quality floor` (staged, but not
worth paying for), `consensus unreachable` (the frames disagreed beyond recovery),
`classifier returned no candidates` (the model would not name it) and `budget exhausted`. A confident
result is labelled **Fishial AI**; the VIAME species field remains unchanged. The stored JSON
includes the full tally, frame numbers, scores, margins, raw responses, and abstention reasons.
The gallery updates results without replacing playing videos. The session breakdown includes
counts and mean confidence by identified species, review count, and image-call usage.

**Hard ceiling: `N * M + FISHIAL_MAX_API_RETRIES` image requests per session.**
`SpeciesIdentifier._reserve()` uses a conditional SQL increment immediately before *every*
recognition request, including 401/timeout/429/5xx retries, and commits the increment with a
per-frame attempt journal before sending. Retries also consume a separately derived, durable,
session-wide retry allowance in that JSON. Auth requests are not image calls. On restart,
uncertain sends remain spent and their frames abstain; they are never replayed. A crash just
before sending may conservatively overcount one reservation. Remaining crops stop at the ceiling.
No prediction table was necessary. Capture, claiming, heartbeat, and uploaded-video processing
retain their existing behavior. Long provider calls can increase processing lag; bounded
capture queues still apply. A forced worker kill can prevent completion, but cannot refund calls.

Defaults below use the `FISHIAL_` environment prefix. They favor abstention and bounded spending;
the model scores are not calibrated probabilities of correctness.

| Setting | Default | Rationale |
| --- | --- | --- |
| `ENABLED` | `false` | Requires explicit deployment and session opt-in. |
| `API_BASE_URL` | `https://api-recognition.fishial.ai/v2` | Current documented HTTPS contract. |
| `CLIENT_ID`, `CLIENT_SECRET` | unset | No embedded credentials. |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Bounds each outbound request. |
| `DEFAULT_FRAMES_PER_FISH` | `5` | Allows agreement across several views; range 1–20. |
| `MAX_FISH_PER_SESSION` | `20` | Caps operator enrollment; deployment range 1–200. |
| `MIN_FRAME_CONFIDENCE` | `0.50` | Floor only. At the detector's own threshold; rejects 0% of reference detections. |
| `MIN_CROP_PIXELS` | `40` | Floor only. Coral City's 10th percentile bbox short side; rejects 9.7% there, 0.5% on SmartBay 3. |
| `EDGE_MARGIN_PIXELS` | `4` | Rejects fish clipped by the frame edges. |
| `CROP_MARGIN` | `1.15` | Keeps nearby fins without excessive background; full-frame crops rejected. |
| `BLUR_MIN_VARIANCE` | `0` | Disabled: no sample-validated blur threshold; tune for the camera. |
| `MIN_FRAME_SEPARATION_SECONDS` | `0.6` | Spreads views over capture time, including across worker reconstruction. |
| `MIN_FRAMES_TO_VOTE` | `3` | Prevents guessing from a short glimpse. |
| `MIN_VOTES` | `3` | Requires repeated agreement. |
| `VOTE_RATIO` | `0.6` | Requires a majority including failed image attempts in the denominator. |
| `MIN_SPECIES_SCORE` | `0.5` | Rejects weak winning predictions in addition to agreement. |
| `MIN_FRAME_MARGIN` | `0` | Optional top-two ambiguity gate; default adds no unvalidated score-gap threshold. |
| `MAX_API_RETRIES` | `2` | At most two extra image calls across the entire session. |
| `CANDIDATE_POOL_SIZE` | `0` (auto) | `max(3 x fish_target, fish_target + 8)`, fixed at session start. Staging is free, so keep far more candidates than can be paid for. |
| `MAX_STAGED_FRAMES_PER_CANDIDATE` | `0` (auto) | `frames_per_fish + 2`. Spare frames cost nothing and give the selector and stop rules room. |
| `MAX_STAGED_BYTES` | `536870912` | Generous staging is free in API terms, not in disk terms. Weakest candidates are evicted first. |
| `QUALITY_FLOOR` | `0.0` | Admits everything. The blend's pixel terms are not yet calibrated against stored data; calibrate with the replay harness before raising it. |
| `QUALITY_WEIGHTS` | see `.env.example` | Crop size and sharpness dominate; detector confidence is weakest because it measures "is this a fish", not "is this crop identifiable". |
| `LATE_RESERVE_FISH` | `2` | Slots held for fish that arrive late; clamped below the fish target and always released at shutdown. |
| `RESERVE_AFTER_SECONDS` | `180` | When the reserve opens during a running session. |
| `MAX_EMPTY_RESPONSES` | `2` | Once a track starts coming back unnamed, the rest almost certainly will too. `0` disables. |
| `OBJECT_MATCH_MIN_IOU` | `0.5` | Minimum overlap with the stored target box before a multi-object response is attributed to this track. |
| `OBJECT_MATCH_MIN_MARGIN` | `0.2` | How far the best match must beat the runner-up; below this the frame abstains. |
| `REGION_FILTER_ENABLED` | `true` | Safe on by default: a camera with no configured region already filters nothing. |
| `PREPROCESS` | `none` | **Ships off.** No evidence yet that underwater correction helps this classifier; see below. |
| `CLAHE_CLIP` | `2.0` | Contrast limit when CLAHE is enabled. |
| `UPSCALE_SHORT_SIDE` | `0` | Disabled. Never downscales and never changes aspect ratio. |
| `KEEP_STAGED_CROPS` | `false` | Retains fish imagery on the output volume for offline replay. |

#### Tuning identification yield

Every raw Fishial response is persisted in `fishial_votes_json`, so a configuration change can be
evaluated against real stored data **without spending a single API call**:

```
python scripts/fishial_replay.py --dry-run --session <session-id>
```

It re-runs parsing, the multi-object matcher, the regional filter, voting, consensus and the stop
rules over the stored responses, and reports per track the old outcome versus the new one, plus
the calls the stop rules would have spent against the calls actually spent. Sweep a threshold with
`--set`, which accepts any `fishial_*` setting and validates it:

```
python scripts/fishial_replay.py --dry-run --session <id> --set fishial_max_empty_responses=1
```

Where to reach for each knob:

- **Nothing is ever staged / everything is "insufficient clear frames"** — the floor is above the
  data. Measure the detector's confidence and bbox short-side percentiles for that camera and set
  `MIN_FRAME_CONFIDENCE` and `MIN_CROP_PIXELS` below them. A floor rejecting more than ~25% of
  detections is doing selection's job and should be lowered.
- **The wrong fish get identified** — raise `QUALITY_FLOOR` (calibrate it with `--dry-run` first;
  it starts at 0.0) or widen `CANDIDATE_POOL_SIZE` so selection has more to choose from.
- **The budget runs out before the interesting fish appear** — raise `LATE_RESERVE_FISH` or lower
  `RESERVE_AFTER_SECONDS`.
- **Lots of calls, few answers** — check the `declined` count in `GET /live/{id}/species`. A high
  count means the model would not name these fish, which is not something a threshold can fix;
  lower `MAX_EMPTY_RESPONSES` to stop paying for it sooner.
- **Implausible species appear** — give the camera a `region` in the source registry and add its
  species list to `app/services/species_region.py`.

`--replay <directory> --max-calls <n>` is the only mode that spends money. It refuses to run
without an explicit ceiling and prints a running call count. Use it to A/B `PREPROCESS` variants
over stored crops, which requires `KEEP_STAGED_CROPS=true` on the run that produced them, since
the scratch wipe otherwise removes them.

**Preprocessing ships off (`PREPROCESS=none`), and should stay off until data says otherwise.**
Dim, green, low-contrast footage is a plausible reason the classifier declines to name fish, but
that is a hypothesis, not a measurement: no stored crops exist from any completed session, so no
A/B has been run. Turn on `KEEP_STAGED_CROPS`, capture a session, then use `--replay` to compare
variants under a small fixed budget before changing the default.

The adapter follows Fishial's [v2 API reference](https://docs.fishial.ai/api/api_reference)
and [tutorial](https://docs.fishial.ai/api/api_tutorial), checked on 2026-09-10: JSON credentials
to `/v2/auth`, a bearer token with a 600-second lifetime, then JPEG bytes directly to
`/v2/recognize`. Species IDs and certainty values resolve through `definitions[species_id].scientificName`.
This replaces the implementation prompt's older v1 signed-upload sketch. Token refresh occurs
before expiry and once after 401, subject to remaining image retry budget. Contract tests use
`httpx.MockTransport`; no tests contact Fishial, the internet, or cameras.

Migration `0007_live_species_quality` adds three session columns
(`species_id_candidate_pool_size`, `species_id_calls_saved`, `species_id_region`) and one track
column (`fishial_quality_score`), and widens the `fishial_state` check constraint to include
`candidate`. Postgres cannot alter a check constraint in place, so it is dropped and recreated in
both directions; the downgrade first retires any `candidate` rows to `disabled` so the narrower
constraint cannot fail on real data.

Migration `0006_live_species_id` adds five session columns (`species_id_enabled`,
`species_id_fish_target`, `species_id_frames_per_fish`, `species_id_fish_enrolled`,
`species_id_api_calls`) and six track columns (`fishial_state`, `fishial_species`,
`fishial_species_confidence`, `fishial_frames_used`, `fishial_votes_json`,
`fishial_completed_at`), with defaults, a state constraint, and a reversible downgrade.
`GET /live/sources` includes Fishial availability/defaults; `POST /live/start` accepts
`species_id_fish_target` and `species_id_frames_per_fish`. Session and track responses include
progress/results, and `GET /live/{session_id}/species` aggregates the whole session.

Small additions beyond the prompt's file list: Compose passes the same Fishial configuration to
both relevant services, and stale-worker recovery finishes or abstains outstanding identification
before removing its scratch crops. Deliberate conservative refinements include rejecting tied
winners, multiple-fish responses, full-frame crops, and non-HTTPS provider URLs. Call counts are
reserved before sending (rather than incremented afterwards) to make the crash ceiling reliable.

API: `GET /live/sources` (registry, defaults, and which camera is currently running),
`POST /live/start` with `{"source": "<key>"}`, `POST /live/{session_id}/stop`,
`GET /live/latest?source=<key>` (omitted `source` uses the default key), and
`GET /live/{session_id}/{status,activity,tracks,clips}`. Session payloads carry `source_key` and
`source_label`.
Track/clip lists accept `limit` and `offset`; tracks also accept `status=active|finalized`.
`GET /live/{session_id}/annotated-stream` serves MJPEG; add `?snapshot=true` for the latest JPEG.
Before the first frame it returns 503 with `Retry-After: 2`.
`GET /live/tracks/{track_id}/clip` and `/crop` serve generated media. Stopping is asynchronous while
an in-flight detector exits (bounded by its timeout).

## Security Defaults

- Uploaded filenames are retained only as display metadata.
- Stored video filenames are generated UUIDs.
- Annotated video filenames are generated UUIDs under `OUTPUT_ROOT`.
- Fish clips are written under `OUTPUT_ROOT/clips` using video and track UUIDs, and paths are validated to stay inside it.
- Upload size and extensions are configurable.
- The API does not expose arbitrary filesystem path ingestion.
- Public job responses omit worker log paths and never return log contents or the configured shell command.
- VIAME is run through a fixed/admin-configured command, not user-provided shell input.
- User-controlled paths are not interpolated unquoted into shell commands.
- Per-job stdout and stderr are stored under UUID job directories.
- Error messages are bounded before persistence.

## Non-Goals For Version 1

- Species recognition from scratch
- Individual fish re-identification
- Fish biomass estimation
- Stereo length measurement
- Live RTSP ingestion
- Custom model training
- Mobile app
