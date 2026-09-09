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

## Live-Camera Compatibility

Live RTSP ingestion is not implemented in v1. The intended future path is:

```text
RTSP camera
-> FFmpeg
-> 5-30 minute MP4 chunks
-> processing queue
-> VIAME
```

The service layer already separates video registration from HTTP upload through
`register_video_from_path`, so future chunk recorders can register filesystem videos without going
through the upload endpoint.

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
