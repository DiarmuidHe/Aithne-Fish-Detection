# Product layer implementation and validation

Validated on 9 September 2026.

## Result

The FastAPI dashboard now supports saved track review, per-video and batch CSV exports,
multi-video processing/annotation actions, and five analytics views. Rejected and needs-review
tracks remain in the database. Original VIAME confidences and observations are preserved.
Human acceptance decisions affect counts, accepted exports, timelines, and annotations.

Annotation frames remain exclusively the exact frames recorded in FishDetection rows joined
to FishTrack. The VIAME parser, frame offset configuration, downsampling, and GPU startup
configuration were not changed. No prediction, interpolation, or frame-holding was introduced.

## Changed files

| Files | Purpose |
| --- | --- |
| `app/db/models.py`, `app/db/migrations/versions/0004_track_review.py` | Review state/time, source processing job, constraint, and migration/backfill |
| `app/schemas/track.py`, `app/api/tracks.py` | Review API validation, completed-video guard, original observation inspection |
| `app/api/videos.py` | Review-aware summaries, protected source-video streaming, annotation serialization using video lock |
| `app/api/batch.py` | Deduplicated batch processing/annotation, safe per-video failures |
| `app/api/exports.py`, `app/services/reporting.py` | CSV streaming, safe allowlisted provenance, source-job attribution, formula protection |
| `app/api/analytics.py` | Library counts, bounded time/confidence/duration distributions, missing-data reporting |
| `app/services/fish_counter.py`, `app/services/video_annotator.py` | Shared acceptance decisions; exact database annotation frames retained |
| `app/workers/processing_worker.py` | Save the producing job ID on new tracks |
| `app/main.py` | Register new API routers |
| `app/templates/dashboard.html`, `app/static/dashboard.js`, `app/static/dashboard.css`, `app/static/product.js` | Batch controls, review inspector, exports, charts, responsive layout |
| `tests/test_product.py` | Review/export/batch/analytics/concurrency/migration regression coverage |
| `scripts/validate_product_mock.py` | Isolated, repeatable mock end-to-end validation and optional dashboard |
| `.gitignore` | Exclude generated validation data and portable validation tools |
| `README.md` | Review semantics, API exports, batch usage, analytics, startup/shutdown, local prerequisites |
| `docs/product-layer-validation.md` | This change/validation report |

## Checks completed

- `.venv\Scripts\python.exe -m pytest`: **34 passed**. Two upstream FastAPI/Starlette
  dependency deprecation warnings; no test failures.
- Original dashboard/API/parser/counting/annotation tests remain passing.
- Simultaneous batch and single-video HTTP requests against file-backed SQLite return the same
  active job. The existing PostgreSQL/SQLite partial unique index remains unchanged.
- Upgrade from migration 0003 with existing results preserves detections and assigns only a
  completed job as their source, even when a failed retry exists.
- Isolated mock end-to-end run: all migrations, two video uploads, batch queueing, worker claims,
  processing, database results, batch annotations, MP4 retrieval, CSV export, and analytics passed.
- `docker compose config --quiet`: passed.
- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet`: passed.
- Browser verification on desktop and at 768 × 1024: review rejection and low-confidence acceptance
  update counts; source observation selection seeks to its recorded frame; CSV download works;
  batch processing reaches completion; batch annotation reports each result; annotated H.264 video
  plays with stored database boxes. Tablet overflow found during QA was fixed.
- Portable FFmpeg was used only under ignored `data/product-validation/` for local video metadata
  and browser playback. No system FFmpeg installation was changed.

## Validation limits and operational notes

Docker's Linux engine was unavailable (`dockerDesktopLinuxEngine` named pipe missing), including
after an elevated inspection. Therefore this session could not run real VIAME/GPU end-to-end
validation or apply/test the new migration against running PostgreSQL. No plain Compose startup
was used for real VIAME validation. The required command remains:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The production PostgreSQL database and existing uploaded footage were not modified. Mock browser
validation used separate generated footage and SQLite databases in `data/product-validation/`.
The temporary validation server was stopped after verification to release port 8000.

Review is a current per-track decision rather than an actor-attributed audit history. Reprocessing
successfully replaces tracks, as before, so new results start unreviewed. Export a reviewed dataset
before reprocessing when retention is needed. Annotation batches are synchronous, sequential actions;
keep the page open until they finish. Processing jobs remain durable after the page closes.
