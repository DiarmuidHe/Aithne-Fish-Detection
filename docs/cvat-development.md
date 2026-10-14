# CVAT development review

The website can mirror selected video uploads to a local CVAT Community server. It sends the original video first. After the first successful VIAME processing run, it adds fish tracks with frame boxes, VIAME track IDs, suggested class, and per-frame confidence. A person can then correct the tracks in CVAT. The sync is one-way: CVAT edits are never overwritten by another processing run and are not automatically imported back into Aithne.

## Local setup on Windows

CVAT Community v2.78.0 is kept under the ignored `data/cvat-community` directory. The tracked `docker-compose.cvat-local.yml` override lets both the browser (`localhost`) and Aithne containers (`host.docker.internal`) reach it.

```powershell
git clone --depth 1 --branch v2.78.0 https://github.com/cvat-ai/cvat.git data/cvat-community
docker compose -f data/cvat-community/docker-compose.yml -f docker-compose.cvat-local.yml up -d
powershell -ExecutionPolicy Bypass -File scripts/setup_cvat_dev.ps1
```

The setup script creates an `aithne-dev` CVAT account with a generated password and writes `CVAT_DEV_MODE=true`, the local URLs, and credentials into the ignored project `.env`. Do not commit that file. View the password locally in `.env` when signing in at [CVAT](http://localhost:8080). Run the script once; if the account or settings already exist, keep the existing password.

Then rebuild and start the Aithne API and worker so both receive the new dependency and environment settings:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build api worker
```

Use the base `docker-compose.yml` alone for the mock detector. Both modes support CVAT. The API applies migration `0011_cvat_sync` at startup.

## Review flow

1. Open the Aithne website. The upload dialog shows a **Development · CVAT** checkbox when the mode and credentials are configured; it starts checked.
2. Upload a video. Aithne stores it and creates a CVAT task in the background. The video detail page shows the task link and sync state.
3. Start VIAME processing from the video detail page. Its first successful result is imported to the empty CVAT task as suggested fish tracks.
4. Open the CVAT task and correct missing fish, false positives, boxes, tracks, and suggested class values. CVAT holds the human-reviewed annotations.

If CVAT is temporarily unavailable, the video stays in Aithne. Use **Send to CVAT** on its detail page to retry. If the CVAT task already contains annotations, automatic import stops to protect the reviewer’s work. Turning `CVAT_DEV_MODE` off hides the controls and stops new syncs. Previously created CVAT tasks remain there.

The status endpoint is `GET /cvat/status`. Credentials stay server-side.

## Bounded review and verified export

The video detail page now shows a 30 minute review queue after a CVAT task exists. For the SmartBay Cam 2 pilot it proposes 12 distributed source frames (0, 10, …, 110 seconds) and full 48–51 and 88–91 second windows. Open the CVAT task and check **every fish box** in each selected frame, including whether an apparently empty frame is genuinely empty. In the two windows, inspect every intervening source frame and correct track identities. Use the video detail page to assign each reviewed CVAT track a Galway Bay catalogue species or `unknown`. Record the actual review minutes and mark the corresponding region complete only after the work is done. The page rejects a total above 30 minutes. The hotspot list ranks likely misses and overlapping IDs for inspection; it is not ground truth.

The completion record stores source frame bounds, separate box/ID/species flags, review time, and a hash of the CVAT boxes and track IDs (plus species decisions where checked). If a CVAT annotation or checked species decision changes later, the exporter stops until that region is reviewed and completed again. **Only completed regions** can supply training labels, verified empty frames, and benchmark truth. The rest of the VIAME task remains unreviewed suggestions. Undo completion from the detail page if a region needs more work.

Create a JSON manifest under the shared `data` volume with `video_id`, recording session `group`, and `split` for each source. Start with the pilot in `train`; add independent sessions for `val` and `test` before judging accuracy. Groups cannot cross splits. Export from the API container:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec api python scripts/reviewed_dataset.py /data/training/review-manifest.json /data/training/reviewed-v1
```

The output contains source-frame images, YOLO fish labels (including empty label files for verified fish-free frames), `metadata.jsonl`, `tracks_truth.jsonl` for ID-checked frames, `species_truth.jsonl`, a dataset YAML, and a report. It refuses to overwrite an existing export. Keep each export as a versioned snapshot.

For a custom model's unreviewed predictions, create a **separate** CVAT task so the corrected source task stays intact:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec api python scripts/import_model_suggestions.py VIDEO_ID candidate-v1 /data/outputs/fish-track-v1/tracks.csv
```
