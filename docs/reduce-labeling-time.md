# Reduce fish box correction time (experimental)

The production VIAME run and its CVAT task remain the baseline. This workflow makes a **separate, unreviewed candidate**: VIAME tracks plus Fishial fish detections that VIAME did not cover. It groups nearby Fishial boxes into short tracks and discards one-frame suggestions. It does not infer species or certify empty frames.

An optional CVAT import mode ends track visibility at the first missed detector sample when a VIAME track has a long gap. This avoids drawing interpolated fish boxes across a multi-sample absence. It can also hide a fish that the detector briefly missed, so compare it in a separate task. The review queue now lists long track gaps as places to inspect. Existing CVAT tasks and annotations are unchanged.

To compare this with your current automatic labels without running a second detector, import the original VIAME CSV into a **new task** with the gap mode enabled. Use the video's actual UUID and completed job path:

```powershell
docker compose exec api python scripts/import_model_suggestions.py VIDEO_UUID viame-gap-candidate /data/jobs/JOB/RUN/viame_tracks.csv --frame-offset -1 --break-long-gaps
```

The API container must contain the updated code (rebuild it if needed). The default VIAME-to-CVAT import path has not changed.

1. Run Fishial offline on the **same source video** as a completed VIAME job. Use a stride that covers the frames you intend to review. For a 30 fps source with VIAME sampled near 10 fps, `--stride 3` is a bounded first pass. Increase `--max-frames` enough to cover the desired span; this value counts processed frames, not source frames. See [external-fish-datasets.md](external-fish-datasets.md) for the GPU worker command, weight provenance, and setup.

   ```powershell
   .venv\Scripts\python.exe scripts/external_fish.py infer-fishial data/models/fishial/model.ts data/uploads/VIDEO.mov data/outputs/fishial-VIDEO --max-frames 600 --stride 3 --device auto
   ```

2. Combine its CSV with the **unchanged** completed VIAME CSV. The default `--baseline-offset -1` converts VIAME's one-based source frame IDs to the zero-based source frame IDs used by Fishial and CVAT. Check this offset for the particular VIAME job. The output directory must be new.

   ```powershell
   .venv\Scripts\python.exe scripts/fuse_fish_suggestions.py data/jobs/JOB/RUN/viame_tracks.csv data/outputs/fishial-VIDEO/candidates.csv data/outputs/fused-VIDEO
   ```

   Read `data/outputs/fused-VIDEO/report.json`. `fishial_added_tracks` and `fishial_added_boxes` count **suggestions**, not confirmed missed fish. The output CSV preserves VIAME's valid boxes and track IDs under a `viame-` prefix. Fishial extras have a `fishial-extra-` prefix, at least two observations, confidence at least 0.35, nearby box overlap for linking, and a box aspect ratio no greater than 5. The aspect gate prevents an observed seagrass failure mode but could also omit elongated fish. Defaults are not calibrated against SmartBay truth.

   Before importing a candidate, visually inspect each new track:

   ```powershell
   .venv\Scripts\python.exe scripts/preview_fused_suggestions.py data/uploads/VIDEO.mov data/outputs/fused-VIDEO/candidates.csv data/outputs/fused-VIDEO/extras-contact-sheet.jpg
   ```

3. If CVAT development integration is configured, import this CSV into a **new candidate task**. Use a new name for each version. This never writes into the existing SmartBay task.

   ```powershell
   docker compose exec api python scripts/import_model_suggestions.py VIDEO_UUID viame-fishial-v1 /data/outputs/fused-VIDEO/candidates.csv
   ```

4. Time the same bounded review on the VIAME task and candidate task. Correct all boxes on the chosen frames, including empty frames. Then export reviewed truth as described in [cvat-development.md](cvat-development.md) and score both CSVs with `scripts/reviewed_benchmark.py`. Use `frame_offset: -1` for the original VIAME CSV and `frame_offset: 0` for the fused CSV. Compare held-out recall, false positives, ID switches, and **actual minutes spent correcting**. Do not promote the candidate or skip reviewing frames based on its suggestions alone.

The separate candidate may save time if its persistent Fishial extras are real VIAME misses. It can also add false boxes. In one 51-second local clip, the first fusion produced 12 added tracks and a box-shape gate reduced that to four. The contact sheets showed **seagrass in the added boxes**, including all four after the gate, so **neither candidate was imported into CVAT**. The contact sheets are at `data/outputs/fused-reduction-full-20261011/extras-contact-sheet.jpg` and `data/outputs/fused-reduction-shape-gate-20261011/extras-contact-sheet.jpg`. These are visual diagnostics, not reviewed truth. The current repository has no reviewed SmartBay test set, so time saving and accuracy are not yet established. Brackish data can support later training, but validation must use reviewed SmartBay recordings from independent sessions.
