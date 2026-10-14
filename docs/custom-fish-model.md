# Local fish model workflow

For new work, use the [bounded CVAT review and verified export](cvat-development.md#bounded-review-and-verified-export) first. The older VIAME CSV manifest below is useful for auditing pseudo-labels; setting a whole source to `reviewed: true` is unsuitable when only selected frames were checked. `scripts/reviewed_dataset.py` exports exactly the completed source frames, including verified negatives, checked IDs, and species decisions.

The existing API and workers still use VIAME. Their completed jobs retain the original `viame_tracks.csv` in `data/jobs/<job>/<run>/`. This workflow reuses `app.services.viame_parser`, pairs those CSVs with the original `data/uploads` files via each job's `input_list.txt`, and writes generated files under `data/training` and `data/models`. It does not modify source media, VIAME output, database rows, or the existing dashboard.

## 1. Inventory and review

From the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe scripts/fish_model.py inventory --output data/training/inventory.json
```

The current local inventory finds 24 CSV/video pairs across 11 distinct videos and one observed label, `fish`. Thirteen alternate runs over the same videos are marked `enabled: false`; inspect them and select one annotation file per video before building, or merge corrections into a new CSV. The draft is deliberately marked `reviewed: false` and assigned to `train`; these are automatic pseudo-labels. Inspect the `video` and `annotations` paths and preview boxes before changing review status. The production VIAME writer records source frame 1 as its first frame, so the inventory sets `frame_offset: -1` for its CSVs. Other CSVs default to zero offset. Verify this for each source. `source` inside many VIAME rows is `input_list.txt`, so use the job's input list or the manifest pair to identify the real video.

Manifest shape:

```json
{
  "class_mapping": {"fish": "fish", "cod": "cod"},
  "sources": [
    {"video": "C:/path/to/video.mp4", "annotations": "C:/path/to/viame_tracks.csv",
     "group": "deployment-A", "split": "train", "reviewed": false, "frame_offset": -1}
  ]
}
```

Set `group` to the source video or, better, a deployment/camera session shared by related clips. All entries with the same group must use one split. Assign reviewed videos to `val` and `test`; only reviewed labels are suitable for evaluation. If annotations are numbered at a different rate from the original video, set `annotation_fps` explicitly; the builder maps annotation frame time to the nearest source frame. Do not set it merely because VIAME sampled fewer frames per second: production CSV frame numbers already refer to source frames.

For corrections, copy a CSV to a new file, correct or remove wrong rows in that copy, and point the manifest at it. Keep the original VIAME CSV. Set `reviewed: true` only after all labelled frames in that source have been checked. You can retain species by mapping distinct VIAME labels to distinct classes. Mapping several labels to `fish` makes a single-class model. Unmapped labels are flagged and excluded.

## 2. Build data and inspect it

```powershell
.venv\Scripts\python.exe scripts/fish_model.py preview data/training/inventory.json data/training/viame-previews --limit 20
.venv\Scripts\python.exe scripts/fish_model.py dataset data/training/inventory.json data/training/fish-v1
```

The preview command overlays all valid imported boxes on the original frames, including low-confidence detections in red. It does not certify them. The dataset builder defaults to reviewed sources only. During initial experimentation, `--include-pseudo` permits high-confidence unreviewed detections in the **training** split. A pseudo frame containing any low-confidence, invalid or unmapped detection is excluded as a whole so a known fish is not silently treated as background. Validation/test remain reviewed. `--low-confidence 0.6` changes the threshold for flagging and pseudo-label inclusion. The builder never creates fish-free negative examples from frames lacking VIAME detections.

The output contains `images/{train,val,test}`, normalized YOLO `labels`, `dataset.yaml`, `classes.json`, `metadata.jsonl` (source video, source frame, VIAME frame, original label, track ID, confidence, review state), `review_issues.json`, up to 20 box previews per split, and `report.json`. It flags malformed rows, invalid/out-of-frame boxes, duplicates, low confidence, unmapped labels, inconsistent track labels, missing/decode-failed frames and dimension mismatches. The command refuses to overwrite an existing output directory. Inspect `previews` and resolve `review_issues.json` before training. Empty `val` or `test` means more reviewed videos are needed; the current unreviewed inventory alone cannot train a validated model.

An initial audit of the 11 active real-video pairs is saved at `data/training/audit-v1`: 3,237 detections were below confidence 0.6 and two CSV rows were malformed. It produced zero training frames because every source remains unreviewed. Twenty real overlay images are saved at `data/training/viame-previews`; inspect these before approving labels. Some boxes visually appear questionable, which is why the CSVs remain pseudo-labels.

## 3. Train and evaluate locally

The optional detector is COCO-pretrained YOLO11n, a small model suited to first GPU experiments. The app has no PyTorch or Ultralytics installed currently. Use the [PyTorch selector](https://docs.pytorch.org/get-started/locally/) to install the CUDA build appropriate for this machine, then install the optional project extra with `.venv\Scripts\python.exe -m pip install -e ".[train]"`. Verify CUDA with `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`. The pretrained weight file is downloaded on first use unless you supply a local path to `--weights`; use a local weight file for an entirely offline run.

```powershell
.venv\Scripts\python.exe scripts/fish_model.py train data/training/fish-v1 --require-cuda --weights C:\models\yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --lr 0.01
.venv\Scripts\python.exe scripts/fish_model.py evaluate data/models/fish-YYYYMMDDTHHMMSSZ/weights/best.pt data/training/fish-v1 data/outputs/fish-eval-v1
```

Training creates a timestamped run in `data/models` with `weights/best.pt`, `weights/last.pt`, Ultralytics metrics, class mapping and run configuration. Resume an interrupted run with `train data/training/fish-v1 --resume data/models/<run>/weights/last.pt`. Evaluation requires a reviewed test split and writes mAP@50, mAP@50:95, precision/recall, false-positive and missed-fish counts at IoU .5, per-class counts, observed FPS, and prediction previews. These counts cover annotated frames only. Tracking identity metrics require reviewed track-ID ground truth and are not reported by this command.

```powershell
.venv\Scripts\python.exe scripts/fish_model.py track data/models/<run>/weights/best.pt data/uploads/<new-video>.mp4 data/outputs/fish-track-v1 --tracker bytetrack.yaml --conf 0.25
```

This runs local inference with ByteTrack, saves `tracked.mp4`, `tracks.csv` in VIAME-style columns (zero-based frame numbers), and a run report. These predictions are **unreviewed**. Use `scripts/import_model_suggestions.py` to put them in a separate CVAT task, then repeat bounded human review. `botsort.yaml` is also available. The existing app's VIAME workflow continues to run independently while the custom model is assessed.

## Compare VIAME candidates and enhancement

Preserve the existing `viame_tracks.csv` as the baseline. After the first bounded export, run offline candidates with `scripts/viame_review_experiment.py run VIDEO OUTPUT` inside the worker container, changing one setting at a time: `--fps 12.5`, `--score`, ByteTrack thresholds/buffer, or `--pipeline` for an installed newer fish model pack. Each run saves settings, processing time, and a separate CSV; it never changes the app's processing settings. A newer pack must be installed and its pipeline supplied before that candidate can run.

Only after reviewed truth exists, `scripts/viame_review_experiment.py paired-clahe VIDEO OUTPUT` creates matched re-encoded original and modest CLAHE videos. Run the same VIAME settings on each, using identical source-frame labels to compare them. Do not route the existing FUnIE-GAN crop preprocessing through full-frame VIAME based on appearance alone. The original media and review labels remain the reference.

Score candidates with `scripts/reviewed_benchmark.py DATASET CANDIDATES_JSON OUTPUT`. The JSON has `candidates` in order, with `baseline` first; each entry needs `name`, `videos` mapping every reviewed video ID to `{"csv":"/data/path.csv","frame_offset":-1}`, plus measured `review_minutes` and `processing_seconds`. Review minutes must be timed separately for each candidate's suggestions on comparable footage. The report counts fish true positives, false boxes, misses, checked-frame ID switches, and held-out test metrics. It only marks a candidate promotion-ready after at least six independent recording groups, a test recall gain of at least 5 percentage points at IoU 0.5, a correction-time reduction of at least 20%, and no rise in test ID switches. The first SmartBay clip is a diagnostic pilot, not an accuracy claim.

Ultralytics code and model weights use [AGPL-3.0 or Enterprise terms](https://docs.ultralytics.com/help/contributing/). Review those terms before distributing or incorporating the custom model into a commercial service. No cloud API is used by these commands, but a pretrained weight download may occur unless supplied locally.

## Current limits

The repository currently contains VIAME CSV outputs, not DIVE JSON, COCO JSON or KW18 annotation files; this importer supports the format actually present, VIAME CSV. The custom model commands are separate local commands and are not yet selectable in the web UI or processing worker. No reviewed validation/test set or trained custom weights exist, so GPU training, inference, comparison against VIAME, and IDF1/ID-switch evaluation have not been demonstrated. Provide reviewed source-video-level train/validation/test annotations to proceed.
