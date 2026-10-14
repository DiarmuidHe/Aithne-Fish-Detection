# External fish candidates (offline, optional)

These commands live in `scripts/external_fish.py`. They do not change the production VIAME pipeline, its settings, or the existing SmartBay CVAT task. All new predictions are suggestions pending review. The outputs below belong under ignored `data/` directories.

## 1. Brackish detection frames

The authors' [Brackish Dataset](https://vbn.aau.dk/en/datasets/the-brackish-dataset/) is linked to [Kaggle](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset). The pinned Kaggle **version 4** download is [CC BY-SA 4.0](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset) in its API metadata. Its ZIP SHA256 is `95de9551dd3792e4c690b59ea8dfe9c4ab59dae4c7ad32d994bd6820a75630da`. The archive contains 89 AVI recordings, `annotations/annotations_YOLO/*.txt`, and `scripts/Brackish.names`, with one-based classes **1 fish, 2 small_fish, 3 crab, 4 shrimp, 5 jellyfish, 6 starfish**. Its so-called YOLO boxes are **pixel xywh at 960×540**, not normalized YOLO training labels. The direct archive importer handles this exact release, clips edge-crossing boxes, and records every source recording and one-based frame number.

```powershell
.venv\Scripts\python.exe scripts/external_fish.py fetch-brackish data/external/brackish-v4.zip
.venv\Scripts\python.exe scripts/external_fish.py import-brackish-archive data/external/brackish-v4.zip data/training/brackish-fish-v4
```

The archive's original train/valid/test annotation CSVs mix frames from **all 89 recordings across splits**. The importer ignores those frame-level splits and assigns complete recordings by a stable hash: 75 train, 4 validation, and 10 test recordings in v4. It extracts and resizes frames to 960×540, maps only fish and small_fish to target `0 fish`, and writes single-class YOLO `images/`, `labels/`, `dataset.yaml`, `classes.json`, `metadata.jsonl`, and `report.json`. The report records archive and class-list hashes; frame metadata records video and label hashes. For a bounded first run, add `--max-sequences 1 --max-frames-per-sequence 60`; this output is only a smoke subset.

For a full import, the existing custom model script can train on the resulting YOLO dataset with a small first configuration:

```powershell
.venv\Scripts\python.exe scripts/fish_model.py train data/training/brackish-fish-v4 --weights yolo11n.pt --epochs 30 --imgsz 640 --batch 4 --require-cuda
```

The full v4 import has 15,084 frames (13,015 train, 569 validation, 1,500 test) and 13,814 fish boxes. All 89 AVIs are 1920×1080 at 15 FPS, and every one of their frames has a label file. Of those, 4,089 label files are empty. The package's own `scripts/createDummyYOLOAnnotations.py` writes an empty file "for frames with no detections", and nothing states that those frames were checked as empty. The importer therefore marks them `unverified_empty: true`, and `fish_model.py evaluate` reports predictions on them separately, excluding them from mAP and from false positives. Frames with crabs or other classes but no fish are annotated frames and are scored normally.

The full archive's published annotations set `reviewed: true` **for the Brackish publisher's annotation scope** and `reviewed_for_smartbay: false`; this lets the existing training and external test commands run without treating them as reviewed SmartBay annotations. Training needs the optional Ultralytics/PyTorch dependencies and a CUDA-enabled environment. The 60-frame smoke subset has no validation/test recordings and is intentionally refused by the training guard.

**Murky copies and a darkened test set.** `scripts/murk_augmentation.py` makes seeded, degraded copies of the **train** split only. Each copy gets lower brightness and contrast, a green/blue water tint, backscatter `I = J·t + B·(1 − t)` with a varied transmission `t` and veil colour `B`, and mild blur and noise. The copies keep each frame's recording and labels, and the folder ships a contact sheet. `synthetic-test` darkens the test split with **disjoint** ranges (darker, bluer, blurrier and noisier, but less hazy than training), as a clearly labelled synthetic robustness check rather than real murky water. `fish_model.py train --extra-train` adds such train-only folders and refuses any folder with a non-train row. `--extra-train-fraction` uses a stable-hash share of each folder. On Windows, `--workers` and `--val-workers` bound the dataloader processes, at about 1 GB of host memory each.

```powershell
.venv\Scripts\python.exe scripts/murk_augmentation.py augment-train data/training/brackish-fish-v4 data/training/brackish-fish-v4-murk --seed 20261011
.venv\Scripts\python.exe scripts/murk_augmentation.py synthetic-test data/training/brackish-fish-v4 data/training/brackish-fish-v4-dark-test --seed 7
.venv\Scripts\python.exe scripts/fish_model.py train data/training/brackish-fish-v4 --weights data/models/ultralytics/yolo11s.pt --epochs 10 --imgsz 640 --batch 16 --seed 0 --workers 4 --val-workers 2 --extra-train data/training/brackish-fish-v4-murk --require-cuda
.venv\Scripts\python.exe scripts/fish_model.py evaluate MODEL.pt data/training/brackish-fish-v4-dark-test data/outputs/dark-test-eval --save-predictions --workers 2
```

Results of the A/B comparison are in [murky-water-results.md](murky-water-results.md). Ultralytics code and weights are AGPL-3.0.

The authors also [describe AAU and COCO annotations](https://vap.aau.dk/the-brackish-dataset/). If you already have an **extracted, normalized five-column YOLO frame export**, use the manifest importer below. **Create a six-entry JSON class map from that exact export's class list.** The example [Roboflow class map](brackish-roboflow-class-map.json) is for its `0 crab, 1 fish, 2 jellyfish, 3 shrimp, 4 small_fish, 5 starfish` ordering. It does not apply to the official archive.

Create a CSV manifest with columns `image,label,sequence,frame,split`. Each row identifies an original recording and source frame; `split` is `train`, `val`, or `test`. To generate one from filenames containing recording and frame numbers, run:

```powershell
.venv\Scripts\python.exe scripts/external_fish.py manifest-brackish data/external/brackish data/external/brackish-manifest.csv --sequence-regex '(?P<sequence>.+)_(?P<frame>\d+)$'
.venv\Scripts\python.exe scripts/external_fish.py import-brackish data/external/brackish data/external/brackish-manifest.csv data/external/brackish-class-map.json data/training/brackish-fish-v1
```

Inspect the generated CSV first. The regex must recover the **actual source recording**, not a frame batch or an arbitrary export prefix. If the export has lost recording identity, build the manifest from original recording metadata; do not train on its frame-level random splits. The generator assigns each complete sequence to a stable 80/10/10 split, ignoring any source split folders. The importer independently rejects a sequence listed in multiple splits, duplicate `(sequence, frame)` pairs, and identical images crossing splits. Validation/test here measure performance on external Brackish data only.

## 2. BrackishMOT tracking benchmark

[BrackishMOT](https://vbn.aau.dk/en/datasets/brackishmot/) is linked to a [Kaggle release](https://www.kaggle.com/datasets/maltepedersen/brackishmot). The pinned version 4 ZIP downloads without authentication: 1,962,308,947 bytes, SHA256 `e717dc1438aab65eaccb59b135b58f50ca7af09043ef7effbf9fd2825410c3ff`. Kaggle's metadata says CC BY-SA 4.0. **The package itself contains no licence or readme file**, so that metadata is the only licence statement; a secondary inventory's CC BY-NC-SA 4.0 is not supported by the package. It holds 98 real sequences (`BrackishMOT/train` 78, `BrackishMOT/test` 20), all 1920×1080 at 15 FPS, plus synthetic `brackishMOTSynth*` sets and background videos that are never extracted. `brackishMOT-54`'s `seqinfo.ini` claims 610 frames, but it ships 200 images and its ground truth ends at frame 200. Two of its ground-truth rows in `brackishMOT-92` have track ID −1 (a box with no identity); the importer excludes them and lists them in `report.json`.

```powershell
.venv\Scripts\python.exe scripts/external_fish.py fetch-brackish-mot data/external/brackishmot-v4.zip
.venv\Scripts\python.exe scripts/external_fish.py extract-brackish-mot data/external/brackishmot-v4.zip data/external/brackishmot
.venv\Scripts\python.exe scripts/external_fish.py import-brackish-mot data/external/brackishmot data/training/brackishmot-v1
.venv\Scripts\python.exe scripts/external_fish.py evaluate-mot data/training/brackishmot-v1 data/outputs/my-tracker-mot data/outputs/my-tracker-mot/evaluation.json
```

The importer expects each recording at `<sequence>/gt/gt.txt` in MOTChallenge columns: `frame,id,left,top,width,height,confidence,class,visibility`. Source frames are **one-based** and IDs remain scoped to each sequence. Its class map is 1 fish, 2 crab, 3 shrimp, 4 starfish, 5 small_fish, 6 jellyfish; classes 1 and 5 become fish. 64 of the 98 sequences contain fish; the other 34 hold only crabs, starfish and other classes, so any fish track there is a false positive. No images or tracking IDs are copied into the detector training set.

Place one candidate MOT file at `data/outputs/my-tracker-mot/<sequence>.txt` for every imported sequence, with one-based `frame,id,left,top,width,height,score` rows. Use an empty file for a sequence with no candidate boxes. If the tracker analysed only some frames (10 or 4 FPS of 15 FPS footage), add `analysed_frames.json` mapping each sequence to its one-based analysed frames: only those sequences and frames are scored, unanalysed frames are not misses, and a prediction on an unanalysed frame is an error. The evaluator reports, at IoU 0.5:

- **`identity`** (official-style): IDF1, IDP, IDR with one global min-cost ID assignment per sequence, following TrackEval's Identity metric.
- **`clear`** (official-style): CLEAR MOT TP/FP/FN, ID switches and MOTA with per-frame optimal assignment that prefers continuing pairs, following TrackEval's CLEAR rules, plus ID switches per ground-truth track.
- **`counts`, `mota`, `precision`, `recall`** (diagnostic): the original greedy per-frame pairing, kept for continuity.

These are re-implementations of TrackEval's rules, not TrackEval itself; HOTA is not computed because TrackEval is not a project dependency. The assignment routine is a small tested Hungarian implementation, so no SciPy is needed. Keep identities within sequence boundaries.

**Overlap with the detection archive.** `brackish-mot-overlap` shortlists archive recordings by thumbnail distance. It then aligns each BrackishMOT sequence against them over every frame, using the cosine similarity of consecutive-frame differences of 64×36 grey thumbnails, which cancels the static camera's background. The acceptance rule was fixed before the run: similarity ≥ 0.6, at least 0.2 above any other recording or offset. Results on v4:

- **64 sequences overlap decisively.** 62 are whole recordings at offset 0. Two are later cuts, at offsets 200 and 380, of the 41-second recording `2019-03-19_17-07-53…`, which explains `brackishMOT-54`'s 610-frame `seqinfo.ini`.
- **25 sequences** are below that bar, but each one's best alignment is offset 0 in a single recording, far above the runner-up. They are almost certainly the same recordings; 24 of them are training recordings.
- **2 sequences** (`brackishMOT-91`, `brackishMOT-96`) had no candidate long enough to align.
- **7 sequences** match nothing in the archive.

In total, 87 of the 89 archive recordings appear in BrackishMOT: **BrackishMOT was cut from the same 2019 recordings as the detection archive.** Any Brackish-trained detector must pass `leakage_guard` before it is scored on BrackishMOT. The guard drops sequences cut from training or validation recordings and sequences whose origin is undetermined. With the v4 split, that leaves 17 scorable sequences: 10 from test recordings and the 7 unmatched ones.

```powershell
.venv\Scripts\python.exe scripts/external_fish.py brackish-mot-overlap data/external/brackish-v4.zip data/external/brackishmot data/training/brackishmot-v1 data/outputs/brackishmot-overlap.json
```

`scripts/murky_water_tracker.py` replays VIAME's own ByteTrack over saved detections to sweep tracker settings on BrackishMOT; see [murky-water-results.md](murky-water-results.md).

## 3. Fishial offline detector and CVAT suggestions

The pinned artifact is Fishial's [YOLOv12 medium TorchScript detector](https://github.com/Wye-Foundation/Fishial-Fish-Identification#-pre-trained-models), archive version `2` dated 2025-05-25 in its `info.json`. The direct [archive URL](https://storage.googleapis.com/fishial-ml-resources/detector_v10_m5.zip) has SHA256 `54839fa0c688c42c4114b251aff036a9b8a1288cf61db7d2956786a0238fa47a`; its `model.ts` has SHA256 `78b7c1b2eabd563e7a5d98135e791e4e6fa75c0dd88e119133c3a3f5260bdbbd`. `fetch-fishial` verifies both and extracts only the weight file. The source repository is [MIT licensed](https://github.com/Wye-Foundation/Fishial-Fish-Identification/blob/main/LICENSE); the checkpoint is linked there without a separate explicit weight license statement.

The verified GPU smoke used the existing worker image ID `sha256:9e63d3a4165b132de064d05b7bf26dcf339cf8028c3c52fd254f065c7a3808a7` with Python 3.12.3, PyTorch `2.12.0a0+git0d62256`, OpenCV 5.0.0, and NumPy 2.0.2. These are the pinned tested runtime versions for this integration. `report.json` records the effective versions for each new inference run.

```powershell
.venv\Scripts\python.exe scripts/external_fish.py fetch-fishial data/models/fishial
.venv\Scripts\python.exe scripts/external_fish.py infer-fishial data/models/fishial/model.ts data/uploads/VIDEO.mp4 data/outputs/fishial-VIDEO --max-frames 60 --stride 2 --confidence 0.25 --device auto
```

Install a PyTorch build appropriate for the machine plus OpenCV and NumPy, or use the existing VIAME GPU worker environment as shown below. Inference is batch one, 640×640, FP32, with IoU 0.3 NMS; `--max-frames 60 --stride 2` bounds the first 8 GB GPU run. The TorchScript export yields box coordinates and scores, not track IDs; each CSV row gets a unique one-frame suggestion ID. The result is `candidates.csv` in the project's VIAME-style candidate columns plus `report.json` with model, video, settings, and hashes.

After confirming the video exists in the app database and CVAT development integration is configured, create a **separate** CVAT task:

```powershell
docker compose exec api python scripts/external_fish.py cvat-suggestions VIDEO_UUID fishial-v12 /data/outputs/fishial-VIDEO/candidates.csv
```

The command reuses the existing `import_model_suggestions` safety path: `Aithne candidate <video UUID> fishial-v12` is separate from `Aithne <video UUID> ...`. A retry will not overwrite annotations already present in that candidate task. It never writes to the original SmartBay task. Rebuild the API image after adding this script, or run the command in an API environment with the current source mounted.

For the existing GPU worker image, with source mounted read-only and data already mounted at `/data`:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps --volume "${PWD}:/app:ro" --entrypoint bash worker -lc 'source /opt/noaa/viame/setup_viame.sh; export PYTHONPATH=/app:/usr/local/lib/python3.12/dist-packages:$PYTHONPATH; cd /app; python3 scripts/external_fish.py infer-fishial /data/models/fishial/model.ts /data/uploads/VIDEO.mp4 /data/outputs/fishial-VIDEO --max-frames 60 --stride 2 --device cuda:0'
```

## 4. FathomNet Megalodon for dark deep-sea footage

[FathomNet Megalodon](https://huggingface.co/FathomNet/megalodon) is MBARI's Ultralytics YOLOv8x detector, fine-tuned on all public FathomNet localizations. It has **one class, `object`**, so it boxes fish, crabs, jellyfish and other animals alike. Its output is labelled `object` and is never called fish. Hugging Face now redirects `FathomNet/megalodon` to `FathomNet/megalodon-2023-yolov8`. `fetch-megalodon` downloads `mbari-megalodon-yolov8x.pt` at commit `f312604d2f59dcccde74499cf0e0ae0647331f01`: 136,714,110 bytes, SHA256 `24fe6d9808fdd44b64d2fbfc23bbe6455520eb7b392d3664c92cbcdf88a09196`. The model card's metadata at that commit says `license: cc-by-4.0`; Ultralytics code is AGPL-3.0. The `.pt` file is a pickle, so loading it runs code: `infer-megalodon` refuses any file whose SHA256 differs, before importing Ultralytics.

```powershell
.venv\Scripts\python.exe scripts/external_fish.py fetch-megalodon data/models/megalodon
.venv\Scripts\python.exe scripts/external_fish.py infer-megalodon data/models/megalodon/mbari-megalodon-yolov8x.pt data/uploads/VIDEO.mp4 data/outputs/megalodon-VIDEO --max-frames 120 --confidence 0.25
```

`infer-megalodon` writes the same `candidates.csv` and `report.json` as `infer-fishial`, with class `object`, IDs `megalodon-<frame>-<index>`, and the checkpoint's training image size unless `--imgsz` is given. Both commands share one frame loop, and Fishial's output is unchanged. The `cvat-suggestions` command accepts these candidates for a separate CVAT candidate task.

## 5. OpenAqua assessment

[OpenAqua's README](https://github.com/White-cat-ed/OpenAqua#-license-and-disclaimer) explicitly places its **dataset, annotations, splits, and protocols** under CC BY-NC-SA 4.0 and says its curated data is strictly for academic, non-commercial research. Its raw FishNet/iNaturalist images retain their original creators' rights. The [repository badge](https://github.com/White-cat-ed/OpenAqua) labels the code license CC0-1.0, but the README gives no separate license for its linked YOLO11x or other pretrained weights. Do not infer commercial weight rights from the code badge; verify the exact checkpoint terms with the authors before use. OpenAqua is intentionally absent from all commands and defaults here.

## Provenance and first smoke runs

The portal pages themselves do not list archive checksums or licenses; the Kaggle metadata above supplies release versions and license labels. A secondary [dataset inventory](https://github.com/kevinsbarnard/fish-datasets) calls BrackishMOT CC BY-NC-SA 4.0, which conflicts with Kaggle's current CC BY-SA 4.0 label. The downloaded BrackishMOT package contains no licence file of its own, so Kaggle's label is the only statement. The direct Brackish importer and `extract-brackish-mot` both check their exact archive SHA256; the MOT importer also records source ground-truth hashes.

The **real Brackish v4 archive** was downloaded to `data/external/brackish-v4.zip`. A bounded direct import of its first recording and first 60 frames produced 20 fish boxes at `data/outputs/brackish-official-smoke-final-20261011/`. A separate extracted-format smoke used one existing upload frame and a **synthetic** label in `data/outputs/brackish-format-smoke-20261011/`; it checks mechanics only. The Fishial GPU smoke processed one existing video frame and emitted one unreviewed candidate in `data/outputs/fishial-external-smoke-final-20261011/`. None of these runs establishes SmartBay accuracy.
