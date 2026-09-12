# FUnIE-GAN integration and offline results — 2026-09-11

Optional `FISHIAL_PREPROCESS=funie_gan` is implemented and disabled. Production remains
`none`. No camera capture, Fishial authentication, image request or paid replay was run.
There are no labelled end-to-end species accuracy results.

## Implementation and invariants

- `app/services/fish_enhancement.py` owns byte-preserving `none`, existing classical
  transforms and lazy, process-cached FUnIE inference with serialized access. A replaceable
  boundary makes ordinary tests independent of PyTorch, weights, cameras and network.
- `LiveTracker` stages the clean JPEG once. `SpeciesIdentifier` enhances only selected
  frames, validates bytes/provenance, commits pre-send metadata, then uses the existing
  reservation and identification path. Retries reuse the exact same payload. Local failures
  reserve zero calls, abstain, and use the existing consensus stop logic.
- Normalized `expected_box` is passed unchanged, including edge-clamped targets. Offline
  response replay uses submitted dimensions from new audits when converting that box.
- The durable session ceiling remains `fish_target * frames_per_fish + fishial_max_api_retries`.
  There is no new production Fishial call site. Vote, confidence, ratio, regional and
  multi-object match rules are unchanged. No database migration was needed.
- Enhanced images are not retained by live inference. Original scratch cleanup and the
  existing explicit retained-crop policy remain in force. Replay/benchmark output is an
  explicit retention action and contains the exact paired JPEGs for review.
- Paid replay supports `--preprocess-modes none,funie_gan`. Crop-first scheduling holds
  back retry allowance for the remaining sides and refuses to begin a crop that cannot
  fit all sides. SQLite freezes modes, crop hashes, boxes, model id/hash, evaluation settings
  and ceiling; changed specifications are rejected before modifying an existing journal.
  Uncertain sends remain spent and are not repeated after restart.

Small compatibility change: classical pixel transforms now operate on the decoded original
staged JPEG, rather than the pre-JPEG pixels. `none` always returns original bytes and ignores
the optional upscale setting, as required by this integration contract. FUnIE always outputs
256×256 and never applies classical corrections. This does not affect the default path.

## Provenance and runtime

The [upstream repository](https://github.com/xahidbuffon/FUnIE-GAN) was pinned at commit
`8f934c834c94e007b00866186b9ee624dc2b7b69`. Only the generator and its UNet blocks from
`PyTorch/nets/funiegan.py` were adapted (import/super cleanup; architecture unchanged).
`app/vendor/funie_gan/LICENSE` preserves the MIT notice and attribution to Md Jahidul Islam.
`provenance.json` is included in Python package data.

Artifact: [`PyTorch/models/funie_generator.pth`](https://raw.githubusercontent.com/xahidbuffon/FUnIE-GAN/8f934c834c94e007b00866186b9ee624dc2b7b69/PyTorch/models/funie_generator.pth).
Size: **28,105,229 bytes**. SHA-256:

```text
e4fcb50e03868f1683c244b1c125def30df335ec43d9bb001b8700c03b8f2bd6
```

The application never downloads models. `scripts/funie_model.py --output
data/models/funie/funie_generator.pth` is the explicit HTTPS administrator helper; it
verifies the pinned digest and byte size before atomic rename. Production hashes the
same in-memory file bytes it safely loads with `weights_only=True`, checks strict
state-dict compatibility, sets evaluation/no-gradient mode, and validates a warm-up output
before the live worker can claim a session. Missing, mismatched, corrupt, incompatible or
unsupported-device configurations fail with the fixed local `preprocessing failed` error.
No legacy pickle conversion was needed.

The installed VIAME environment contains **PyTorch 2.12.0a0+git0d62256**, **Pillow 12.3.0**,
Python **3.12.3**, with functional CUDA. Its libraries are under
`/opt/noaa/viame/lib/python3.12/site-packages`; ordinary system Python does not see them
until `setup_viame.sh` is sourced. The conditional GPU live-worker launcher activates this
existing environment only for FUnIE. No Torch/TensorFlow/ONNX dependency was installed in
the API, base development environment or other worker. Reuse adds **0 runtime bytes**,
so an ONNX export or separate service would not reduce the incremental runtime cost.

The measured existing worker image is **8,026,217,121 bytes**, image id
`sha256:d29bde3ee1ee69ec23f89cac03e49b7f6467c4131d4f19511ae343c10f3fdda2`.
Verification mounted the current code read-only into this image: **0 image bytes added**,
no pull/rebuild and no running service replacement. A future code rebuild adds only the
small Python/script/licence changes; that layer delta was not measured. The model is
mounted read-only at `/models`, adding 28,105,229 bytes to local artifact storage.

JPEG BGR is converted to RGB, resized with Pillow bicubic to 256×256 (matching upstream's
PIL transform), converted to float CHW/batch in [-1,1], and inferred in `torch.inference_mode()`.
Finite output is clamped to tanh's [-1,1], denormalized with round-to-nearest uint8,
converted back to BGR, and JPEG encoded at quality 95. The reference test script saves a
joint input/output montage with dynamic normalization; this adapter deliberately uses
the prompt's fixed tanh-range conversion for an individual submitted crop, with no
per-image contrast stretch.

## Offline measurements

The discovery pass found **two clean retained crops**: both originals from
`smartbay-screen-20260910`. No retained production live crop directory supplied more.
Tests and annotated media are excluded. Each device used one excluded warm-up and five
measurements per crop, ten timed transformations total, with **zero failures**.
Hardware: **NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB** under Docker/WSL.

| Measurement | CPU | CUDA |
| --- | ---: | ---: |
| Median total latency | 48.02 ms | 46.55 ms |
| p95 total latency | 59.98 ms | 116.29 ms |
| Median decode | 1.07 ms | 0.87 ms |
| Median inference including tensor transfer | 42.64 ms | 42.73 ms |
| Median encode | 1.21 ms | 0.87 ms |
| Peak process RSS | 666,451,968 B | 1,039,101,952 B |
| Peak Torch GPU allocation | 0 B | 80,864,256 B |
| Peak Torch GPU reservation | 0 B | 106,954,752 B |

RSS is whole-process high-water memory, including import/model warm-up; it is not an
incremental model-only measurement. Torch GPU figures exclude driver/context allocations
and other processes. Total latency also includes resize, color conversion and validation.
These are small-sample deployment measurements, not a loaded-worker throughput guarantee.
CPU and CUDA produce different JPEG hashes, so the future comparison explicitly pins CPU.

| Crop | Original dimensions | Submitted dimensions | Original SHA-256 | CPU enhanced SHA-256 |
| --- | --- | --- | --- | --- |
| 1 | 395×245 | 256×256 | `f4f7a1c89cc11a3406c5b8411555346b26f563a9c6d90ecd45dcd3aee35c13ee` | `5bccadeaeb67f77eda2aa5ea40d9bcbea908fbb16da22e4b56c647803493c528` |
| 2 | 513×299 | 256×256 | `3f43f6c4e1ba5429c6695aa6d208b91e50a85199799a139f0ea43b48c4def061` | `9b0fbbc4b20e588fb776848665017e18d9900f804c3b96247599687ecaa0c254` |

Artifacts, relative to the repository:

- `data/outputs/fishial-experiments/funie-offline-20260911/offline-report-cpu.json`
- `data/outputs/fishial-experiments/funie-offline-20260911/offline-report-cuda.json`
- `data/outputs/fishial-experiments/funie-offline-20260911/pairs/{cpu,cuda}/{none,funie_gan}/crop-{1,2}.jpg`

The JSON contains every timing sample, original/submitted hash (including CUDA hashes),
dimensions, expected box and provenance. Visual inspection confirms the transformation
changes color substantially and does not recover a sharp species-defining image. No
brightness, UIQM, PSNR or SSIM score is used as an accuracy claim or promotion gate.

Reproduce with the command in the Fishial README section, using `--device cpu` or `cuda`.
Explicit output retention is confined to the experiment directory. The model is ignored
by Git; the two originals and the completed older six-call journal were not edited.

## Verification

Baseline: **109 passed** around crop staging, live identification and replay.
Focused feature checks: **138 passed, 1 optional smoke skipped** before the final
additional safe-loader/device/startup tests. Full final ordinary suite:

```powershell
.venv\Scripts\python.exe -m pytest --basetemp=C:/tmp/funie-pt
```

**327 passed, 1 skipped** in 21.78 seconds. The optional real-model pytest test skips
unless `FUNIE_SMOKE=1` is explicit and its local artifact/runtime are present. The same
real CPU smoke also ran separately inside the existing VIAME image and **passed**:
safe load, eval/no gradients, inference mode, deterministic output, valid 256×256 JPEG.
Ordinary tests ran in the Windows development environment without PyTorch or Pillow.

Regression coverage includes exact original bytes, classical transform parity, lazy
imports/cache, incompatible/hash-mismatched weights, device/startup validation, finite
output checks, staging without enhancement, zero reservations on preprocessing failure,
401/429/500 retries with one enhancement per frame, unchanged edge-clamped box, compact
provenance, scratch cleanup, selection/consensus budget invariants, paired retry holdback,
labelled counts, frozen model hash and restart safety. Existing response-only dry-run
tests passed without network access.

Ruff passes for the integration modules/scripts/tests. Inspection of `app/config.py` and
`app/services/live_monitor.py` also reported three pre-existing unrelated lint findings
(quoted annotation, UTC alias, broad exception); those were preserved. Compose validation
passes. The first full-suite attempt was blocked solely by sandbox permissions on the
specified temp directory; the successful rerun used the same command with access granted.
Remaining warnings are existing framework/migration deprecations and pytest-cache access.

## Future paired evaluation — separate paid authorization required

Prepared directory: `data/outputs/fishial-experiments/funie-paired-20260911`, containing
copies of the two original crops and their manifest. Each crop has an optional
`ground_truth_scientific_name: null`; fill with verified labels before a labelled run.
Do not invent labels from the enhanced image. The directory has no replay journal or
spent calls. Do not use `smartbay-screen-20260910/replay.sqlite` for this experiment.

After separate approval, this exact command allows **at most four paid image calls**:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --no-deps --env FISHIAL_PREPROCESS=funie_gan --volume "${PWD}:/app:ro" --entrypoint bash live-worker /app/scripts/live_worker_viame.sh scripts/fishial_replay.py --replay /data/outputs/fishial-experiments/funie-paired-20260911 --preprocess-modes none,funie_gan --max-calls 4 --set fishial_funie_device=cpu --set fishial_max_api_retries=0
```

Retries are explicitly disabled for this tiny screen so both crops can receive both
sides within four attempts. The journal's four-call ceiling still includes every possible
image send, including authentication-related retries. Restarting cannot create another
allowance. Labels/modes/model/settings/bytes cannot be changed under that journal.

Each side reports detected objects, nonempty species, accepted top name, certainty,
regional rejection, target IoU, usable answer and preprocessing latency. Correct/wrong/
abstained counts apply only to labelled crops; an unlabelled usable answer proves no accuracy.
Even two labelled crops would be a screen, not representative production evidence.
Promotion requires the broader labelled gate in the implementation prompt, unchanged
abstention rules/call ceilings and acceptable worker resource cost. **`none` remains the
justified production default.** There is no missing artifact or licence blocker.

## Follow-up: live video decoder regression

Enabling FUnIE exposed a runtime conflict missed by the initial image-only smoke test:
`setup_viame.sh` placed VIAME OpenCV 4.9.0 (without FFmpeg video decoding) ahead of the
application OpenCV 5.0.0 (with FFmpeg). Captured MP4 segments therefore yielded zero
decoded frames, even though model loading and JPEG inference worked.

The launcher now preserves the application's Python/site-package precedence while
adding VIAME's ML libraries. The detector still sources VIAME in its separate subprocess.
Startup checks OpenCV's FFmpeg backend before claiming sessions. The new
`scripts/live_runtime_smoke.py` ran through this exact launcher and decoded ten synthetic
H.264 frames plus completed deterministic FUnIE inference in the same process.

The earlier timing and image-hash tables remain historical measurements of the initial
VIAME OpenCV environment; they do not establish latency or pixel parity for the corrected
OpenCV 5 runtime. Future benchmark and paid-replay commands above use the corrected
launcher. No paid replay was run as part of this fix.

Fix verification: 73 focused tests passed (one opt-in model test skipped); the full
suite passed 330 tests with one skip. The rebuilt image also passed the combined
H.264/FUnIE smoke without a code bind mount. Only the idle live-worker container was
recreated; the API, database and processing worker stayed running. Failed camera
sessions need a new start. Local tests set `FISHIAL_PREPROCESS=none` in the test
process because the user's `.env` now enables FUnIE using Compose-supplied model defaults.
