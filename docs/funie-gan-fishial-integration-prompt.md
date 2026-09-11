# Astra implementation prompt: add FUnIE-GAN before Fishial at bounded cost

Use this prompt with `gpt-6-astra`. Start with `reasoning.effort=low`; raise it to
`medium` only if the runtime/model-format decision remains genuinely ambiguous after
inspecting the named files. Keep the prompt prefix stable if the work spans multiple
turns so prompt caching can apply.

---

Implement an optional, production-safe integration of
[`xahidbuffon/FUnIE-GAN`](https://github.com/xahidbuffon/FUnIE-GAN) in this repository.
Enhance the clean crop of an already detected fish immediately before that crop is
submitted to Fishial AI for species recognition. The goal is to increase the number
of correct, usable Fishial species answers on degraded underwater footage without
increasing the existing paid image-call ceiling or weakening abstention rules.

Carry this task through implementation, focused tests, documentation, and a local
offline benchmark. Make reasonable in-scope assumptions and continue without asking
questions unless a missing model artifact, licence problem, or destructive/external
action truly blocks the work. Do not stop after presenting a plan.

## Outcome and completion bar

The work is complete when all of these are true:

1. `FISHIAL_PREPROCESS=funie_gan` is a supported but **off-by-default** mode.
2. FUnIE-GAN runs only on Fishial crops selected for a paid recognition attempt. It
   does not process full video frames, VIAME detector input, annotated media, every
   detection, or every unpaid candidate.
3. The original crop is staged once. Enhancement is performed lazily, before the
   budget reservation and network request. One crop is enhanced at most once per
   recognition attempt; Fishial's internal retries reuse the same enhanced bytes.
4. Missing, corrupt, hash-mismatched, or incompatible weights cause a clear local
   failure before any Fishial image-call reservation. No bad/partial enhancement is
   uploaded and no call is charged for a preprocessing failure.
5. The session's existing hard ceiling remains exactly:
   `fish_target * frames_per_fish + fishial_max_api_retries`. FUnIE-GAN adds zero
   calls and creates no new Fishial call site.
6. Existing modes (`none`, `white_balance`, `clahe`, `both`) preserve their current
   behavior unless a small refactor is required to make lazy preprocessing possible.
7. Unit tests never download weights, contact GitHub, call Fishial, open a camera, or
   require a GPU. The optional real-model smoke test is explicit and skips cleanly
   when its artifact/runtime is absent.
8. Production defaults stay at `FISHIAL_PREPROCESS=none`. Promotion to FUnIE-GAN is
   permitted only by a paired evaluation using known species labels; an increase in
   brightness, UIQM, or non-empty answers alone is not proof of species accuracy.

## Read only the relevant context first

Inspect these files before editing; do not rescan the entire repository or re-read
the older implementation prompt unless a named invariant is unclear:

- `app/services/species_quality.py` — current `preprocess_crop` implementation.
- `app/services/live_monitor.py` — clean original-resolution crop staging,
  `expected_box`, quality ranking, scratch retention, and the current preprocessing
  call site.
- `app/services/live_species.py` — the only production identification loop and its
  durable `_reserve()` journal.
- `app/services/fishial.py` — Fishial v2 adapter, retries, target-box matching, and
  redaction guarantees.
- `scripts/fishial_replay.py` and `scripts/fishial_replay_journal.py` — bounded,
  restart-safe A/B tooling.
- `scripts/fishial_capture.py` — collection of clean crops without Fishial calls.
- `app/config.py`, `.env.example`, `docker-compose.yml`,
  `docker-compose.gpu.yml`, `Dockerfile.worker.viame`, and `pyproject.toml`.
- `tests/test_fishial.py`, `tests/test_live_monitor.py`,
  `tests/test_live_species.py`, and `tests/test_fishial_experiment.py`.
- The Fishial section of `README.md` and
  `docs/smartbay-automated-experiment.md`.

Preserve unrelated working-tree changes. Do not reformat or rewrite files outside
this feature.

## Verified upstream facts and constraints

The upstream project is MIT-licensed and provides TensorFlow and PyTorch training
pipelines. Its PyTorch reference uses PyTorch 1.6/Python 3.8, loads
`models/funie_generator.pth`, resizes RGB input to `256 x 256` with bicubic
interpolation, converts to a tensor, normalizes each channel from `[0,1]` to
`[-1,1]`, calls `GeneratorFunieGAN` in evaluation mode, and converts the generated
tensor back to an image. The repository says the TensorFlow/Keras implementation is
the official implementation for reproducing paper results. Treat this as legacy
reference code, not as a dependency specification for this Python 3.11 application.

Primary sources:

- Repository and performance claims:
  https://github.com/xahidbuffon/FUnIE-GAN
- PyTorch reference path:
  https://github.com/xahidbuffon/FUnIE-GAN/tree/master/PyTorch
- Paper/preprint:
  https://arxiv.org/abs/1903.09766
- Licence:
  https://github.com/xahidbuffon/FUnIE-GAN/blob/master/LICENSE

The paper reports improvements on underwater visual-perception tasks. That does not
establish an improvement for this Fishial classifier, these cameras, or fine-grained
species identification. GAN enhancement can hallucinate or suppress markings used
to distinguish species. Therefore accuracy must be measured end to end, and the new
mode must ship disabled.

## Existing invariants that must survive

- Only tight, clean, unannotated crops of already detected fish may leave the app.
  Never send a full frame, camera URL, annotated image, or operator image.
- `SpeciesIdentifier` remains the sole production caller of `FishialClient.identify`.
- `_reserve()` commits each image attempt before network egress, including retries.
  Uncertain sends remain spent and are never replayed after restart.
- Failed/ambiguous predictions abstain. Do not lower vote count, score, vote ratio,
  regional filtering, or multi-object IoU thresholds to make results look better.
- `expected_box` is normalized to the crop. Preserve the exact stored box through
  preprocessing; do not recenter it or derive it again from a detection row.
- Candidate ranking, late budget reserve, early stopping, regional filtering,
  scratch cleanup, and response redaction remain intact.
- Logs and exceptions must contain no credential, token, endpoint containing secret
  material, image bytes, or private crop path.
- `fishial_keep_staged_crops=false` remains the default. Do not create a second
  retention path for enhanced imagery.

## Implementation design

### 1. Add a small enhancement boundary

Create a focused module such as `app/services/fish_enhancement.py` with a narrow
interface that can be replaced by a test double. Keep OpenCV-only modes separate
from the optional learned-model dependency.

Required behavior:

- `none` returns the original bytes without decode/re-encode.
- Classical modes reproduce the current pixel transforms.
- `funie_gan` lazily loads one generator per worker process, sets `eval()`, disables
  gradients with `torch.inference_mode()`, and serializes access if the chosen
  runtime is unsafe for concurrent inference.
- Follow the upstream preprocessing/postprocessing contract exactly unless a test
  proves a necessary compatibility correction: JPEG BGR -> RGB, bicubic `256 x
  256`, float tensor in `[-1,1]`, batch dimension, generator inference, finite-value
  check, clamp to the documented range, denormalize to 8-bit RGB, RGB -> BGR, JPEG
  encode at a documented quality.
- Do not silently apply a second CLAHE, white-balance, sharpening, or upscaling pass
  around FUnIE-GAN. Additional combinations require their own measured A/B mode.
- Return enhanced JPEG bytes plus compact metadata: mode, model identifier, model
  SHA-256, input/output dimensions, inference milliseconds, and whether processing
  succeeded. Do not return or log pixel data.
- Keep normalized `expected_box` unchanged. The upstream square resize changes pixel
  dimensions but not normalized coordinates. Add a regression test for an
  off-center, edge-clamped fish.

Prefer the smallest maintainable inference runtime that works in the existing VIAME
GPU worker. First inventory what the base image already contains. Do not add PyTorch
to the core API image or base development dependency just because the reference code
uses it. If the VIAME worker already has a compatible PyTorch, reuse it. Otherwise,
compare these options and choose the smallest verified one:

1. a `funie` optional dependency installed only in the GPU worker;
2. a pinned ONNX export plus an already-compatible ONNX runtime;
3. a separate enhancement image only if dependency conflicts cannot be isolated.

Do not create a network service when an in-process adapter works. Do not install both
TensorFlow and PyTorch. Record the decision and measured image-size impact in the
final report.

### 2. Pin provenance and keep runtime offline

Do not clone GitHub or download weights at application startup. Resolve and record:

- the exact upstream commit used;
- which implementation/architecture file was adapted;
- the exact weight filename and source URL;
- the weight SHA-256 and byte size;
- the MIT licence notice and attribution.

Vendor only the minimum source required by the selected runtime, with its licence,
or write a small compatible adapter if that is cleaner. Do not copy training code,
discriminators, datasets, notebooks, evaluation bundles, or unrelated models.

Use an explicit model path and required hash. A helper may download the artifact for
an administrator, but it must require an explicit command, use HTTPS, write to a
temporary file, verify SHA-256 before atomic rename, and never be called by normal
startup or tests. Prefer mounting the model read-only into the live worker instead of
baking a large binary into every application image. Do not commit a large weight
file unless this repository already has an intentional model-artifact policy that
requires it.

Suggested settings; adjust names only when the current configuration style gives a
clearer equivalent:

- `fishial_preprocess`: extend allowed values with `funie_gan`; default `none`.
- `fishial_funie_model_path`: optional path, required only for `funie_gan`.
- `fishial_funie_model_sha256`: required 64-character lowercase digest when enabled.
- `fishial_funie_device`: `auto | cpu | cuda`, default `auto`.
- `fishial_funie_jpeg_quality`: bounded integer, default `95`.

Validate syntax in `Settings`. Validate model existence, digest, state-dict shape,
and device compatibility when the live worker initializes the selected enhancer,
before a live session can reserve a Fishial image call. The API service must still
start without the optional ML runtime when the mode is `none`.

Use `torch.load(..., map_location=..., weights_only=True)` when supported. Do not
unpickle an untrusted arbitrary object. If legacy weights cannot load safely, convert
them once in an explicit administrator tool and document the conversion; do not
weaken safe loading in production.

### 3. Move preprocessing to the last free point

`LiveTracker._stage_species_crop` currently applies preprocessing while staging many
unpaid candidates. Refactor so it stores the clean original JPEG and quality data.
Apply the selected preprocessing in `SpeciesIdentifier._identify_track` only after a
track/frame is chosen, and immediately **before** `_reserve()`.

The order for one selected frame must be:

```text
read clean staged JPEG
-> preprocess/enhance locally
-> validate encoded JPEG and metadata
-> append a pre-send audit record without image bytes
-> reserve the Fishial image attempt durably
-> call Fishial with those exact bytes and the unchanged expected_box
-> persist the normal response/vote audit
```

If decode, model load, inference, range validation, or JPEG encoding fails, record a
fixed local reason such as `preprocessing failed`, do not call `_reserve()`, and do
not contact Fishial for that frame. Continue only if another staged frame can still
satisfy consensus; otherwise end as review required using the existing stop logic.
Do not fall back silently to the original bytes under a request labelled
`funie_gan`, because that makes A/B results and provenance untrustworthy.

Persist only compact per-frame provenance in `fishial_votes_json`: selected mode,
model id/hash, original and submitted-byte SHA-256, dimensions, latency, and fixed
failure reason. Never persist the generated image bytes in JSON. Existing scratch
and optional retention rules own image lifetime.

### 4. Extend the bounded replay harness

Keep the existing `--dry-run` path network-free. It replays stored Fishial responses,
not pixels, so it cannot evaluate a new image transform.

Extend paid replay with an explicit list such as:

```text
--preprocess-modes none,funie_gan
```

Requirements:

- Refuse an unknown/duplicate mode and refuse replay without positive
  `--max-calls` exactly as today.
- Freeze the mode list, crop hashes, expected boxes, model hash, settings, and call
  ceiling in the existing SQLite journal. A changed spec must not reuse a journal.
- Schedule complete pairs (`none`, then `funie_gan` for the same crop) before moving
  to the next crop so a retry cannot consume the entire budget on unpaired samples.
- Save exact submitted variants below the experiment directory for human review,
  subject to the existing explicit retention behavior.
- Never alter or reuse
  `data/outputs/fishial-experiments/smartbay-screen-20260910/replay.sqlite`; it is a
  completed six-call experiment. Create a new directory for FUnIE-GAN.
- Report object detected, non-empty species list, accepted top species, certainty,
  regional rejection, target IoU, usable answer, and preprocessing latency for each
  side of each pair.
- Support optional ground-truth scientific name per crop in `manifest.json`. Report
  top-1 correct, wrong, and abstained counts only where a ground-truth label exists.
  Do not equate `usable=true` with correct.

No real Fishial calls are authorized by this implementation task. Build the command
and dry-run/local benchmark, then stop. A human must separately approve a paid replay.

### 5. Cost and quality gates

The integration must optimize the whole decision, not image aesthetics.

Produce an offline report over every clean retained crop currently available (at
minimum the two SmartBay crops in
`data/outputs/fishial-experiments/smartbay-screen-20260910/` if present):

- original and enhanced dimensions/hashes;
- decode, inference, and encode latency after one warm-up;
- median and p95 latency;
- peak process RSS and GPU memory when measurable;
- model/runtime/container size added;
- failure count;
- a side-by-side contact sheet or saved paired images for human inspection;
- optional no-reference metrics clearly labelled as diagnostic only.

Do not use PSNR/SSIM without paired ground truth. Do not use UIQM or prettier color as
a promotion gate. The production promotion gate is a representative labelled crop
set, evaluated on the same crops in both modes, with:

- more correct accepted Fishial identifications, or equal correct identifications
  with fewer abstentions;
- no increase in confidently wrong species;
- no weakening of vote/region/object-match rules;
- identical Fishial request ceiling;
- acceptable worker latency and memory under the measured deployment hardware.

If labels are unavailable, finish with the feature implemented and disabled, report
what can be measured, and give the exact bounded command for a future labelled A/B.

## Tests

Add focused tests with a fake enhancer and fake Fishial client. Cover at least:

- `none` returns byte-identical input without constructing the learned enhancer;
- existing classical transforms remain deterministic and preserve expected output;
- configuration accepts `funie_gan`, rejects invalid mode/device/hash/quality, and
  requires model settings only when that mode is selected;
- learned runtime is imported lazily and the model is loaded once per worker;
- generator is in evaluation/inference mode and output is finite, clamped, valid
  JPEG, and deterministic for a fixed fixture;
- the original staged JPEG is clean and unchanged; enhancement is not invoked for
  candidates that are never selected;
- enhancement happens before `_reserve()`; a preprocessing failure makes zero
  reservations and zero Fishial calls;
- one successful frame invokes the enhancer once even if the Fishial adapter retries
  the same submitted bytes after a 401/429/5xx;
- the submitted payload is the enhanced payload, while `none` submits the original;
- normalized `expected_box` is unchanged, including an edge-clamped example;
- audit provenance contains hashes/latency/model id but no bytes, secrets, URLs, or
  private absolute paths;
- scratch cleanup and `KEEP_STAGED_CROPS=false` behavior remain unchanged;
- replay rejects missing call ceilings and changed journals, schedules complete
  pairs, resumes without duplicate calls, and records model provenance;
- the existing budget-invariant test still proves total `identify` calls never
  exceed `fish_target * frames_per_fish + fishial_max_api_retries`;
- the entire ordinary test suite passes without torch, a model file, network, camera,
  or GPU.

If a real artifact is available, add a separately marked smoke test that checks the
known SHA-256, loads safely on CPU, transforms one tiny fixture, and verifies a valid
deterministic JPEG. It must skip with a precise reason when the optional artifact or
runtime is absent.

## Documentation and deployment

Update `.env.example` and the Fishial section of `README.md` with:

- why FUnIE-GAN operates only on selected Fishial crops;
- exact setup/pinning/checksum instructions;
- CPU/GPU selection and expected startup failure modes;
- that the feature ships off and why GAN output can harm species markings;
- the offline benchmark command;
- the separate, explicitly paid paired-replay command and its hard ceiling;
- model and code licence attribution;
- how to remove/disable the optional artifact cleanly.

Update only the live worker deployment path that needs the runtime/model. The API,
mock worker, migrations, VIAME detection/tracking graph, annotated video, and frontend
should not gain a learned-model dependency. Avoid a database migration unless the
existing JSON audit cannot represent the required provenance; explain any migration
before adding it.

## Verification order

1. Run the smallest existing tests around crop staging, species identification, and
   replay before editing to establish a baseline.
2. Implement the dependency-free boundary and lazy call ordering with fakes.
3. Run the focused tests.
4. Add the optional real runtime/artifact and run the CPU smoke test if available.
5. Run the local offline benchmark. Make no Fishial call.
6. Run the full Python suite using the repository's short Windows temp path:

   ```powershell
   .venv\Scripts\python.exe -m pytest --basetemp=C:/tmp/funie-pt
   ```

7. Run the configured lint/checks relevant to changed files. Build the GPU image only
   if needed to verify the chosen optional runtime; do not pull/rebuild unrelated
   services.

Once a targeted check and the full suite pass, do not repeat them unless another
change or failure justifies it.

## Final report

Lead with the implemented outcome. Then report, concisely:

- files and behavior changed;
- exact upstream commit, model source, SHA-256, byte size, and licence handling;
- runtime choice and why it was the lowest-cost compatible option;
- proof that enhancement is lazy and happens before reservation;
- proof that paid-call ceilings and abstention behavior are unchanged;
- focused/full test results and any skipped real-model test;
- offline latency, memory, image/container-size results, and artifact locations;
- whether labelled end-to-end evidence exists; if it does not, state that
  `FISHIAL_PREPROCESS=none` remains the justified production default;
- the exact future paired-replay command and its maximum possible paid calls;
- any blocker or scope change.

Do not claim that FUnIE-GAN improves Fishial species accuracy unless the paired,
labelled end-to-end gate above passes.
