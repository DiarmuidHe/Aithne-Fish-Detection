#!/usr/bin/env bash
# Prepare or inspect SAM 3 labelling and RF-DETR training on an AWS Deep Learning PyTorch AMI
# (docs/smartbay-rolling-training-prompt.md, Part 2). --check installs and writes nothing.
#   scripts/cloud_labelling/bootstrap_labelling.sh [install|--check] [cut-window.mp4]
# Given a cut window, it ends with a 30-frame SAM 3 smoke run on it (results printed only).
set -euo pipefail

MODE="${1:-install}"
SMOKE="${2:-}"
if [[ "$MODE" != "install" && "$MODE" != "--check" ]]; then
  echo "usage: $0 [install|--check] [cut-window.mp4]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_LABELLING_PACKAGES:-$REPO_ROOT/.labelling-python-packages}"
LOCK="$REPO_ROOT/scripts/cloud-labelling-train-requirements.txt"
MODEL="$REPO_ROOT/data/models/sam3/sam3.pt"
EXPECTED_MODEL_BYTES=3450062241
EXPECTED_MODEL_SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "missing AMI Python: $BASE_PYTHON" >&2
  exit 1
fi
if [[ "$(findmnt -n -o TARGET --target "$REPO_ROOT")" == /opt/dlami/nvme* ]]; then
  echo "$REPO_ROOT is on the instance store, which is wiped on stop; use /home/ubuntu" >&2
  exit 1
fi

echo "Repository: $REPO_ROOT"
echo "Filesystem: $(findmnt -n -o SOURCE,FSTYPE,TARGET --target "$REPO_ROOT")"
echo "Free: $(df -h --output=avail "$REPO_ROOT" | tail -n 1 | tr -d ' ') of $(df -h --output=size "$REPO_ROOT" | tail -n 1 | tr -d ' ')"
"$BASE_PYTHON" --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "$MODE" == "install" ]]; then
  if ! command -v ffmpeg >/dev/null; then
    # Unattended upgrades often hold the apt lock just after boot.
    sudo apt-get -o DPkg::Lock::Timeout=300 update -q
    sudo apt-get -o DPkg::Lock::Timeout=300 install -y -q ffmpeg
  fi
  mkdir -p "$PACKAGES"
  "$BASE_PYTHON" -m pip install --upgrade --target "$PACKAGES" --no-deps --require-hashes -r "$LOCK"
fi
command -v ffmpeg >/dev/null || { echo "missing ffmpeg; run $0 install" >&2; exit 1; }
ffmpeg -hide_banner -version | head -n 1

export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
cd "$REPO_ROOT"

# Every module the sam3, static, labels and student stages import, including the lazy ones.
"$BASE_PYTHON" - <<'PY'
import importlib
import json
import sys

needed = ("torch", "torchvision", "cv2", "numpy", "psutil", "lap", "clip", "timm", "ultralytics.models.sam",
          "ultralytics.models.sam.predict", "ultralytics.trackers.bot_sort", "ultralytics.trackers.utils.gmc",
          "rfdetr", "pytorch_lightning", "app.services.box_tracking", "app.services.static_filter",
          "scripts.sam3_stages", "scripts.student_detector")
failed = {}
for name in needed:
    try:
        importlib.import_module(name)
    except Exception as error:  # noqa: BLE001 - report every failure, then stop
        failed[name] = f"{type(error).__name__}: {error}"[:300]
if not failed:
    from rfdetr import RFDETRNano, training  # noqa: F401 - the names student_detector uses
    from pytorch_lightning import Callback  # noqa: F401
    from scripts.sam3_autolabel import missing_requirements
    failed |= {f"sam3 requirement {k}": v for k, v in missing_requirements().items()}
print(json.dumps({"python": sys.version, "failed_imports": failed}, indent=2))
if failed:
    raise SystemExit(1)
import torch, torchvision
print(json.dumps({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

if [[ ! -f "$MODEL" ]]; then
  echo "missing SAM 3 checkpoint: $MODEL" >&2
  exit 1
fi
[[ -f "$(dirname "$MODEL")/LICENSE" ]] || { echo "missing Meta's LICENSE beside $MODEL" >&2; exit 1; }
ACTUAL_BYTES="$(stat -c %s "$MODEL")"
[[ "$ACTUAL_BYTES" == "$EXPECTED_MODEL_BYTES" ]] || {
  echo "checkpoint size $ACTUAL_BYTES != $EXPECTED_MODEL_BYTES" >&2; exit 1;
}
ACTUAL_SHA="$(sha256sum "$MODEL" | cut -d' ' -f1)"
[[ "$ACTUAL_SHA" == "$EXPECTED_MODEL_SHA" ]] || {
  echo "checkpoint SHA256 $ACTUAL_SHA != $EXPECTED_MODEL_SHA" >&2; exit 1;
}
echo "SAM 3 checkpoint verified: $ACTUAL_SHA"
"$BASE_PYTHON" -c "import json; from scripts.student_detector import verify_rfdetr; print(json.dumps(verify_rfdetr()))"
echo "RF-DETR Nano checkpoint verified."

if [[ -n "$SMOKE" ]]; then
  "$BASE_PYTHON" - "$SMOKE" <<'PY'
import json
import os
import sys
from pathlib import Path

from scripts.sam3_botsort import OFFLINE, PROMPTS

os.environ.update(OFFLINE)
import torch

from scripts.sam3_autolabel import SAM3_CHECKPOINT, build_predictor, detect_video
from scripts.sam3_stages import DEFAULT_SETTINGS

predictor, _ = build_predictor(SAM3_CHECKPOINT, **DEFAULT_SETTINGS)
frames = list(detect_video(predictor, Path(sys.argv[1]), PROMPTS["fish"], max_frames=30))
seconds = frames[-1]["seconds"] if frames else 0
print(json.dumps({"smoke_video": sys.argv[1], "frames": len(frames), "boxes": sum(len(f["boxes"]) for f in frames),
                  "fps": round(len(frames) / seconds, 2) if seconds else None,
                  "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}, indent=2))
if len(frames) != 30:
    raise SystemExit(f"SAM 3 smoke run read {len(frames)} frames, expected 30")
PY
fi
echo "Labelling check passed."
