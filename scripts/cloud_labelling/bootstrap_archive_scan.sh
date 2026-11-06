#!/usr/bin/env bash
# Prepare or inspect the SmartBay archive scanner on an AWS Deep Learning PyTorch AMI.
set -euo pipefail

MODE="${1:-install}"
if [[ "$MODE" != "install" && "$MODE" != "--check" ]]; then
  echo "usage: $0 [install|--check]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_ARCHIVE_PACKAGES:-$REPO_ROOT/.archive-python-packages}"
MODEL="$REPO_ROOT/data/models/sam3/sam3.pt"
EXPECTED_MODEL_BYTES=3450062241
EXPECTED_MODEL_SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "missing AMI Python: $BASE_PYTHON" >&2
  exit 1
fi

echo "Repository: $REPO_ROOT"
echo "Filesystem: $(findmnt -n -o SOURCE,FSTYPE,TARGET --target "$REPO_ROOT")"
"$BASE_PYTHON" --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "$MODE" == "install" ]]; then
  mkdir -p "$PACKAGES"
  "$BASE_PYTHON" -m pip install --upgrade --target "$PACKAGES" --no-deps --require-hashes \
    -r "$REPO_ROOT/scripts/cloud-labelling-archive-requirements.txt"
fi

export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$BASE_PYTHON"

"$PYTHON" - <<'PY'
import importlib.util
import json
import sys

needed = ("torch", "torchvision", "cv2", "ultralytics", "timm", "clip", "psutil")
missing = [name for name in needed if importlib.util.find_spec(name) is None]
print(json.dumps({"python": sys.version, "missing_modules": missing}, indent=2))
if missing:
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
ACTUAL_BYTES="$(stat -c %s "$MODEL")"
ACTUAL_SHA="$(sha256sum "$MODEL" | cut -d' ' -f1)"
[[ "$ACTUAL_BYTES" == "$EXPECTED_MODEL_BYTES" ]] || {
  echo "checkpoint size $ACTUAL_BYTES != $EXPECTED_MODEL_BYTES" >&2; exit 1;
}
[[ "$ACTUAL_SHA" == "$EXPECTED_MODEL_SHA" ]] || {
  echo "checkpoint SHA256 $ACTUAL_SHA != $EXPECTED_MODEL_SHA" >&2; exit 1;
}
echo "SAM 3 checkpoint verified: $ACTUAL_SHA"
echo "Archive scanner check passed."
