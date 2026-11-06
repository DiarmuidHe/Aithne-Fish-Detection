#!/usr/bin/env bash
# Run one resumable SmartBay scan. Use inside tmux; a lock prevents duplicate GPU runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_ARCHIVE_PACKAGES:-$REPO_ROOT/.archive-python-packages}"
OUTPUT="${AITHNE_ARCHIVE_OUTPUT:-$REPO_ROOT/data/archive_scan}"
LOCK="$OUTPUT/.cloud-scan.lock"

if [[ ! -x "$PYTHON" || ! -d "$PACKAGES/ultralytics" ]]; then
  echo "scanner environment is incomplete; run scripts/cloud_labelling/bootstrap_archive_scan.sh first" >&2
  exit 1
fi
mkdir -p "$OUTPUT" "$REPO_ROOT/data/ultralytics"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another archive scan holds $LOCK" >&2
  exit 1
fi

export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

if [[ $# -eq 0 ]]; then
  set -- --camera 1 --year 2025 --all-pngs --max-mp4s 3 \
    --retry-incomplete 3 --retry-delay-seconds 10 --max-days 2
fi

echo "Started $(date -u +%FT%TZ)"
echo "Output: $OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -m scripts.smartbay_archive_scan.workflow --output "$OUTPUT" "$@"
echo "Finished $(date -u +%FT%TZ)"
