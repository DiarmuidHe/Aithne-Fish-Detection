#!/usr/bin/env bash
# Part 4: label a round's windows with run_windows workers. Two side by side gave more frames
# per dollar than one (Part 3 probe); the second starts once the first has loaded SAM 3, which
# is built on the CPU first. Use inside tmux; a lock prevents a second copy.
#   scripts/cloud_labelling/run_labelling.sh [round folder] [workers]
# AITHNE_GUARD_WHEN_DONE=<minutes> replaces the shutdown guard with one that many minutes after
# the last worker ends, so a finished run doesn't sit billing until the old deadline.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-data/outputs/smartbay-rounds/round-1}"
WORKERS="${2:-2}"
STAGGER="${AITHNE_STAGGER_SECONDS:-30}"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_LABELLING_PACKAGES:-$REPO_ROOT/.labelling-python-packages}"

cd "$REPO_ROOT"
if [[ ! -d "$PACKAGES/rfdetr" || ! -f "$ROOT/journal.json" ]]; then
  echo "run bootstrap_labelling.sh and seed $ROOT/journal.json first" >&2
  exit 1
fi
mkdir -p "$ROOT/run_windows"
exec 9>"$ROOT/.labelling.lock"
if ! flock -n 9; then
  echo "another labelling run holds $ROOT/.labelling.lock" >&2
  exit 1
fi

export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
export PYTHONUNBUFFERED=1

echo "Started $(date -u +%FT%TZ): $WORKERS workers on $ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
pids=()
for ((k = 0; k < WORKERS; k++)); do
  if ((k)); then sleep "$STAGGER"; fi
  "$PYTHON" -m scripts.cloud_labelling.run_windows --root "$ROOT" --worker "$k/$WORKERS" \
    > "$ROOT/run_windows/worker-$k.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "Finished $(date -u +%FT%TZ), status $status"
tail -n 12 "$ROOT"/run_windows/worker-*.log
if [[ -n "${AITHNE_GUARD_WHEN_DONE:-}" ]]; then
  sudo shutdown -h "+$AITHNE_GUARD_WHEN_DONE"
fi
exit "$status"
