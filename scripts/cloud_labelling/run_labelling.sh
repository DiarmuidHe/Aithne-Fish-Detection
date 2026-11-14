#!/usr/bin/env bash
# Part 4: label a round's windows with run_windows workers. Two side by side gave more frames
# per dollar than one (Part 3 probe); the second starts once the first has loaded SAM 3, which
# is built on the CPU first. Use inside tmux; a lock prevents a second copy.
#   scripts/cloud_labelling/run_labelling.sh [round folder] [workers]
# AITHNE_GUARD_WHEN_DONE=<minutes> replaces the shutdown guard with one that many minutes after
# the last worker ends, so a finished run doesn't sit billing until the old deadline.
# AITHNE_SHARD=i/n makes this instance shard i of n (round 2, 2026-11-14): its workers take
# rows i*WORKERS .. i*WORKERS+WORKERS-1 of n*WORKERS, so n instances split one to_label.csv.
# AITHNE_TO_LABEL=<csv> gives the workers that list instead of <round>/selection/to_label.csv: an
# instance's share when a running round is split (round2.py split).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-data/outputs/smartbay-rounds/round-1}"
WORKERS="${2:-2}"
SHARD="${AITHNE_SHARD:-0/1}"
SHARD_INDEX="${SHARD%/*}"
SHARDS="${SHARD#*/}"
if [[ ! "$SHARD_INDEX" =~ ^[0-9]+$ || ! "$SHARDS" =~ ^[0-9]+$ ]] || ((SHARD_INDEX >= SHARDS)); then
  echo "AITHNE_SHARD must be i/n with 0 <= i < n, not $SHARD" >&2
  exit 2
fi
STAGGER="${AITHNE_STAGGER_SECONDS:-30}"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_LABELLING_PACKAGES:-$REPO_ROOT/.labelling-python-packages}"

cd "$REPO_ROOT"
if [[ ! -d "$PACKAGES/rfdetr" || ! -f "$ROOT/journal.json" ]]; then
  echo "run bootstrap_labelling.sh and seed $ROOT/journal.json first" >&2
  exit 1
fi
TO_LABEL="${AITHNE_TO_LABEL:-}"
if [[ -n "$TO_LABEL" && ! -f "$TO_LABEL" ]]; then
  echo "AITHNE_TO_LABEL=$TO_LABEL is not a file" >&2
  exit 2
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

echo "Started $(date -u +%FT%TZ): $WORKERS workers on $ROOT, shard $SHARD${TO_LABEL:+, rows from $TO_LABEL}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
pids=()
for ((k = 0; k < WORKERS; k++)); do
  if ((k)); then sleep "$STAGGER"; fi
  global=$((SHARD_INDEX * WORKERS + k))
  "$PYTHON" -m scripts.cloud_labelling.run_windows --root "$ROOT" --worker "$global/$((SHARDS * WORKERS))" \
    ${TO_LABEL:+--to-label "$TO_LABEL"} > "$ROOT/run_windows/worker-$global.log" 2>&1 &
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
