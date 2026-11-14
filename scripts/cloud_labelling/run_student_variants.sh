#!/usr/bin/env bash
# The student comparison on a new instance, unattended (docs/student-variants.md). Use inside tmux.
#   scripts/cloud_labelling/run_student_variants.sh [ARM ...]
#       fetch checkpoints, restore round 1's windows, rebuild its dataset, then train and evaluate
#       each arm in turn (default: nano-gentle medium), then compare them all with round 1.
# An arm that fails is logged and skipped, so the arms before it still reach the comparison.
# The shutdown guard covers setup plus every arm's training budget and evaluation (the
# guard-minutes stage); AITHNE_GUARD_MINUTES overrides it. It re-arms for 30 minutes when the run
# ends or fails, so a finished or broken run doesn't sit billing.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AITHNE_ROUND:-data/outputs/smartbay-rounds/round-1}"
OUT="${AITHNE_OUT:-data/outputs/smartbay-rounds/round-1-variants}"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_LABELLING_PACKAGES:-$REPO_ROOT/.labelling-python-packages}"
if [[ $# -gt 0 ]]; then ARMS=("$@"); else ARMS=(nano-gentle medium); fi
ARM_FLAGS=()
for arm in "${ARMS[@]}"; do ARM_FLAGS+=(--arm "$arm"); done

cd "$REPO_ROOT"
if [[ ! -d "$PACKAGES/rfdetr" || ! -f "$ROOT/journal.json" ]]; then
  echo "unpack the bundle and run AITHNE_NO_SAM3=1 bootstrap_labelling.sh install first" >&2
  exit 1
fi
export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
export PYTHONUNBUFFERED=1
if ! GUARD_MINUTES="${AITHNE_GUARD_MINUTES:-$("$PYTHON" -m scripts.student_variants guard-minutes "${ARM_FLAGS[@]}")}"; then
  echo "unknown arm in: ${ARMS[*]}" >&2
  exit 2
fi
mkdir -p "$OUT/logs"
exec 9>"$OUT/.run.lock"
if ! flock -n 9; then
  echo "another run holds $OUT/.run.lock" >&2
  exit 1
fi
log() { echo "$(date -u +%FT%TZ) $*"; }

sudo shutdown -h "+$GUARD_MINUTES"
log "guard set for $GUARD_MINUTES minutes; arms: ${ARMS[*]}"
trap 'sudo shutdown -h +30; log "guard re-armed: the instance stops in 30 minutes"' EXIT

step() {  # log name, stage arguments...
  local name="$1"; shift
  log "start $name"
  if ! "$PYTHON" -m scripts.student_variants "$@" --root "$ROOT" --out "$OUT" > "$OUT/logs/$name.log" 2>&1; then
    log "FAILED $name (see $OUT/logs/$name.log)"; tail -n 20 "$OUT/logs/$name.log"
    return 1
  fi
  log "done $name"; tail -n 30 "$OUT/logs/$name.log"
}

step weights weights "${ARM_FLAGS[@]}" || exit 1
step restore restore || exit 1
step data data || exit 1
finished=0
for arm in "${ARMS[@]}"; do
  if step "train-$arm" train --arm "$arm" && step "eval-$arm" eval --arm "$arm"; then
    finished=$((finished + 1))
  fi
done
if [[ $finished -eq 0 ]]; then
  log "no arm finished: nothing to compare"
  exit 1
fi
step compare compare || exit 1
log "finished $finished of ${#ARMS[@]} arms: copy $OUT back; comparison.json has every student"
