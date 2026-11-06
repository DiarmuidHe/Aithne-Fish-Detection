#!/usr/bin/env bash
# Parts 5-7 of docs/smartbay-rolling-training-prompt.md, unattended (approved 2026-10-30): wait
# for the labelling tmux session to end, extend the shutdown guard, run the automatic checks,
# build the day-split dataset, train and evaluate the student, then re-arm the guard for 30
# minutes. Any failure stops the chain and also re-arms the guard. Use inside tmux.
#   scripts/cloud_labelling/run_after_labelling.sh [round folder]
# AITHNE_AFTER_GUARD_MINUTES (default 270: checks, dataset, the 3 h training budget, evaluation
# and a margin) and AITHNE_MIN_LABELLED (default 0.9: below it nothing is trained).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-data/outputs/smartbay-rounds/round-1}"
GUARD_MINUTES="${AITHNE_AFTER_GUARD_MINUTES:-270}"
MIN_LABELLED="${AITHNE_MIN_LABELLED:-0.9}"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_LABELLING_PACKAGES:-$REPO_ROOT/.labelling-python-packages}"

cd "$REPO_ROOT"
export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
export PYTHONUNBUFFERED=1
log() { echo "$(date -u +%FT%TZ) $*"; }

log "waiting for the labelling session to end"
while tmux has-session -t labelling 2>/dev/null; do sleep 60; done
sudo shutdown -h "+$GUARD_MINUTES"
log "labelling ended; guard set for $GUARD_MINUTES minutes"
trap 'sudo shutdown -h +30; log "guard re-armed: the instance stops in 30 minutes"' EXIT

step() {  # name, command...
  local name="$1"; shift
  log "start $name"
  if ! "$@" > "$ROOT/after/$name.log" 2>&1; then
    log "FAILED $name (see $ROOT/after/$name.log)"; tail -n 20 "$ROOT/after/$name.log"
    exit 1
  fi
  log "done $name"; tail -n 5 "$ROOT/after/$name.log"
}

mkdir -p "$ROOT/after"
step checks "$PYTHON" -m scripts.cloud_labelling.round_checks --root "$ROOT" --min-labelled "$MIN_LABELLED"
step data "$PYTHON" -m scripts.student_detector round-data --root "$ROOT"
step train "$PYTHON" -m scripts.student_detector round-train --root "$ROOT"
step eval "$PYTHON" -m scripts.student_detector round-eval --root "$ROOT"
log "Parts 5-7 finished"
