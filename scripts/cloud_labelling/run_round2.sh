#!/usr/bin/env bash
# Round 2 on an instance launched from the round 2 image (docs/smartbay-round-2.md). Use in tmux.
#   scripts/cloud_labelling/run_round2.sh label <i> <n>
#       label shard i of n of round 2's to_label.csv with two SAM 3 workers, then stop.
#   scripts/cloud_labelling/run_round2.sh train
#       restore every window from the public archive, build the dataset (round 1's validation days
#       kept), train RF-DETR Nano and evaluate it per camera, then stop.
#   scripts/cloud_labelling/run_round2.sh after-label
#       when one instance labels everything: wait for the tmux session "label" to end, merge round 1's
#       labels (round1-bundle) with this instance's into round-2-train, stop unless at least 90% of the
#       new windows were labelled, then train as above on round-2-train.
#       AITHNE_EXTRA_SHARDS="<folder> ..." (a running round split with round2.py split): after
#       labelling, also wait up to AITHNE_SHARD_WAIT_MINUTES (240) for each folder's .complete,
#       written by relay_shards.sh once that instance's results are copied in, and merge those too.
# Each sets a shutdown guard first (AITHNE_GUARD_MINUTES overrides: 240 for label, 480 for train and,
# once labelling has ended, for after-label) and re-arms it for 20 minutes when it ends or fails, so
# a finished or broken run doesn't sit billing.
set -uo pipefail

MODE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="data/outputs/smartbay-rounds/round-2"
ROUND1="data/outputs/smartbay-rounds/round-1"
MIN_LABELLED_SHARE="0.9"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_LABELLING_PACKAGES:-$REPO_ROOT/.labelling-python-packages}"
case "$MODE" in
  label)
    SHARD_INDEX="${2:-}"; SHARDS="${3:-}"
    if [[ ! "$SHARD_INDEX" =~ ^[0-9]+$ || ! "$SHARDS" =~ ^[0-9]+$ ]] || ((SHARD_INDEX >= SHARDS)); then
      echo "usage: $0 label <i> <n>, with 0 <= i < n" >&2; exit 2
    fi
    GUARD_MINUTES="${AITHNE_GUARD_MINUTES:-240}" ;;
  train|after-label) GUARD_MINUTES="${AITHNE_GUARD_MINUTES:-480}" ;;
  *) echo "usage: $0 label <i> <n> | train | after-label" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
if [[ ! -d "$PACKAGES/rfdetr" || ! -f "$ROOT/journal.json" ]]; then
  echo "unpack the round 2 bundle into $REPO_ROOT first (no $ROOT/journal.json)" >&2
  exit 1
fi
export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
log() { echo "$(date -u +%FT%TZ) $*"; }
EXTRA=()  # after-label: the other labellers' folders, as --shard arguments

if [[ "$MODE" == "after-label" ]]; then
  if [[ ! -f "$ROUND1/journal.json" ]]; then
    echo "unpack round 1's labels (round2 round1-bundle) first: no $ROUND1/journal.json" >&2; exit 1
  fi
  # Its guard would cut labelling short, so it is set only once labelling has ended (and has
  # re-armed its own 20-minute guard, which this one then replaces)
  log "waiting for the labelling session to end"
  while tmux has-session -t label 2>/dev/null; do sleep 60; done
  sleep 5
  if [[ -n "${AITHNE_EXTRA_SHARDS:-}" ]]; then
    WAIT_MINUTES="${AITHNE_SHARD_WAIT_MINUTES:-240}"
    sudo shutdown -h "+$((WAIT_MINUTES + 30))"
    log "waiting up to $WAIT_MINUTES minutes for the other labellers' results: $AITHNE_EXTRA_SHARDS"
    deadline=$((SECONDS + WAIT_MINUTES * 60))
    for shard in $AITHNE_EXTRA_SHARDS; do
      while [[ ! -f "$shard/.complete" ]] && ((SECONDS < deadline)); do sleep 60; done
      if [[ -f "$shard/.complete" ]]; then
        EXTRA+=(--shard "$shard"); log "results from $shard are here"
      else
        log "no results from $shard: merging without them"
      fi
    done
  fi
fi

sudo shutdown -h "+$GUARD_MINUTES"
log "guard set for $GUARD_MINUTES minutes"
trap 'sudo shutdown -h +20; log "guard re-armed: the instance stops in 20 minutes"' EXIT

if [[ "$MODE" == "label" ]]; then
  log "labelling shard $SHARD_INDEX of $SHARDS"
  AITHNE_SHARD="$SHARD_INDEX/$SHARDS" scripts/cloud_labelling/run_labelling.sh "$ROOT" 2
  log "labelling finished with status $?: copy $ROOT back to the PC"
  exit 0
fi

step() {  # name, command...
  local name="$1"; shift
  log "start $name"
  if ! "$@" > "$ROOT/logs/$name.log" 2>&1; then
    log "FAILED $name (see $ROOT/logs/$name.log)"; tail -n 20 "$ROOT/logs/$name.log"
    exit 1
  fi
  log "done $name"; tail -n 15 "$ROOT/logs/$name.log"
}
if [[ "$MODE" == "after-label" ]]; then
  TRAIN_ROOT="data/outputs/smartbay-rounds/round-2-train"
  step merge "$PYTHON" -m scripts.cloud_labelling.round2 merge --selection "$ROOT/selection" --shard "$ROOT" \
    ${EXTRA[@]+"${EXTRA[@]}"} --out "$TRAIN_ROOT" --round1 "$ROUND1"
  if ! "$PYTHON" - "$TRAIN_ROOT/merge_summary.json" "$ROOT/selection/to_label.csv" "$MIN_LABELLED_SHARE" <<'PY'
import csv, json, sys
summary = json.load(open(sys.argv[1]))
new = sum(1 for _ in csv.DictReader(open(sys.argv[2], newline="")))
done = summary["labelled"] - (summary["windows"] - new)  # Labels stage complete; every earlier window is
share = done / new
print(f"{done} of {new} new windows labelled ({share:.0%}); kept {summary['kept']} of {summary['windows']}")
sys.exit(0 if share >= float(sys.argv[3]) else 1)
PY
  then
    log "too few new windows labelled to train on: stopping"; exit 1
  fi
  ROOT="$TRAIN_ROOT"
  mkdir -p "$ROOT/logs"
fi
# Restore needs the network for the public archive; every later stage runs offline
step restore "$PYTHON" -m scripts.hard_negatives restore --root "$ROOT" --out "$ROOT" --workers 4
step train-and-eval "$PYTHON" -m scripts.student_detector round --root "$ROOT" --validation-from "$ROUND1/journal.json"
log "training finished: copy $ROOT back to the PC; student/round_eval.json has by_camera"
