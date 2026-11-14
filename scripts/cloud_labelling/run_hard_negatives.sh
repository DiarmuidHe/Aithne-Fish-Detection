#!/usr/bin/env bash
# Hard negatives on a new instance, unattended (docs/smartbay-hard-negatives.md). Use inside tmux.
#   scripts/cloud_labelling/run_hard_negatives.sh mine
#       restore the cut windows, mine, write the review sheets, then stop for a person to look.
#   scripts/cloud_labelling/run_hard_negatives.sh train "<the reviewer's approval>"
#       build the dataset, train and evaluate, then stop.
# Each run sets a shutdown guard first and re-arms it for 30 minutes when it ends or fails, so a
# finished or broken run doesn't sit billing. AITHNE_GUARD_MINUTES overrides the guard: the
# defaults are 240 (about 2 h of work) and 330 (about 3 h 15 min of work). AITHNE_LIMIT runs mine
# on that many windows only, as a quick check. AITHNE_USE_EXCLUDED=1 also trains on zones from
# windows the frozen exclusions left out.
set -uo pipefail

MODE="${1:-}"
APPROVAL="${2:-}"
if [[ "$MODE" != "mine" && ! ( "$MODE" == "train" && -n "$APPROVAL" ) ]]; then
  echo "usage: $0 mine | train \"<the reviewer's approval>\"" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AITHNE_ROUND:-data/outputs/smartbay-rounds/round-1}"
OUT="${AITHNE_OUT:-data/outputs/smartbay-rounds/round-1-hn}"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_LABELLING_PACKAGES:-$REPO_ROOT/.labelling-python-packages}"
if [[ "$MODE" == "mine" ]]; then GUARD_MINUTES="${AITHNE_GUARD_MINUTES:-240}"; else GUARD_MINUTES="${AITHNE_GUARD_MINUTES:-330}"; fi

cd "$REPO_ROOT"
if [[ ! -d "$PACKAGES/rfdetr" || ! -f "$ROOT/journal.json" ]]; then
  echo "unpack the bundle and run AITHNE_NO_SAM3=1 bootstrap_labelling.sh install first" >&2
  exit 1
fi
mkdir -p "$OUT/logs"
exec 9>"$OUT/.run.lock"
if ! flock -n 9; then
  echo "another run holds $OUT/.run.lock" >&2
  exit 1
fi

export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
export PYTHONUNBUFFERED=1
log() { echo "$(date -u +%FT%TZ) $*"; }

sudo shutdown -h "+$GUARD_MINUTES"
log "guard set for $GUARD_MINUTES minutes"
trap 'sudo shutdown -h +30; log "guard re-armed: the instance stops in 30 minutes"' EXIT

step() {  # name, stage arguments...
  local name="$1"; shift
  log "start $name"
  if ! "$PYTHON" -m scripts.hard_negatives "$name" --root "$ROOT" --out "$OUT" "$@" > "$OUT/logs/$name.log" 2>&1; then
    log "FAILED $name (see $OUT/logs/$name.log)"; tail -n 20 "$OUT/logs/$name.log"
    exit 1
  fi
  log "done $name"; tail -n 30 "$OUT/logs/$name.log"
}

if [[ "$MODE" == "mine" ]]; then
  step restore ${AITHNE_LIMIT:+--limit "$AITHNE_LIMIT"}
  step mine ${AITHNE_LIMIT:+--limit "$AITHNE_LIMIT"}
  log "mining finished: copy $OUT back and look at review/*.jpg"
else
  step data --approval "$APPROVAL" ${AITHNE_USE_EXCLUDED:+--use-excluded}
  step train
  step eval
  log "training finished: copy $OUT back; comparison.json has both students"
fi
