#!/usr/bin/env bash
# Scan a list of one camera's archive days, one day at a time, with the same settings as Cam 2's
# first sample (30 PNGs and 3 MP4s a day, seed 17). Round 2 uses it for 36 more Cam 2 days, picked
# on the PC by `smartbay_label_selection --pick-scan-days`. Use inside tmux.
#   scripts/cloud_labelling/run_scan_days.sh <camera number> <days file>
# A day that comes back incomplete is tried again, up to 4 times; then the run goes on to the
# next day and lists it at the end. Sets a shutdown guard first (AITHNE_GUARD_MINUTES, default
# 150) and re-arms it for AITHNE_GUARD_WHEN_DONE minutes (default 60) when it ends, so the
# instance waits for the window selection but never sits billing for long.
set -uo pipefail

CAMERA_NUMBER="${1:-}"
DAYS_FILE="${2:-}"
if [[ ! "$CAMERA_NUMBER" =~ ^[12]$ || ! -f "$DAYS_FILE" ]]; then
  echo "usage: $0 <camera number> <days file>" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AITHNE_BASE_PYTHON:-/opt/pytorch/bin/python}"
PACKAGES="${AITHNE_ARCHIVE_PACKAGES:-$REPO_ROOT/.archive-python-packages}"
OUTPUT="${AITHNE_ARCHIVE_OUTPUT:-$REPO_ROOT/data/archive_scan}"
CAMERA="SmartBayCam$CAMERA_NUMBER"
if [[ ! -x "$PYTHON" || ! -d "$PACKAGES/ultralytics" ]]; then
  echo "scanner environment is incomplete; run scripts/cloud_labelling/bootstrap_archive_scan.sh first" >&2
  exit 1
fi
mkdir -p "$OUTPUT" "$REPO_ROOT/data/ultralytics"
exec 9>"$OUTPUT/.cloud-scan.lock"
if ! flock -n 9; then
  echo "another archive scan holds $OUTPUT/.cloud-scan.lock" >&2
  exit 1
fi
export YOLO_OFFLINE=true
export YOLO_CONFIG_DIR="$REPO_ROOT/data/ultralytics"
export PYTHONPATH="$PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
log() { echo "$(date -u +%FT%TZ) $*"; }

sudo shutdown -h "+${AITHNE_GUARD_MINUTES:-150}"
trap 'sudo shutdown -h "+${AITHNE_GUARD_WHEN_DONE:-60}"; log "guard re-armed for ${AITHNE_GUARD_WHEN_DONE:-60} minutes"' EXIT

complete() {  # day: every selected PNG and MP4 processed, from the day's own report
  "$PYTHON" - "$OUTPUT/${CAMERA}_$1/summary.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
s = json.loads(path.read_text()) if path.exists() else {}
ok = s and s["processed_pngs"] == s["selected_pngs"] and s["processed_mp4s"] == s["selected_mp4s"]
sys.exit(0 if ok else 1)
PY
}

mapfile -t DAYS < <(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' "$DAYS_FILE")
log "scanning ${#DAYS[@]} $CAMERA days into $OUTPUT"
failed=()
n=0
for day in "${DAYS[@]}"; do
  n=$((n + 1))
  for attempt in 1 2 3 4; do
    if complete "$day"; then break; fi
    "$PYTHON" -m scripts.smartbay_archive_scan.workflow --output "$OUTPUT" --camera "$CAMERA_NUMBER" --day "$day" \
      --max-pngs 30 --max-mp4s 3 --seed 17 > "$OUTPUT/${CAMERA}_scan_$day.log" 2>&1 || true
    complete "$day" && break
    sleep 10
  done
  if complete "$day"; then log "done $day ($n of ${#DAYS[@]})"; else log "INCOMPLETE $day"; failed+=("$day"); fi
done
log "finished: $(( ${#DAYS[@]} - ${#failed[@]} )) of ${#DAYS[@]} days complete${failed:+; incomplete: ${failed[*]}}"
