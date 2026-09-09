#!/usr/bin/env bash
set -euo pipefail

VIAME_ROOT="${VIAME_ROOT:-/opt/noaa/viame}"
VIAME_SETUP_SCRIPT="${VIAME_SETUP_SCRIPT:-$VIAME_ROOT/setup_viame.sh}"
VIAME_TRACKER_PIPELINE="${VIAME_TRACKER_PIPELINE:-$VIAME_ROOT/configs/pipelines/tracker_default_fish_fusion.pipe}"

test -f "$VIAME_SETUP_SCRIPT"
test -f "$VIAME_TRACKER_PIPELINE"

# shellcheck disable=SC1090
source "$VIAME_SETUP_SCRIPT"
viame -h >/dev/null
kwiver runner --help >/dev/null

printf 'VIAME setup OK\n'
printf 'Tracker pipeline: %s\n' "$VIAME_TRACKER_PIPELINE"

