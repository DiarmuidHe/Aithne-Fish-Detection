#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: %s /path/to/video.mp4 [output.csv]\n' "$0" >&2
  exit 2
fi

INPUT_VIDEO="$1"
OUTPUT_CSV="${2:-viame_tracks.csv}"
VIAME_ROOT="${VIAME_ROOT:-/opt/noaa/viame}"
VIAME_SETUP_SCRIPT="${VIAME_SETUP_SCRIPT:-$VIAME_ROOT/setup_viame.sh}"
VIAME_PIPELINE="${VIAME_TRACKER_PIPELINE:-$VIAME_ROOT/configs/pipelines/tracker_default_fish_fusion.pipe}"
VIAME_DOWNSAMPLE_FPS="${VIAME_DOWNSAMPLE_FPS:-10}"
VIAME_DETECTOR_SCORE_THRESHOLD="${VIAME_DETECTOR_SCORE_THRESHOLD:-0.10}"
VIAME_TRACKER_HIGH_THRESHOLD="${VIAME_TRACKER_HIGH_THRESHOLD:-0.45}"
VIAME_TRACKER_LOW_THRESHOLD="${VIAME_TRACKER_LOW_THRESHOLD:-0.10}"
VIAME_TRACKER_NEW_TRACK_THRESHOLD="${VIAME_TRACKER_NEW_TRACK_THRESHOLD:-0.50}"
VIAME_TRACKER_BUFFER_FRAMES="${VIAME_TRACKER_BUFFER_FRAMES:-30}"

test -f "$INPUT_VIDEO"
test -f "$VIAME_SETUP_SCRIPT"
test -f "$VIAME_PIPELINE"

# shellcheck disable=SC1090
source "$VIAME_SETUP_SCRIPT"

kwiver runner -I "$VIAME_ROOT/configs/pipelines" "$VIAME_PIPELINE" \
  -s "input:video_filename=$INPUT_VIDEO" \
  -s "input:video_reader:type=vidl_ffmpeg" \
  -s "downsampler:target_frame_rate=$VIAME_DOWNSAMPLE_FPS" \
  -s "downsampler:renumber_frames=false" \
  -s "detector:detector:onnx:score_thresh=$VIAME_DETECTOR_SCORE_THRESHOLD" \
  -s "tracker:track_objects:bytetrack:high_thresh=$VIAME_TRACKER_HIGH_THRESHOLD" \
  -s "tracker:track_objects:bytetrack:low_thresh=$VIAME_TRACKER_LOW_THRESHOLD" \
  -s "tracker:track_objects:bytetrack:new_track_thresh=$VIAME_TRACKER_NEW_TRACK_THRESHOLD" \
  -s "tracker:track_objects:bytetrack:track_buffer=$VIAME_TRACKER_BUFFER_FRAMES" \
  -s "track_writer:file_name=$OUTPUT_CSV"
