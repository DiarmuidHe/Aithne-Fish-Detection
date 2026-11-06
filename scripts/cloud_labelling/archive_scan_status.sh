#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${AITHNE_ARCHIVE_OUTPUT:-$REPO_ROOT/data/archive_scan}"

nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if [[ -f "$OUTPUT/SmartBayCam1_2025/year_summary.json" ]]; then
  python3 -m json.tool "$OUTPUT/SmartBayCam1_2025/year_summary.json"
else
  echo "No Cam 1 2025 summary at $OUTPUT yet."
fi
echo "Disk:"
df -h "$REPO_ROOT" /opt/dlami/nvme
echo "tmux:"
tmux list-sessions 2>/dev/null || true
