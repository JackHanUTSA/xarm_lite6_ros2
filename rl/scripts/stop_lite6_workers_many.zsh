#!/usr/bin/env zsh
set -euo pipefail

pids=($(pgrep -f "isaac_bridge/scripts/lite6_reach_worker.py" || true))
if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No Lite6 workers running"
  exit 0
fi

echo "Stopping ${#pids[@]} Lite6 workers: ${pids[@]}"
for pid in $pids; do
  kill "$pid" || true
done
