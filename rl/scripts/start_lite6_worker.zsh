#!/usr/bin/env zsh
set -euo pipefail
HOST=${1:-127.0.0.1}
PORT=${2:-5555}
LOG=${3:-/tmp/lite6_worker.log}
ISAAC_PY="$HOME/isaacsim/isaac-sim-4.2.0/python.sh"
SCRIPT="$HOME/ws_xarm/isaac_bridge/scripts/lite6_reach_worker.py"

echo "Starting Lite6 Isaac worker on ${HOST}:${PORT}"
nohup "$ISAAC_PY" "$SCRIPT" >"$LOG" 2>&1 &
PID=$!
echo $PID > /tmp/lite6_worker.pid
sleep 1

echo "PID=$PID"
echo "log=$LOG"
