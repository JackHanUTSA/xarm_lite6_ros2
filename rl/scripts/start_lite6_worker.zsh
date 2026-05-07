#!/usr/bin/env zsh
set -euo pipefail
HOST=${1:-127.0.0.1}
PORT=${2:-5555}
LOG=${3:-/tmp/lite6_worker.log}
ISAAC_PY="$HOME/isaacsim/isaac-sim-4.2.0/python.sh"
SCRIPT="$HOME/ws_xarm/isaac_bridge/scripts/lite6_reach_worker.py"

# GUI toggle: set LITE6_GUI=1 to show Isaac Sim window (requires display/VNC).
# Example: LITE6_GUI=1 ./scripts/start_lite6_worker.zsh

# Idempotent: if something is already listening, don't start another worker.
if ss -ltn 2>/dev/null | grep -q "${HOST}:${PORT}"; then
  echo "Lite6 Isaac worker already listening on ${HOST}:${PORT} (not starting a new one)"
  exit 0
fi

echo "Starting Lite6 Isaac worker on ${HOST}:${PORT}"
nohup "$ISAAC_PY" "$SCRIPT" --host "$HOST" --port "$PORT" >"$LOG" 2>&1 &
PID=$!
echo $PID > "/tmp/lite6_worker_${PORT}.pid"
echo 'WAITING_FOR_LISTEN...'
# Wait up to 120s for the TCP port to start listening.
for i in {1..240}; do
  if ss -ltn 2>/dev/null | grep -q "${HOST}:${PORT}"; then
    break
  fi
  sleep 0.5
done

echo "PID=$PID"
echo "log=$LOG"
