#!/usr/bin/env zsh
set -euo pipefail

# DreamerV4 worker: loads USD robot with gripper (fixed geometry).

HOST=${1:-127.0.0.1}
PORT=${2:-5555}
LOG=${3:-/tmp/lite6_worker_v4.log}

# If already listening, do nothing.
if ss -ltn 2>/dev/null | grep -q ":${PORT}"; then
  echo "Lite6 Isaac worker V4 already listening on ${HOST}:${PORT} (not starting a new one)"
  exit 0
fi

echo "Starting Lite6 Isaac worker V4 on ${HOST}:${PORT}"

# Isaac python launcher
ISAAC_PY=${ISAAC_PY:-"$HOME/isaacsim/isaac-sim-4.2.0/python.sh"}
WORKER=${WORKER:-"$HOME/ws_xarm/isaac_bridge/scripts/lite6_reach_worker_v4.py"}

nohup "$ISAAC_PY" "$WORKER" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 &
PID=$!

echo $PID > /tmp/lite6_worker_v4_${PORT}.pid

echo "PID=${PID}"
echo "log=${LOG}"

# wait for listen (Isaac startup can be slow)
for i in {1..240}; do
  if ss -ltn 2>/dev/null | grep -q ":${PORT}"; then
    echo "LISTEN ${HOST}:${PORT}"
    exit 0
  fi
  sleep 1
  if ! kill -0 $PID 2>/dev/null; then
    echo "WORKER_V4_EXITED"
    exit 1
  fi
  if (( i % 20 == 0 )); then
    echo "WAITING_FOR_LISTEN..."
  fi
done

echo "TIMEOUT_WAITING_FOR_LISTEN"
exit 1
