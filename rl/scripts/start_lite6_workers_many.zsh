#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <host> <port_base> <num_workers> [log_dir]"
  exit 2
fi

HOST="$1"
PORT_BASE="$2"
N="$3"
LOGDIR="${4:-/tmp/lite6_workers}"
mkdir -p "$LOGDIR"

for i in $(seq 0 $((N-1))); do
  port=$((PORT_BASE + i))
  log="$LOGDIR/worker_${port}.log"
  echo "Starting worker $i on ${HOST}:${port} log=${log}"
  # re-use existing single-worker start script but allow port override
  ./rl/scripts/start_lite6_worker.zsh "$HOST" "$port" "$log"
  sleep 0.2
done
