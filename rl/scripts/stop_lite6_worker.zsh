#!/usr/bin/env zsh
set -euo pipefail
PIDFILE=/tmp/lite6_worker.pid
if [[ ! -f $PIDFILE ]]; then
  echo "No pidfile at $PIDFILE"; exit 0
fi
PID=$(cat $PIDFILE)
if kill -0 $PID 2>/dev/null; then
  echo "Stopping worker PID=$PID"
  kill $PID
else
  echo "Worker not running (PID=$PID)"
fi
rm -f $PIDFILE
