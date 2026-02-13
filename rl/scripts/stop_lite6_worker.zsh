#!/usr/bin/env zsh
set -euo pipefail
PORT=${1:-}

kill_pidfile() {
  local pf="$1"
  if [[ -f "$pf" ]]; then
    local pid=$(cat "$pf" || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping worker PID=$pid (${pf})"
      kill "$pid" || true
    fi
    rm -f "$pf" || true
  fi
}

if [[ -n "$PORT" ]]; then
  kill_pidfile "/tmp/lite6_worker_${PORT}.pid"
  exit 0
fi

# Try stop all known pidfiles
for pf in /tmp/lite6_worker_*.pid; do
  [[ -e "$pf" ]] || break
  kill_pidfile "$pf"
done
