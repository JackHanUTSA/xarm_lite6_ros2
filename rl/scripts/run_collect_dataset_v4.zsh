#!/usr/bin/env zsh
set -euo pipefail

# DreamerV4 (Dreamer4) dataset collection run
export LITE6_LOGDIR=${LITE6_LOGDIR:-"$HOME/ws_xarm/rl/logdir_v4"}
export LITE6_VENV=${LITE6_VENV:-"$HOME/ws_xarm/rl/.venv/bin/activate"}

# Start worker (idempotent)
$HOME/ws_xarm/rl/scripts/start_lite6_worker_v4.zsh

# Wait until the port is actually listening (the worker script can return before ss sees it)
for i in {1..240}; do
  if ss -ltn 2>/dev/null | grep -q ":5555"; then
    break
  fi
  sleep 1
  if (( i % 20 == 0 )); then
    echo "WAITING_FOR_WORKER_LISTEN..."
  fi
done

source "$LITE6_VENV"

cd $HOME/ws_xarm/rl
export PYTHONPATH="$HOME/ws_xarm/rl:${PYTHONPATH:-}"
python scripts/collect_lite6_dataset_v4.py \
  --episodes ${EPISODES:-5} \
  --max_steps ${MAX_STEPS:-200} \
  --logdir "$LITE6_LOGDIR"
