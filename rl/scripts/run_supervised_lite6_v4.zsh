#!/usr/bin/env zsh
set -euo pipefail

# Supervisor settings (DreamerV4)
export LITE6_LOGDIR=${LITE6_LOGDIR:-"$HOME/ws_xarm/rl/logdir_v4"}
export LITE6_CHUNK=${LITE6_CHUNK:-2000}
export LITE6_VENV=${LITE6_VENV:-"$HOME/ws_xarm/rl/.venv/bin/activate"}

# Start worker (idempotent)
$HOME/ws_xarm/rl/scripts/start_lite6_worker.zsh

source "$LITE6_VENV"

# NOTE: this will fail until DreamerV4 code is vendored.
python3 $HOME/ws_xarm/rl/scripts/supervised_train_lite6_v4.py
