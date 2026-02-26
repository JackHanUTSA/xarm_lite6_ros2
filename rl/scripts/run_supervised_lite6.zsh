#!/usr/bin/env zsh
set -euo pipefail

# Supervisor settings
export LITE6_LOGDIR=${LITE6_LOGDIR:-"$HOME/ws_xarm/rl/logdir"}
export LITE6_CHUNK=${LITE6_CHUNK:-2000}
export LITE6_VENV=${LITE6_VENV:-"$HOME/ws_xarm/rl/.venv/bin/activate"}

# Start worker (idempotent)
$HOME/ws_xarm/rl/scripts/start_lite6_worker.zsh

# Activate venv so ruamel.yaml and dreamerv3 deps are available
source "$LITE6_VENV"

# Run supervisor
python3 $HOME/ws_xarm/rl/scripts/supervised_train_lite6.py
