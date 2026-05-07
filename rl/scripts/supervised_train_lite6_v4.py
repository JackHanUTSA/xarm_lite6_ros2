"""Supervisor for Lite6 DreamerV4 training.

This mirrors scripts/supervised_train_lite6.py but calls dreamerv4_lite6_main.py.
"""

import os
import subprocess
import sys
from datetime import datetime


def main():
    logdir = os.environ.get('LITE6_LOGDIR', os.path.expanduser('~/ws_xarm/rl/logdir_v4'))

    # Keep args consistent with current V3 supervisor defaults.
    cmd = [
        sys.executable,
        os.path.expanduser('~/ws_xarm/rl/dreamerv4_lite6_main.py'),
        '--task', 'lite6_reach',
        '--logdir', logdir,
        '--run.steps', os.environ.get('LITE6_RUN_STEPS', '6830'),
        '--run.envs', os.environ.get('LITE6_RUN_ENVS', '1'),
        '--env.lite6.host', os.environ.get('LITE6_HOST', '127.0.0.1'),
        '--env.lite6.port', os.environ.get('LITE6_PORT', '5555'),
        '--env.lite6.video_every', os.environ.get('LITE6_VIDEO_EVERY', '500'),
        '--env.lite6.download_dir', os.environ.get('LITE6_DOWNLOAD_DIR', os.path.expanduser('~/Downloads')),
        '--env.lite6.download_prefix', os.environ.get('LITE6_DOWNLOAD_PREFIX', 'robotarm training video (left+visgate)'),
    ]

    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] launching: {' '.join(cmd)}', flush=True)

    # Stream stdout/stderr
    p = subprocess.Popen(cmd)
    return p.wait()


if __name__ == '__main__':
    raise SystemExit(main())
