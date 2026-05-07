"""DreamerV4 Lite6 entrypoint.

This mirrors `dreamerv3_lite6_main.py` but calls DreamerV4.

Once you vendor DreamerV4 under third_party/embodied/dreamerv4,
this script should run training the same way as V3.

Example:
  source rl/.venv/bin/activate
  python rl/dreamerv4_lite6_main.py --task lite6_reach --run.steps 2000 --run.envs 1
"""

import importlib
import os
import sys

# Ensure local third_party is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.join(ROOT, 'third_party')
if THIRD_PARTY not in sys.path:
    sys.path.insert(0, THIRD_PARTY)

# Import Lite6 env registration (same pattern as V3)
# This file expects your existing lite6 env modules are importable.
import envs.lite6_reach_env  # noqa: F401
import envs.embodied_lite6_reach  # noqa: F401


def main(argv=None):
    dv4 = importlib.import_module('embodied.dreamerv4.main')
    return dv4.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
