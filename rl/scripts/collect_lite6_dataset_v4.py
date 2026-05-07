"""Collect Lite6 rollouts into DreamerV4 (Dreamer4) dataset format (raw).

This runs a simple random policy to generate episodes and relies on the Isaac
worker to write frames+steps.jsonl into logdir_v4/dataset_v4/.

Usage:
  source .venv/bin/activate
  python scripts/collect_lite6_dataset_v4.py --episodes 50
"""

import argparse
import numpy as np

from lite6_rpc_env_v4 import Lite6RPCEnvV4


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=5)
  ap.add_argument('--max_steps', type=int, default=200)
  ap.add_argument('--logdir', type=str, default='')
  ap.add_argument('--host', type=str, default='127.0.0.1')
  ap.add_argument('--port', type=int, default=5555)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--task-variant', type=str, default='reach')
  ap.add_argument('--ball-radius', type=float, default=0.03)
  ap.add_argument('--grasp-dist-thresh', type=float, default=0.04)
  ap.add_argument('--lift-height-thresh', type=float, default=0.08)
  args = ap.parse_args()

  rng = np.random.RandomState(args.seed)
  env = Lite6RPCEnvV4(
    task='lite6_reach',
    host=args.host,
    port=args.port,
    logdir=args.logdir,
    record_v4_enabled=True,
    record_v4_dir='dataset_v4',
    record_v4_every=1,
    video_every=0,
    task_variant=args.task_variant,
    ball_radius=args.ball_radius,
    grasp_dist_thresh=args.grasp_dist_thresh,
    lift_height_thresh=args.lift_height_thresh,
  )

  try:
    for ep in range(args.episodes):
      obs = env.reset()
      ret = 0.0
      for t in range(args.max_steps):
        act = rng.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)
        obs = env.step({'action': act, 'reset': False})
        ret += float(obs['reward'])
        if obs['is_last']:
          break
      print(f'EP {ep+1}/{args.episodes} steps={t+1} return={ret:.3f}')
  finally:
    env.close()


if __name__ == '__main__':
  main()
