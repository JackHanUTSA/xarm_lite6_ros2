#!/usr/bin/env python3
"""Build Dreamer4 WMDataset demo .pt from Lite6 dataset_v4 recordings.

Creates <out_dir>/<task>.pt containing:
  - episode: int64 (N,)
  - action:  float32 (N, action_dim)  (pads extra dims with 0)
  - reward:  float32 (N,)

Ordering matches the frame shard exporter: episodes sorted by name, frames sorted
by frame index within each episode.

Notes:
- steps.jsonl includes a first step with action=None (t=0). We encode that as NaN
  so WMDataset will treat it as invalid transition.
- This is enough to run train_dynamics.py with --use_actions (action-conditioned)
  using WMDataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', type=str, required=True, help='path to dataset_v4 root')
    ap.add_argument('--out_dir', type=str, required=True, help='output dir for <task>.pt')
    ap.add_argument('--task', type=str, default='lite6_reach')
    ap.add_argument('--action_dim', type=int, default=16)
    args = ap.parse_args()

    inp = Path(args.inp).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    actions = []
    rewards = []

    ep_id = 0
    total_steps = 0

    for ep in sorted(inp.glob('episode_*')):
        steps_path = ep / 'steps.jsonl'
        frames_dir = ep / 'frames'
        if not steps_path.exists() or not frames_dir.is_dir():
            continue

        # Build a dict record_step -> (action,reward)
        step_map = {}
        with steps_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rs = int(obj.get('record_step', obj.get('t', 0)))
                act = obj.get('action', None)
                rew = obj.get('reward', 0.0)
                if rew is None:
                    rew = float('nan')
                step_map[rs] = (act, float(rew))

        # For each frame, append aligned step info
        frame_paths = sorted(frames_dir.glob('frame_*.png'))
        for fp in frame_paths:
            # parse frame_000123.png
            idx = int(fp.stem.split('_')[-1])
            act, rew = step_map.get(idx, (None, math.nan))

            a = np.zeros((args.action_dim,), dtype=np.float32)
            if act is None:
                a[:] = np.nan
            else:
                act = np.asarray(act, dtype=np.float32)
                a[: min(args.action_dim, act.shape[0])] = act[: min(args.action_dim, act.shape[0])]

            episodes.append(ep_id)
            actions.append(a)
            rewards.append(rew)
            total_steps += 1

        ep_id += 1

    if total_steps == 0:
        raise SystemExit(f'No episodes/frames found under {inp}')

    ep_t = torch.tensor(episodes, dtype=torch.int64)
    act_t = torch.tensor(np.stack(actions, axis=0), dtype=torch.float32)
    rew_t = torch.tensor(rewards, dtype=torch.float32)

    out_path = out_dir / f'{args.task}.pt'
    torch.save({'episode': ep_t, 'action': act_t, 'reward': rew_t}, out_path)

    print(f'[OK] wrote {out_path}')
    print(f'  N={total_steps} episodes={ep_id} action_dim={args.action_dim}')
    print(f'  action_nan_steps={(~torch.isfinite(act_t)).any(dim=-1).sum().item()}')


if __name__ == '__main__':
    main()
