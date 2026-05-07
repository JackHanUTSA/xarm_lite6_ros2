#!/usr/bin/env python3
"""Export Lite6 dataset_v4 episodes (png frames) into Dreamer4 sharded format.

Input (from Lite6RPCEnvV4 recording):
  logdir_v4/dataset_v4/episode_XXXXXXXXX/frames/frame_000123.png

Output (Dreamer4 sharded format expected by third_party/dreamer_v4_pytorch):
  <out_root>/<task>/<task>_shard0000.pt with {"frames": uint8 (N,3,128,128)}

Notes:
- This exporter concatenates frames across episodes in sorted order.
- It resizes frames to 128x128.
- It ignores actions for now (Dreamer4 pytorch code can train tokenizer/dynamics without actions).

Usage:
  python scripts/export_lite6_dataset_v4_to_dreamer4_shards.py \
    --inp ~/ws_xarm/rl/logdir_v4/dataset_v4 \
    --out ~/ws_xarm/rl/dreamer4_shards \
    --task lite6_reach \
    --shard_size 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image


def iter_frame_paths(dataset_root: Path):
    # dataset_root/episode_*/frames/frame_*.png
    for ep in sorted(dataset_root.glob('episode_*')):
        frames_dir = ep / 'frames'
        if not frames_dir.is_dir():
            continue
        for fp in sorted(frames_dir.glob('frame_*.png')):
            yield fp


def safe_save(out_path: Path, frames_u8: torch.Tensor):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    torch.save({'frames': frames_u8.contiguous()}, tmp)
    tmp.replace(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', type=str, required=True, help='path to dataset_v4 directory')
    ap.add_argument('--out', type=str, required=True, help='output root for shards')
    ap.add_argument('--task', type=str, default='lite6_reach')
    ap.add_argument('--H', type=int, default=128)
    ap.add_argument('--W', type=int, default=128)
    ap.add_argument('--shard_size', type=int, default=2048)
    ap.add_argument('--max_frames', type=int, default=0, help='0 = no limit')
    args = ap.parse_args()

    inp = Path(args.inp).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    task = args.task

    assert inp.is_dir(), f'input not found: {inp}'

    out_task = out_root / task
    out_task.mkdir(parents=True, exist_ok=True)

    buf = []
    shard_idx = 0
    total = 0

    def flush(frames_list):
        nonlocal shard_idx
        frames = torch.cat(frames_list, dim=0)  # (N,3,H,W) uint8
        while frames.shape[0] >= args.shard_size:
            to_save = frames[: args.shard_size]
            frames = frames[args.shard_size :]
            out_path = out_task / f'{task}_shard{shard_idx:04d}.pt'
            safe_save(out_path, to_save)
            print(f'[OK] saved {out_path} frames={to_save.shape[0]}')
            shard_idx += 1
        return [frames] if frames.numel() else []

    for fp in iter_frame_paths(inp):
        img = read_image(str(fp))  # (C,H,W) uint8
        if img.ndim != 3 or img.shape[0] != 3:
            continue

        # resize once
        x = img.to(torch.float32).unsqueeze(0) / 255.0
        x = F.interpolate(x, size=(args.H, args.W), mode='bilinear', align_corners=False)
        u8 = (x.clamp(0, 1) * 255.0).to(torch.uint8).squeeze(0)  # (3,H,W)
        buf.append(u8.unsqueeze(0))

        total += 1
        if args.max_frames and total >= args.max_frames:
            break

        if sum(t.shape[0] for t in buf) >= args.shard_size:
            buf = flush(buf)

        if total % 500 == 0:
            print(f'...processed {total} frames')

    # final flush
    if buf:
        frames = torch.cat(buf, dim=0)
        out_path = out_task / f'{task}_shard{shard_idx:04d}.pt'
        safe_save(out_path, frames)
        print(f'[OK] saved final {out_path} frames={frames.shape[0]}')

    print(f'Done. total_frames={total} shards_written={shard_idx + (1 if buf else 0)} out={out_root}')


if __name__ == '__main__':
    main()
