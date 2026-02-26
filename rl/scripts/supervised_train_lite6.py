#!/usr/bin/env python3
"""Chunked DreamerV3 training with safe auto-patches.

Design goals:
- Run training in chunks (default 2000 steps).
- After each chunk, read latest metrics/videos and apply a small allowlisted patch.
- Restart training from latest checkpoint automatically.

This is a *rule-based* supervisor ("Level 1") with conservative updates.
You can later replace propose_patch() with a VLM/LLM call.
"""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


@dataclass
class Bounds:
    lo: float
    hi: float


def clamp(x: float, b: Bounds) -> float:
    return max(b.lo, min(b.hi, x))


def read_last_jsonl(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        # Read from end (small files anyway)
        lines = path.read_text().strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    except Exception:
        return None


def find_latest_checkpoint(logdir: Path) -> Optional[Path]:
    cp = logdir / 'checkpoint.pkl'
    if not cp.exists():
        return None
    # checkpoint.pkl/<timestamp>/agent.pkl
    candidates = sorted(cp.glob('*/agent.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent if candidates else None


def find_latest_clip(logdir: Path) -> Optional[Path]:
    eps = logdir / 'episodes'
    if not eps.exists():
        return None
    clips = sorted(eps.glob('clip_*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    return clips[0] if clips else None


def parse_config_yaml(logdir: Path) -> Dict[str, Any]:
    # Minimal parsing: the config.yaml is a YAML, but we only need env.lite6 keys.
    # We'll use ruamel if available, otherwise do a tiny regex-based fallback.
    cfgp = logdir / 'config.yaml'
    if not cfgp.exists():
        return {}
    try:
        from ruamel.yaml import YAML
        y = YAML(typ='safe')
        return y.load(cfgp.read_text()) or {}
    except Exception:
        return {}


def propose_patch(state: Dict[str, Any]) -> Tuple[Dict[str, float], str, float]:
    """Return (patch, why, confidence).

    Level-1 conservative rules:
    - If success is rare AND distance not improving, shrink target_radius a bit.
    - If policy looks jerky (high vis_dist changes not available yet), increase reward_w_du.

    NOTE: We don't have jerk stats yet. We use proxy: if episode score is noisy and doesn't improve,
    increase w_du slightly (bounded).
    """

    env = state.get('env', {})
    # Defaults if missing
    w_du = float(env.get('reward_w_du', 0.0) or 0.0)
    w_u = float(env.get('reward_w_u', 0.0) or 0.0)
    rad = float(env.get('target_radius', 0.0) or 0.0)

    last_score = state.get('last_episode_score')
    prev_score = state.get('prev_episode_score')

    # Bounds
    b_wdu = Bounds(0.0, 0.2)
    b_wu = Bounds(0.0, 0.05)
    b_rad = Bounds(0.02, 0.25)

    patch: Dict[str, float] = {}

    # Heuristic 1: if not improving, add a bit more smoothing.
    if last_score is not None and prev_score is not None:
        # If score got worse by a margin, increase smoothness penalty (small step).
        if last_score < prev_score - 2.0:
            patch['reward_w_du'] = clamp(w_du * 1.2 + 0.005, b_wdu)

    # Heuristic 2: if it's still very rough, also add tiny action magnitude penalty.
    if 'reward_w_du' in patch and w_u == 0.0:
        patch['reward_w_u'] = clamp(0.001, b_wu)

    # Heuristic 3: if we have no curriculum radius set, set a reasonable starting radius.
    if rad <= 0.0:
        patch['target_radius'] = clamp(0.08, b_rad)

    if not patch:
        return {}, 'no safe change triggered', 0.0

    why = f"auto-patch: {patch}"
    confidence = 0.7
    return patch, why, confidence


def apply_patch_to_config(logdir: Path, patch: Dict[str, float]):
    cfgp = logdir / 'config.yaml'
    if not cfgp.exists():
        raise FileNotFoundError(cfgp)
    from ruamel.yaml import YAML
    y = YAML()
    cfg = y.load(cfgp.read_text())
    cfg.setdefault('env', {})
    cfg['env'].setdefault('lite6', {})
    for k, v in patch.items():
        cfg['env']['lite6'][k] = float(v)
    tmp = cfgp.with_suffix('.yaml.tmp')
    with tmp.open('w') as f:
        y.dump(cfg, f)
    tmp.replace(cfgp)


def run_train(venv_activate: str, logdir: Path, steps: int, checkpoint_dir: Optional[Path]):
    # Ensure we run from the rl project directory so dreamerv3_lite6_main.py is found.
    proj = Path.home() / 'ws_xarm/rl'
    cmd = [
        'zsh', '-lc',
        f"cd {proj} && source {venv_activate} && python dreamerv3_lite6_main.py --task lite6_reach --logdir {logdir} --run.steps {steps} --run.envs 1" + (
            f" --run.from_checkpoint {checkpoint_dir}" if checkpoint_dir else ""
        ) + " --env.lite6.host 127.0.0.1 --env.lite6.port 5555 --env.lite6.video_every 500 --env.lite6.download_dir ~/Downloads --env.lite6.download_prefix 'robotarm training video (left+visgate)'"
    ]
    subprocess.run(cmd, check=False)


def main():
    root = Path(os.environ.get('LITE6_LOGDIR', str(Path.home() / 'ws_xarm/rl/logdir')))
    chunk = int(os.environ.get('LITE6_CHUNK', '2000'))
    venv_activate = os.environ.get('LITE6_VENV', str(Path.home() / 'ws_xarm/rl/.venv/bin/activate'))

    # Ensure worker is up
    subprocess.run(['zsh', '-lc', '~/ws_xarm/rl/scripts/start_lite6_worker.zsh'], check=False)

    metrics = root / 'metrics.jsonl'
    scores = root / 'scores.jsonl'

    prev_score = None

    while True:
        # Determine current step from metrics.
        lastm = read_last_jsonl(metrics) or {}
        cur_step = int(lastm.get('step', 0) or 0)
        target_steps = max(cur_step + chunk, chunk)

        cpdir = find_latest_checkpoint(root)
        run_train(venv_activate, root, target_steps, cpdir)

        # After run exits, update state.
        last_score_entry = read_last_jsonl(scores) or {}
        last_score = last_score_entry.get('episode/score') or last_score_entry.get('episode/score'.replace('/', '/')) or last_score_entry.get('episode/score')
        if last_score is None:
            last_score = last_score_entry.get('episode/score')
        if last_score is None:
            last_score = last_score_entry.get('episode/score')
        if last_score is None:
            last_score = last_score_entry.get('episode/score')

        cfg = parse_config_yaml(root)
        env = (cfg or {}).get('env', {}).get('lite6', {})

        state = {
            'env': env,
            'last_episode_score': float(last_score) if last_score is not None else None,
            'prev_episode_score': float(prev_score) if prev_score is not None else None,
            'latest_clip': str(find_latest_clip(root) or ''),
        }

        patch, why, conf = propose_patch(state)

        # Level-1: require confidence and at least one metric present.
        if patch and conf >= 0.6:
            apply_patch_to_config(root, patch)
            # Save audit record
            audit = root / 'supervisor_patches.jsonl'
            rec = {
                'time': time.time(),
                'step': target_steps,
                'patch': patch,
                'why': why,
                'confidence': conf,
                'state': state,
            }
            with audit.open('a') as f:
                f.write(json.dumps(rec) + '\n')

        prev_score = state['last_episode_score']

        # Small pause to avoid tight loop if training exits immediately.
        time.sleep(2)


if __name__ == '__main__':
    main()
