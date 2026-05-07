#!/usr/bin/env python3
"""Chunked DreamerV3 training with safe auto-patches.

Design goals:
- Run training in chunks (default 2000 steps).
- After each chunk, read latest metrics/videos and apply a small allowlisted patch.
- Restart training from latest checkpoint automatically.

This is a *rule-based* supervisor ("Level 1") with conservative updates.
You can later replace propose_patch() with a VLM/LLM call.
"""

import base64
import json
import os
import re
import subprocess
import time
import urllib.request
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
    # checkpoint.pkl/<timestamp>/agent.pkl  (Dreamer expects a *file* for --run.from_checkpoint)
    candidates = sorted(cp.glob('*/agent.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_clip(logdir: Path) -> Optional[Path]:
    eps = logdir / 'episodes'
    if not eps.exists():
        return None
    clips = sorted(eps.glob('clip_*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    return clips[0] if clips else None


def find_latest_episode_video(logdir: Path) -> Optional[Path]:
    eps = logdir / 'episodes'
    if not eps.exists():
        return None
    vids = sorted(eps.glob('ep_*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    return vids[0] if vids else None


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


def _read_fold_guard_triggers(path: Path, max_lines: int = 2000) -> int:
    """Count recent fold-guard triggers from worker log."""
    try:
        if not path.exists():
            return 0
        lines = path.read_text(errors='ignore').splitlines()
        lines = lines[-max_lines:]
        return sum(1 for ln in lines if 'FOLD_GUARD_TRIGGER' in ln)
    except Exception:
        return 0


def _read_joint_telemetry(path: Path, max_lines: int = 300) -> Dict[str, Any]:
    """Summarize recent joint telemetry for the LLM.

    Returns summary dict with per-joint min/max/mean/std for q and |dq|.
    """
    if not path.exists():
        return {'available': False}
    rows = []
    try:
        lines = path.read_text(errors='ignore').splitlines()[-max_lines:]
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    except Exception:
        return {'available': False}

    if not rows:
        return {'available': False}

    qs = [r.get('q') for r in rows if isinstance(r.get('q'), list) and len(r.get('q')) == 6]
    if not qs:
        return {'available': False}

    import numpy as _np
    q = _np.array(qs, dtype=float)
    dq = _np.diff(q, axis=0)
    if dq.size == 0:
        dq_abs = _np.zeros((1, 6))
    else:
        dq_abs = _np.abs(dq)

    summary = {
        'available': True,
        'n': int(q.shape[0]),
        'q_min': q.min(axis=0).round(4).tolist(),
        'q_max': q.max(axis=0).round(4).tolist(),
        'q_mean': q.mean(axis=0).round(4).tolist(),
        'q_std': q.std(axis=0).round(4).tolist(),
        'dq_abs_max': dq_abs.max(axis=0).round(4).tolist(),
        'dq_abs_mean': dq_abs.mean(axis=0).round(4).tolist(),
    }
    return summary


def _ollama_chat(model: str, prompt: str, timeout_s: int = 60) -> str:
    """Call local Ollama (text only) and return stdout text."""
    cmd = ['ollama', 'run', model, prompt]
    out = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    text = (out.stdout or b'').decode('utf-8', errors='ignore')
    if not text.strip():
        # include stderr for debugging in logs
        err = (out.stderr or b'').decode('utf-8', errors='ignore')
        return err.strip()
    return text.strip()


def _ollama_generate_vlm(model: str, prompt: str, images_b64: list[str], timeout_s: int = 120) -> str:
    """Call Ollama multimodal generate endpoint with base64 images."""
    payload = {
        'model': model,
        'prompt': prompt,
        'images': images_b64,
        'stream': False,
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:11434/api/generate', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode('utf-8', errors='ignore')
    obj = json.loads(body)
    return (obj.get('response') or '').strip()


def _ffprobe_duration_s(video: Path) -> Optional[float]:
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(video)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if out.returncode != 0:
            return None
        s = out.stdout.decode().strip()
        return float(s) if s else None
    except Exception:
        return None


def _extract_frames(video: Path, outdir: Path, n: int = 8) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    dur = _ffprobe_duration_s(video)
    frames: list[Path] = []

    if dur is None or dur <= 0:
        # Fallback: sample first 8 seconds.
        times = [float(i) for i in range(n)]
    else:
        # Evenly spaced, avoid first/last 5%.
        lo = 0.05 * dur
        hi = 0.95 * dur
        if hi <= lo:
            lo, hi = 0.0, dur
        times = [lo + (hi - lo) * (i / max(n - 1, 1)) for i in range(n)]

    for i, t in enumerate(times):
        outp = outdir / f'frame_{i:02d}.jpg'
        subprocess.run(
            ['ffmpeg', '-y', '-ss', f'{t:.3f}', '-i', str(video), '-frames:v', '1', '-q:v', '2', str(outp)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if outp.exists():
            frames.append(outp)

    return frames


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM output."""
    try:
        # Try direct
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find a JSON block
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def review_episode_video_for_patch(logdir: Path, state: Dict[str, Any]) -> Tuple[Dict[str, float], str, float]:
    """Use a local VLM to review the most recent ep_*.mp4 and propose a safe patch.

    Returns (patch, why, confidence). Empty patch means no change.

    Triggering is controlled by the caller (e.g., every N episodes).
    """

    video = find_latest_episode_video(logdir)
    if not video or not video.exists():
        return {}, 'no episode video found', 0.0

    model = os.environ.get('LITE6_VLM_MODEL', 'qwen3-vl:32b')
    nframes = int(os.environ.get('LITE6_VLM_FRAMES', '10'))
    timeout = int(os.environ.get('LITE6_VLM_TIMEOUT', '180'))

    tmp = logdir / '.vlm_frames'
    frames = _extract_frames(video, tmp, n=nframes)
    if not frames:
        return {}, 'failed to extract frames', 0.0

    images_b64: list[str] = []
    for fp in frames:
        try:
            images_b64.append(base64.b64encode(fp.read_bytes()).decode('ascii'))
        except Exception:
            continue

    env = state.get('env', {})

    prompt = (
        "You are a robotic arm RL training supervisor. Review the frames from a single episode video. "
        "Goal: smooth motion, avoid self-collision/folding, while still reaching the target.\n\n"
        "Return ONLY valid JSON with schema:\n"
        "{\n"
        "  \"observations\": {\"folding\": boolean, \"jerky\": boolean, \"progress\": boolean, \"notes\": string},\n"
        "  \"patch\": {\"action_scale\"?: number, \"reward_w_du\"?: number, \"reward_w_u\"?: number, \"target_radius\"?: number},\n"
        "  \"why\": string,\n"
        "  \"confidence\": number\n"
        "}\n\n"
        f"Context: last_score={state.get('last_episode_score')}, fold_triggers_recent={state.get('fold_triggers_recent')}\n"
        f"Joint telemetry summary: {state.get('joint_telemetry')}\n"
        f"Current config: action_scale={env.get('action_scale')}, reward_w_du={env.get('reward_w_du')}, reward_w_u={env.get('reward_w_u')}, target_radius={env.get('target_radius')}\n"
        "Guidelines: If folding/self-collision is visible, prefer lowering action_scale and/or increasing reward_w_du. "
        "If motion is jerky, increase reward_w_du slightly. If motion is too timid and progress is false, slightly increase action_scale."
    )

    try:
        raw = _ollama_generate_vlm(model, prompt, images_b64, timeout_s=timeout)
    except Exception as e:
        return {}, f'vlm call failed: {e}', 0.0

    obj = _parse_llm_json(raw) or {}
    patch_in = obj.get('patch') if isinstance(obj, dict) else None
    why = (obj.get('why') if isinstance(obj, dict) else None) or 'vlm'
    conf = float(obj.get('confidence') if isinstance(obj, dict) else 0.0)

    if not isinstance(patch_in, dict) or not patch_in:
        return {}, f'no patch (vlm output): {raw[:200]}', 0.0

    # Validate with same allowlist/bounds as text LLM patcher.
    b_action = Bounds(0.01, 0.08)
    b_wdu = Bounds(0.0, 0.2)
    b_wu = Bounds(0.0, 0.05)
    b_rad = Bounds(0.02, 0.25)

    action_scale = float(env.get('action_scale', 0.06) or 0.06)
    max_action_mult = 1.25

    patch: Dict[str, float] = {}
    if 'action_scale' in patch_in:
        v = clamp(float(patch_in['action_scale']), b_action)
        v = clamp(v, Bounds(action_scale / max_action_mult, action_scale * max_action_mult))
        patch['action_scale'] = v
    if 'reward_w_du' in patch_in:
        patch['reward_w_du'] = clamp(float(patch_in['reward_w_du']), b_wdu)
    if 'reward_w_u' in patch_in:
        patch['reward_w_u'] = clamp(float(patch_in['reward_w_u']), b_wu)
    if 'target_radius' in patch_in:
        patch['target_radius'] = clamp(float(patch_in['target_radius']), b_rad)

    if not patch:
        return {}, 'vlm patch rejected by allowlist/bounds', 0.0

    # Persist review record
    review_log = logdir / 'supervisor_video_reviews.jsonl'
    rec = {
        'time': time.time(),
        'video': str(video),
        'frames': [str(p) for p in frames],
        'vlm_model': model,
        'why': why,
        'confidence': conf,
        'patch': patch,
        'state': state,
    }
    try:
        with review_log.open('a') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception:
        pass

    return patch, why, conf


def propose_patch(state: Dict[str, Any]) -> Tuple[Dict[str, float], str, float]:
    """Return (patch, why, confidence).

    LLM-in-the-loop supervisor (local Ollama) with strict allowlist + bounds.

    Objective: optimize motion (smoother / less folding) while keeping score improving.
    """

    env = state.get('env', {})

    # Current values
    action_scale = float(env.get('action_scale', 0.06) or 0.06)
    w_du = float(env.get('reward_w_du', 0.0) or 0.0)
    w_u = float(env.get('reward_w_u', 0.0) or 0.0)
    rad = float(env.get('target_radius', 0.0) or 0.0)

    # Bounds (hard)
    b_action = Bounds(0.01, 0.08)
    b_wdu = Bounds(0.0, 0.2)
    b_wu = Bounds(0.0, 0.05)
    b_rad = Bounds(0.02, 0.25)

    # Change limits (per patch)
    max_action_mult = 1.25

    model = os.environ.get('LITE6_LLM_MODEL', 'qwen3:latest')

    prompt = (
        "You are a cautious RL training supervisor. Propose at most ONE small config patch to improve motion smoothness "
        "and reduce fold/self-collision events, while maintaining or improving episode score.\n\n"
        "You must output ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"patch\": {\"action_scale\"?: number, \"reward_w_du\"?: number, \"reward_w_u\"?: number, \"target_radius\"?: number},\n"
        "  \"why\": string,\n"
        "  \"confidence\": number\n"
        "}\n\n"
        f"Current config: action_scale={action_scale}, reward_w_du={w_du}, reward_w_u={w_u}, target_radius={rad}\n"
        f"Recent scores: prev={state.get('prev_episode_score')}, last={state.get('last_episode_score')}\n"
        f"Recent fold_guard_triggers={state.get('fold_triggers_recent')}\n"
        f"Recent joint telemetry summary (q over time): {state.get('joint_telemetry')}\n"
        f"Notes: If fold triggers > 0, prioritize reducing action_scale and/or increasing reward_w_du.\n"
    )

    raw = _ollama_chat(model, prompt, timeout_s=int(os.environ.get('LITE6_LLM_TIMEOUT', '60')))
    obj = _parse_llm_json(raw) or {}
    patch_in = obj.get('patch') if isinstance(obj, dict) else None
    why = (obj.get('why') if isinstance(obj, dict) else None) or 'llm'
    conf = float(obj.get('confidence') if isinstance(obj, dict) else 0.0)

    if not isinstance(patch_in, dict) or not patch_in:
        return {}, f'no patch (llm output): {raw[:200]}', 0.0

    # Allowlist + validation
    patch: Dict[str, float] = {}

    if 'action_scale' in patch_in:
        v = float(patch_in['action_scale'])
        # limit relative change
        v = clamp(v, b_action)
        v = clamp(v, Bounds(action_scale / max_action_mult, action_scale * max_action_mult))
        patch['action_scale'] = v

    if 'reward_w_du' in patch_in:
        patch['reward_w_du'] = clamp(float(patch_in['reward_w_du']), b_wdu)

    if 'reward_w_u' in patch_in:
        patch['reward_w_u'] = clamp(float(patch_in['reward_w_u']), b_wu)

    if 'target_radius' in patch_in:
        patch['target_radius'] = clamp(float(patch_in['target_radius']), b_rad)

    if not patch:
        return {}, 'patch rejected by allowlist/bounds', 0.0

    return patch, why, conf


def apply_patch_to_config(logdir: Path, patch: Dict[str, float]) -> bool:
    """Apply patch to config.yaml. Returns True if any value changed."""
    cfgp = logdir / 'config.yaml'
    if not cfgp.exists():
        raise FileNotFoundError(cfgp)
    from ruamel.yaml import YAML
    y = YAML()
    cfg = y.load(cfgp.read_text())
    cfg.setdefault('env', {})
    cfg['env'].setdefault('lite6', {})

    changed = False
    for k, v in patch.items():
        cur = cfg['env']['lite6'].get(k, None)
        try:
            curf = float(cur)
        except Exception:
            curf = None
        vf = float(v)
        if curf is None or abs(curf - vf) > 1e-9:
            cfg['env']['lite6'][k] = vf
            changed = True

    if not changed:
        return False

    tmp = cfgp.with_suffix('.yaml.tmp')
    with tmp.open('w') as f:
        y.dump(cfg, f)
    tmp.replace(cfgp)
    return True


def run_train(venv_activate: str, logdir: Path, steps: int):
    # Ensure we run from the rl project directory so dreamerv3_lite6_main.py is found.
    # IMPORTANT: Do NOT pass --run.from_checkpoint here.
    # The embodied training loop already uses elements.Checkpoint(logdir/'checkpoint.pkl')
    # and will resume from that automatically (cp.load_or_save()).
    proj = Path.home() / 'ws_xarm/rl'
    task = os.environ.get('LITE6_TASK', 'lite6_reach')
    task_variant = os.environ.get('LITE6_TASK_VARIANT', 'reach')
    ball_radius = os.environ.get('LITE6_BALL_RADIUS', '0.03')
    grasp_dist_thresh = os.environ.get('LITE6_GRASP_DIST_THRESH', '0.04')
    lift_height_thresh = os.environ.get('LITE6_LIFT_HEIGHT_THRESH', '0.08')
    cmd = [
        'zsh', '-lc',
        f"cd {proj} && source {venv_activate} && python dreamerv3_lite6_main.py --task {task} --logdir {logdir} --run.steps {steps} --run.envs 1"
        + " --env.lite6.host 127.0.0.1 --env.lite6.port 5555 --env.lite6.video_every ${LITE6_VIDEO_EVERY:-500} --env.lite6.download_dir ~/Downloads --env.lite6.download_prefix 'robotarm training video (left+visgate)'"
        + f" --env.lite6.task_variant {task_variant} --env.lite6.ball_radius {ball_radius} --env.lite6.grasp_dist_thresh {grasp_dist_thresh} --env.lite6.lift_height_thresh {lift_height_thresh}"
    ]
    subprocess.run(cmd, check=False)


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.run(['pgrep', '-f', pattern], check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if out.returncode != 0:
            return []
        return [int(x) for x in out.stdout.decode().split() if x.strip().isdigit()]
    except Exception:
        return []


def _kill(pids: list[int]):
    for p in pids:
        try:
            os.kill(p, 9)
        except Exception:
            pass


def _ensure_single_process(pattern: str, keep: str = 'newest') -> tuple[int | None, list[int]]:
    """Ensure only one process matching pattern exists. Returns (kept_pid, killed_pids)."""
    pids = _pgrep(pattern)
    if len(pids) <= 1:
        return (pids[0] if pids else None), []

    # Keep newest by start time (ps lstart), fallback to max pid.
    kept = None
    if keep == 'newest':
        try:
            ps = subprocess.run(['ps', '-o', 'pid=,lstart=', '-p', ','.join(map(str, pids))], check=False, stdout=subprocess.PIPE)
            rows = []
            for line in ps.stdout.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                pid = int(parts[0])
                # lstart: Day Mon DD HH:MM:SS YYYY
                # Use string sort as a rough proxy (good enough)
                lstart = ' '.join(parts[1:])
                rows.append((lstart, pid))
            rows.sort()
            kept = rows[-1][1]
        except Exception:
            kept = max(pids)
    else:
        kept = max(pids)

    killed = [p for p in pids if p != kept]
    _kill(killed)
    return kept, killed


def _fix_checkpoint_latest(logdir: Path) -> bool:
    """If checkpoint.pkl/latest points to missing dir, remove latest. Returns True if changed."""
    latest = logdir / 'checkpoint.pkl' / 'latest'
    if not latest.exists():
        return False
    try:
        target = latest.read_text().strip()
        if not target:
            latest.unlink(missing_ok=True)
            return True
        path = logdir / 'checkpoint.pkl' / target
        if not path.exists():
            latest.unlink(missing_ok=True)
            return True
    except Exception:
        try:
            latest.unlink(missing_ok=True)
            return True
        except Exception:
            return False
    return False


def _ensure_worker_listening(host: str = '127.0.0.1', port: int = 5555) -> bool:
    """Best-effort check that worker port is listening; if not, try to start."""
    try:
        out = subprocess.run(['zsh', '-lc', f'lsof -iTCP:{port} -sTCP:LISTEN -nP >/dev/null 2>&1'], check=False)
        if out.returncode == 0:
            return True
    except Exception:
        pass
    subprocess.run(['zsh', '-lc', '~/ws_xarm/rl/scripts/start_lite6_worker.zsh'], check=False)
    return True


def main():
    root = Path(os.environ.get('LITE6_LOGDIR', str(Path.home() / 'ws_xarm/rl/logdir_v3')))
    chunk = int(os.environ.get('LITE6_CHUNK', '2000'))
    venv_activate = os.environ.get('LITE6_VENV', str(Path.home() / 'ws_xarm/rl/.venv/bin/activate'))

    # Ensure worker is up
    _ensure_worker_listening('127.0.0.1', 5555)

    metrics = root / 'metrics.jsonl'
    scores = root / 'scores.jsonl'

    prev_score = None
    episodes_seen = 0
    last_scores_len = 0

    patch_every = int(os.environ.get('LITE6_PATCH_EVERY_EPISODES', '5'))
    video_review_every = int(os.environ.get('LITE6_VIDEO_REVIEW_EVERY_EPISODES', '2'))
    worker_log = Path(os.environ.get('LITE6_WORKER_LOG', '/tmp/lite6_worker.log'))
    telemetry_path = Path(os.environ.get('LITE6_TELEMETRY_PATH', '/tmp/lite6_joint_telemetry.jsonl'))

    # Conservative autopilot watchdog settings
    autopilot = os.environ.get('LITE6_AUTOPILOT', 'conservative')
    llm_model = os.environ.get('LITE6_LLM_MODEL', 'qwen3:latest')

    healthy_before_tune_s = int(float(os.environ.get('LITE6_HEALTHY_BEFORE_TUNE_MIN', '20')) * 60)
    healthy_since = time.time()

    last_reviewed_video = ''

    while True:
        # Conservative watchdog (no LLM autonomy beyond patch cadence).
        if autopilot == 'conservative':
            _ensure_worker_listening('127.0.0.1', 5555)
            fixed_latest = _fix_checkpoint_latest(root)
            # Kill duplicates if they appear.
            kept_d, killed_d = _ensure_single_process(r'dreamerv3_lite6_main.py')
            kept_s, killed_s = _ensure_single_process(r'scripts/supervised_train_lite6.py')

            # Health gating: only allow tuning after the system stays stable for a window.
            # Any infra fix (killing duplicates / fixing broken latest) resets the timer.
            infra_changed = fixed_latest or bool(killed_d) or bool(killed_s)
            if infra_changed or kept_d is None or kept_s is None:
                healthy_since = time.time()

        # Determine current step from metrics.
        lastm = read_last_jsonl(metrics) or {}
        cur_step = int(lastm.get('step', 0) or 0)
        target_steps = max(cur_step + chunk, chunk)

        run_train(venv_activate, root, target_steps)

        # After run exits, update state.
        # Episode counter: use score file length as a proxy.
        try:
            scores_len = len(scores.read_text().splitlines()) if scores.exists() else 0
        except Exception:
            scores_len = 0
        if scores_len > last_scores_len:
            episodes_seen += (scores_len - last_scores_len)
            last_scores_len = scores_len

        last_score_entry = read_last_jsonl(scores) or {}
        last_score = last_score_entry.get('episode/score')

        cfg = parse_config_yaml(root)
        env = (cfg or {}).get('env', {}).get('lite6', {})

        joint_summary = _read_joint_telemetry(telemetry_path)

        state = {
            'env': env,
            'last_episode_score': float(last_score) if last_score is not None else None,
            'prev_episode_score': float(prev_score) if prev_score is not None else None,
            'latest_clip': str(find_latest_clip(root) or ''),
            'episodes_seen': int(episodes_seen),
            'fold_triggers_recent': int(_read_fold_guard_triggers(worker_log)),
            'joint_telemetry': joint_summary,
        }

        patch = {}
        why = 'skip'
        conf = 0.0

        # Allow tuning only after system has been stable for a while.
        tune_ok = (time.time() - healthy_since) >= healthy_before_tune_s

        # 1) Video-based review (every N episodes) using local VLM.
        if tune_ok and video_review_every > 0 and episodes_seen > 0 and (episodes_seen % video_review_every) == 0:
            vid = find_latest_episode_video(root)
            if vid is not None and str(vid) != last_reviewed_video:
                vpatch, vwhy, vconf = review_episode_video_for_patch(root, state)
                last_reviewed_video = str(vid)
                if vpatch and vconf >= 0.6:
                    patch, why, conf = vpatch, vwhy, vconf

        # 2) Text LLM patching (every N episodes) if video didn't trigger a patch.
        if tune_ok and (not patch) and patch_every > 0 and episodes_seen > 0 and (episodes_seen % patch_every) == 0:
            patch, why, conf = propose_patch(state)

        # Require confidence and at least one metric present.
        if patch and conf >= 0.6:
            changed = apply_patch_to_config(root, patch)
            if changed:
                # Save audit record
                audit = root / 'supervisor_patches.jsonl'
                rec = {
                    'time': time.time(),
                    'step': target_steps,
                    'episodes_seen': episodes_seen,
                    'patch': patch,
                    'why': why,
                    'confidence': conf,
                    'state': state,
                    'llm_model': os.environ.get('LITE6_LLM_MODEL', 'qwen3:latest'),
                    'vlm_model': os.environ.get('LITE6_VLM_MODEL', 'qwen3-vl:32b'),
                }
                with audit.open('a') as f:
                    f.write(json.dumps(rec) + '\n')

        prev_score = state['last_episode_score']

        # Small pause to avoid tight loop if training exits immediately.
        time.sleep(2)


if __name__ == '__main__':
    main()
