import socket
import json

# Run-global step counter (shared across all Lite6RPCEnv instances in this process).
_RUN_GLOBAL_STEP = 0
_RUN_GLOBAL_STEP_LAST = 0

import elements
import embodied
import numpy as np


def _recvall(sock, n):
  buf = b''
  while len(buf) < n:
    chunk = sock.recv(n - len(buf))
    if not chunk:
      raise ConnectionError('socket closed')
    buf += chunk
  return buf


def _send(sock, obj):
  data = json.dumps(obj).encode('utf-8')
  sock.sendall(len(data).to_bytes(4, 'big') + data)


def _recv(sock):
  n = int.from_bytes(_recvall(sock, 4), 'big')
  return json.loads(_recvall(sock, n).decode('utf-8'))




def _copy_video_to_downloads(src_path: str, step_count: int, download_dir: str, prefix: str):
  try:
    import os
    import shutil
    d = os.path.expanduser(download_dir or '')
    if not d:
      return None
    os.makedirs(d, exist_ok=True)
    safe = (prefix or 'robotarm training video').replace('/', '_')
    dst = os.path.join(d, f"{safe} - step_{step_count:09d}.mp4")
    shutil.copy2(src_path, dst)
    return dst
  except Exception:
    return None

class Lite6RPCEnv(embodied.Env):
  """Embodied env proxying to Isaac worker over TCP."""

  def __init__(
    self,
    task,
    index=0,
    host='127.0.0.1',
    port=5555,
    timeout=30.0,
    logdir='',
    video_fps=30,
    video_w=640,
    video_h=480,
    video_seconds=20,
    video_every=0,
    download_dir='~/Downloads',
    download_prefix='robotarm training video',
    # Task variant / grasp config.
    task_variant='reach',
    ball_radius=None,
    grasp_dist_thresh=None,
    lift_height_thresh=None,
    # Patchable knobs forwarded to the worker on reset.
    action_scale=None,
    reward_w_u=None,
    reward_w_du=None,
    target_radius=None,
  ):
    self._task = task
    self._index = int(index)
    self._host = host
    self._port = int(port)
    self._timeout = float(timeout)
    self._logdir = str(logdir)
    self._video = dict(fps=int(video_fps), w=int(video_w), h=int(video_h), seconds=int(video_seconds))
    self._video_every = int(video_every)
    self._step_count = 0
    self._download_dir = str(download_dir)
    self._download_prefix = str(download_prefix)
    self._sock = None
    self._done = True

    self._task_variant = str(task_variant)
    self._ball_radius = None if ball_radius is None else float(ball_radius)
    self._grasp_dist_thresh = None if grasp_dist_thresh is None else float(grasp_dist_thresh)
    self._lift_height_thresh = None if lift_height_thresh is None else float(lift_height_thresh)

    self._action_scale = None if action_scale is None else float(action_scale)
    self._reward_w_u = None if reward_w_u is None else float(reward_w_u)
    self._reward_w_du = None if reward_w_du is None else float(reward_w_du)
    self._target_radius = None if target_radius is None else float(target_radius)

  def _connect(self):
    if self._sock is not None:
      return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(self._timeout)
    sock.connect((self._host, self._port))
    sock.settimeout(None)
    self._sock = sock

  @property
  def obs_space(self):
    return {
      'q': elements.Space(np.float32, (6,)),
      'ee_pos': elements.Space(np.float32, (3,)),
      'target_pos': elements.Space(np.float32, (3,)),
      'dist': elements.Space(np.float32),
      'vis_dist_px': elements.Space(np.float32),
      # Use uint8 for binary signals to avoid JAX one_hot(bool) issues.
      'success_ee': elements.Space(np.uint8, (), 0, 1),
      'success_vis': elements.Space(np.uint8, (), 0, 1),
      'reward': elements.Space(np.float32),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
      'reset': elements.Space(bool),
      'action': elements.Space(np.float32, (6,), -1.0, 1.0),
    }

  def step(self, action):
    self._connect()
    if action['reset'] or self._done:
      cfg = {
        'task_variant': getattr(self, '_task_variant', 'reach'),
        'ball_radius': getattr(self, '_ball_radius', None),
        'grasp_dist_thresh': getattr(self, '_grasp_dist_thresh', None),
        'lift_height_thresh': getattr(self, '_lift_height_thresh', None),
        'action_scale': getattr(self, '_action_scale', None),
        'reward_w_u': getattr(self, '_reward_w_u', None),
        'reward_w_du': getattr(self, '_reward_w_du', None),
        'target_radius': getattr(self, '_target_radius', None),
      }
      cfg = {k: v for k, v in cfg.items() if v is not None}
      _send(self._sock, {
        'cmd': 'reset',
        'task': self._task,
        'logdir': self._logdir,
        'video': self._video,
        'video_every': self._video_every,
        'download': {'dir': self._download_dir, 'prefix': self._download_prefix},
        'cfg': cfg,
      })
      msg = _recv(self._sock)
      self._done = False
      return self._format(msg, is_first=True)

    act = np.asarray(action['action'], np.float32).reshape((6,))
    global _RUN_GLOBAL_STEP, _RUN_GLOBAL_STEP_LAST
    if self._index == 0:
      _RUN_GLOBAL_STEP += 1
      _RUN_GLOBAL_STEP_LAST = _RUN_GLOBAL_STEP
    gstep = _RUN_GLOBAL_STEP_LAST
    _send(self._sock, {'cmd': 'step', 'action': act.tolist(), 'global_step': gstep})
    msg = _recv(self._sock)
    self._done = bool(msg.get('is_last', False))
    return self._format(msg)

  def _format(self, msg, is_first=False):
    vis = msg.get('vis_dist_px', None)
    return {
      'q': np.asarray(msg['q'], np.float32),
      'ee_pos': np.asarray(msg['ee_pos'], np.float32),
      'target_pos': np.asarray(msg['target_pos'], np.float32),
      'dist': np.float32(msg.get('dist', 0.0)),
      'vis_dist_px': np.float32(1e9 if vis is None else vis),
      'success_ee': np.uint8(1 if msg.get('success_ee', False) else 0),
      'success_vis': np.uint8(1 if msg.get('success_vis', False) else 0),
      'reward': np.float32(msg.get('reward', 0.0)),
      'is_first': bool(is_first),
      'is_last': bool(msg.get('is_last', False)),
      'is_terminal': bool(msg.get('is_terminal', False)),
    }

  def close(self):
    if self._sock is not None:
      try:
        _send(self._sock, {'cmd': 'close'})
      except Exception:
        pass
      try:
        self._sock.close()
      except Exception:
        pass
      self._sock = None
