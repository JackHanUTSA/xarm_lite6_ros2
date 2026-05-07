"""Lite6 RPC env (DreamerV4 data recording variant).

This file is intentionally separate from lite6_rpc_env.py (DreamerV3) to keep the
pipelines clean.

Adds `record_v4` config in reset() messages so the Isaac worker writes
frame+action+proprio datasets under logdir_v4.
"""

import socket
import json

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


class Lite6RPCEnvV4(embodied.Env):
  """Embodied env proxying to Isaac worker over TCP (DreamerV4 recording mode)."""

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
    # Recording options (DreamerV4)
    record_v4_enabled=True,
    record_v4_dir='dataset_v4',
    record_v4_every=1,
    # Task variant / grasp config
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

    self._record_v4 = {
      'enabled': bool(record_v4_enabled),
      'dir': str(record_v4_dir),
      'every': int(record_v4_every),
    }

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
      'action': elements.Space(np.float32, (6,), -1.0, 1.0),
      'reset': elements.Space(bool),
    }

  def close(self):
    try:
      if self._sock is not None:
        _send(self._sock, {'cmd': 'close'})
        self._sock.close()
    except Exception:
      pass
    self._sock = None

  def reset(self):
    self._connect()
    self._step_count = 0
    cfg = {}
    cfg['task_variant'] = self._task_variant
    if self._ball_radius is not None:
      cfg['ball_radius'] = float(self._ball_radius)
    if self._grasp_dist_thresh is not None:
      cfg['grasp_dist_thresh'] = float(self._grasp_dist_thresh)
    if self._lift_height_thresh is not None:
      cfg['lift_height_thresh'] = float(self._lift_height_thresh)
    if self._action_scale is not None:
      cfg['action_scale'] = float(self._action_scale)
    if self._reward_w_u is not None:
      cfg['reward_w_u'] = float(self._reward_w_u)
    if self._reward_w_du is not None:
      cfg['reward_w_du'] = float(self._reward_w_du)
    if self._target_radius is not None:
      cfg['target_radius'] = float(self._target_radius)

    _send(self._sock, {
      'cmd': 'reset',
      'logdir': self._logdir,
      'video': self._video,
      'video_every': self._video_every,
      'download': {'dir': self._download_dir, 'prefix': self._download_prefix},
      'record_v4': self._record_v4,
      'cfg': cfg,
    })
    obs = _recv(self._sock)
    self._done = False
    return self._wrap_obs(obs, is_first=True)

  def step(self, action):
    if action.get('reset', False) or self._done:
      return self.reset()

    act = action['action']
    self._connect()
    self._step_count += 1
    _send(self._sock, {
      'cmd': 'step',
      'action': [float(x) for x in np.array(act, np.float32).tolist()],
      'global_step': None,
    })
    obs = _recv(self._sock)
    self._done = bool(obs.get('is_last', False))
    return self._wrap_obs(obs, is_first=False)

  def _wrap_obs(self, obs, is_first=False):
    out = {
      'q': np.array(obs.get('q', [0]*6), np.float32),
      'ee_pos': np.array(obs.get('ee_pos', [0]*3), np.float32),
      'target_pos': np.array(obs.get('target_pos', [0]*3), np.float32),
      'dist': np.float32(obs.get('dist', 0.0)),
      'vis_dist_px': np.float32(obs.get('vis_dist_px', 0.0) or 0.0),
      'success_ee': np.uint8(1 if obs.get('success_ee', False) else 0),
      'success_vis': np.uint8(1 if obs.get('success_vis', False) else 0),
      'reward': np.float32(obs.get('reward', 0.0)),
      'is_first': bool(is_first),
      'is_last': bool(obs.get('is_last', False)),
      'is_terminal': bool(obs.get('is_terminal', False)),
    }
    return out
