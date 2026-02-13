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

  def __init__(self, task, host='127.0.0.1', port=5555, timeout=30.0, logdir='', video_fps=30, video_w=640, video_h=480, video_seconds=20, video_every=0, download_dir='~/Downloads', download_prefix='robotarm training video'):
    self._task = task
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
      _send(self._sock, {'cmd': 'reset', 'task': self._task, 'logdir': self._logdir, 'video': self._video, 'video_every': self._video_every})
      msg = _recv(self._sock)
      self._done = False
      return self._format(msg, is_first=True)

    act = np.asarray(action['action'], np.float32).reshape((6,))
    _send(self._sock, {'cmd': 'step', 'action': act.tolist()})
    msg = _recv(self._sock)
    self._done = bool(msg.get('is_last', False))
    return self._format(msg)

  def _format(self, msg, is_first=False):
    return {
      'q': np.asarray(msg['q'], np.float32),
      'ee_pos': np.asarray(msg['ee_pos'], np.float32),
      'target_pos': np.asarray(msg['target_pos'], np.float32),
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