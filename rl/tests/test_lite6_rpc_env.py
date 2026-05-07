import sys
from pathlib import Path

ROOT = Path('/home/r91/ws_xarm/rl')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party'))

import lite6_rpc_env as mod


def test_lite6_rpc_env_reset_forwards_ball_grasp_task_variant(monkeypatch):
    sent = []

    class DummyEnv(mod.Lite6RPCEnv):
        def _connect(self):
            self._sock = object()

    monkeypatch.setattr(mod, '_send', lambda sock, obj: sent.append(obj))
    monkeypatch.setattr(mod, '_recv', lambda sock: {
        'q': [0] * 6,
        'ee_pos': [0, 0, 0],
        'target_pos': [0, 0, 0],
        'dist': 0.0,
        'reward': 0.0,
        'is_last': False,
        'is_terminal': False,
    })

    env = DummyEnv(
        task='lite6_grasp_ball',
        host='127.0.0.1',
        port=5555,
        task_variant='ball_grasp',
        ball_radius=0.03,
        grasp_dist_thresh=0.04,
        lift_height_thresh=0.08,
    )
    env.step({'reset': True, 'action': [0.0] * 6})

    assert sent[0]['cmd'] == 'reset'
    assert sent[0]['task'] == 'lite6_grasp_ball'
    assert sent[0]['cfg']['task_variant'] == 'ball_grasp'
    assert sent[0]['cfg']['ball_radius'] == 0.03
    assert sent[0]['cfg']['grasp_dist_thresh'] == 0.04
    assert sent[0]['cfg']['lift_height_thresh'] == 0.08
