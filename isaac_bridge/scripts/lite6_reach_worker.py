#!/usr/bin/env python3
"""Headless Isaac worker: exposes reset/step over TCP for Lite6 reach.

Launch with Isaac Sim's python.sh:
  ~/isaacsim/isaac-sim-4.2.0/python.sh ~/ws_xarm/isaac_bridge/scripts/lite6_reach_worker.py

Protocol: length-prefixed JSON messages.
Client sends:
  {"cmd":"reset"}
  {"cmd":"step", "action":[6 floats]}
Worker replies with:
  {"q":[6], "ee_pos":[3], "target_pos":[3], "reward":float, "is_last":bool, "is_terminal":bool}
"""

import socket
import json
import time
from dataclasses import dataclass

import numpy as np

from omni.isaac.kit import SimulationApp


def recvall(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('socket closed')
        buf += chunk
    return buf


def send_msg(sock, obj):
    data = json.dumps(obj).encode('utf-8')
    sock.sendall(len(data).to_bytes(4, 'big') + data)


def recv_msg(sock):
    n = int.from_bytes(recvall(sock, 4), 'big')
    return json.loads(recvall(sock, n).decode('utf-8'))


@dataclass
class ReachConfig:
    episode_len: int = 200
    action_scale: float = 0.03
    settle_steps: int = 2
    x_min: float = 0.20
    x_max: float = 0.45
    y_min: float = -0.20
    y_max: float = 0.20
    z_min: float = 0.12
    z_max: float = 0.40


class Lite6ReachSim:
    def __init__(self, cfg: ReachConfig):
        self.cfg = cfg
        self.sim = None
        self.world = None
        self.art = None
        self.ee_path = None
        self.q = np.zeros((6,), np.float32)
        self.target = np.zeros((3,), np.float32)
        self.t = 0
        self._rng = np.random.RandomState(0)

    def start(self):
        self.sim = SimulationApp({"headless": True})

        import omni.usd
        from pxr import UsdPhysics
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.importer.urdf import _urdf
        import omni.isaac.core.utils.stage as stage_utils
        from omni.isaac.core.utils.xforms import get_world_pose

        self._get_world_pose = get_world_pose

        stage_utils.create_new_stage()
        stage = omni.usd.get_context().get_stage()

        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        urdf_path = "/home/r91/ws_xarm/lite6_isaac.urdf"
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False

        importer = _urdf.acquire_urdf_interface()
        ok = importer.import_robot(urdf_path, import_config)
        if not ok:
            raise RuntimeError("URDF import failed")

        self.world = World(stage_units_in_meters=1.0)
        self.world.reset()

        default_prim = stage.GetDefaultPrim()
        robot_path = str(default_prim.GetPath())

        art_path = robot_path.rstrip('/') + "/root_joint"
        self.art = Articulation(prim_path=art_path)
        self.art.initialize()

        self.ee_path = robot_path.rstrip('/') + "/link_eef"

        # settle
        for _ in range(10):
            self.world.step(render=False)

        self._randomize_target()
        self.reset()

    def close(self):
        if self.sim is not None:
            try:
                self.sim.close()
            except Exception:
                pass
            self.sim = None

    def _randomize_target(self):
        # change rng seed over time
        self._rng.seed(int(time.time() * 1e6) % (2**32 - 1))
        self.target = np.array([
            self._rng.uniform(self.cfg.x_min, self.cfg.x_max),
            self._rng.uniform(self.cfg.y_min, self.cfg.y_max),
            self._rng.uniform(self.cfg.z_min, self.cfg.z_max),
        ], np.float32)

    def _ee_pos(self):
        pos, _ = self._get_world_pose(self.ee_path)
        return np.array(pos, np.float32)

    def reset(self):
        self.t = 0
        self.q[:] = 0.0
        self._randomize_target()
        self.art.set_joint_positions(self.q)
        for _ in range(self.cfg.settle_steps):
            self.world.step(render=False)
        ee = self._ee_pos()
        dist = float(np.linalg.norm(ee - self.target))
        return {
            'q': self.q.tolist(),
            'ee_pos': ee.tolist(),
            'target_pos': self.target.tolist(),
            'reward': float(-dist),
            'is_last': False,
            'is_terminal': False,
        }

    def step(self, action):
        a = np.clip(np.array(action, np.float32), -1.0, 1.0)
        dq = a * self.cfg.action_scale
        self.q = self.q + dq
        self.art.set_joint_positions(self.q)
        for _ in range(self.cfg.settle_steps):
            self.world.step(render=False)
        self.t += 1
        ee = self._ee_pos()
        dist = float(np.linalg.norm(ee - self.target))
        done = self.t >= self.cfg.episode_len
        return {
            'q': self.q.tolist(),
            'ee_pos': ee.tolist(),
            'target_pos': self.target.tolist(),
            'reward': float(-dist),
            'is_last': bool(done),
            'is_terminal': False,
        }


def serve(host='127.0.0.1', port=5555):
    sim = Lite6ReachSim(ReachConfig())
    sim.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(1)
    print(f"LITE6_WORKER_LISTEN {host}:{port}", flush=True)

    conn = None
    try:
        conn, addr = srv.accept()
        print(f"LITE6_WORKER_CLIENT {addr}", flush=True)
        while True:
            msg = recv_msg(conn)
            cmd = msg.get('cmd')
            if cmd == 'reset':
                send_msg(conn, sim.reset())
            elif cmd == 'step':
                send_msg(conn, sim.step(msg['action']))
            elif cmd == 'close':
                break
            else:
                send_msg(conn, {'error': f'unknown cmd {cmd}'})
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        try:
            srv.close()
        except Exception:
            pass
        sim.close()


if __name__ == '__main__':
    serve()
