#!/usr/bin/env python3
"""Headless Isaac worker: exposes reset/step over TCP for Lite6 reach.

Adds per-episode video saving: on episode end, saves an MP4 of exactly N seconds
(default 20s) by resampling frames to target_fps*seconds.

Launch:
  ~/isaacsim/isaac-sim-4.2.0/python.sh ~/ws_xarm/isaac_bridge/scripts/lite6_reach_worker.py
"""

import socket
import json
import time
import os
import subprocess
from dataclasses import dataclass

import numpy as np

from isaacsim import SimulationApp


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


class VideoRecorder:
    def __init__(self):
        self.enabled = False
        self.fps = 30
        self.w = 640
        self.h = 480
        self.seconds = 20
        self.logdir = ''
        self.episode_idx = 0
        self.frames = []
        self.rep = None
        self.annot = None
        self.rp = None
        self.cam = None

    def configure(self, logdir: str, video: dict):
        self.logdir = str(logdir or '')
        self.enabled = bool(self.logdir)
        if not self.enabled:
            return
        self.fps = int(video.get('fps', 30))
        self.w = int(video.get('w', 640))
        self.h = int(video.get('h', 480))
        self.seconds = int(video.get('seconds', 20))

    def setup_rep(self, stage):
        if not self.enabled:
            return
        import omni.replicator.core as rep
        from pxr import UsdGeom, Sdf

        self.rep = rep
        # Create a camera
        cam_path = Sdf.Path('/World/Lite6Cam')
        if not stage.GetPrimAtPath(cam_path):
            rep.create.camera(position=(0.9, 0.0, 0.7), look_at=(0.0, 0.0, 0.25), name='Lite6Cam')
        self.cam = '/World/Lite6Cam'

        self.rp = rep.create.render_product(self.cam, (self.w, self.h))
        self.annot = rep.AnnotatorRegistry.get_annotator('rgb')
        self.annot.attach([self.rp])
        # no writers; we pull frames directly

    def reset_episode(self):
        if not self.enabled:
            return
        self.frames = []

    def capture(self):
        if not self.enabled or self.annot is None:
            return
        # Replicator updates
        self.rep.orchestrator.step()
        data = self.annot.get_data()
        if data is None:
            return
        # rgba -> rgb
        rgb = np.asarray(data)[..., :3].copy()
        self.frames.append(rgb)

    def save_episode(self, name_prefix='ep'):
        if not self.enabled:
            return None
        if not self.frames:
            return None

        target = int(self.fps * self.seconds)
        n = len(self.frames)
        # resample indices to exactly target frames
        idx = np.linspace(0, max(n - 1, 0), num=target)
        idx = np.clip(np.round(idx).astype(int), 0, max(n - 1, 0))
        frames = [self.frames[i] for i in idx]

        out_dir = os.path.join(self.logdir, 'episodes')
        os.makedirs(out_dir, exist_ok=True)
        self.episode_idx += 1
        mp4_path = os.path.join(out_dir, f'{name_prefix}_{self.episode_idx:06d}.mp4')

        tmp_dir = os.path.join(out_dir, f'.tmp_{name_prefix}_{self.episode_idx:06d}')
        os.makedirs(tmp_dir, exist_ok=True)

        import imageio.v2 as imageio
        for k, fr in enumerate(frames, 1):
            imageio.imwrite(os.path.join(tmp_dir, f'frame_{k:06d}.png'), fr)

        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.fps),
            '-i', os.path.join(tmp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            mp4_path,
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # cleanup pngs
        try:
            for name in os.listdir(tmp_dir):
                if name.endswith('.png'):
                    os.remove(os.path.join(tmp_dir, name))
            os.rmdir(tmp_dir)
        except Exception:
            pass

        return mp4_path


class Lite6ReachSim:
    def __init__(self, cfg: ReachConfig):
        self.cfg = cfg
        self.app = None
        self.sim = None
        self.art = None
        self.stage = None
        self.stage_path = None
        self.ee_path = None
        self.q = np.zeros((6,), np.float32)
        self.target = np.zeros((3,), np.float32)
        self.t = 0
        self._rng = np.random.RandomState(0)
        self._get_world_pose = None
        self.video = VideoRecorder()

    def start(self):
        ws = "/home/r91/ws_xarm"
        urdf_path = f"{ws}/isaac_bridge/lite6_isaac.urdf"

        self.app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

        import omni
        import omni.kit.commands
        from omni.isaac.core import SimulationContext
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.xforms import get_world_pose
        from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics

        self._get_world_pose = get_world_pose

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.distance_scale = 1

        status, stage_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
            get_articulation_root=True,
        )
        if not status:
            raise RuntimeError("URDF import failed")

        self.stage = omni.usd.get_context().get_stage()
        self.stage_path = stage_path

        scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath("/physicsScene"))

        light = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/DistantLight"))
        light.CreateIntensityAttr(500)

        self.app.update()

        self.sim = SimulationContext(stage_units_in_meters=1.0)
        self.sim.initialize_physics()

        self.art = Articulation(prim_path=self.stage_path)
        self.art.initialize()
        if not self.art.handles_initialized:
            raise RuntimeError(f"{self.stage_path} is not an articulation")

        self.ee_path = None
        for prim in self.stage.Traverse():
            p = prim.GetPath().pathString
            if p.endswith("/link_eef"):
                self.ee_path = p
                break
        if not self.ee_path:
            raise RuntimeError("Could not find link_eef prim")

        for _ in range(10):
            self.sim.step(render=self.video.enabled)

    def close(self):
        if self.app is not None:
            try:
                self.app.close()
            except Exception:
                pass
            self.app = None

    def _randomize_target(self):
        self._rng.seed(int(time.time() * 1e6) % (2**32 - 1))
        self.target = np.array([
            self._rng.uniform(self.cfg.x_min, self.cfg.x_max),
            self._rng.uniform(self.cfg.y_min, self.cfg.y_max),
            self._rng.uniform(self.cfg.z_min, self.cfg.z_max),
        ], np.float32)

    def _ee_pos(self):
        pos, _ = self._get_world_pose(self.ee_path)
        return np.array(pos, np.float32)

    def reset(self, logdir='', video=None):
        # configure video capture on first reset
        if video is None:
            video = {}
        self.video.configure(logdir, video)
        if self.video.enabled:
            print(f'VIDEO_ENABLED logdir={self.video.logdir} fps={self.video.fps} size={self.video.w}x{self.video.h} seconds={self.video.seconds}', flush=True)
        if self.video.enabled and self.video.annot is None:
            self.video.setup_rep(self.stage)
        self.video.reset_episode()

        self.t = 0
        self.q[:] = 0.0
        self._randomize_target()
        self.art.set_joint_positions(self.q)
        for _ in range(self.cfg.settle_steps):
            self.sim.step(render=self.video.enabled)
        self.video.capture()

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
            self.sim.step(render=self.video.enabled)
        self.t += 1
        self.video.capture()

        ee = self._ee_pos()
        dist = float(np.linalg.norm(ee - self.target))
        done = self.t >= self.cfg.episode_len

        mp4 = None
        if done:
            mp4 = self.video.save_episode('ep')

        if mp4:
            print(f'VIDEO_SAVED {mp4}', flush=True)

        out = {
            'q': self.q.tolist(),
            'ee_pos': ee.tolist(),
            'target_pos': self.target.tolist(),
            'reward': float(-dist),
            'is_last': bool(done),
            'is_terminal': False,
        }
        if mp4:
            out['video_path'] = mp4
        return out


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
                send_msg(conn, sim.reset(msg.get('logdir',''), msg.get('video', {})))
            elif cmd == 'step':
                send_msg(conn, sim.step(msg['action']))
            elif cmd == 'save_video':
                # Save a clip from currently buffered frames (resampled to N seconds).
                mp4 = sim.video.save_episode()
                send_msg(conn, {'ok': True, 'video_path': mp4})
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