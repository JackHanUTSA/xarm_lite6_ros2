#!/usr/bin/env python3
"""Headless Isaac worker (DreamerV4): Lite6 reach + gripper (auto open/close) + V4 dataset recording.

Adds per-episode video saving: on episode end, saves an MP4 of exactly N seconds
(default 20s) by resampling frames to target_fps*seconds.

Launch:
  ~/isaacsim/isaac-sim-4.2.0/python.sh ~/ws_xarm/isaac_bridge/scripts/lite6_reach_worker_v4.py
"""

import socket
import json
import time
import math
import faulthandler
import traceback
import os
import subprocess
import select
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

    # Collision handling
    # If enabled, any collision (self or world) will terminate the episode and apply
    # an additional negative reward.
    collision_terminate: bool = True
    collision_penalty_self: float = 50.0
    collision_penalty_world: float = 20.0

    # Contact-force based collision monitor (polls per step; does not rely on PhysX events)
    contact_monitor: bool = True
    contact_force_thresh: float = 1.0

    # Posture-based self-collision guard (heuristic fallback).
    fold_guard: bool = True
    fold_q1_thresh: float = 2.2
    fold_q2_thresh: float = 2.2
    fold_penalty: float = 50.0

    # Telemetry logging (for LLM supervisor)
    telemetry_enabled: bool = True
    telemetry_every: int = 5
    telemetry_path: str = '/tmp/lite6_joint_telemetry.jsonl'

    # Action scaling (radians per env step). Applied as: q <- q + clip(a,-1,1)*action_scale
    action_scale: float = 0.06

    # Reward shaping weights
    # reward = -dist - w_u*||a||^2 - w_du*||a - a_prev||^2
    reward_w_u: float = 0.0
    reward_w_du: float = 0.0

    settle_steps: int = 2

    # Target space (absolute ranges)
    x_min: float = 0.20
    x_max: float = 0.45
    y_min: float = -0.20
    y_max: float = 0.20
    z_min: float = 0.12
    z_max: float = 0.40

    # Curriculum: optional target half-range (meters) around the center of the above ranges.
    # If set to >0, x/y sampling becomes center +/- target_radius.
    target_radius: float = 0.0

    # Task selection.
    task_variant: str = 'reach'  # 'reach' or 'ball_grasp'
    ball_radius: float = 0.03
    grasp_dist_thresh: float = 0.04
    lift_height_thresh: float = 0.08
    grasp_attach_z_offset: float = 0.02


class VideoRecorder:
    def __init__(self):
        self.enabled = False
        self.fps = 30
        self.w = 640
        self.h = 480
        self.seconds = 20
        self.logdir = ''
        self.episode_idx = 0
        self.video_every = 0
        self.download_dir = ''
        self.download_prefix = 'robotarm training video'

        # DreamerV4 dataset recording (frames + actions + proprio).
        # Enabled only when `record_v4` is provided on reset() and video is enabled.
        self.record_enabled = False
        self.record_root = ''
        self.record_every = 1
        self.record_episode_dir = ''
        self.record_step = 0
        self.record_f_steps = None
        self.global_step = 0  # run-global step counter (not reset per episode)
        self.frames = []
        self.rep = None
        self.annot = None
        self.bbox_annot = None
        self.rp = None
        self.cam = None
        self.debug_lines = []
        self._overlay = lambda _rgb, _lines: None

        # Visual success helpers
        self.ee_prim_path = '/World/Markers/EE'
        self.target_prim_path = '/World/Markers/Target'
        self.last_vis_dist_px = None
        self._warned_bbox = False

    def configure(self, logdir: str, video: dict, video_every: int = 0, download=None, record_v4=None, app=None):
        self.logdir = str(logdir or '')
        self.enabled = bool(self.logdir)
        self.app = app
        if not self.enabled:
            return
        self.fps = int(video.get('fps', 30))
        self.w = int(video.get('w', 640))
        self.h = int(video.get('h', 480))
        self.seconds = int(video.get('seconds', 20))
        self.video_every = int(video_every or 0)
        download = download or {}
        self.download_dir = str(download.get('dir','') or '')
        self.download_prefix = str(download.get('prefix','robotarm training video') or 'robotarm training video')

        # Recording config (DreamerV4 pipeline).
        record_v4 = record_v4 or {}
        self.record_enabled = bool(record_v4.get('enabled', False))
        self.record_every = int(record_v4.get('every', 1) or 1)
        if self.record_every <= 0:
            self.record_every = 1
        rec_dir = str(record_v4.get('dir', '') or '')
        self.record_root = os.path.join(self.logdir, rec_dir) if (self.record_enabled and rec_dir) else ''
        if self.record_enabled and not self.record_root:
            # If enabled but no dir provided, default to a clearly-labeled folder.
            self.record_root = os.path.join(self.logdir, 'dataset_v4')
        if self.record_enabled:
            os.makedirs(self.record_root, exist_ok=True)


    def setup_rep(self, stage):
        if not self.enabled:
            return
        try:
            import omni.replicator.core as rep
            from pxr import Gf, Sdf, UsdGeom
            self.rep = rep
            # Ensure /World exists
            world = stage.GetPrimAtPath(Sdf.Path('/World'))
            if not world:
                UsdGeom.Xform.Define(stage, Sdf.Path('/World'))
            # Create camera via Replicator (handles orientation robustly)
            eye = tuple(self.eye) if self.eye is not None else (2.0, 2.0, 1.5)
            look = tuple(self.look) if self.look is not None else (0.0, 0.0, 0.3)
            cam = rep.create.camera(position=eye, look_at=look, focal_length=24.0)
            self.cam = cam
            # Add a dome light so the arm isn't black in headless renders
            try:
                rep.create.light(light_type='dome', intensity=300)
            except Exception:
                pass
            self.rp = rep.create.render_product(self.cam, (self.w, self.h))
            self.annot = rep.AnnotatorRegistry.get_annotator('rgb')
            self.annot.attach([self.rp])

            # For fast "visual reach" checks, try to enable 2D tight bboxes.
            try:
                try:
                    self.bbox_annot = rep.AnnotatorRegistry.get_annotator('bbox_2d_tight')
                except Exception:
                    self.bbox_annot = rep.AnnotatorRegistry.get_annotator('bounding_box_2d_tight')
                self.bbox_annot.attach([self.rp])
            except Exception:
                self.bbox_annot = None
        except Exception as e:
            # Disable video if replicator/camera setup fails
            print(f'VIDEO_SETUP_FAILED {e}', flush=True)
            self.enabled = False
            self.annot = None
            self.rep = None
            self.rp = None
            self.cam = None
    def reset_episode(self):
        if not self.enabled:
            return
        self.frames = []

        # Start a new recording episode directory if recording is enabled.
        if self.record_enabled and self.record_root:
            self.record_step = 0
            # Use the *next* episode_idx (since save_episode increments later); keep separate counter here.
            # We base it on global_step to avoid collisions across restarts.
            ep_name = f'episode_{int(self.global_step):09d}'
            self.record_episode_dir = os.path.join(self.record_root, ep_name)
            os.makedirs(os.path.join(self.record_episode_dir, 'frames'), exist_ok=True)
            self.record_f_steps = open(os.path.join(self.record_episode_dir, 'steps.jsonl'), 'a')
            meta = {
                'created_at': time.time(),
                'fps': int(self.fps),
                'w': int(self.w),
                'h': int(self.h),
                'seconds': int(self.seconds),
            }
            try:
                with open(os.path.join(self.record_episode_dir, 'meta.json'), 'w') as f:
                    f.write(json.dumps(meta))
            except Exception:
                pass
        else:
            self.record_episode_dir = ''
            self.record_f_steps = None

    def record_step_v4(self, rgb, step_obj: dict):
        if not self.record_enabled or not self.record_episode_dir or self.record_f_steps is None:
            return
        try:
            if (self.record_step % int(self.record_every)) == 0 and rgb is not None:
                # Write frame
                import imageio.v2 as imageio
                fpath = os.path.join(self.record_episode_dir, 'frames', f'frame_{self.record_step:06d}.png')
                imageio.imwrite(fpath, rgb)
            step_obj = dict(step_obj)
            step_obj['record_step'] = int(self.record_step)
            self.record_f_steps.write(json.dumps(step_obj) + '\n')
            self.record_f_steps.flush()
        except Exception:
            pass
        finally:
            self.record_step += 1

    def capture(self):
        if not self.enabled or self.annot is None:
            return
        # Replicator updates
        self.rep.orchestrator.step()

        # RGB frame
        data = self.annot.get_data()
        if data is None:
            return
        # rgba -> rgb
        rgb = np.asarray(data)[..., :3].copy()

        # Optional: compute visual distance between EE and target markers (pixel-space)
        self.last_vis_dist_px = None
        if self.bbox_annot is not None:
            try:
                bb = self.bbox_annot.get_data()
                prim_paths = None
                rects = None
                if isinstance(bb, dict):
                    info = bb.get('info') or {}
                    prim_paths = info.get('primPaths') or info.get('prim_paths')
                    rects = bb.get('data') or bb.get('rects')
                if prim_paths and rects is not None:
                    prim_paths = list(prim_paths)

                    def center_for(path):
                        if path not in prim_paths:
                            return None
                        i = prim_paths.index(path)
                        r = rects[i]
                        try:
                            x0, y0, x1, y1 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                        except Exception:
                            return None
                        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)

                    ce = center_for(self.ee_prim_path)
                    ct = center_for(self.target_prim_path)
                    if ce is not None and ct is not None:
                        dx = float(ce[0] - ct[0])
                        dy = float(ce[1] - ct[1])
                        self.last_vis_dist_px = float((dx * dx + dy * dy) ** 0.5)
            except Exception as e:
                if not self._warned_bbox:
                    print(f'VIDEO_BBOX_FAILED {e}', flush=True)
                    self._warned_bbox = True

        if self.debug_lines:
            self._overlay(rgb, self.debug_lines)
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

        # Close recording file if open (end of episode).
        try:
            if self.record_f_steps is not None:
                self.record_f_steps.close()
        except Exception:
            pass
        self.record_f_steps = None

        return mp4_path




class GripperController:
    """Best-effort gripper controller for xArm6-with-gripper USD.

    We auto-toggle open/close (discrete) on a fixed period during V4 dataset runs.
    """

    def __init__(self):
        self.enabled = False
        self.joint_ids = []
        self.lower = None
        self.upper = None
        self.state = 0  # 0=open, 1=close
        self.period = 50

    def setup(self, art, period: int = 50):
        self.enabled = False
        self.period = int(period or 50)
        try:
            names = list(getattr(art, 'joint_names', []) or [])
        except Exception:
            names = []
        if not names:
            try:
                names = list(art.get_joint_names())
            except Exception:
                names = []

        # Heuristic: select joints that look like gripper/finger.
        cand = []
        for i, n in enumerate(names):
            nl = str(n).lower()
            if 'gripper' in nl or 'finger' in nl or 'left' in nl and 'finger' in nl or 'right' in nl and 'finger' in nl:
                cand.append(i)
        if not cand:
            try:
                print('GRIPPER_JOINT_NAMES ' + ' | '.join([str(x) for x in names[:60]]), flush=True)
            except Exception:
                pass
            return

        self.joint_ids = cand
        # Joint limits
        try:
            lim = art.get_joint_limits()
            # expected shape (N,2)
            self.lower = lim[:, 0]
            self.upper = lim[:, 1]
        except Exception:
            self.lower = None
            self.upper = None

        self.enabled = True

    def _target_for_state(self, state: int):
        # If we have limits: open -> upper, close -> lower (best-effort).
        if self.lower is not None and self.upper is not None:
            tgt = self.upper if state == 0 else self.lower
            return tgt
        return None

    def maybe_toggle(self, global_step: int):
        if not self.enabled:
            return None
        if self.period > 0 and (int(global_step) % int(self.period)) == 0:
            self.state = 1 - int(self.state)
        return int(self.state)

    def apply(self, art, state: int):
        if not self.enabled:
            return
        tgt = self._target_for_state(int(state))
        if tgt is None:
            return
        try:
            # Set position targets for only the gripper joints
            # Use full-size target vector if API requires.
            ids = self.joint_ids
            # Some Isaac versions accept indices+values.
            try:
                art.set_joint_position_targets(tgt[ids], joint_indices=ids)
            except Exception:
                # Fallback: set full vector
                cur = art.get_joint_position_targets()
                cur = cur.copy()
                cur[ids] = tgt[ids]
                art.set_joint_position_targets(cur)
        except Exception:
            pass
class Lite6ReachSim:
    EE_SUCCESS_M: float = 0.01
    VIS_SUCCESS_PX: float = 9.0
    VIS_GATE_M: float = 0.03
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
        self.gripper = GripperController()
        self.gripper.enabled = False  # enabled when a gripper is detected in start()
        self.ball = np.zeros((3,), np.float32)
        self.ball_initial_z = 0.0
        self.ball_attached = False
        self.ball_marker_xf = None

        # Collision state
        self._contact_sub = None
        self._robot_prefix = None
        self._coll_self = False
        self._coll_world = False

        # Contact-force polling (Isaac Core view)
        self._rb_view = None

    def _setup_contact_reporting(self):
        """Enable PhysX contact reports for bodies and subscribe to events.

        This is best-effort and aims to catch both self-collisions and collisions
        with the world (table/ground)."""
        try:
            import omni
            from pxr import PhysxSchema, UsdPhysics
            from omni.physx import get_physx_interface

            self._robot_prefix = str(self.stage_path)

            # Apply contact report API to all rigid bodies under the robot.
            for prim in self.stage.Traverse():
                p = prim.GetPath().pathString
                if not p.startswith(self._robot_prefix + '/'):
                    continue
                try:
                    if not UsdPhysics.RigidBodyAPI(prim):
                        continue
                except Exception:
                    # Some prims may not support the schema query.
                    continue
                try:
                    api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                    # Low threshold so we see contacts even for gentle taps.
                    if api and api.GetThresholdAttr():
                        api.GetThresholdAttr().Set(0.0)
                except Exception:
                    pass

            # Subscribe to contact reports.
            # API surface differs across Isaac Sim/Kit versions, so try a few.
            physx = get_physx_interface()
            try:
                from omni.physx import get_physx_simulation_interface
                physx_sim = get_physx_simulation_interface()
            except Exception:
                physx_sim = None

            def _extract_paths(evt):
                # Try to robustly extract prim paths for the 2 actors involved.
                a = b = None
                # Common attribute names (vary by Isaac/Kit versions).
                for k in ('actor0', 'body0', 'rigid_body0', 'prim_path0', 'path0'):
                    if hasattr(evt, k):
                        a = getattr(evt, k)
                        break
                for k in ('actor1', 'body1', 'rigid_body1', 'prim_path1', 'path1'):
                    if hasattr(evt, k):
                        b = getattr(evt, k)
                        break
                # Some versions deliver a dict-like event.
                if isinstance(evt, dict):
                    a = a or evt.get('actor0') or evt.get('body0') or evt.get('path0')
                    b = b or evt.get('actor1') or evt.get('body1') or evt.get('path1')
                # Convert to strings.
                try:
                    a = a.pathString  # pxr.Sdf.Path
                except Exception:
                    pass
                try:
                    b = b.pathString
                except Exception:
                    pass
                if a is not None:
                    a = str(a)
                if b is not None:
                    b = str(b)
                return a, b

            def _on_contact(evt):
                try:
                    a, b = _extract_paths(evt)
                    if not a or not b:
                        return
                    rp = self._robot_prefix
                    a_is = a.startswith(rp)
                    b_is = b.startswith(rp)
                    if not (a_is or b_is):
                        return
                    if a_is and b_is:
                        self._coll_self = True
                    else:
                        self._coll_world = True
                except Exception:
                    return

            # Store subscription handle so it doesn't get GC'd.
            sub = None
            if hasattr(physx, 'subscribe_contact_report_events'):
                sub = physx.subscribe_contact_report_events(_on_contact)
            elif physx_sim is not None and hasattr(physx_sim, 'subscribe_contact_report_events'):
                sub = physx_sim.subscribe_contact_report_events(_on_contact)
            else:
                # Dump available subscribe methods to the log for debugging.
                try:
                    avail = [n for n in dir(physx) if 'subscribe' in n]
                except Exception:
                    avail = []
                try:
                    avail_sim = [n for n in dir(physx_sim) if 'subscribe' in n] if physx_sim is not None else []
                except Exception:
                    avail_sim = []
                print(f'CONTACT_REPORT_NO_SUBSCRIBE physx={avail} sim={avail_sim}', flush=True)
                sub = None
            self._contact_sub = sub
            if self._contact_sub is not None:
                print('CONTACT_REPORT_SUBSCRIBED', flush=True)
        except Exception as e:
            print(f'CONTACT_REPORT_SETUP_FAILED {e}', flush=True)
            self._contact_sub = None

    def _clear_collisions(self):
        self._coll_self = False
        self._coll_world = False

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

        # DreamerV4: use the same URDF articulation as V3 (ensures joints are controllable)
        ws = "/home/r91/ws_xarm"
        urdf_path = f"{ws}/isaac_bridge/lite6_isaac.urdf"

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

        # Some Kit versions return a list of prim paths.
        if isinstance(stage_path, (list, tuple)):
            stage_path = stage_path[0] if stage_path else stage_path
        stage_path = str(stage_path)

        self.stage = omni.usd.get_context().get_stage()




        self.stage = omni.usd.get_context().get_stage()
        # Debug visual: small red cube to verify camera/render pipeline
        try:
            from pxr import UsdGeom, Sdf, Gf
            cube_path = Sdf.Path('/World/DebugCube')
            if not self.stage.GetPrimAtPath(cube_path):
                cube = UsdGeom.Cube.Define(self.stage, cube_path)
                cube.CreateSizeAttr(0.1)
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.30))
                prim = cube.GetPrim()
                if prim.HasAttribute('primvars:displayColor'):
                    prim.GetAttribute('primvars:displayColor').Set([Gf.Vec3f(1.0,0.0,0.0)])
                else:
                    prim.CreateAttribute('primvars:displayColor', Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(1.0,0.0,0.0)])
        except Exception:
            pass
        self.stage_path = stage_path

        # Create marker prims for fast visual success checks.
        self.ee_marker_xf = None
        self.target_marker_xf = None
        try:
            from pxr import UsdGeom, Gf
            # Ensure parent prims exist
            UsdGeom.Xform.Define(self.stage, Sdf.Path('/World/Markers'))

            def _make_sphere(path, rgb):
                prim = self.stage.GetPrimAtPath(Sdf.Path(path))
                if not prim:
                    sph = UsdGeom.Sphere.Define(self.stage, Sdf.Path(path))
                    sph.CreateRadiusAttr(0.015)
                    prim = sph.GetPrim()
                xf = UsdGeom.Xformable(prim)
                # Create a translate op if missing
                ops = xf.GetOrderedXformOps()
                if not ops:
                    op = xf.AddTranslateOp()
                else:
                    op = ops[0]
                # Color
                if prim.HasAttribute('primvars:displayColor'):
                    prim.GetAttribute('primvars:displayColor').Set([Gf.Vec3f(*rgb)])
                else:
                    prim.CreateAttribute('primvars:displayColor', Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(*rgb)])
                return op

            # Green EE marker, Blue target marker, Orange ball marker
            self.ee_marker_xf = _make_sphere('/World/Markers/EE', (0.0, 1.0, 0.0))
            self.target_marker_xf = _make_sphere('/World/Markers/Target', (0.2, 0.4, 1.0))
            self.ball_marker_xf = _make_sphere('/World/Markers/Ball', (1.0, 0.55, 0.10))
        except Exception:
            pass

        scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath("/physicsScene"))

        light = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/DistantLight"))
        light.CreateIntensityAttr(150)

        self.app.update()
        # Render background: black (improves contrast)
        try:
            import carb
            s = carb.settings.get_settings()
            # Try common clear color settings (varies by Kit version)
            s.set('/rtx/clearColor', [0.0, 0.0, 0.0, 1.0])
            s.set('/app/viewport/clearColor', [0.0, 0.0, 0.0, 1.0])
        except Exception:
            pass

        self.sim = SimulationContext(stage_units_in_meters=1.0)
        self.sim.initialize_physics()

        self.art = Articulation(prim_path=self.stage_path)
        self.art.initialize()
        if not self.art.handles_initialized:
            raise RuntimeError(f"{self.stage_path} is not an articulation")

        # Gripper: enable if joints are found.
        try:
            self.gripper.setup(self.art, period=int(os.environ.get('LITE6_GRIPPER_PERIOD', '50')))
            if self.gripper.enabled:
                print(f'GRIPPER_ENABLED joints={len(self.gripper.joint_ids)} period={self.gripper.period}', flush=True)
            else:
                print('GRIPPER_NOT_FOUND', flush=True)
        except Exception as e:
            print(f'GRIPPER_SETUP_FAILED {e}', flush=True)

        # Enable collision monitoring.
        # 1) Prefer per-step contact-force polling via an Isaac Core view (more robust across versions).
        # Delay view creation until after a few sim steps so prims are fully created.
        self._rb_view = None

        # 2) Also attempt PhysX contact report subscription (optional; may not exist in this build).
        self._setup_contact_reporting()

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

        # Create contact-force view after startup steps.
        try:
            from omni.isaac.core.prims import RigidPrimView
            # RigidPrimView globbing support varies; single-level wildcard is reliable.
            expr = f"{self.stage_path}/*"
            self._rb_view = RigidPrimView(
                prim_paths_expr=expr,
                name="lite6_rigid_bodies",
                track_contact_forces=True,
            )
            self._rb_view.initialize()
            print(f'CONTACT_MONITOR_VIEW_READY {expr}', flush=True)
        except Exception as e:
            self._rb_view = None
            print(f'CONTACT_MONITOR_VIEW_FAILED {e}', flush=True)

    def close(self):
        if self.app is not None:
            try:
                self.app.close()
            except Exception:
                pass
            self.app = None

    def _apply_q(self, q):
        # Use position targets to drive joints (more reliable than set_joint_positions alone).
        if hasattr(self.art, 'set_joint_position_targets'):
            try:
                self.art.set_joint_position_targets(q)
            except Exception:
                pass
        try:
            self.art.set_joint_positions(q)
        except Exception:
            pass

    def _randomize_target(self):
        self._rng.seed(int(time.time() * 1e6) % (2**32 - 1))

        # Optional curriculum radius around the center of the configured ranges.
        # Only affects x/y; z remains sampled from z_min/z_max.
        cx = 0.5 * (float(self.cfg.x_min) + float(self.cfg.x_max))
        cy = 0.5 * (float(self.cfg.y_min) + float(self.cfg.y_max))
        rad = float(getattr(self.cfg, 'target_radius', 0.0) or 0.0)
        if rad > 0:
            x_min, x_max = cx - rad, cx + rad
            y_min, y_max = cy - rad, cy + rad
        else:
            x_min, x_max = float(self.cfg.x_min), float(self.cfg.x_max)
            y_min, y_max = float(self.cfg.y_min), float(self.cfg.y_max)

        self.target = np.array([
            self._rng.uniform(x_min, x_max),
            self._rng.uniform(y_min, y_max),
            self._rng.uniform(float(self.cfg.z_min), float(self.cfg.z_max)),
        ], np.float32)

    def _randomize_ball(self):
        self._rng.seed(int(time.time() * 1e6) % (2**32 - 1))
        cx = 0.5 * (float(self.cfg.x_min) + float(self.cfg.x_max))
        cy = 0.5 * (float(self.cfg.y_min) + float(self.cfg.y_max))
        rad = float(getattr(self.cfg, 'target_radius', 0.0) or 0.0)
        if rad > 0:
            x_min, x_max = cx - rad, cx + rad
            y_min, y_max = cy - rad, cy + rad
        else:
            x_min, x_max = float(self.cfg.x_min), float(self.cfg.x_max)
            y_min, y_max = float(self.cfg.y_min), float(self.cfg.y_max)
        z = max(0.03, float(self.cfg.z_min) * 0.35)
        self.ball = np.array([
            self._rng.uniform(x_min, x_max),
            self._rng.uniform(y_min, y_max),
            z,
        ], np.float32)
        self.ball_initial_z = float(self.ball[2])
        self.target = self.ball.copy()
        self.ball_attached = False

    def _task_variant(self):
        return str(getattr(self.cfg, 'task_variant', 'reach') or 'reach')

    def _update_ball_pose(self, ee=None):
        if ee is None:
            ee = self._ee_pos()
        if self._task_variant() != 'ball_grasp':
            return
        if self.ball_attached:
            self.ball = np.array([
                float(ee[0]),
                float(ee[1]),
                float(ee[2]) - float(getattr(self.cfg, 'grasp_attach_z_offset', 0.02) or 0.02),
            ], np.float32)
        self.target = self.ball.copy()

    def _ee_pos(self):
        pos, _ = self._get_world_pose(self.ee_path)
        return np.array(pos, np.float32)

    def reset(self, logdir='', video=None, video_every=0, download=None, record_v4=None, cfg_patch=None):
        # Clear any latched collision flags from previous episodes.
        self._clear_collisions()

        # Apply per-run config patches from the RL side (safe, allowlisted fields only).
        cfg_patch = cfg_patch or {}
        try:
            if 'task_variant' in cfg_patch:
                self.cfg.task_variant = str(cfg_patch['task_variant'])
            if 'ball_radius' in cfg_patch:
                self.cfg.ball_radius = float(cfg_patch['ball_radius'])
            if 'grasp_dist_thresh' in cfg_patch:
                self.cfg.grasp_dist_thresh = float(cfg_patch['grasp_dist_thresh'])
            if 'lift_height_thresh' in cfg_patch:
                self.cfg.lift_height_thresh = float(cfg_patch['lift_height_thresh'])
            if 'action_scale' in cfg_patch:
                self.cfg.action_scale = float(cfg_patch['action_scale'])
            if 'reward_w_u' in cfg_patch:
                self.cfg.reward_w_u = float(cfg_patch['reward_w_u'])
            if 'reward_w_du' in cfg_patch:
                self.cfg.reward_w_du = float(cfg_patch['reward_w_du'])
            if 'target_radius' in cfg_patch:
                self.cfg.target_radius = float(cfg_patch['target_radius'])
        except Exception:
            pass

        # configure video capture on first reset
        if video is None:
            video = {}
        self.video.configure(logdir, video, video_every, download, record_v4)
        if self.video.enabled:
            print(f'VIDEO_ENABLED logdir={self.video.logdir} fps={self.video.fps} size={self.video.w}x{self.video.h} seconds={self.video.seconds}', flush=True)
        if self.video.enabled and self.video.annot is None:
            # Aim camera at the robot base (articulation root) if possible
            try:
                base, _ = self._get_world_pose(self.stage_path)
                bx, by, bz = float(base[0]), float(base[1]), float(base[2])
                # Side view (left): place camera at -Y, low height
                self.video.look = (bx, by, bz + 0.25)
                self.video.eye = (bx + 0.10, by - 1.20, bz + 0.25)
            except Exception:
                pass
            self.video.setup_rep(self.stage)
        self.video.reset_episode()

        self.t = 0
        self.q[:] = 0.0
        self._prev_action = np.zeros((6,), np.float32)
        self.ball_attached = False
        # V4: reset gripper to OPEN
        try:
            self.gripper.state = 0
            if self.gripper.enabled:
                self.gripper.apply(self.art, 0)
        except Exception:
            pass
        if self._task_variant() == 'ball_grasp':
            self._randomize_ball()
        else:
            self._randomize_target()
        self._apply_q(self.q)
        for _ in range(self.cfg.settle_steps):
            self.sim.step(render=self.video.enabled)

        ee = self._ee_pos()
        if self._task_variant() == 'ball_grasp':
            self._update_ball_pose(ee)
        dist = float(np.linalg.norm(ee - self.target))
        # Update marker positions (best-effort)
        try:
            if self.ee_marker_xf is not None:
                self.ee_marker_xf.Set((float(ee[0]), float(ee[1]), float(ee[2])))
            if self.target_marker_xf is not None:
                self.target_marker_xf.Set((float(self.target[0]), float(self.target[1]), float(self.target[2])))
            if self.ball_marker_xf is not None and self._task_variant() == 'ball_grasp':
                self.ball_marker_xf.Set((float(self.ball[0]), float(self.ball[1]), float(self.ball[2])))
        except Exception:
            pass

        # Capture after markers are placed
        self.video.capture()
        # Record DreamerV4 dataset step (initial observation)
        try:
            rgb = self.video.frames[-1] if self.video.frames else None
            self.video.record_step_v4(rgb, {
                'time': time.time(),
                'global_step': int(self.video.global_step),
                't': int(self.t),
                'q': [float(x) for x in self.q.tolist()],
                'ee_pos': [float(x) for x in ee.tolist()],
                'target_pos': [float(x) for x in self.target.tolist()],
                'action': None,
                'reward': float(-dist),
                'is_last': False,
                'is_terminal': False,
            })
        except Exception:
            pass

        if self.video.enabled:
            self.video.debug_lines = [
                f'step={self.video.global_step} t={self.t}',
                f'dist={dist:.3f}',
            ]
        return {
            'q': self.q.tolist(),
            'ee_pos': ee.tolist(),
            'target_pos': self.target.tolist(),
            'reward': float(-dist),
            'is_last': False,
            'is_terminal': False,
        }

    def step(self, action, global_step=None):
        # Clear collision flags for this step.
        # The contact callback may latch them during sim.step() calls below.
        self._clear_collisions()

        a = np.clip(np.array(action, np.float32), -1.0, 1.0)
        dq = a * float(self.cfg.action_scale)
        self.q = self.q + dq
        # Safety: clamp joints to a conservative range to prevent fold-up collapse.
        # (These are generic bounds; tune to your Lite6 URDF limits if needed.)
        self.q = np.clip(self.q, -2.8, 2.8)

        # Heuristic fold guard (self-collision prevention): terminate early when the
        # arm enters very folded configurations.
        if bool(getattr(self.cfg, 'fold_guard', True)):
            try:
                q1 = float(self.q[1])
                q2 = float(self.q[2])
                if abs(q1) > float(getattr(self.cfg, 'fold_q1_thresh', 2.2)) and abs(q2) > float(getattr(self.cfg, 'fold_q2_thresh', 2.2)):
                    self._coll_self = True
                    print(f'FOLD_GUARD_TRIGGER q1={q1:.3f} q2={q2:.3f} t={self.t} gstep={self.video.global_step}', flush=True)
            except Exception:
                pass

        self._apply_q(self.q)
        for _ in range(self.cfg.settle_steps):
            self.sim.step(render=self.video.enabled)

        # Poll contact forces (robust collision monitor).
        if bool(getattr(self.cfg, 'contact_monitor', True)) and self._rb_view is not None:
            try:
                forces = self._rb_view.get_net_contact_forces()
                # forces: (N,3)
                maxf = float(np.max(np.linalg.norm(forces, axis=-1))) if forces is not None and len(forces) else 0.0
                if maxf > float(getattr(self.cfg, 'contact_force_thresh', 1.0) or 1.0):
                    self._coll_world = True
            except Exception:
                pass

        self.t += 1
        if global_step is None:
            self.video.global_step += 1
        else:
            self.video.global_step = int(global_step)

        ee = self._ee_pos()

        # V4: gripper policy for task variants.
        gripper_cmd = None
        try:
            if self._task_variant() == 'ball_grasp':
                ball_dist = float(np.linalg.norm(ee - self.ball))
                if self.ball_attached:
                    self.gripper.state = 1
                elif ball_dist < float(getattr(self.cfg, 'grasp_dist_thresh', 0.04) or 0.04):
                    self.gripper.state = 1
                else:
                    self.gripper.state = 0
                gripper_cmd = int(self.gripper.state)
                if self.gripper.enabled:
                    self.gripper.apply(self.art, gripper_cmd)
            elif self.gripper.enabled:
                gripper_cmd = self.gripper.maybe_toggle(self.video.global_step)
                if gripper_cmd is not None:
                    self.gripper.apply(self.art, gripper_cmd)
        except Exception:
            pass

        if self._task_variant() == 'ball_grasp':
            ball_dist = float(np.linalg.norm(ee - self.ball))
            if (not self.ball_attached) and int(self.gripper.state) == 1 and ball_dist < float(getattr(self.cfg, 'grasp_dist_thresh', 0.04) or 0.04):
                self.ball_attached = True
            self._update_ball_pose(ee)
        dist = float(np.linalg.norm(ee - self.target))
        # Update marker positions (best-effort)
        try:
            if self.ee_marker_xf is not None:
                self.ee_marker_xf.Set((float(ee[0]), float(ee[1]), float(ee[2])))
            if self.target_marker_xf is not None:
                self.target_marker_xf.Set((float(self.target[0]), float(self.target[1]), float(self.target[2])))
            if self.ball_marker_xf is not None and self._task_variant() == 'ball_grasp':
                self.ball_marker_xf.Set((float(self.ball[0]), float(self.ball[1]), float(self.ball[2])))
        except Exception:
            pass

        self.video.capture()

        # Periodic clip saving (e.g., every 100 steps)
        if self.video.enabled and self.video.video_every > 0 and (self.video.global_step % self.video.video_every) == 0:
            mp4c = self.video.save_episode('dreamer_v4_clip')
            if mp4c and self.video.download_dir:
                try:
                    import shutil, os
                    d = os.path.expanduser(self.video.download_dir)
                    os.makedirs(d, exist_ok=True)
                    safe = self.video.download_prefix.replace('/', '_')
                    dst = os.path.join(d, f"{safe} - step_{self.video.global_step:09d}.mp4")
                    shutil.copy2(mp4c, dst)
                except Exception:
                    pass


        # Visual task metrics.
        vis_dist = self.video.last_vis_dist_px
        success_ee = dist < float(self.EE_SUCCESS_M)
        success_vis = (vis_dist is not None) and (vis_dist < float(self.VIS_SUCCESS_PX)) and (dist < float(self.VIS_GATE_M))
        success_grasp = False
        success_lift = False
        if self._task_variant() == 'ball_grasp':
            success_grasp = bool(self.ball_attached)
            success_lift = bool(self.ball_attached and float(self.ball[2]) > float(self.ball_initial_z) + float(getattr(self.cfg, 'lift_height_thresh', 0.08) or 0.08))

        if self.video.enabled:
            self.video.debug_lines = [
                f'step={self.video.global_step} t={self.t}',
                f'dist={dist:.3f}',
                f'vis_px={vis_dist:.1f}' if vis_dist is not None else 'vis_px=NA',
                f'|a|={float(np.linalg.norm(a)):.3f}',
                f'succ_ee={int(success_ee)} succ_vis={int(success_vis)} grasp={int(success_grasp)} lift={int(success_lift)}',
            ]

        # Collision detection (best-effort): terminate and penalize.
        coll_self = bool(getattr(self, '_coll_self', False))
        coll_world = bool(getattr(self, '_coll_world', False))
        coll = coll_self or coll_world

        task_success = success_lift if self._task_variant() == 'ball_grasp' else (success_ee or success_vis)
        done = (self.t >= self.cfg.episode_len) or task_success or (bool(getattr(self.cfg, 'collision_terminate', True)) and coll)

        mp4 = None
        if done:
            mp4 = self.video.save_episode('dreamer_v4_ep')

        if mp4:
            print(f'VIDEO_SAVED {mp4}', flush=True)

        # Reward: task + smoothness shaping + collision penalties
        w_u = float(getattr(self.cfg, 'reward_w_u', 0.0) or 0.0)
        w_du = float(getattr(self.cfg, 'reward_w_du', 0.0) or 0.0)
        du = a - getattr(self, '_prev_action', np.zeros_like(a))
        if self._task_variant() == 'ball_grasp':
            rew = float(-dist)
            if self.ball_attached:
                rew += 2.0
                rew += max(0.0, float(self.ball[2]) - float(self.ball_initial_z)) * 15.0
            if success_lift:
                rew += 10.0
        else:
            rew = float(-dist)
        rew -= w_u * float(np.sum(a * a)) + w_du * float(np.sum(du * du))

        if coll_self:
            # If fold guard triggered, allow overriding penalty.
            rew -= float(getattr(self.cfg, 'fold_penalty', None) or getattr(self.cfg, 'collision_penalty_self', 50.0) or 50.0)
        elif coll_world:
            rew -= float(getattr(self.cfg, 'collision_penalty_world', 20.0) or 20.0)

        self._prev_action = a.copy()

        # Telemetry: log joint/action/reward for supervisor debugging.
        try:
            if bool(getattr(self.cfg, 'telemetry_enabled', True)):
                every = int(getattr(self.cfg, 'telemetry_every', 5) or 5)
                if every <= 0:
                    every = 1
                if (self.video.global_step % every) == 0:
                    rec = {
                        'time': time.time(),
                        'global_step': int(self.video.global_step),
                        't': int(self.t),
                        'q': [float(x) for x in self.q.tolist()],
                        'a': [float(x) for x in a.tolist()],
                        'dist': float(dist),
                        'reward': float(rew),
                        'success_ee': bool(success_ee),
                        'success_vis': bool(success_vis),
                        'collision_self': bool(coll_self),
                        'collision_world': bool(coll_world),
                    }
                    p = str(getattr(self.cfg, 'telemetry_path', '/tmp/lite6_joint_telemetry.jsonl') or '/tmp/lite6_joint_telemetry.jsonl')
                    with open(p, 'a') as f:
                        f.write(json.dumps(rec) + '\n')
        except Exception:
            pass


        # Record DreamerV4 dataset step (obs+action+reward+done)
        try:
            rgb = self.video.frames[-1] if self.video.frames else None
            self.video.record_step_v4(rgb, {
                'time': time.time(),
                'global_step': int(self.video.global_step),
                't': int(self.t),
                'q': [float(x) for x in self.q.tolist()],
                'ee_pos': [float(x) for x in ee.tolist()],
                'target_pos': [float(x) for x in self.target.tolist()],
                'action': [float(x) for x in a.tolist()],
                'dist': float(dist),
                'vis_dist_px': None if vis_dist is None else float(vis_dist),
                'success_ee': bool(success_ee),
                'success_vis': bool(success_vis),
                'success_grasp': bool(success_grasp),
                'success_lift': bool(success_lift),
                'task_variant': self._task_variant(),
                'ball_pos': None if self._task_variant() != 'ball_grasp' else [float(x) for x in self.ball.tolist()],
                'collision_self': bool(coll_self),
                'collision_world': bool(coll_world),
                'reward': float(rew),
                'is_last': bool(done),
                'is_terminal': bool(coll),
                'gripper_cmd': None if gripper_cmd is None else int(gripper_cmd),
                'gripper_state': int(self.gripper.state),
            })
        except Exception:
            pass
        out = {
            'q': self.q.tolist(),
            'ee_pos': ee.tolist(),
            'target_pos': self.target.tolist(),
            'dist': float(dist),
            'vis_dist_px': None if vis_dist is None else float(vis_dist),
            'success_ee': bool(success_ee),
            'success_vis': bool(success_vis),
            'success_grasp': bool(success_grasp),
            'success_lift': bool(success_lift),
            'task_variant': self._task_variant(),
            'ball_pos': None if self._task_variant() != 'ball_grasp' else [float(x) for x in self.ball.tolist()],
            'collision_self': bool(coll_self),
            'collision_world': bool(coll_world),
            'reward': float(rew),
            'is_last': bool(done),
            # Treat collisions as terminal.
            'is_terminal': bool(coll),
        }
        if mp4:
            out['video_path'] = mp4
        return out


def serve(host='127.0.0.1', port=5555):
    # Note: Isaac can shutdown without a Python traceback.
    # This wrapper forces any uncaught exception to be logged.
    
    """Serve RPC requests.

    We must keep the Kit app updating even when idle; otherwise Isaac Sim may
    decide to shut down. So we run accept() with a timeout and call app.update().
    """
    sim = Lite6ReachSim(ReachConfig())
    sim.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(16)
    srv.settimeout(0.5)
    print(f"LITE6_WORKER_LISTEN {host}:{port}", flush=True)

    conn = None
    try:
        while True:
            # idle tick to keep Kit responsive
            try:
                sim.app.update()
            except Exception:
                pass

            if conn is None:
                try:
                    conn, addr = srv.accept()
                    print(f"LITE6_WORKER_CLIENT {addr}", flush=True)
                except socket.timeout:
                    continue
                except (OSError, ConnectionError):
                    continue

            try:
                r, _, _ = select.select([conn], [], [], 0.0)
                if not r:
                    continue
                msg = recv_msg(conn)
                cmd = msg.get('cmd')
                if cmd == 'reset':
                    send_msg(conn, sim.reset(
                        msg.get('logdir',''),
                        msg.get('video', {}),
                        msg.get('video_every', 0),
                        msg.get('download', None),
                        msg.get('record_v4', None),
                        msg.get('cfg', None),
                    ))
                elif cmd == 'step':
                    send_msg(conn, sim.step(msg['action'], msg.get('global_step', None)))
                elif cmd == 'save_video':
                    mp4 = sim.video.save_episode('dreamer_v4_clip')
                    send_msg(conn, {'ok': True, 'video_path': mp4})
                elif cmd == 'close':
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                else:
                    send_msg(conn, {'error': f'unknown cmd {cmd}'})
            except (ConnectionError, OSError):
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass
                conn = None
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print('WORKER_FATAL', repr(e), flush=True)
        traceback.print_exc()
        raise

        pass
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
    try:
        faulthandler.enable()
    except Exception:
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5555)
    args = ap.parse_args()
    serve(args.host, args.port)
