#!/usr/bin/env python3
"""Rollout smoke test for Lite6 reach task in Isaac Sim headless.

Validates sim control loop before DreamerV3 training.

Obs (simple): q(6) + ee_pos(3) + target(3)
Action: joint position deltas (6)
Episode: 200 steps
"""

import os
from pathlib import Path

import numpy as np

import builtins
print = lambda *a, **k: builtins.print(*a, **k, flush=True)


from isaacsim import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": True}

WS = Path(os.path.expanduser("~/ws_xarm"))
URDF_PATH = WS / "isaac_bridge" / "lite6_isaac.urdf"

EE_LINK_NAME = "link_eef"

BOUNDS = {
    "x": (0.20, 0.45),
    "y": (-0.20, 0.20),
    "z": (0.12, 0.40),
}


def find_prim_path_by_suffix(stage, suffix: str):
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if p.endswith("/" + suffix):
            return p
    return None


def get_joint_names(art) -> list:
    # Isaac versions differ; try common accessors.
    for attr in ("joint_names", "dof_names"):
        if hasattr(art, attr):
            v = getattr(art, attr)
            if isinstance(v, (list, tuple)) and v:
                return list(v)
    for fn in ("get_joint_names", "get_dof_names"):
        if hasattr(art, fn):
            try:
                v = getattr(art, fn)()
                if isinstance(v, (list, tuple)) and v:
                    return list(v)
            except Exception:
                pass
    return []


def main():
    if not URDF_PATH.exists():
        print(f"URDF not found: {URDF_PATH}")
        return 2

    app = SimulationApp(CONFIG)

    import carb
    import omni
    import omni.kit.commands
    from omni.isaac.core import SimulationContext
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.xforms import get_world_pose
    from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics

    # Import URDF
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1

    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(URDF_PATH),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        carb.log_error("URDF import failed")
        app.close()
        return 2

    stage = omni.usd.get_context().get_stage()

    # Physics scene (must exist before Articulation.initialize)
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))

    # Light
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    light.CreateIntensityAttr(500)

    # Make sure stage updates are applied
    app.update()

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()

    # Articulation init
    art = Articulation(prim_path=stage_path)
    art.initialize()
    if not art.handles_initialized:
        print(f"ERROR: {stage_path} is not an articulation")
        app.close()
        return 2

    ee_path = find_prim_path_by_suffix(stage, EE_LINK_NAME)
    if ee_path is None:
        print(f"ERROR: could not find EE prim ending with /{EE_LINK_NAME}")
        app.close()
        return 2

    joint_names = get_joint_names(art)
    dof = len(joint_names)

    print("ART_PATH", stage_path)
    print("EE_PATH", ee_path)
    print("JOINT_NAMES_LEN", dof)
    print("JOINT_NAMES", joint_names)

    # Use first 6 joints (Lite6)
    idx = list(range(6))

    rng = np.random.default_rng(0)
    target = np.array([
        rng.uniform(*BOUNDS['x']),
        rng.uniform(*BOUNDS['y']),
        rng.uniform(*BOUNDS['z']),
    ], dtype=np.float32)

    sim.play()

    # Warm-up
    for _ in range(5):
        sim.step(render=False)
        app.update()

    N = 200
    delta_lim = 0.10
    dists = []

    for t in range(N):
        q_full = np.array(art.get_joint_positions(), dtype=np.float32)
        q = q_full[idx]
        a = rng.uniform(-delta_lim, delta_lim, size=(6,)).astype(np.float32)
        q2 = np.clip(q + a, -3.14, 3.14)
        q_full[idx] = q2

        # set desired joints
        art.set_joint_positions(q_full)

        sim.step(render=False)
        app.update()

        pos, _ = get_world_pose(ee_path)
        ee = np.array(pos, dtype=np.float32)
        dist = float(np.linalg.norm(ee - target))
        dists.append(dist)

        if (t + 1) % 50 == 0:
            print(f"step={t+1} dist={dist:.4f}")

    dists = np.array(dists, dtype=np.float32)
    print("ROLL_OUT_DONE")
    print("dist_min", float(dists.min()))
    print("dist_mean", float(dists.mean()))
    print("dist_last", float(dists[-1]))

    sim.stop()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
