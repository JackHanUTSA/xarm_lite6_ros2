#!/usr/bin/env python3
"""Smoke test: load Lite6 URDF into Isaac Sim headless and run random actions.

Prints distance-to-target over steps.

This is a stepping stone to DreamerV3 training.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# Isaac Sim bootstrap
from isaacsim import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": True}

WS = Path(os.path.expanduser("~/ws_xarm"))
URDF_PATH = WS / "isaac_bridge" / "lite6_isaac.urdf"

ROBOT_TOPIC_HINT = "Lite6"
EE_LINK = "link_eef"  # auto-chosen from URDF


def main():
    if not URDF_PATH.exists():
        print(f"URDF not found: {URDF_PATH}")
        return 2

    app = SimulationApp(CONFIG)

    import omni
    import omni.kit.commands
    import carb
    from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
    from omni.isaac.core import SimulationContext

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

    # Physics scene
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))

    # Light
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    light.CreateIntensityAttr(500)

    # Start sim
    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()

    # TODO: Find articulation + joints + ee prim and implement stepping.
    # For now just step a bit to ensure import is stable.
    for i in range(120):
        sim.step(render=False)
        app.update()

    sim.stop()
    app.close()
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
