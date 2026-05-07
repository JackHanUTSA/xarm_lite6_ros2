import os
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

# Headless config
CONFIG = {"renderer": "RayTracedLighting", "headless": True}

ROBOT_PRIM_PATH = "/Lite6"

# Start Isaac Sim app
simulation_app = SimulationApp(CONFIG)

import carb
import omni
import omni.graph.core as og
import omni.kit.commands
import usdrt.Sdf

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import extensions, prims
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics

# Enable ROS2 bridge
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

# --- Import URDF ---
ws_xarm = Path(os.path.expanduser("~/ws_xarm"))
urdf_path = ws_xarm / "isaac_bridge" / "lite6_isaac.urdf"
if not urdf_path.exists():
    carb.log_error(f"URDF not found: {urdf_path}. Run gen_lite6_urdf.sh first.")
    simulation_app.close()
    sys.exit(1)

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = True   # fixed base arm
import_config.distance_scale = 1

status, stage_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=str(urdf_path),
    import_config=import_config,
    get_articulation_root=True,
)
if not status:
    carb.log_error("URDF import failed")
    simulation_app.close()
    sys.exit(1)

# Move imported prim to a known path if needed
# stage_path is the articulation root path returned by importer

stage = omni.usd.get_context().get_stage()

# Physics scene
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")

# Ground
omni.kit.commands.execute(
    "AddGroundPlaneCommand",
    stage=stage,
    planePath="/groundPlane",
    axis="Z",
    size=10.0,
    position=Gf.Vec3f(0, 0, 0),
    color=Gf.Vec3f(0.5),
)

# Light
light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
light.CreateIntensityAttr(500)

# Create ROS ActionGraph
GRAPH_PATH = "/ActionGraph"

og.Controller.edit(
    {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
            ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
            ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
            ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
            ("Context.outputs:context", "PublishJointState.inputs:context"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("ArticulationController.inputs:robotPath", stage_path),
            ("PublishJointState.inputs:topicName", "isaac_joint_states"),
            ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
            ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(stage_path)]),
        ],
    },
)

simulation_app.update()

# Simulation loop
simulation_context = SimulationContext(stage_units_in_meters=1.0)
simulation_context.initialize_physics()
simulation_context.play()

carb.log_info(f"ROS2 bridge ready. Publishing JointState on /isaac_joint_states, subscribing commands on /isaac_joint_commands")
print("ISAAC_ROS_READY")

max_steps = int(os.environ.get("ISAAC_MAX_STEPS", "0"))
steps = 0

while simulation_app.is_running():
    if max_steps and steps >= max_steps:
        break
    simulation_context.step(render=False)
    simulation_app.update()
    og.Controller.set(og.Controller.attribute(f"{GRAPH_PATH}/OnImpulseEvent.state:enableImpulse"), True)
    steps += 1

simulation_context.stop()
simulation_app.close()
