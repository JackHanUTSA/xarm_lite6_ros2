from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .camera_manager import CameraManager
from .observation_rules import ObservationRulesEngine, MotionRule
from .motion_observer import MotionObserver
from .motion_analyzer import MotionAnalyzer
from .world_reconstruction import WorldReconstructionEngine
from .urdf_builder import URDFBuilder
from .robot_interface import build_robot_interface
from .command_motion_dataset import CommandMotionDataset

from ..robot_discovery.robot_spec import RobotSpec, JointSpec


@dataclass
class RealWorldIntegrator:
    """Orchestrates: cameras → motion capture → 3-D reconstruction → URDF build → (USD) → RobotSpec."""

    rtsp_urls: List[str]
    calib_dir: str = "./calib"
    assets_dir: str = "./robot_assets"
    n_record_eps: int = 500
    n_recon_eps: int = 100

    def run(
        self,
        robot_interface: str,
        robot_name: str = "reconstructed_arm",
        extra_rules: Optional[List[MotionRule]] = None,
        *,
        skip_cameras: bool = False,
        skip_recording: bool = False,
    ) -> RobotSpec:
        print(f"[RealWorldIntegrator] Starting pipeline for: {robot_interface}")

        cam_mgr = None
        if not skip_cameras:
            cam_mgr = CameraManager(rtsp_urls=self.rtsp_urls, calib_dir=self.calib_dir)

        robot = build_robot_interface(robot_interface)

        rules = ObservationRulesEngine(extra_rules=extra_rules)
        rules.print_rules()

        assets_dir = Path(self.assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = str(assets_dir / "motion_dataset.h5")
        if skip_recording or cam_mgr is None:
            # Defer recording; create an empty dataset placeholder path.
            dataset = CommandMotionDataset(dataset_path)
        else:
            observer = MotionObserver(
                cam_mgr,
                rules,
                robot,
                dataset_path=dataset_path,
            )
            dataset = observer.record_exploration(n_episodes=self.n_record_eps)

        # If we skipped recording, we don't have a dataset/URDF yet. Return a minimal spec.
        if skip_recording:
            dof = robot.dof
            joints = [
                JointSpec(
                    name=f"joint_{i}",
                    joint_type="revolute",
                    lower_limit=-3.14,
                    upper_limit=3.14,
                    max_velocity=2.175,
                    max_effort=87.0,
                )
                for i in range(dof)
            ]
            return RobotSpec(
                name=robot_name,
                manufacturer="real_world",
                model_id=robot_name,
                dof=dof,
                reach_m=0.85,
                payload_kg=5.0,
                urdf_path="",
                urdf_url=None,
                local_usd_path="",
                joints=joints,
                confidence=0.0,
                sources_used=["ros2_interface_only"],
            )

        analyzer = MotionAnalyzer(dataset)
        vlm_kin = analyzer.analyse(n_episodes=30)

        recon = WorldReconstructionEngine(dataset, cam_mgr, output_dir=str(assets_dir / "reconstruction"))
        recon.reconstruct(n_episodes=self.n_recon_eps)

        builder = URDFBuilder(recon, dataset, vlm_kin, output_dir=str(assets_dir), robot_name=robot_name)
        urdf_path = builder.build()

        # USD conversion is intentionally stubbed here; integrate inside Isaac Sim python.
        usd_path = urdf_path

        jstats = dataset.joint_limit_stats()
        dof = len(jstats["min"])
        joints = [
            JointSpec(
                name=f"joint_{i}",
                joint_type="revolute",
                lower_limit=float(jstats["min"][i]),
                upper_limit=float(jstats["max"][i]),
                max_velocity=2.175,
                max_effort=87.0,
            )
            for i in range(dof)
        ]

        spec = RobotSpec(
            name=robot_name,
            manufacturer="real_world",
            model_id=robot_name,
            dof=dof,
            reach_m=float(vlm_kin.get("workspace_radius_m", 0.85) or 0.85),
            payload_kg=5.0,
            urdf_path=urdf_path,
            urdf_url=None,
            local_usd_path=usd_path,
            joints=joints,
            confidence=float(vlm_kin.get("confidence", 0.0) or 0.0),
            sources_used=["real_world_cameras", "motion_observer"],
        )

        print(f"[RealWorldIntegrator] Done — RobotSpec DOF={dof}")
        return spec
