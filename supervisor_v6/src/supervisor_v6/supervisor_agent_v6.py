from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .real_world.real_world_integrator import RealWorldIntegrator
from .robot_discovery.robot_spec import RobotSpec


@dataclass
class SupervisorAgentV6:
    """Top-level facade for Supervisor Agent V6.

    Spec: `supervisor.setup_from_real_world('robot_interface_name')`.
    """

    rtsp_urls: Optional[List[str]] = None
    calib_dir: str = "./calib"
    assets_dir: str = "./robot_assets"
    n_record_eps: int = 500
    n_recon_eps: int = 100

    def setup_from_real_world(
        self,
        robot_interface_name: str,
        robot_name: str = "reconstructed_arm",
        *,
        skip_cameras: bool = False,
        skip_recording: bool = False,
    ) -> RobotSpec:
        integrator = RealWorldIntegrator(
            rtsp_urls=self.rtsp_urls or [],
            calib_dir=self.calib_dir,
            assets_dir=self.assets_dir,
            n_record_eps=self.n_record_eps,
            n_recon_eps=self.n_recon_eps,
        )
        return integrator.run(
            robot_interface=robot_interface_name,
            robot_name=robot_name,
            skip_cameras=skip_cameras,
            skip_recording=skip_recording,
        )
