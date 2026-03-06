from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


Mode = Literal["dryrun", "capture", "reconstruct", "build_urdf", "isaac", "full"]


class CameraOptions(BaseModel):
    mode: Literal["auto", "manual"] = "auto"
    rtsp_urls: List[str] = Field(default_factory=list)


class RecordingOptions(BaseModel):
    n_episodes: int = 500
    episode_dt_s: float = 2.0
    record_hz: float = 30.0
    strategy: str = "random_joint_delta"


class ReconstructionOptions(BaseModel):
    voxel_m: float = 0.005
    tri_target: int = 20000
    max_links: int = 8


class UrdOptions(BaseModel):
    density_kg_m3: float = 2700.0
    collision_tri_target: int = 500


class IsaacOptions(BaseModel):
    enable: bool = True
    merge_fixed_joints: bool = True


class JobOptions(BaseModel):
    cameras: CameraOptions = Field(default_factory=CameraOptions)
    recording: RecordingOptions = Field(default_factory=RecordingOptions)
    rules_profile: str = "default"
    reconstruction: ReconstructionOptions = Field(default_factory=ReconstructionOptions)
    urdf: UrdOptions = Field(default_factory=UrdOptions)
    isaac: IsaacOptions = Field(default_factory=IsaacOptions)


class CreateJobRequest(BaseModel):
    robot: str = "ros2:/xarm"
    name: str = "reconstructed_arm"
    mode: Mode = "full"
    out_dir: str = "robot_assets/reconstructed_arm"
    options: JobOptions = Field(default_factory=JobOptions)


class Artifact(BaseModel):
    name: str
    path: str
    mime: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "finished", "error", "stopped"] = "queued"
    stage: str = "init"
    message: str = ""
    pct: Optional[float] = None
    out_dir: str
    artifacts: List[Artifact] = Field(default_factory=list)
    error: Optional[str] = None


class StopJobResponse(BaseModel):
    job_id: str
    ok: bool
