from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class JointSpec:
    name: str
    joint_type: str  # 'revolute'|'prismatic'|'fixed'
    lower_limit: float
    upper_limit: float
    max_velocity: float = 0.0
    max_effort: float = 0.0


@dataclass
class RobotSpec:
    name: str
    manufacturer: str
    model_id: str
    dof: int
    reach_m: float
    payload_kg: float
    urdf_path: str
    urdf_url: Optional[str]
    local_usd_path: str
    joints: List[JointSpec]
    confidence: float = 0.0
    sources_used: List[str] = None
