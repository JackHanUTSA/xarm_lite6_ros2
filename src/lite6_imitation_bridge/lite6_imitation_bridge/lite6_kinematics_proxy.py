from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Lite6KinematicsProxy:
    workspace_min: np.ndarray = field(default_factory=lambda: np.array([0.20, -0.20, 0.05], dtype=float))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.60, 0.30, 0.30], dtype=float))
    nominal_joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-np.pi, np.pi),
            "joint_2": (-2.0, 2.0),
            "joint_3": (-2.6, 2.6),
            "joint_4": (-np.pi, np.pi),
            "joint_5": (-2.0, 2.0),
            "joint_6": (-np.pi, np.pi),
        }
    )

    def __post_init__(self) -> None:
        workspace_min = np.asarray(self.workspace_min, dtype=float)
        workspace_max = np.asarray(self.workspace_max, dtype=float)
        if workspace_min.shape != (3,) or workspace_max.shape != (3,):
            raise ValueError("workspace_min and workspace_max must have shape (3,)")
        if not np.all(np.isfinite(workspace_min)) or not np.all(np.isfinite(workspace_max)):
            raise ValueError("workspace bounds must contain only finite values")
        if np.any(workspace_min > workspace_max):
            raise ValueError("workspace_min must be <= workspace_max elementwise")
        object.__setattr__(self, "workspace_min", workspace_min)
        object.__setattr__(self, "workspace_max", workspace_max)

    def clamp_position(self, position: np.ndarray) -> tuple[np.ndarray, bool]:
        position_array = np.asarray(position, dtype=float)
        if position_array.shape != (3,):
            raise ValueError("position must have shape (3,)")
        clamped = np.clip(position_array, self.workspace_min, self.workspace_max)
        return clamped, not np.allclose(clamped, position_array)

    def is_within_workspace(self, position: np.ndarray) -> bool:
        position_array = np.asarray(position, dtype=float)
        if position_array.shape != (3,):
            raise ValueError("position must have shape (3,)")
        return bool(np.all(position_array >= self.workspace_min) and np.all(position_array <= self.workspace_max))

    def validate_target(self, position: np.ndarray, approach_direction: np.ndarray) -> dict[str, Any]:
        position_array = np.asarray(position, dtype=float)
        approach_array = np.asarray(approach_direction, dtype=float)
        if position_array.shape != (3,) or approach_array.shape != (3,):
            raise ValueError("position and approach_direction must have shape (3,)")
        direction_norm = float(np.linalg.norm(approach_array))
        return {
            "workspace_valid": self.is_within_workspace(position_array),
            "approach_valid": np.isfinite(direction_norm) and direction_norm > 0.0,
            "joint_limits": self.nominal_joint_limits,
        }
