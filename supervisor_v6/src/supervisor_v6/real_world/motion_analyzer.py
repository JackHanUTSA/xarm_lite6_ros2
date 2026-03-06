from __future__ import annotations

from typing import Optional

import numpy as np

from .command_motion_dataset import CommandMotionDataset, Episode


class MotionAnalyzer:
    """Uses a VLM to extract kinematic info from motion episode frames.

    Scaffold: provides method signatures and a placeholder merge.
    """

    ANALYSIS_PROMPT = """
You are a robot kinematics expert. Analyse these images of a robot arm in motion
and return a JSON object with these fields:
{
  'estimated_dof': int,
  'link_lengths_m': [float, ...],
  'end_effector_type': str,
  'base_mount': str,
  'workspace_radius_m': float,
  'observed_joint_axes': ['revolute'|'prismatic', ...],
  'confidence': float
}
Return ONLY valid JSON.
""".strip()

    def __init__(self, dataset: CommandMotionDataset):
        self.dataset = dataset

    def analyse(self, n_episodes: int = 20) -> dict:
        # TODO: integrate anthropic client or other VLM.
        # Placeholder: just return joint-based DOF guess.
        try:
            ep0 = self.dataset.load_episode(0)
            dof = int(ep0.joint_states.shape[1])
        except Exception:
            dof = None
        return {
            "estimated_dof": dof,
            "workspace_radius_m": 0.85,
            "link_lengths_m": None,
            "end_effector_type": None,
            "confidence": 0.0,
            "n_analyses": 0,
        }

    def _sample_frames(self, ep: Episode, n: int) -> list:
        cam_id = next(iter(ep.images))
        frames = ep.images[cam_id]
        idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in idxs]

    def _call_vlm(self, imgs: list) -> Optional[dict]:
        raise NotImplementedError
