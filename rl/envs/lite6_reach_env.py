"""Lite6 Reach environment in Isaac Sim (headless-friendly).

Obs (simple):
  - joint positions (6)
  - end-effector position (3)
  - target position (3)
Action:
  - joint position deltas (6), clipped

This file provides a minimal Gymnasium-like API: reset(), step(action).

NOTE: This is a scaffold. It expects an Isaac Sim app + stage with the Lite6 articulation loaded.
We will wire the articulation + ee frame lookup in the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class WorkspaceBounds:
    x: Tuple[float, float] = (0.20, 0.45)
    y: Tuple[float, float] = (-0.20, 0.20)
    z: Tuple[float, float] = (0.12, 0.40)


@dataclass
class EnvConfig:
    episode_len: int = 200
    action_delta_limit: float = 0.10  # rad per step
    joint_limit: float = 3.14  # crude clip; will refine from articulation
    bounds: WorkspaceBounds = WorkspaceBounds()
    dt: float = 1.0 / 60.0


class Lite6ReachEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.t = 0
        self.target = np.zeros(3, dtype=np.float32)

        # placeholders to be wired to Isaac objects
        self._art = None
        self._joint_names = None
        self._ee_prim_path = None

    def seed(self, seed: int):
        self.np_random = np.random.default_rng(seed)

    def set_isaac_handles(self, articulation, joint_names, ee_prim_path: str):
        """Call after importing the robot in Isaac."""
        self._art = articulation
        self._joint_names = list(joint_names)
        self._ee_prim_path = ee_prim_path

    def _sample_target(self) -> np.ndarray:
        b = self.cfg.bounds
        return np.array([
            self.np_random.uniform(*b.x),
            self.np_random.uniform(*b.y),
            self.np_random.uniform(*b.z),
        ], dtype=np.float32)

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        if not hasattr(self, 'np_random'):
            self.seed(0)
        self.target = self._sample_target()

        # TODO: set robot to home pose in Isaac
        obs = self._get_obs()
        return obs

    def step(self, action: np.ndarray):
        assert action.shape == (6,)
        self.t += 1

        # TODO: apply action to articulation joint positions
        # - read current joints
        # - add clipped deltas
        # - write targets
        # - step sim

        obs = self._get_obs()
        ee = obs['ee_pos']
        dist = float(np.linalg.norm(ee - self.target))
        reward = -dist
        done = self.t >= self.cfg.episode_len
        info = {'dist': dist}
        return obs, reward, done, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # TODO: pull from Isaac articulation + ee prim
        q = np.zeros(6, dtype=np.float32)
        ee = np.zeros(3, dtype=np.float32)
        obs = {
            'q': q,
            'ee_pos': ee,
            'target_pos': self.target.copy(),
            'obs': np.concatenate([q, ee, self.target], axis=0),
        }
        return obs
