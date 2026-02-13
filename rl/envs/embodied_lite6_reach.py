"""Embodied environment wrapper for Lite6 reach in Isaac.

This is a minimal adapter so DreamerV3 can call the environment.
We'll implement the actual Isaac step/read in a separate process runner.

For now, this file defines the spec and a placeholder class to be filled.
"""

import numpy as np


class Lite6ReachEmbodied:
    """Placeholder. Will be implemented to use Isaac Sim for stepping."""

    def __init__(self, episode_len=200):
        self.episode_len = episode_len
        self.t = 0

    @property
    def obs_space(self):
        # q(6) + ee(3) + target(3)
        return {
            'obs': (12,),
        }

    @property
    def act_space(self):
        return {
            'action': (6,),
        }

    def reset(self):
        self.t = 0
        obs = np.zeros((12,), np.float32)
        return {'obs': obs, 'is_first': True, 'is_last': False, 'is_terminal': False}

    def step(self, action):
        self.t += 1
        obs = np.zeros((12,), np.float32)
        done = self.t >= self.episode_len
        return {
            'obs': obs,
            'reward': 0.0,
            'is_first': False,
            'is_last': done,
            'is_terminal': False,
        }
