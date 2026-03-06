from __future__ import annotations

import time
from typing import List

import numpy as np

from .camera_manager import CameraManager
from .observation_rules import ObservationRulesEngine
from .command_motion_dataset import CommandMotionDataset
from .robot_interface import RobotInterface


class MotionObserver:
    """Orchestrates motion capture: issues commands, reads joint states, grabs synced frames."""

    def __init__(
        self,
        camera_manager: CameraManager,
        rules_engine: ObservationRulesEngine,
        robot_interface: RobotInterface,
        dataset_path: str = "./motion_dataset.h5",
        episode_dt_s: float = 2.0,
        record_hz: float = 30.0,
    ):
        self.cam = camera_manager
        self.rules = rules_engine
        self.robot = robot_interface
        self.dataset = CommandMotionDataset(dataset_path)
        self.ep_dt = episode_dt_s
        self.hz = record_hz
        self.dt = 1.0 / record_hz

    def record_exploration(self, n_episodes: int = 500, strategy: str = "random_joint_delta") -> CommandMotionDataset:
        dof = self.robot.dof
        recorded = 0
        while recorded < n_episodes:
            if strategy == "random_joint_delta":
                cmd = np.random.uniform(-0.3, 0.3, dof)
            elif strategy == "sinusoidal":
                t = recorded * self.ep_dt
                cmd = 0.3 * np.sin(0.5 * t + np.linspace(0, np.pi, dof))
            else:
                cmd = self.robot.get_teleop_command()

            joint_states: List[np.ndarray] = []
            frames_list = []

            t0 = time.monotonic()
            self.robot.send_command(cmd)
            while time.monotonic() - t0 < self.ep_dt:
                js = self.robot.read_joint_states()
                synced = self.cam.grab_synced()
                joint_states.append(js)
                frames_list.append(synced)
                time.sleep(self.dt)

            self.robot.send_command(np.zeros(dof))

            js_arr = np.stack(joint_states, axis=0)
            frames0 = frames_list[0].frames if frames_list else {}
            valid, failed = self.rules.validate(cmd, js_arr, frames0)
            if valid:
                self.dataset.store_episode(cmd, js_arr, frames_list, episode_id=recorded)
                recorded += 1
            else:
                print(f"[MotionObserver] Episode skipped — failed rules: {failed}")

        return self.dataset
