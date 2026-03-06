from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


class RobotInterface(ABC):
    @property
    @abstractmethod
    def dof(self) -> int: ...

    @abstractmethod
    def read_joint_states(self) -> np.ndarray: ...

    @abstractmethod
    def send_command(self, delta: np.ndarray): ...

    def get_teleop_command(self) -> np.ndarray:
        return np.zeros(self.dof)


class ROS2RobotInterface(RobotInterface):
    """ROS2 adapter.

    Expects:
    - JointState on <namespace>/joint_states
    - JointTrajectory on <namespace>/joint_command

    NOTE: This is a scaffold. You must ensure rclpy init/spin model is appropriate
    for your process (single-threaded vs executor).
    """

    def __init__(self, namespace: str):
        try:
            import rclpy
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "rclpy is not available in this Python env. Run inside a ROS2-sourced environment "
                "(e.g., `source /opt/ros/<distro>/setup.zsh`) or install rclpy for Python 3.10."
            ) from e

        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory

        self._rclpy = rclpy
        self._JointState = JointState
        self._JointTrajectory = JointTrajectory

        rclpy.init(args=None)
        self.node = rclpy.create_node("supervisor_v6_observer")
        self._js: Optional[np.ndarray] = None
        self.namespace = namespace

        self.sub = self.node.create_subscription(
            JointState,
            f"{namespace}/joint_states",
            lambda msg: setattr(self, "_js", np.array(msg.position, dtype=np.float32)),
            10,
        )
        self.pub = self.node.create_publisher(JointTrajectory, f"{namespace}/joint_command", 10)

    @property
    def dof(self) -> int:
        # Don't hang forever if the joint_states topic isn't publishing.
        import time

        t0 = time.time()
        while self._js is None and (time.time() - t0) < 2.0:
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
        if self._js is None:
            # Reasonable default; override in your robot-specific adapter later.
            return 7
        return int(len(self._js))

    def read_joint_states(self) -> np.ndarray:
        self._rclpy.spin_once(self.node, timeout_sec=0.01)
        if self._js is None:
            return np.zeros(7, dtype=np.float32)
        return self._js

    def send_command(self, delta: np.ndarray):
        from trajectory_msgs.msg import JointTrajectoryPoint

        msg = self._JointTrajectory()
        pt = JointTrajectoryPoint()
        current = self.read_joint_states()
        pt.positions = (current + delta).tolist()
        msg.points = [pt]
        self.pub.publish(msg)


def build_robot_interface(robot_interface: str) -> RobotInterface:
    """Factory.

    Conventions:
    - 'ros2:/namespace'
    - '/namespace' (treated as ROS2)
    """
    if robot_interface.startswith("ros2:"):
        ns = robot_interface.split("ros2:", 1)[1]
        return ROS2RobotInterface(namespace=ns)
    # default ROS2
    return ROS2RobotInterface(namespace=robot_interface)
