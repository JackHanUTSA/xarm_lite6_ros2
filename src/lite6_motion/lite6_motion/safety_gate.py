from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class MotionLimits:
    joint_position_limit: float = 3.14
    stale_state_timeout_sec: float = 0.5


@dataclass
class RobotHealth:
    enabled: bool
    has_error: bool
    mode: int
    state: int
    joint_names: List[str]
    joint_positions: List[float]
    last_state_age_sec: float


class SafetyGate:
    def __init__(self, limits: MotionLimits | None = None):
        self.limits = limits or MotionLimits()

    def state_is_fresh(self, health: RobotHealth) -> bool:
        return health.last_state_age_sec <= self.limits.stale_state_timeout_sec

    def validate_joint_targets(self, joint_targets: List[float]) -> bool:
        if len(joint_targets) != 6:
            return False
        return all(abs(value) <= self.limits.joint_position_limit for value in joint_targets)

    def can_execute_motion(self, health: RobotHealth, joint_targets: List[float]) -> Tuple[bool, str]:
        if not health.enabled:
            return False, 'motion not enabled'
        if health.has_error:
            return False, 'robot has active error'
        if not self.state_is_fresh(health):
            return False, 'state is stale'
        if health.mode != 0:
            return False, 'robot mode is not ready'
        if health.state != 0:
            return False, 'robot state is not ready'
        if not self.validate_joint_targets(joint_targets):
            return False, 'joint target exceeds configured limits'
        return True, 'ok'

    def build_status(self, health: RobotHealth) -> dict:
        ready, reason = self.can_execute_motion(health, health.joint_positions[:6] if health.joint_positions else [0.0] * 6)
        return {
            'ready': ready,
            'reason': reason,
            'enabled': health.enabled,
            'has_error': health.has_error,
            'mode': health.mode,
            'state': health.state,
            'joint_count': len(health.joint_positions),
            'joint_names': list(health.joint_names),
            'joint_positions': list(health.joint_positions),
            'last_state_age_sec': health.last_state_age_sec,
        }
