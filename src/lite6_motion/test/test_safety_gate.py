from lite6_motion.safety_gate import MotionLimits, RobotHealth, SafetyGate


def test_robot_cannot_move_when_state_is_stale():
    gate = SafetyGate(MotionLimits(joint_position_limit=3.14, stale_state_timeout_sec=0.5))
    health = RobotHealth(
        enabled=True,
        has_error=False,
        mode=0,
        state=0,
        joint_names=[f"joint{i}" for i in range(1, 7)],
        joint_positions=[0.0] * 6,
        last_state_age_sec=2.0,
    )

    allowed, reason = gate.can_execute_motion(health, [0.0] * 6)

    assert allowed is False
    assert reason == "state is stale"


def test_robot_rejects_joint_targets_outside_limit():
    gate = SafetyGate(MotionLimits(joint_position_limit=3.14, stale_state_timeout_sec=0.5))
    health = RobotHealth(
        enabled=True,
        has_error=False,
        mode=0,
        state=0,
        joint_names=[f"joint{i}" for i in range(1, 7)],
        joint_positions=[0.0] * 6,
        last_state_age_sec=0.1,
    )

    allowed, reason = gate.can_execute_motion(health, [0.0, 0.0, 0.0, 0.0, 0.0, 3.5])

    assert allowed is False
    assert reason == "joint target exceeds configured limits"


def test_robot_accepts_healthy_state_and_valid_joint_targets():
    gate = SafetyGate(MotionLimits(joint_position_limit=3.14, stale_state_timeout_sec=0.5))
    health = RobotHealth(
        enabled=True,
        has_error=False,
        mode=0,
        state=0,
        joint_names=[f"joint{i}" for i in range(1, 7)],
        joint_positions=[0.0] * 6,
        last_state_age_sec=0.05,
    )

    allowed, reason = gate.can_execute_motion(health, [0.1, -0.2, 0.3, 0.0, 0.2, -0.1])

    assert allowed is True
    assert reason == "ok"


def test_status_payload_includes_readiness_reason_and_joint_snapshot():
    gate = SafetyGate(MotionLimits(joint_position_limit=3.14, stale_state_timeout_sec=0.5))
    health = RobotHealth(
        enabled=False,
        has_error=False,
        mode=0,
        state=0,
        joint_names=[f"joint{i}" for i in range(1, 7)],
        joint_positions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        last_state_age_sec=0.05,
    )

    status = gate.build_status(health)

    assert status["ready"] is False
    assert status["reason"] == "motion not enabled"
    assert status["joint_count"] == 6
    assert status["joint_names"] == [f"joint{i}" for i in range(1, 7)]
    assert status["joint_positions"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
