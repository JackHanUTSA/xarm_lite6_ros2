from lite6_motion.robot_client import Lite6RobotClient


def test_prepare_sequence_contains_expected_vendor_calls():
    client = Lite6RobotClient()

    calls = client.prepare_sequence()

    assert [call.service_name for call in calls] == [
        "/ufactory/clean_error",
        "/ufactory/clean_warn",
        "/ufactory/motion_enable",
        "/ufactory/set_mode",
        "/ufactory/set_state",
    ]
    assert calls[2].payload == {"id": 8, "data": 1}
    assert calls[3].payload == {"data": 0}
    assert calls[4].payload == {"data": 0}


def test_move_joint_payload_is_absolute_and_uses_limits():
    client = Lite6RobotClient(default_speed=0.25, default_acc=0.5, default_timeout=15.0)

    payload = client.build_move_joint_payload([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    assert payload["angles"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert payload["speed"] == 0.25
    assert payload["acc"] == 0.5
    assert payload["timeout"] == 15.0
    assert payload["relative"] is False


def test_home_payload_returns_to_zero_joint_pose():
    client = Lite6RobotClient()

    payload = client.build_home_payload()

    assert payload["angles"] == [0.0] * 6
    assert payload["relative"] is False
