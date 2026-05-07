import json
from pathlib import Path

import pytest

from lite6_imitation_bridge.execute_lite6_targets import (
    build_execute_demo_payload,
    execute_demo,
    preview_demo,
)


def _write_demo(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")



def test_preview_demo_stays_non_live(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_demo(
        jsonl_path,
        [
            {"frame_index": 0, "valid": True, "joint_positions": [0, 0, 0, 0, 0, 0]},
            {"frame_index": 1, "valid": False, "rejection_reasons": ["step_distance_exceeded"]},
        ],
    )

    payload = preview_demo(jsonl_path)

    assert payload["mode"] == "preview"
    assert payload["dry_run"] is True
    assert payload["live_execution"] is False
    assert payload["execution_supported"] is False
    assert payload["preview_message_count"] >= 2
    assert payload["publish_topic"] != "/lite6_motion/joint_command"



def test_execute_demo_refuses_unsupported_target_representation(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_demo(
        jsonl_path,
        [{"frame_index": 0, "valid": True, "target_position": [0.4, 0.1, 0.2], "approach_direction": [0, 1, 0]}],
    )

    result = execute_demo(
        jsonl_path,
        operator_confirmed=True,
        publish_joint_command=lambda _joint_positions: None,
        read_motion_status=lambda: {"ready": True, "reason": "ok"},
        request_stop=lambda: None,
        sleep_fn=lambda _duration: None,
        monotonic_time=lambda: 0.0,
    )

    assert result["started_execution"] is False
    assert result["execution_supported"] is False
    assert result["stop_requested"] is True
    assert "unsupported target representation" in result["reason"]



def test_execute_demo_publishes_joint_frames_when_motion_status_stays_ready(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_demo(
        jsonl_path,
        [
            {"frame_index": 0, "valid": True, "joint_positions": [0, 0, 0, 0, 0, 0]},
            {"frame_index": 1, "valid": True, "joint_positions": [0.1, 0, 0, 0, 0, 0]},
        ],
    )
    published = []
    stop_calls = []

    result = execute_demo(
        jsonl_path,
        operator_confirmed=True,
        publish_joint_command=lambda joint_positions: published.append(joint_positions),
        read_motion_status=lambda: {"ready": True, "reason": "ok"},
        request_stop=lambda: stop_calls.append("stop"),
        sleep_fn=lambda _duration: None,
        monotonic_time=lambda: 0.0,
        rate_hz=4.0,
    )

    assert result["started_execution"] is True
    assert result["completed_steps"] == 2
    assert result["stop_requested"] is False
    assert stop_calls == []
    assert published == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]



def test_execute_demo_stops_when_any_invalid_frame_is_present(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_demo(
        jsonl_path,
        [
            {"frame_index": 0, "valid": True, "joint_positions": [0, 0, 0, 0, 0, 0]},
            {"frame_index": 1, "valid": False, "rejection_reasons": ["step_distance_exceeded"]},
        ],
    )
    stop_calls = []

    result = execute_demo(
        jsonl_path,
        operator_confirmed=True,
        publish_joint_command=lambda _joint_positions: None,
        read_motion_status=lambda: {"ready": True, "reason": "ok"},
        request_stop=lambda: stop_calls.append("stop"),
        sleep_fn=lambda _duration: None,
        monotonic_time=lambda: 0.0,
    )

    assert result["started_execution"] is False
    assert result["execution_supported"] is False
    assert result["stop_requested"] is True
    assert "record marked invalid" in result["reason"]
    assert stop_calls == ["stop"]



def test_build_execute_demo_payload_requires_operator_confirmation(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_demo(jsonl_path, [{"frame_index": 0, "valid": True, "joint_positions": [0, 0, 0, 0, 0, 0]}])

    payload = build_execute_demo_payload(jsonl_path, operator_confirmed=False)

    assert payload["mode"] == "execute"
    assert payload["dry_run"] is False
    assert payload["live_execution"] is True
    assert payload["operator_confirmation_required"] is True
    assert payload["execution_supported"] is True
    assert payload["motion_command_topic"] == "/lite6_motion/joint_command"
