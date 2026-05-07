import json

import pytest

from lite6_imitation_bridge.trajectory_scheduler import (
    SchedulerStop,
    load_joint_target_file,
    run_scheduled_steps,
    schedule_joint_targets,
)


def test_scheduler_uses_bounded_rate_and_stops_before_next_frame(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"frame_index": 0, "valid": True, "joint_positions": [0, 0, 0, 0, 0, 0]}),
                json.dumps({"frame_index": 1, "valid": True, "joint_positions": [0.1, 0, 0, 0, 0, 0]}),
                json.dumps({"frame_index": 2, "valid": True, "joint_positions": [0.2, 0, 0, 0, 0, 0]}),
            ]
        )
        + "\n"
    )

    loaded = load_joint_target_file(jsonl_path)
    scheduled = schedule_joint_targets(loaded.executable_targets, rate_hz=5.0)
    published = []
    sleeps = []
    now = {"value": 100.0}

    def monotonic_time():
        return now["value"]

    def sleep_fn(duration):
        sleeps.append(duration)
        now["value"] += duration

    def publish_fn(step):
        published.append((step.frame_index, step.joint_positions))

    stop_checks = {"count": 0}

    def stop_requested():
        stop_checks["count"] += 1
        return stop_checks["count"] >= 5

    result = run_scheduled_steps(
        scheduled,
        publish_step=publish_fn,
        health_check=lambda: {"ready": True, "reason": "ok"},
        stop_requested=stop_requested,
        monotonic_time=monotonic_time,
        sleep_fn=sleep_fn,
    )

    assert [frame_index for frame_index, _ in published] == [0, 1]
    assert sleeps == pytest.approx([0.2])
    assert result.completed_steps == 2
    assert result.stop_reason == "stop requested"



def test_scheduler_rechecks_stop_and_health_after_sleep_before_publish():
    scheduled = schedule_joint_targets(
        [
            {"frame_index": 0, "joint_positions": [0, 0, 0, 0, 0, 0]},
            {"frame_index": 1, "joint_positions": [0.1, 0, 0, 0, 0, 0]},
        ],
        rate_hz=5.0,
    )
    published = []
    now = {"value": 0.0}
    stop_state = {"value": False}
    health_state = {"ready": True, "reason": "ok"}

    def monotonic_time():
        return now["value"]

    def sleep_fn(duration):
        now["value"] += duration
        stop_state["value"] = True
        health_state["ready"] = False
        health_state["reason"] = "state is stale"

    result = run_scheduled_steps(
        scheduled,
        publish_step=lambda step: published.append(step.frame_index),
        health_check=lambda: dict(health_state),
        stop_requested=lambda: stop_state["value"],
        monotonic_time=monotonic_time,
        sleep_fn=sleep_fn,
    )

    assert published == [0]
    assert result.completed_steps == 1
    assert result.stop_reason == "stop requested"



def test_load_joint_target_file_rejects_invalid_and_unsupported_frames(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"frame_index": 0, "valid": False, "rejection_reasons": ["step_distance_exceeded"]}),
                json.dumps({"frame_index": 1, "valid": True, "target_position": [0.4, 0.1, 0.2]}),
                json.dumps({"frame_index": 2, "valid": True, "joint_positions": [0, 0, 0, 0, 0, 0]}),
            ]
        )
        + "\n"
    )

    loaded = load_joint_target_file(jsonl_path)

    assert [target.frame_index for target in loaded.executable_targets] == [2]
    assert loaded.rejections == [
        {"frame_index": 0, "reason": "record marked invalid: step_distance_exceeded"},
        {"frame_index": 1, "reason": "unsupported target representation: joint_positions required"},
    ]



def test_run_scheduled_steps_stops_on_unhealthy_state():
    scheduled = schedule_joint_targets(
        [{"frame_index": 0, "joint_positions": [0, 0, 0, 0, 0, 0]}, {"frame_index": 1, "joint_positions": [0.1, 0, 0, 0, 0, 0]}],
        rate_hz=10.0,
    )
    published = []

    def publish_fn(step):
        published.append(step.frame_index)

    health = iter([
        {"ready": True, "reason": "ok"},
        {"ready": True, "reason": "ok"},
        {"ready": False, "reason": "state is stale"},
    ])

    result = run_scheduled_steps(
        scheduled,
        publish_step=publish_fn,
        health_check=lambda: next(health),
        stop_requested=lambda: False,
        monotonic_time=lambda: 0.0,
        sleep_fn=lambda _duration: None,
    )

    assert published == [0]
    assert result.completed_steps == 1
    assert result.stop_reason == "state is stale"
