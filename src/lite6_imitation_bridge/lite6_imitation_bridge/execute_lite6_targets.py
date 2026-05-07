from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from lite6_imitation_bridge.preview_targets import STATUS_TOPIC, build_preview_status_payload, summarize_preview_artifact
from lite6_imitation_bridge.publish_joint_command_preview import build_preview_message_sequence
from lite6_imitation_bridge.trajectory_scheduler import load_joint_target_file, run_scheduled_steps, schedule_joint_targets

MOTION_STATUS_TOPIC = "/lite6_motion/status"
MOTION_COMMAND_TOPIC = "/lite6_motion/joint_command"
MOTION_STOP_SERVICE = "/lite6_motion/stop"
DEFAULT_RATE_HZ = 2.0


def build_execute_demo_payload(source_path: str | Path, operator_confirmed: bool, rate_hz: float = DEFAULT_RATE_HZ) -> dict[str, Any]:
    summary = summarize_preview_artifact(source_path)
    loaded = load_joint_target_file(source_path)
    has_unsupported_targets = any(
        rejection["reason"] == "unsupported target representation: joint_positions required"
        for rejection in loaded.rejections
    )
    has_invalid_targets = any(rejection["reason"].startswith("record marked invalid") for rejection in loaded.rejections)
    execution_supported = len(loaded.executable_targets) > 0 and not has_unsupported_targets and not has_invalid_targets
    return {
        **build_preview_status_payload(summary, operator_confirmed=operator_confirmed),
        "mode": "execute",
        "dry_run": False,
        "live_execution": True,
        "execution_supported": execution_supported,
        "operator_confirmed": bool(operator_confirmed),
        "operator_confirmation_required": not bool(operator_confirmed),
        "motion_status_topic": MOTION_STATUS_TOPIC,
        "motion_command_topic": MOTION_COMMAND_TOPIC,
        "stop_service": MOTION_STOP_SERVICE,
        "rate_hz": float(rate_hz),
        "executable_frame_count": len(loaded.executable_targets),
        "rejections": loaded.rejections,
    }


def preview_demo(source_path: str | Path) -> dict[str, Any]:
    summary = summarize_preview_artifact(source_path)
    payload = build_execute_demo_payload(source_path, operator_confirmed=False)
    messages = build_preview_message_sequence(source_path, summary=summary, operator_confirmed=False)
    return {
        **build_preview_status_payload(summary, operator_confirmed=False),
        "source_path": str(Path(source_path).expanduser().resolve()),
        "execution_supported": bool(payload["execution_supported"]),
        "preview_message_count": len(messages),
        "publish_topic": messages[1]["topic"] if len(messages) > 1 else STATUS_TOPIC,
        "rejections": payload["rejections"],
    }


def execute_demo(
    source_path: str | Path,
    operator_confirmed: bool,
    publish_joint_command: Callable[[list[float]], None],
    read_motion_status: Callable[[], dict[str, Any]],
    request_stop: Callable[[], None],
    stop_requested: Callable[[], bool] = lambda: False,
    rate_hz: float = DEFAULT_RATE_HZ,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_time: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    payload = build_execute_demo_payload(source_path, operator_confirmed=operator_confirmed, rate_hz=rate_hz)
    if not operator_confirmed:
        return {
            **payload,
            "started_execution": False,
            "completed_steps": 0,
            "stop_requested": False,
            "reason": "operator confirmation required",
        }

    unsupported_rejections = [
        rejection
        for rejection in payload["rejections"]
        if rejection["reason"] == "unsupported target representation: joint_positions required"
    ]
    invalid_rejections = [
        rejection
        for rejection in payload["rejections"]
        if rejection["reason"].startswith("record marked invalid")
    ]
    if unsupported_rejections or invalid_rejections:
        request_stop()
        primary_rejection = unsupported_rejections[0] if unsupported_rejections else invalid_rejections[0]
        return {
            **payload,
            "started_execution": False,
            "completed_steps": 0,
            "stop_requested": True,
            "execution_supported": False,
            "reason": primary_rejection["reason"],
        }

    loaded = load_joint_target_file(source_path)
    scheduled_steps = schedule_joint_targets(loaded.executable_targets, rate_hz=rate_hz)

    result = run_scheduled_steps(
        scheduled_steps,
        publish_step=lambda step: publish_joint_command(list(step.joint_positions)),
        health_check=read_motion_status,
        stop_requested=stop_requested,
        monotonic_time=monotonic_time,
        sleep_fn=sleep_fn,
    )
    stop_was_requested = result.stop_reason != "complete"
    if stop_was_requested and result.stop_reason != "stop requested":
        request_stop()

    return {
        **payload,
        "started_execution": True,
        "completed_steps": result.completed_steps,
        "stop_requested": stop_was_requested,
        "reason": result.stop_reason,
        "last_completed_frame_index": result.last_completed_frame_index,
    }


def _read_motion_status_once(status_topic: str = MOTION_STATUS_TOPIC) -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import String

    class OneShotStatus(Node):
        def __init__(self):
            super().__init__("lite6_execute_demo_status_reader")
            self.payload = None
            qos = QoSProfile(depth=1)
            qos.reliability = QoSReliabilityPolicy.RELIABLE
            qos.durability = QoSDurabilityPolicy.VOLATILE
            self.create_subscription(String, status_topic, self._on_status, qos)

        def _on_status(self, msg):
            self.payload = json.loads(msg.data)

    rclpy.init(args=None)
    node = OneShotStatus()
    try:
        deadline = time.time() + 2.0
        while rclpy.ok() and node.payload is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.payload is None:
            return {"ready": False, "reason": "no lite6_motion status received"}
        return dict(node.payload)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def _ros_publish_joint_command(joint_positions: list[float], command_topic: str = MOTION_COMMAND_TOPIC) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init(args=None)
    node = Node("lite6_execute_demo_joint_command")
    pub = node.create_publisher(JointState, command_topic, 10)
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline and pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.05)
        if pub.get_subscription_count() == 0:
            raise RuntimeError(f"no subscribers on {command_topic}")
        msg = JointState()
        msg.name = [f"joint{i}" for i in range(1, 7)]
        msg.position = [float(value) for value in joint_positions]
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def _ros_request_stop(stop_service: str = MOTION_STOP_SERVICE) -> None:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger

    rclpy.init(args=None)
    node = Node("lite6_execute_demo_stop")
    client = node.create_client(Trigger, stop_service)
    try:
        if not client.wait_for_service(timeout_sec=1.0):
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preview or conservatively execute Lite6 joint target demos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview_parser = subparsers.add_parser("preview_demo")
    preview_parser.add_argument("source_path")

    execute_parser = subparsers.add_parser("execute_demo")
    execute_parser.add_argument("source_path")
    execute_parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    execute_parser.add_argument("--confirm", action="store_true")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--service", default=MOTION_STOP_SERVICE)

    args = parser.parse_args(argv)

    if args.command == "preview_demo":
        print(json.dumps(preview_demo(args.source_path), indent=2, sort_keys=True))
        return
    if args.command == "stop":
        _ros_request_stop(args.service)
        print(json.dumps({"stop_service": args.service, "stop_requested": True}, sort_keys=True))
        return

    result = execute_demo(
        args.source_path,
        operator_confirmed=bool(args.confirm),
        rate_hz=float(args.rate_hz),
        publish_joint_command=_ros_publish_joint_command,
        read_motion_status=_read_motion_status_once,
        request_stop=_ros_request_stop,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
