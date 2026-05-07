from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lite6_imitation_bridge.preview_targets import (
    PREVIEW_TARGETS_TOPIC,
    STATUS_TOPIC,
    build_preview_status_payload,
    load_preview_records,
    summarize_preview_artifact,
)


def _preview_frame_payload(record: dict[str, Any], resolved_source_path: Path, operator_confirmed: bool) -> dict[str, Any]:
    payload = {
        **record,
        "source_path": str(resolved_source_path),
        "mode": "preview",
        "dry_run": True,
        "live_execution": False,
        "operator_confirmed": bool(operator_confirmed),
    }
    return payload


def build_preview_message_sequence(
    source_path: str | Path,
    summary: dict[str, Any] | None = None,
    operator_confirmed: bool = False,
) -> list[dict[str, Any]]:
    resolved_source_path = Path(source_path).expanduser().resolve()
    preview_summary = summary or summarize_preview_artifact(resolved_source_path)
    records = load_preview_records(resolved_source_path)

    start_status = build_preview_status_payload(preview_summary, operator_confirmed=operator_confirmed)
    start_status["stage"] = "preview_started"

    messages: list[dict[str, Any]] = [{"topic": STATUS_TOPIC, "payload": start_status}]
    for record in records:
        messages.append(
            {
                "topic": PREVIEW_TARGETS_TOPIC,
                "payload": _preview_frame_payload(record, resolved_source_path, operator_confirmed),
            }
        )

    end_status = build_preview_status_payload(preview_summary, operator_confirmed=operator_confirmed)
    end_status["stage"] = "preview_complete"
    end_status["published_preview_frame_count"] = len(records)
    messages.append({"topic": STATUS_TOPIC, "payload": end_status})
    return messages


def publish_preview_messages(messages: list[dict[str, Any]], status_topic: str = STATUS_TOPIC, preview_topic: str = PREVIEW_TARGETS_TOPIC) -> int:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init(args=None)
    node = Node("lite6_imitation_preview_publisher")
    status_pub = node.create_publisher(String, status_topic, 10)
    preview_pub = node.create_publisher(String, preview_topic, 10)
    published_count = 0
    try:
        for message in messages:
            msg = String()
            msg.data = json.dumps(message["payload"], sort_keys=True)
            topic = message["topic"]
            if topic == status_topic:
                status_pub.publish(msg)
            elif topic == preview_topic:
                preview_pub.publish(msg)
            else:
                continue
            published_count += 1
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return published_count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit dry-run Lite6 preview messages without touching live command topics")
    parser.add_argument("source_path", help="Path to lite6_targets.jsonl or lite6_targets_report.json")
    parser.add_argument("--ros-publish", action="store_true", help="Publish preview/status payloads to ROS String topics instead of printing them")
    args = parser.parse_args(argv)
    messages = build_preview_message_sequence(args.source_path)
    if args.ros_publish:
        published_count = publish_preview_messages(messages)
        print(json.dumps({"published_count": published_count, "status_topic": STATUS_TOPIC, "preview_topic": PREVIEW_TARGETS_TOPIC}, sort_keys=True))
        return
    for message in messages:
        print(json.dumps(message, sort_keys=True))


if __name__ == "__main__":
    main()
