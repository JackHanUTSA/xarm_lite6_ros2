import json
from pathlib import Path

from lite6_imitation_bridge.preview_targets import summarize_preview_artifact
from lite6_imitation_bridge.publish_joint_command_preview import build_preview_message_sequence


def _write_preview_jsonl(path: Path) -> None:
    records = [
        {
            "frame_index": 0,
            "valid": True,
            "clamped": False,
            "confidence": 1.0,
            "target_position": [0.4, 0.1, 0.1],
            "approach_direction": [0.0, 1.0, 0.0],
            "rejection_reasons": [],
        },
        {
            "frame_index": 1,
            "valid": False,
            "clamped": False,
            "confidence": 0.0,
            "target_position": None,
            "approach_direction": None,
            "rejection_reasons": ["step_distance_exceeded"],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_build_preview_message_sequence_stays_off_live_topics(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_preview_jsonl(jsonl_path)
    summary = summarize_preview_artifact(jsonl_path)

    messages = build_preview_message_sequence(jsonl_path, summary=summary, operator_confirmed=False)

    assert [message["topic"] for message in messages] == [
        "/lite6_imitation/status",
        "/lite6_imitation/preview_targets",
        "/lite6_imitation/preview_targets",
        "/lite6_imitation/status",
    ]
    assert all(message["topic"] != "/lite6_motion/joint_command" for message in messages)
    assert all(message["topic"] != "/lite6_imitation/live_targets" for message in messages)

    start_status = messages[0]["payload"]
    assert start_status["stage"] == "preview_started"
    assert start_status["live_execution"] is False
    assert start_status["operator_confirmation_required"] is True

    first_frame = messages[1]["payload"]
    assert first_frame["frame_index"] == 0
    assert first_frame["valid"] is True
    assert first_frame["dry_run"] is True

    end_status = messages[-1]["payload"]
    assert end_status["stage"] == "preview_complete"
    assert end_status["published_preview_frame_count"] == 2



def test_build_preview_message_sequence_overrides_conflicting_record_flags(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "valid": True,
                "mode": "live",
                "dry_run": False,
                "live_execution": True,
                "operator_confirmed": True,
            }
        )
        + "\n"
    )

    messages = build_preview_message_sequence(jsonl_path, operator_confirmed=False)
    preview_payload = messages[1]["payload"]

    assert preview_payload["mode"] == "preview"
    assert preview_payload["dry_run"] is True
    assert preview_payload["live_execution"] is False
    assert preview_payload["operator_confirmed"] is False
