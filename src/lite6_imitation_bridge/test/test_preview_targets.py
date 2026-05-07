import json
from pathlib import Path

import pytest

from lite6_imitation_bridge.preview_targets import (
    build_preview_status_payload,
    format_preview_summary,
    summarize_preview_artifact,
)


def _write_preview_jsonl(path: Path) -> None:
    records = [
        {
            "frame_index": 0,
            "valid": True,
            "clamped": False,
            "confidence": 1.0,
            "workspace_valid": True,
            "rejection_reasons": [],
        },
        {
            "frame_index": 1,
            "valid": True,
            "clamped": True,
            "confidence": 0.75,
            "workspace_valid": True,
            "rejection_reasons": ["clamped_to_workspace"],
        },
        {
            "frame_index": 2,
            "valid": False,
            "clamped": False,
            "confidence": 0.0,
            "workspace_valid": False,
            "rejection_reasons": ["invalid_human_features: missing right wrist"],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_summarize_preview_artifact_from_jsonl(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_preview_jsonl(jsonl_path)

    summary = summarize_preview_artifact(jsonl_path)

    assert summary["source_path"] == str(jsonl_path)
    assert summary["frame_count"] == 3
    assert summary["valid_frame_count"] == 2
    assert summary["invalid_frame_count"] == 1
    assert summary["clamped_frame_count"] == 1
    assert summary["frame_index_range"] == [0, 2]
    assert summary["workspace_valid_frame_count"] == 2
    assert summary["confidence"] == pytest.approx({"min": 0.0, "max": 1.0, "mean": 0.5833333333333334})
    assert summary["rejection_reason_counts"] == {
        "clamped_to_workspace": 1,
        "invalid_human_features: missing right wrist": 1,
    }


def test_build_preview_status_payload_marks_preview_only(tmp_path):
    jsonl_path = tmp_path / "lite6_targets.jsonl"
    _write_preview_jsonl(jsonl_path)
    summary = summarize_preview_artifact(jsonl_path)

    payload = build_preview_status_payload(summary, operator_confirmed=False)

    assert payload["mode"] == "preview"
    assert payload["dry_run"] is True
    assert payload["live_execution"] is False
    assert payload["operator_confirmed"] is False
    assert payload["operator_confirmation_required"] is True
    assert payload["status_topic"] == "/lite6_imitation/status"
    assert payload["preview_topic"] == "/lite6_imitation/preview_targets"
    assert payload["live_topic"] == "/lite6_imitation/live_targets"
    assert "frames=3" in format_preview_summary(summary)



def test_build_preview_status_payload_overrides_conflicting_summary_flags():
    payload = build_preview_status_payload(
        {
            "mode": "live",
            "dry_run": False,
            "live_execution": True,
            "operator_confirmed": True,
            "status_topic": "/bad/status",
            "preview_topic": "/bad/preview",
            "live_topic": "/bad/live",
        },
        operator_confirmed=False,
    )

    assert payload["mode"] == "preview"
    assert payload["dry_run"] is True
    assert payload["live_execution"] is False
    assert payload["operator_confirmed"] is False
    assert payload["status_topic"] == "/lite6_imitation/status"
    assert payload["preview_topic"] == "/lite6_imitation/preview_targets"
    assert payload["live_topic"] == "/lite6_imitation/live_targets"
