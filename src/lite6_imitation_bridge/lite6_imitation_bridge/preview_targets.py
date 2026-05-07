from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

STATUS_TOPIC = "/lite6_imitation/status"
PREVIEW_TARGETS_TOPIC = "/lite6_imitation/preview_targets"
LIVE_TARGETS_TOPIC = "/lite6_imitation/live_targets"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return None if np.isnan(numeric) else numeric
    return value


def load_preview_records(source_path: str | Path) -> list[dict[str, Any]]:
    path = Path(source_path).expanduser().resolve()
    if path.suffix != ".jsonl":
        if path.name == "lite6_targets_report.json":
            candidate = path.with_name("lite6_targets.jsonl")
            if candidate.exists():
                path = candidate
            else:
                return []
        else:
            return []

    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def summarize_preview_records(records: list[dict[str, Any]], source_path: str | Path | None = None) -> dict[str, Any]:
    frame_indices = [int(record.get("frame_index", index)) for index, record in enumerate(records)]
    frame_count = len(records)
    valid_frame_count = sum(bool(record.get("valid", False)) for record in records)
    invalid_frame_count = frame_count - valid_frame_count
    clamped_frame_count = sum(bool(record.get("clamped", False)) for record in records)
    workspace_valid_frame_count = sum(bool(record.get("workspace_valid", False)) for record in records)
    confidence_values = np.asarray([float(record.get("confidence", 0.0)) for record in records], dtype=float)
    rejection_reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record.get("rejection_reasons", []):
            rejection_reason_counts[str(reason)] = rejection_reason_counts.get(str(reason), 0) + 1

    frame_index_range = None
    if frame_indices:
        frame_index_range = [min(frame_indices), max(frame_indices)]

    if confidence_values.size == 0:
        confidence_summary = {"min": 0.0, "max": 0.0, "mean": 0.0}
    else:
        confidence_summary = {
            "min": float(np.min(confidence_values)),
            "max": float(np.max(confidence_values)),
            "mean": float(np.mean(confidence_values)),
        }

    return {
        "source_path": None if source_path is None else str(Path(source_path).expanduser().resolve()),
        "frame_count": frame_count,
        "valid_frame_count": valid_frame_count,
        "invalid_frame_count": invalid_frame_count,
        "clamped_frame_count": clamped_frame_count,
        "workspace_valid_frame_count": workspace_valid_frame_count,
        "frame_index_range": frame_index_range,
        "confidence": confidence_summary,
        "rejection_reason_counts": rejection_reason_counts,
        "dry_run": True,
        "live_execution": False,
    }


def summarize_preview_artifact(source_path: str | Path) -> dict[str, Any]:
    path = Path(source_path).expanduser().resolve()
    if path.suffix == ".jsonl":
        return summarize_preview_records(load_preview_records(path), source_path=path)

    if path.name == "lite6_targets_report.json":
        report = json.loads(path.read_text())
        summary = {
            "source_path": str(path),
            "frame_count": int(report.get("frame_count", 0)),
            "valid_frame_count": int(report.get("valid_frame_count", 0)),
            "invalid_frame_count": int(report.get("invalid_frame_count", 0)),
            "clamped_frame_count": int(report.get("clamped_frame_count", 0)),
            "workspace_valid_frame_count": int(report.get("valid_frame_count", 0)),
            "frame_index_range": [0, int(report.get("frame_count", 0)) - 1] if int(report.get("frame_count", 0)) > 0 else None,
            "confidence": {"min": 0.0, "max": 0.0, "mean": 0.0},
            "rejection_reason_counts": {},
            "dry_run": True,
            "live_execution": False,
        }
        jsonl_records = load_preview_records(path)
        if jsonl_records:
            summary.update(summarize_preview_records(jsonl_records, source_path=path))
            summary["source_path"] = str(path)
        return summary

    raise ValueError(f"Unsupported preview artifact: {path}")


def build_preview_status_payload(summary: dict[str, Any], operator_confirmed: bool = False) -> dict[str, Any]:
    payload = {
        **_json_ready(summary),
        "mode": "preview",
        "dry_run": True,
        "live_execution": False,
        "operator_confirmed": bool(operator_confirmed),
        "operator_confirmation_required": not bool(operator_confirmed),
        "status_topic": STATUS_TOPIC,
        "preview_topic": PREVIEW_TARGETS_TOPIC,
        "live_topic": LIVE_TARGETS_TOPIC,
    }
    return payload


def format_preview_summary(summary: dict[str, Any]) -> str:
    frame_range = summary.get("frame_index_range")
    if frame_range is None:
        frame_range_text = "none"
    else:
        frame_range_text = f"{frame_range[0]}-{frame_range[1]}"
    return (
        "preview dry-run | "
        f"frames={summary.get('frame_count', 0)} | "
        f"valid={summary.get('valid_frame_count', 0)} | "
        f"invalid={summary.get('invalid_frame_count', 0)} | "
        f"clamped={summary.get('clamped_frame_count', 0)} | "
        f"range={frame_range_text}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize Lite6 preview target artifacts without commanding hardware")
    parser.add_argument("source_path", help="Path to lite6_targets.jsonl or lite6_targets_report.json")
    args = parser.parse_args(argv)
    summary = summarize_preview_artifact(args.source_path)
    payload = build_preview_status_payload(summary)
    print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    print(format_preview_summary(summary))


if __name__ == "__main__":
    main()
