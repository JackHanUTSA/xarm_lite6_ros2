from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lite6_imitation_bridge.freemocap_loader import load_freemocap_3d_data, normalize_freemocap_recording_path
from lite6_imitation_bridge.human_arm_features import extract_arm_features
from lite6_imitation_bridge.lite6_kinematics_proxy import Lite6KinematicsProxy
from lite6_imitation_bridge.retargeter import retarget_arm_features
from lite6_imitation_bridge.robot_frame_calibration import load_calibration


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


def _default_output_dir(input_path: Path) -> Path:
    if input_path.suffix == ".npy":
        return input_path.parent
    return normalize_freemocap_recording_path(input_path) / "output_data"


def _load_features_from_human_demo(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _load_human_demo_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "human_arm_demo_metadata.json"
    if not metadata_path.exists():
        return {}
    data = json.loads(metadata_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("human_arm_demo_metadata.json must contain a JSON object")
    return data


def _resolve_features(input_path: Path, arm_side: str, active_tracker: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    output_dir = _default_output_dir(input_path)
    human_demo_npz = output_dir / "human_arm_demo.npz"
    if human_demo_npz.exists():
        metadata = _load_human_demo_metadata(output_dir)
        cached_arm_side = metadata.get("arm_side")
        if cached_arm_side is not None and cached_arm_side != arm_side:
            raise ValueError(f"human_arm_demo arm_side mismatch: cached={cached_arm_side} requested={arm_side}")
        features = _load_features_from_human_demo(human_demo_npz)
        features["arm_side"] = arm_side
        return features, "human_arm_demo.npz", metadata | {"frame_count": int(len(features["valid_mask"]))}

    landmarks_3d, loader_metadata = load_freemocap_3d_data(input_path, active_tracker=active_tracker)
    features = extract_arm_features(landmarks_3d, arm_side=arm_side)
    return features, "computed_from_freemocap", loader_metadata


def export_lite6_targets(
    input_path: str | Path,
    arm_side: str = "right",
    output_dir: str | Path | None = None,
    calibration_path: str | Path | None = None,
    active_tracker: str = "mediapipe",
    max_step_distance: float = 0.25,
) -> dict[str, Any]:
    resolved_input_path = Path(input_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(resolved_input_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    features, source_features, loader_metadata = _resolve_features(resolved_input_path, arm_side=arm_side, active_tracker=active_tracker)
    calibration = None if calibration_path is None else load_calibration(calibration_path)
    proxy = Lite6KinematicsProxy()
    retargeted = retarget_arm_features(
        features,
        calibration=calibration,
        kinematics=proxy,
        max_step_distance=max_step_distance,
    )

    jsonl_path = resolved_output_dir / "lite6_targets.jsonl"
    npz_path = resolved_output_dir / "lite6_targets_preview.npz"
    report_path = resolved_output_dir / "lite6_targets_report.json"

    jsonl_path.write_text("\n".join(json.dumps(_json_ready(record), sort_keys=True) for record in retargeted["records"]) + "\n")

    frame_count = retargeted["frame_count"]
    target_positions = np.full((frame_count, 3), np.nan, dtype=float)
    approach_directions = np.full((frame_count, 3), np.nan, dtype=float)
    valid_mask = np.zeros(frame_count, dtype=bool)
    confidence = np.zeros(frame_count, dtype=float)
    step_distances = np.full(frame_count, np.nan, dtype=float)

    for record in retargeted["records"]:
        frame_index = int(record["frame_index"])
        if record["target_position"] is not None:
            target_positions[frame_index] = np.asarray(record["target_position"], dtype=float)
        if record["approach_direction"] is not None:
            approach_directions[frame_index] = np.asarray(record["approach_direction"], dtype=float)
        valid_mask[frame_index] = bool(record["valid"])
        confidence[frame_index] = float(record["confidence"])
        if record["step_distance"] is not None:
            step_distances[frame_index] = float(record["step_distance"])

    np.savez(
        npz_path,
        target_positions=target_positions,
        approach_directions=approach_directions,
        valid_mask=valid_mask,
        confidence=confidence,
        step_distance=step_distances,
    )

    clamped_frame_count = int(sum(bool(record.get("clamped", False)) for record in retargeted["records"]))
    report = {
        "input_path": str(resolved_input_path),
        "output_dir": str(resolved_output_dir),
        "jsonl_path": str(jsonl_path),
        "npz_path": str(npz_path),
        "report_path": str(report_path),
        "frame_count": frame_count,
        "valid_frame_count": int(np.count_nonzero(valid_mask)),
        "invalid_frame_count": int(frame_count - np.count_nonzero(valid_mask)),
        "clamped_frame_count": clamped_frame_count,
        "workspace_bounds": retargeted["workspace_bounds"],
        "calibration_path": None if calibration_path is None else str(Path(calibration_path).expanduser().resolve()),
        "source_features": source_features,
        "max_step_distance": float(max_step_distance),
        "tracker": loader_metadata.get("tracker", active_tracker.strip().lower()),
        "arm_side": arm_side,
    }
    report_path.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export conservative Lite6 retargeting targets from FreeMoCap data")
    parser.add_argument("input_path", help="Recording folder path or direct .npy path")
    parser.add_argument("--arm-side", choices=["left", "right"], default="right")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--active-tracker", default="mediapipe")
    parser.add_argument("--max-step-distance", type=float, default=0.25)
    args = parser.parse_args(argv)

    summary = export_lite6_targets(
        input_path=args.input_path,
        arm_side=args.arm_side,
        output_dir=args.output_dir,
        calibration_path=args.calibration_path,
        active_tracker=args.active_tracker,
        max_step_distance=args.max_step_distance,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
