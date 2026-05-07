from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lite6_imitation_bridge.freemocap_loader import load_freemocap_3d_data, normalize_freemocap_recording_path
from lite6_imitation_bridge.human_arm_features import extract_arm_features


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
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


def export_human_arm_demo(
    input_path: str | Path,
    arm_side: str = "right",
    output_dir: str | Path | None = None,
    active_tracker: str = "mediapipe",
) -> dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve()
    if active_tracker.strip().lower() != "mediapipe":
        raise ValueError("Only the mediapipe tracker is currently supported for human arm feature export")
    landmarks_3d, loader_metadata = load_freemocap_3d_data(input_path, active_tracker=active_tracker)
    features = extract_arm_features(landmarks_3d, arm_side=arm_side)

    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(input_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = resolved_output_dir / "human_arm_demo.jsonl"
    npz_path = resolved_output_dir / "human_arm_demo.npz"
    metadata_path = resolved_output_dir / "human_arm_demo_metadata.json"

    records = []
    frame_count = loader_metadata["frame_count"]
    for frame_index in range(frame_count):
        record = {
            "frame_index": frame_index,
            "arm_side": arm_side,
            "valid": bool(features["valid_mask"][frame_index]),
            "invalid_reason": str(features["invalid_reason"][frame_index]),
            "shoulder_to_elbow": _json_ready(features["shoulder_to_elbow"][frame_index]),
            "elbow_to_wrist": _json_ready(features["elbow_to_wrist"][frame_index]),
            "upper_arm_length": _json_ready(features["upper_arm_length"][frame_index]),
            "forearm_length": _json_ready(features["forearm_length"][frame_index]),
            "elbow_bend_angle": _json_ready(features["elbow_bend_angle"][frame_index]),
            "wrist_direction": _json_ready(features["wrist_direction"][frame_index]),
            "torso_origin": _json_ready(features["torso_origin"][frame_index]),
            "torso_axes": _json_ready(features["torso_axes"][frame_index]),
        }
        records.append(record)

    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    np.savez(
        npz_path,
        shoulder_to_elbow=features["shoulder_to_elbow"],
        elbow_to_wrist=features["elbow_to_wrist"],
        upper_arm_length=features["upper_arm_length"],
        forearm_length=features["forearm_length"],
        elbow_bend_angle=features["elbow_bend_angle"],
        wrist_direction=features["wrist_direction"],
        torso_origin=features["torso_origin"],
        torso_axes=features["torso_axes"],
        valid_mask=features["valid_mask"],
        invalid_reason=features["invalid_reason"],
    )

    metadata = {
        **loader_metadata,
        "arm_side": arm_side,
        "jsonl_path": str(jsonl_path),
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
        "valid_frame_count": int(np.count_nonzero(features["valid_mask"])),
        "invalid_frame_count": int(frame_count - np.count_nonzero(features["valid_mask"])),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {
        "jsonl_path": str(jsonl_path),
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-frame human arm features from FreeMoCap 3D landmarks")
    parser.add_argument("input_path", help="Recording folder path or direct .npy path")
    parser.add_argument("--arm-side", choices=["left", "right"], default="right")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--active-tracker", default="mediapipe")
    args = parser.parse_args()

    result = export_human_arm_demo(
        input_path=args.input_path,
        arm_side=args.arm_side,
        output_dir=args.output_dir,
        active_tracker=args.active_tracker,
    )
    print(json.dumps(result["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
