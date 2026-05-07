from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lite6_imitation_bridge.robot_frame_calibration import (
    estimate_rigid_transform,
    save_calibration,
)


DEFAULT_OUTPUT_PATH = Path("~/ws_xarm/calibration/freemocap_to_lite6.yaml")


def _default_output_path() -> Path:
    return DEFAULT_OUTPUT_PATH.expanduser()


def _load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Calibration manifest must be a JSON object")
    return data


def calibrate_from_manifest(manifest_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    calibration = estimate_rigid_transform(
        source_points=manifest["source_points"],
        target_points=manifest["target_points"],
        source_frame=manifest.get("source_frame", "freemocap"),
        target_frame=manifest.get("target_frame", "lite6_workspace"),
        notes=manifest.get("notes"),
        workspace_origin=manifest.get("workspace_origin"),
        workspace_description=manifest.get("workspace_description"),
        known_pose_name=manifest.get("known_pose_name"),
    )
    saved_path = save_calibration(calibration, output_path or _default_output_path())
    summary = {
        "saved_path": str(saved_path),
        "source_frame": calibration.source_frame,
        "target_frame": calibration.target_frame,
        "point_count": int(len(manifest["source_points"])),
        "fit_rmse": calibration.fit_rmse,
        "known_pose_name": calibration.known_pose_name,
        "workspace_description": calibration.workspace_description,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate a rigid FreeMoCap-to-Lite6 workspace calibration")
    parser.add_argument("manifest_path", help="JSON file containing source_points and target_points")
    parser.add_argument("--output-path", default=None, help="YAML output path for the saved calibration")
    args = parser.parse_args(argv)

    summary = calibrate_from_manifest(args.manifest_path, output_path=args.output_path)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
