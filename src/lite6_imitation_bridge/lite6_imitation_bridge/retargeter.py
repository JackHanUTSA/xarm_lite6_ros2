from __future__ import annotations

from typing import Any

import numpy as np

from lite6_imitation_bridge.lite6_kinematics_proxy import Lite6KinematicsProxy
from lite6_imitation_bridge.robot_frame_calibration import RigidTransformCalibration


DEFAULT_GRIPPER_OPEN_THRESHOLD = 0.05
DEFAULT_APPROACH_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=float)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


def _confidence(clamped: bool, reasons: list[str]) -> float:
    if any(reason.startswith("invalid_human_features") for reason in reasons) or "step_distance_exceeded" in reasons:
        return 0.0
    if clamped:
        return 0.75
    return 1.0


def _optional_scalar(values: Any, frame_index: int) -> float | None:
    if values is None:
        return None
    numeric = float(np.asarray(values, dtype=float)[frame_index])
    if np.isnan(numeric):
        return None
    return numeric


def retarget_arm_features(
    features: dict[str, Any],
    calibration: RigidTransformCalibration | None = None,
    kinematics: Lite6KinematicsProxy | None = None,
    max_step_distance: float = 0.25,
    gripper_open_threshold: float = DEFAULT_GRIPPER_OPEN_THRESHOLD,
) -> dict[str, Any]:
    proxy = kinematics or Lite6KinematicsProxy()
    frame_count = int(len(features["valid_mask"]))
    arm_side = str(features.get("arm_side", "right"))
    records: list[dict[str, Any]] = []
    previous_valid_target: np.ndarray | None = None

    for frame_index in range(frame_count):
        valid_human = bool(np.asarray(features["valid_mask"])[frame_index])
        reasons: list[str] = []
        target_position = None
        approach_direction = None
        workspace_valid = False
        step_distance = None
        clamped = False

        if not valid_human:
            reasons.append(f"invalid_human_features: {str(np.asarray(features['invalid_reason'])[frame_index])}")
        else:
            desired_position = (
                np.asarray(features["torso_origin"][frame_index], dtype=float)
                + np.asarray(features["shoulder_to_elbow"][frame_index], dtype=float)
                + np.asarray(features["elbow_to_wrist"][frame_index], dtype=float)
            )
            if not np.all(np.isfinite(desired_position)):
                reasons.append("invalid_human_features: non_finite_target_input")
            else:
                if calibration is not None:
                    desired_position = calibration.apply(desired_position.reshape(1, 3))[0]
                if not np.all(np.isfinite(desired_position)):
                    reasons.append("invalid_human_features: non_finite_calibrated_target")
                else:
                    clamped_position, clamped = proxy.clamp_position(desired_position)
                    target_position = clamped_position
                    clamp_distance = float(np.linalg.norm(desired_position - clamped_position))
                    if clamped:
                        reasons.append("clamped_to_workspace")
                    try:
                        approach_direction = _normalize(np.asarray(features["wrist_direction"][frame_index], dtype=float))
                    except ValueError:
                        approach_direction = DEFAULT_APPROACH_DIRECTION.copy()
                        reasons.append("defaulted_approach_direction")
                    workspace_valid = proxy.is_within_workspace(target_position)
                    candidate_step_distances = [clamp_distance]
                    if previous_valid_target is not None:
                        candidate_step_distances.append(float(np.linalg.norm(target_position - previous_valid_target)))
                    step_distance = max(candidate_step_distances) if candidate_step_distances else None
                    if step_distance is not None and step_distance > float(max_step_distance):
                        reasons = ["step_distance_exceeded"]
                    if reasons != ["step_distance_exceeded"]:
                        previous_valid_target = target_position.copy()

        elbow_angle = _optional_scalar(features.get("elbow_bend_angle"), frame_index)
        thumb_index_distance = _optional_scalar(features.get("thumb_index_distance"), frame_index)
        elbow_configuration = None
        if elbow_angle is not None:
            elbow_configuration = "extended" if elbow_angle >= 150.0 else "bent"
        gripper_intent = None
        if thumb_index_distance is not None:
            gripper_intent = "open" if thumb_index_distance > gripper_open_threshold else "closed"

        record = {
            "frame_index": frame_index,
            "arm_side": arm_side,
            "valid": valid_human and not any(reason.startswith("invalid_human_features") for reason in reasons) and reasons != ["step_distance_exceeded"],
            "target_position": None if target_position is None else target_position.tolist(),
            "approach_direction": None if approach_direction is None else approach_direction.tolist(),
            "elbow_configuration": elbow_configuration,
            "gripper_intent": gripper_intent,
            "confidence": _confidence(clamped, reasons),
            "rejection_reasons": reasons,
            "workspace_valid": workspace_valid,
            "step_distance": step_distance,
            "clamped": clamped,
        }
        records.append(record)

    return {
        "arm_side": arm_side,
        "frame_count": frame_count,
        "records": records,
        "workspace_bounds": {
            "min": proxy.workspace_min.tolist(),
            "max": proxy.workspace_max.tolist(),
        },
        "max_step_distance": float(max_step_distance),
    }
