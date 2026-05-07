from __future__ import annotations

from typing import Any

import numpy as np

MEDIAPIPE_POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]
MEDIAPIPE_LANDMARK_INDEX = {name: index for index, name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES)}
ARM_CONFIG = {
    "right": {
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
        "index": "right_index",
        "thumb": "right_thumb",
        "opposite_shoulder": "left_shoulder",
        "hip": "right_hip",
    },
    "left": {
        "shoulder": "left_shoulder",
        "elbow": "left_elbow",
        "wrist": "left_wrist",
        "index": "left_index",
        "thumb": "left_thumb",
        "opposite_shoulder": "right_shoulder",
        "hip": "left_hip",
    },
}
EXPECTED_LANDMARK_COUNT = len(MEDIAPIPE_POSE_LANDMARK_NAMES)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0 or np.isnan(norm):
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


def _safe_angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    unit_a = _normalize(vector_a)
    unit_b = _normalize(vector_b)
    cosine = np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _has_nan(point: np.ndarray) -> bool:
    return bool(np.isnan(point).any())


def _build_string_array(values: list[str]) -> np.ndarray:
    max_length = max([len(value) for value in values], default=1)
    return np.asarray(values, dtype=f"<U{max_length}")


def extract_arm_features(landmarks_3d: np.ndarray, arm_side: str = "right") -> dict[str, Any]:
    if arm_side not in ARM_CONFIG:
        raise ValueError(f"Unsupported arm_side '{arm_side}'. Expected one of {sorted(ARM_CONFIG)}")
    if landmarks_3d.ndim != 3 or landmarks_3d.shape[-1] != 3:
        raise ValueError(f"Expected landmark array shaped (frames, points, 3), got {landmarks_3d.shape}")
    if landmarks_3d.shape[1] < EXPECTED_LANDMARK_COUNT:
        raise ValueError(
            f"Expected at least {EXPECTED_LANDMARK_COUNT} landmarks for MediaPipe pose data, got {landmarks_3d.shape[1]}"
        )

    config = ARM_CONFIG[arm_side]
    frame_count = landmarks_3d.shape[0]

    shoulder_to_elbow = np.full((frame_count, 3), np.nan, dtype=float)
    elbow_to_wrist = np.full((frame_count, 3), np.nan, dtype=float)
    upper_arm_length = np.full(frame_count, np.nan, dtype=float)
    forearm_length = np.full(frame_count, np.nan, dtype=float)
    elbow_bend_angle = np.full(frame_count, np.nan, dtype=float)
    wrist_direction = np.full((frame_count, 3), np.nan, dtype=float)
    torso_origin = np.full((frame_count, 3), np.nan, dtype=float)
    torso_axes = np.full((frame_count, 3, 3), np.nan, dtype=float)
    valid_mask = np.zeros(frame_count, dtype=bool)
    invalid_reason: list[str] = ["" for _ in range(frame_count)]

    for frame_index in range(frame_count):
        frame = landmarks_3d[frame_index]
        required_names = [
            config["shoulder"],
            config["elbow"],
            config["wrist"],
            config["index"],
            config["thumb"],
            config["opposite_shoulder"],
            "left_hip",
            "right_hip",
        ]
        missing_names = [
            name for name in required_names if _has_nan(frame[MEDIAPIPE_LANDMARK_INDEX[name]])
        ]
        if missing_names:
            invalid_reason[frame_index] = f"missing required landmarks: {','.join(missing_names)}"
            continue

        shoulder = frame[MEDIAPIPE_LANDMARK_INDEX[config["shoulder"]]]
        elbow = frame[MEDIAPIPE_LANDMARK_INDEX[config["elbow"]]]
        wrist = frame[MEDIAPIPE_LANDMARK_INDEX[config["wrist"]]]
        index_tip = frame[MEDIAPIPE_LANDMARK_INDEX[config["index"]]]
        thumb_tip = frame[MEDIAPIPE_LANDMARK_INDEX[config["thumb"]]]
        opposite_shoulder = frame[MEDIAPIPE_LANDMARK_INDEX[config["opposite_shoulder"]]]
        left_shoulder = frame[MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]]
        right_shoulder = frame[MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]]
        left_hip = frame[MEDIAPIPE_LANDMARK_INDEX["left_hip"]]
        right_hip = frame[MEDIAPIPE_LANDMARK_INDEX["right_hip"]]

        upper_vector = elbow - shoulder
        forearm_vector = wrist - elbow
        wrist_vector = index_tip - wrist
        if np.linalg.norm(wrist_vector) <= 0.0:
            wrist_vector = thumb_tip - wrist

        torso_mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
        torso_mid_hip = 0.5 * (left_hip + right_hip)
        torso_up_seed = torso_mid_shoulder - torso_mid_hip

        try:
            torso_x = _normalize(right_shoulder - left_shoulder)
            torso_y_seed = _normalize(torso_up_seed)
            torso_z = _normalize(np.cross(torso_x, torso_y_seed))
            torso_y = _normalize(np.cross(torso_z, torso_x))
            wrist_dir = _normalize(wrist_vector)
            angle = _safe_angle_degrees(shoulder - elbow, wrist - elbow)
        except ValueError as exc:
            invalid_reason[frame_index] = str(exc)
            continue

        shoulder_to_elbow[frame_index] = upper_vector
        elbow_to_wrist[frame_index] = forearm_vector
        upper_arm_length[frame_index] = np.linalg.norm(upper_vector)
        forearm_length[frame_index] = np.linalg.norm(forearm_vector)
        elbow_bend_angle[frame_index] = angle
        wrist_direction[frame_index] = wrist_dir
        torso_origin[frame_index] = torso_mid_shoulder
        torso_axes[frame_index] = np.vstack((torso_x, torso_y, torso_z))
        valid_mask[frame_index] = True
        invalid_reason[frame_index] = ""

    return {
        "arm_side": arm_side,
        "shoulder_to_elbow": shoulder_to_elbow,
        "elbow_to_wrist": elbow_to_wrist,
        "upper_arm_length": upper_arm_length,
        "forearm_length": forearm_length,
        "elbow_bend_angle": elbow_bend_angle,
        "wrist_direction": wrist_direction,
        "torso_origin": torso_origin,
        "torso_axes": torso_axes,
        "valid_mask": valid_mask,
        "invalid_reason": _build_string_array(invalid_reason),
    }
