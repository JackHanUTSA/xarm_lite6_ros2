import json

import numpy as np
import pytest

from lite6_imitation_bridge.export_lite6_targets import export_lite6_targets
from lite6_imitation_bridge.human_arm_features import MEDIAPIPE_LANDMARK_INDEX
from lite6_imitation_bridge.lite6_kinematics_proxy import Lite6KinematicsProxy
from lite6_imitation_bridge.retargeter import retarget_arm_features
from lite6_imitation_bridge.robot_frame_calibration import RigidTransformCalibration, save_calibration


RIGHT_SHOULDER = MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]
RIGHT_ELBOW = MEDIAPIPE_LANDMARK_INDEX["right_elbow"]
RIGHT_WRIST = MEDIAPIPE_LANDMARK_INDEX["right_wrist"]
RIGHT_INDEX = MEDIAPIPE_LANDMARK_INDEX["right_index"]
RIGHT_THUMB = MEDIAPIPE_LANDMARK_INDEX["right_thumb"]
LEFT_SHOULDER = MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]
LEFT_HIP = MEDIAPIPE_LANDMARK_INDEX["left_hip"]
RIGHT_HIP = MEDIAPIPE_LANDMARK_INDEX["right_hip"]


def _synthetic_features() -> dict[str, np.ndarray]:
    return {
        "arm_side": "right",
        "shoulder_to_elbow": np.array(
            [
                [0.20, 0.00, 0.10],
                [0.40, 0.10, 0.10],
                [0.10, 0.00, 0.10],
                [0.10, 0.00, 0.10],
            ],
            dtype=float,
        ),
        "elbow_to_wrist": np.array(
            [
                [0.20, 0.10, 0.00],
                [0.30, 0.30, 0.20],
                [0.15, 0.10, 0.05],
                [0.75, 0.00, 0.00],
            ],
            dtype=float,
        ),
        "elbow_bend_angle": np.array([90.0, 175.0, 110.0, 100.0], dtype=float),
        "wrist_direction": np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        "torso_origin": np.array(
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00],
                [0.10, 0.00, 0.00],
            ],
            dtype=float,
        ),
        "valid_mask": np.array([True, True, False, True], dtype=bool),
        "invalid_reason": np.array(["", "", "missing right wrist", ""], dtype="<U32"),
        "thumb_index_distance": np.array([0.09, 0.01, np.nan, 0.04], dtype=float),
    }


@pytest.fixture
def synthetic_pose_data():
    data = np.full((3, 33, 3), np.nan, dtype=float)
    data[:, LEFT_SHOULDER] = np.array([[-0.2, 0.0, 0.2]] * 3)
    data[:, RIGHT_HIP] = np.array([[0.2, -0.4, 0.0]] * 3)
    data[:, LEFT_HIP] = np.array([[-0.2, -0.4, 0.0]] * 3)

    data[0, RIGHT_SHOULDER] = [0.2, 0.0, 0.2]
    data[0, RIGHT_ELBOW] = [0.4, 0.0, 0.2]
    data[0, RIGHT_WRIST] = [0.55, 0.10, 0.25]
    data[0, RIGHT_INDEX] = [0.60, 0.15, 0.25]
    data[0, RIGHT_THUMB] = [0.52, 0.05, 0.25]

    data[1, RIGHT_SHOULDER] = [0.2, 0.0, 0.2]
    data[1, RIGHT_ELBOW] = [0.45, 0.05, 0.2]
    data[1, RIGHT_WRIST] = [0.95, 0.60, 0.50]
    data[1, RIGHT_INDEX] = [1.00, 0.65, 0.50]
    data[1, RIGHT_THUMB] = [0.93, 0.58, 0.50]

    data[2, RIGHT_SHOULDER] = [0.2, 0.0, 0.2]
    data[2, RIGHT_ELBOW] = [0.4, 0.0, 0.2]
    data[2, RIGHT_WRIST] = [1.60, 0.10, 0.20]
    data[2, RIGHT_INDEX] = [1.65, 0.10, 0.20]
    data[2, RIGHT_THUMB] = [1.58, 0.10, 0.20]

    return data


def test_retarget_arm_features_clamps_workspace_and_flags_large_jumps():
    features = _synthetic_features()
    proxy = Lite6KinematicsProxy()

    result = retarget_arm_features(features, kinematics=proxy, max_step_distance=0.35)

    assert result["arm_side"] == "right"
    assert result["frame_count"] == 4

    first = result["records"][0]
    assert first["valid"] is True
    np.testing.assert_allclose(first["target_position"], [0.4, 0.1, 0.1])
    np.testing.assert_allclose(first["approach_direction"], [0.0, 1.0, 0.0])
    assert first["elbow_configuration"] == "bent"
    assert first["gripper_intent"] == "open"
    assert first["confidence"] == pytest.approx(1.0)
    assert first["rejection_reasons"] == []

    second = result["records"][1]
    assert second["valid"] is True
    np.testing.assert_allclose(second["target_position"], proxy.workspace_max)
    assert second["gripper_intent"] == "closed"
    assert second["confidence"] == pytest.approx(0.75)
    assert second["rejection_reasons"] == ["clamped_to_workspace"]
    assert second["workspace_valid"] is True

    third = result["records"][2]
    assert third["valid"] is False
    assert third["rejection_reasons"] == ["invalid_human_features: missing right wrist"]
    assert third["confidence"] == pytest.approx(0.0)

    fourth = result["records"][3]
    assert fourth["valid"] is False
    assert fourth["rejection_reasons"] == ["step_distance_exceeded"]
    assert fourth["step_distance"] > 0.35
    assert fourth["workspace_valid"] is True


def test_retarget_arm_features_applies_calibration_before_validation():
    features = _synthetic_features()
    calibration = RigidTransformCalibration(
        rotation_matrix=np.eye(3, dtype=float),
        translation_vector=np.array([0.1, -0.1, 0.2], dtype=float),
        source_frame="freemocap",
        target_frame="lite6_workspace",
    )

    result = retarget_arm_features(features, calibration=calibration, max_step_distance=10.0)

    first = result["records"][0]
    np.testing.assert_allclose(first["target_position"], [0.5, 0.0, 0.3])
    assert first["frame_index"] == 0
    assert result["records"][2]["rejection_reasons"] == ["invalid_human_features: missing right wrist"]


def test_export_lite6_targets_writes_default_artifacts_and_report(tmp_path, synthetic_pose_data):
    recording_dir = tmp_path / "session_targets"
    output_dir = recording_dir / "output_data"
    output_dir.mkdir(parents=True)
    np.save(output_dir / "mediapipe_skeleton_3d.npy", synthetic_pose_data)

    calibration = RigidTransformCalibration(
        rotation_matrix=np.eye(3, dtype=float),
        translation_vector=np.array([0.05, 0.0, 0.0], dtype=float),
        source_frame="freemocap",
        target_frame="lite6_workspace",
    )
    calibration_path = tmp_path / "freemocap_to_lite6.yaml"
    save_calibration(calibration, calibration_path)

    summary = export_lite6_targets(recording_dir, calibration_path=calibration_path, max_step_distance=0.8)

    jsonl_path = output_dir / "lite6_targets.jsonl"
    npz_path = output_dir / "lite6_targets_preview.npz"
    report_path = output_dir / "lite6_targets_report.json"

    assert summary["jsonl_path"] == str(jsonl_path)
    assert summary["npz_path"] == str(npz_path)
    assert summary["report_path"] == str(report_path)
    assert jsonl_path.exists()
    assert npz_path.exists()
    assert report_path.exists()

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 3
    first_record = json.loads(lines[0])
    assert first_record["frame_index"] == 0
    assert first_record["valid"] is True
    assert first_record["workspace_valid"] is True
    assert first_record["gripper_intent"] is None

    second_record = json.loads(lines[1])
    assert second_record["rejection_reasons"] == ["clamped_to_workspace"]

    report = json.loads(report_path.read_text())
    assert report["frame_count"] == 3
    assert report["valid_frame_count"] == 2
    assert report["invalid_frame_count"] == 1
    assert report["clamped_frame_count"] == 2
    assert report["calibration_path"] == str(calibration_path)
    assert report["source_features"] == "computed_from_freemocap"
    assert report["max_step_distance"] == pytest.approx(0.8)
    assert report["workspace_bounds"]["min"] == pytest.approx(Lite6KinematicsProxy().workspace_min.tolist())

    with np.load(npz_path, allow_pickle=False) as archive:
        assert archive["target_positions"].shape == (3, 3)
        assert archive["approach_directions"].shape == (3, 3)
        assert archive["valid_mask"].tolist() == [True, True, False]
        assert np.all(archive["target_positions"][archive["valid_mask"]] <= Lite6KinematicsProxy().workspace_max + 1e-9)


def test_retarget_arm_features_rejects_non_finite_inputs_even_when_valid_mask_is_true():
    features = _synthetic_features()
    features["valid_mask"] = np.array([True, True, True, True], dtype=bool)
    features["shoulder_to_elbow"][2] = np.array([np.nan, 0.0, 0.0], dtype=float)

    result = retarget_arm_features(features, max_step_distance=10.0)

    third = result["records"][2]
    assert third["valid"] is False
    assert third["rejection_reasons"] == ["invalid_human_features: non_finite_target_input"]
    assert third["confidence"] == pytest.approx(0.0)



def test_retarget_arm_features_rejects_non_finite_post_calibration_targets():
    features = _synthetic_features()

    class BadCalibration:
        def apply(self, points):
            return np.full_like(points, np.nan, dtype=float)

    result = retarget_arm_features(features, calibration=BadCalibration(), max_step_distance=10.0)

    first = result["records"][0]
    assert first["valid"] is False
    assert first["rejection_reasons"] == ["invalid_human_features: non_finite_calibrated_target"]
    assert first["confidence"] == pytest.approx(0.0)



def test_export_lite6_targets_rejects_cached_human_demo_arm_side_mismatch(tmp_path):
    recording_dir = tmp_path / "session_cached_features"
    output_dir = recording_dir / "output_data"
    output_dir.mkdir(parents=True)
    np.savez(
        output_dir / "human_arm_demo.npz",
        shoulder_to_elbow=np.zeros((1, 3), dtype=float),
        elbow_to_wrist=np.zeros((1, 3), dtype=float),
        upper_arm_length=np.ones(1, dtype=float),
        forearm_length=np.ones(1, dtype=float),
        elbow_bend_angle=np.array([90.0], dtype=float),
        wrist_direction=np.array([[1.0, 0.0, 0.0]], dtype=float),
        torso_origin=np.zeros((1, 3), dtype=float),
        torso_axes=np.repeat(np.eye(3, dtype=float)[None, :, :], 1, axis=0),
        valid_mask=np.array([True], dtype=bool),
        invalid_reason=np.array([""], dtype="<U1"),
    )
    (output_dir / "human_arm_demo_metadata.json").write_text(
        json.dumps({"arm_side": "right", "tracker": "mediapipe", "frame_count": 1})
    )

    with pytest.raises(ValueError, match="arm_side mismatch"):
        export_lite6_targets(recording_dir, arm_side="left")
