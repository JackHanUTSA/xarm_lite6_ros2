import json
from pathlib import Path

import numpy as np
import pytest

from lite6_imitation_bridge.export_human_arm_demo import export_human_arm_demo
from lite6_imitation_bridge.freemocap_loader import load_freemocap_3d_data
from lite6_imitation_bridge.human_arm_features import (
    MEDIAPIPE_LANDMARK_INDEX,
    extract_arm_features,
)


RIGHT_SHOULDER = MEDIAPIPE_LANDMARK_INDEX["right_shoulder"]
RIGHT_ELBOW = MEDIAPIPE_LANDMARK_INDEX["right_elbow"]
RIGHT_WRIST = MEDIAPIPE_LANDMARK_INDEX["right_wrist"]
RIGHT_INDEX = MEDIAPIPE_LANDMARK_INDEX["right_index"]
RIGHT_THUMB = MEDIAPIPE_LANDMARK_INDEX["right_thumb"]
LEFT_SHOULDER = MEDIAPIPE_LANDMARK_INDEX["left_shoulder"]
LEFT_ELBOW = MEDIAPIPE_LANDMARK_INDEX["left_elbow"]
LEFT_WRIST = MEDIAPIPE_LANDMARK_INDEX["left_wrist"]
LEFT_INDEX = MEDIAPIPE_LANDMARK_INDEX["left_index"]
LEFT_THUMB = MEDIAPIPE_LANDMARK_INDEX["left_thumb"]
LEFT_HIP = MEDIAPIPE_LANDMARK_INDEX["left_hip"]
RIGHT_HIP = MEDIAPIPE_LANDMARK_INDEX["right_hip"]


@pytest.fixture
def synthetic_pose_data():
    data = np.full((3, 33, 3), np.nan, dtype=float)

    data[:, LEFT_SHOULDER] = np.array([[-1.0, 0.0, 0.0]] * 3)
    data[:, RIGHT_HIP] = np.array([[1.0, -2.0, 0.0]] * 3)
    data[:, LEFT_HIP] = np.array([[-1.0, -2.0, 0.0]] * 3)

    data[0, RIGHT_SHOULDER] = [1.0, 0.0, 0.0]
    data[0, RIGHT_ELBOW] = [2.0, 0.0, 0.0]
    data[0, RIGHT_WRIST] = [2.0, 1.0, 0.0]
    data[0, RIGHT_INDEX] = [2.0, 1.5, 0.0]
    data[0, RIGHT_THUMB] = [2.5, 1.0, 0.0]

    data[1, RIGHT_SHOULDER] = [1.0, 0.0, 0.0]
    data[1, RIGHT_ELBOW] = [2.0, 0.0, 0.0]
    data[1, RIGHT_WRIST] = [3.0, 0.0, 0.0]
    data[1, RIGHT_INDEX] = [3.5, 0.0, 0.0]
    data[1, RIGHT_THUMB] = [3.0, 0.5, 0.0]

    data[2, RIGHT_SHOULDER] = [1.0, 0.0, 0.0]
    data[2, RIGHT_ELBOW] = [2.0, 0.0, 0.0]
    data[2, RIGHT_WRIST] = [np.nan, np.nan, np.nan]
    data[2, RIGHT_INDEX] = [np.nan, np.nan, np.nan]
    data[2, RIGHT_THUMB] = [np.nan, np.nan, np.nan]

    data[0, LEFT_ELBOW] = [-2.0, 0.0, 0.0]
    data[0, LEFT_WRIST] = [-2.0, 1.0, 0.0]
    data[0, LEFT_INDEX] = [-2.0, 1.5, 0.0]
    data[0, LEFT_THUMB] = [-2.5, 1.0, 0.0]

    data[1, LEFT_ELBOW] = [-2.0, 0.0, 0.0]
    data[1, LEFT_WRIST] = [-3.0, 0.0, 0.0]
    data[1, LEFT_INDEX] = [-3.5, 0.0, 0.0]
    data[1, LEFT_THUMB] = [-3.0, 0.5, 0.0]

    data[2, LEFT_ELBOW] = [-2.0, 0.0, 0.0]
    data[2, LEFT_WRIST] = [-2.0, 1.0, 0.0]
    data[2, LEFT_INDEX] = [-2.0, 1.5, 0.0]
    data[2, LEFT_THUMB] = [-2.5, 1.0, 0.0]

    return data


def test_extract_arm_features_computes_expected_right_arm_geometry(synthetic_pose_data):
    features = extract_arm_features(synthetic_pose_data, arm_side="right")

    np.testing.assert_allclose(features["shoulder_to_elbow"][0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(features["elbow_to_wrist"][0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(features["upper_arm_length"][:2], [1.0, 1.0])
    np.testing.assert_allclose(features["forearm_length"][:2], [1.0, 1.0])
    np.testing.assert_allclose(features["elbow_bend_angle"][0], 90.0)
    np.testing.assert_allclose(features["elbow_bend_angle"][1], 180.0)
    np.testing.assert_allclose(features["wrist_direction"][0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(features["torso_origin"][0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(features["torso_axes"][0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(features["torso_axes"][0, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(features["torso_axes"][0, 2], [0.0, 0.0, 1.0])
    assert features["valid_mask"].tolist() == [True, True, False]
    assert features["invalid_reason"][2] == "missing required landmarks: right_wrist,right_index,right_thumb"


def test_extract_arm_features_supports_left_arm(synthetic_pose_data):
    features = extract_arm_features(synthetic_pose_data, arm_side="left")

    np.testing.assert_allclose(features["shoulder_to_elbow"][0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(features["elbow_to_wrist"][0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(features["elbow_bend_angle"][0], 90.0)
    assert features["valid_mask"].tolist() == [True, True, True]


def test_loader_supports_direct_npy_and_recording_folder_inputs(tmp_path, synthetic_pose_data):
    direct_path = tmp_path / "sample.npy"
    np.save(direct_path, synthetic_pose_data)

    direct_array, direct_meta = load_freemocap_3d_data(direct_path)

    np.testing.assert_allclose(direct_array, synthetic_pose_data, equal_nan=True)
    assert direct_meta["source_path"] == str(direct_path)
    assert direct_meta["tracker"] == "mediapipe"
    assert direct_meta["frame_count"] == 3
    assert direct_meta["point_count"] == 33

    recording_dir = tmp_path / "session_a"
    output_dir = recording_dir / "output_data"
    output_dir.mkdir(parents=True)
    recording_path = output_dir / "mediapipe_skeleton_3d.npy"
    np.save(recording_path, synthetic_pose_data)

    folder_array, folder_meta = load_freemocap_3d_data(recording_dir)

    np.testing.assert_allclose(folder_array, synthetic_pose_data, equal_nan=True)
    assert folder_meta["source_path"] == str(recording_path)
    assert folder_meta["tracker"] == "mediapipe"
    assert folder_meta["recording_folder_path"] == str(recording_dir)


def test_export_human_arm_demo_writes_default_outputs(tmp_path, synthetic_pose_data):
    recording_dir = tmp_path / "session_b"
    output_dir = recording_dir / "output_data"
    output_dir.mkdir(parents=True)
    np.save(output_dir / "mediapipe_skeleton_3d.npy", synthetic_pose_data)

    export_result = export_human_arm_demo(recording_dir, arm_side="right")

    jsonl_path = recording_dir / "output_data" / "human_arm_demo.jsonl"
    npz_path = recording_dir / "output_data" / "human_arm_demo.npz"
    metadata_path = recording_dir / "output_data" / "human_arm_demo_metadata.json"

    assert export_result["jsonl_path"] == str(jsonl_path)
    assert export_result["npz_path"] == str(npz_path)
    assert export_result["metadata_path"] == str(metadata_path)
    assert jsonl_path.exists()
    assert npz_path.exists()
    assert metadata_path.exists()

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 3
    first_record = json.loads(lines[0])
    assert first_record["frame_index"] == 0
    assert first_record["valid"] is True
    assert first_record["arm_side"] == "right"
    assert first_record["elbow_bend_angle"] == pytest.approx(90.0)

    metadata = json.loads(metadata_path.read_text())
    assert metadata["frame_count"] == 3
    assert metadata["valid_frame_count"] == 2
    assert metadata["tracker"] == "mediapipe"
    assert metadata["arm_side"] == "right"

    with np.load(npz_path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["upper_arm_length"][:2], [1.0, 1.0])
        assert archive["valid_mask"].tolist() == [True, True, False]


@pytest.mark.parametrize("entry_dir_name", ["output_data", "annotated_videos", "synchronized_videos"])
def test_export_human_arm_demo_normalizes_freemocap_subpaths(tmp_path, synthetic_pose_data, entry_dir_name):
    recording_dir = tmp_path / "session_c"
    output_dir = recording_dir / "output_data"
    entry_dir = recording_dir / entry_dir_name
    output_dir.mkdir(parents=True)
    entry_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "mediapipe_skeleton_3d.npy", synthetic_pose_data)

    export_result = export_human_arm_demo(entry_dir, arm_side="right")

    assert export_result["jsonl_path"] == str(output_dir / "human_arm_demo.jsonl")
    assert not (output_dir / "output_data").exists()


def test_extract_arm_features_requires_expected_landmark_count():
    too_small = np.zeros((2, 10, 3), dtype=float)

    with pytest.raises(ValueError, match="Expected at least 33 landmarks"):
        extract_arm_features(too_small, arm_side="right")



def test_export_human_arm_demo_rejects_non_mediapipe_trackers(tmp_path, synthetic_pose_data):
    recording_dir = tmp_path / "session_tracker"
    output_dir = recording_dir / "output_data"
    output_dir.mkdir(parents=True)
    np.save(output_dir / "mytracker_skeleton_3d.npy", synthetic_pose_data)

    with pytest.raises(ValueError, match="Only the mediapipe tracker is currently supported"):
        export_human_arm_demo(recording_dir, arm_side="right", active_tracker="mytracker")



def test_export_human_arm_demo_normalizes_mediapipe_tracker_name(tmp_path, synthetic_pose_data):
    recording_dir = tmp_path / "session_tracker_normalized"
    output_dir = recording_dir / "output_data"
    output_dir.mkdir(parents=True)
    np.save(output_dir / "mediapipe_skeleton_3d.npy", synthetic_pose_data)

    export_result = export_human_arm_demo(recording_dir, arm_side="right", active_tracker=" MediaPipe ")

    assert Path(export_result["jsonl_path"]).exists()



def test_extract_arm_features_uses_shared_torso_frame_for_both_arms(synthetic_pose_data):
    asymmetric_hips = synthetic_pose_data.copy()
    asymmetric_hips[:, LEFT_HIP] = np.array([[-3.0, -3.0, 0.0]] * 3)
    asymmetric_hips[:, RIGHT_HIP] = np.array([[1.0, -1.0, 0.0]] * 3)

    right_features = extract_arm_features(asymmetric_hips, arm_side="right")
    left_features = extract_arm_features(asymmetric_hips, arm_side="left")

    np.testing.assert_allclose(right_features["torso_origin"][:2], left_features["torso_origin"][:2], equal_nan=True)
    np.testing.assert_allclose(right_features["torso_axes"][:2], left_features["torso_axes"][:2], equal_nan=True)
