import json
from pathlib import Path

import numpy as np
import pytest

from lite6_imitation_bridge.calibrate_human_to_robot_frame import calibrate_from_manifest
from lite6_imitation_bridge.robot_frame_calibration import (
    RigidTransformCalibration,
    estimate_rigid_transform,
    load_calibration,
    save_calibration,
)


def _rotation_z(theta_radians: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta_radians), -np.sin(theta_radians), 0.0],
            [np.sin(theta_radians), np.cos(theta_radians), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_estimate_rigid_transform_recovers_known_transform():
    source_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    expected_rotation = _rotation_z(np.deg2rad(90.0))
    expected_translation = np.array([0.35, -0.2, 0.5], dtype=float)
    target_points = (source_points @ expected_rotation.T) + expected_translation

    calibration = estimate_rigid_transform(
        source_points,
        target_points,
        source_frame="freemocap",
        target_frame="lite6_workspace",
        notes="synthetic test",
    )

    np.testing.assert_allclose(calibration.rotation_matrix, expected_rotation, atol=1e-7)
    np.testing.assert_allclose(calibration.translation_vector, expected_translation, atol=1e-7)
    np.testing.assert_allclose(calibration.apply(source_points), target_points, atol=1e-7)
    np.testing.assert_allclose(calibration.inverse().apply(target_points), source_points, atol=1e-7)
    assert calibration.source_frame == "freemocap"
    assert calibration.target_frame == "lite6_workspace"
    assert calibration.notes == "synthetic test"
    assert calibration.fit_rmse == pytest.approx(0.0, abs=1e-10)


def test_save_and_load_round_trip_reproduces_transform(tmp_path):
    calibration = RigidTransformCalibration(
        rotation_matrix=_rotation_z(np.deg2rad(30.0)),
        translation_vector=np.array([0.1, 0.2, 0.3], dtype=float),
        source_frame="freemocap",
        target_frame="lite6_workspace",
        notes="round trip",
        fit_rmse=0.0123,
        workspace_origin=np.array([0.0, 0.0, 0.0], dtype=float),
        workspace_description="Lite6 base frame origin",
        known_pose_name="fiducial_triplet",
    )
    save_path = tmp_path / "freemocap_to_lite6.yaml"

    save_calibration(calibration, save_path)
    loaded = load_calibration(save_path)
    inverse_loaded = loaded.inverse()

    assert save_path.exists()
    np.testing.assert_allclose(loaded.rotation_matrix, calibration.rotation_matrix)
    np.testing.assert_allclose(loaded.translation_vector, calibration.translation_vector)
    np.testing.assert_allclose(loaded.workspace_origin, calibration.workspace_origin)
    assert loaded.source_frame == calibration.source_frame
    assert loaded.target_frame == calibration.target_frame
    assert loaded.notes == calibration.notes
    assert loaded.workspace_description == calibration.workspace_description
    assert loaded.known_pose_name == calibration.known_pose_name
    assert loaded.fit_rmse == calibration.fit_rmse
    assert inverse_loaded.source_frame == calibration.target_frame
    assert inverse_loaded.target_frame == calibration.source_frame
    assert inverse_loaded.workspace_origin is None
    assert inverse_loaded.workspace_description == "Derived inverse transform"


def test_transformed_wrist_trajectory_stays_in_expected_workspace_range():
    wrist_trajectory = np.array(
        [
            [-0.20, -0.10, 0.05],
            [0.00, 0.10, 0.10],
            [0.20, 0.00, 0.15],
            [0.10, -0.20, 0.00],
        ],
        dtype=float,
    )
    calibration = RigidTransformCalibration(
        rotation_matrix=np.eye(3, dtype=float),
        translation_vector=np.array([0.40, 0.00, 0.25], dtype=float),
        source_frame="freemocap",
        target_frame="lite6_workspace",
        notes="workspace range",
    )

    transformed = calibration.apply(wrist_trajectory)

    assert np.all((0.15 <= transformed[:, 0]) & (transformed[:, 0] <= 0.65))
    assert np.all((-0.25 <= transformed[:, 1]) & (transformed[:, 1] <= 0.25))
    assert np.all((0.20 <= transformed[:, 2]) & (transformed[:, 2] <= 0.45))


@pytest.mark.parametrize(
    ("source_points", "target_points", "message"),
    [
        (
            np.zeros((3, 3), dtype=float),
            np.zeros((4, 3), dtype=float),
            "matching shape",
        ),
        (
            np.zeros((2, 3), dtype=float),
            np.zeros((2, 3), dtype=float),
            "at least 3 points",
        ),
        (
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float),
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float),
            "non-collinear",
        ),
        (
            np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            np.zeros((3, 3), dtype=float),
            "finite values",
        ),
    ],
)
def test_estimate_rigid_transform_rejects_invalid_inputs(source_points, target_points, message):
    with pytest.raises(ValueError, match=message):
        estimate_rigid_transform(source_points, target_points)


def test_rigid_transform_rejects_non_orthonormal_rotation_matrix():
    with pytest.raises(ValueError, match="orthonormal"):
        RigidTransformCalibration(
            rotation_matrix=np.array(
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=float,
            ),
            translation_vector=np.array([0.0, 0.0, 0.0], dtype=float),
        )

    with pytest.raises(ValueError, match="must be strings"):
        RigidTransformCalibration(
            rotation_matrix=np.eye(3, dtype=float),
            translation_vector=np.array([0.0, 0.0, 0.0], dtype=float),
            source_frame=123,
            target_frame="lite6_workspace",
        )

    with pytest.raises(ValueError, match="must be >= 0"):
        RigidTransformCalibration(
            rotation_matrix=np.eye(3, dtype=float),
            translation_vector=np.array([0.0, 0.0, 0.0], dtype=float),
            fit_rmse=-0.1,
        )


def test_load_calibration_rejects_schema_invalid_yaml(tmp_path):
    bad_path = tmp_path / "bad_calibration.yaml"
    bad_path.write_text(
        "source_frame: freemocap\n"
        "target_frame: lite6_workspace\n"
        "rotation_matrix: [[1, 0], [0, 1]]\n"
        "translation_vector: [0, 0, 0]\n"
    )

    with pytest.raises(ValueError, match="schema"):
        load_calibration(bad_path)



def test_load_calibration_rejects_malformed_yaml_and_extra_fields(tmp_path):
    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text("rotation_matrix: [1, 0\n")

    with pytest.raises(ValueError, match="YAML"):
        load_calibration(malformed_path)

    extra_field_path = tmp_path / "extra_field.yaml"
    extra_field_path.write_text(
        "source_frame: freemocap\n"
        "target_frame: lite6_workspace\n"
        "rotation_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n"
        "translation_vector: [0, 0, 0]\n"
        "fit_rmse: 0.1\n"
        "unexpected: true\n"
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        load_calibration(extra_field_path)


def test_calibrate_from_manifest_saves_default_output_and_reports_summary(tmp_path, monkeypatch):
    source_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ],
        dtype=float,
    )
    rotation = _rotation_z(np.deg2rad(90.0))
    translation = np.array([0.35, 0.05, 0.12], dtype=float)
    target_points = (source_points @ rotation.T) + translation
    manifest_path = tmp_path / "calibration_points.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_frame": "freemocap",
                "target_frame": "lite6_workspace",
                "notes": "known pose calibration",
                "known_pose_name": "aruco_triplet",
                "workspace_origin": [0.35, 0.05, 0.12],
                "workspace_description": "Lite6 base reference point",
                "source_points": source_points.tolist(),
                "target_points": target_points.tolist(),
            }
        )
    )

    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))

    summary = calibrate_from_manifest(manifest_path)
    saved_path = Path(summary["saved_path"])
    loaded = load_calibration(saved_path)

    assert saved_path == fake_home / "ws_xarm" / "calibration" / "freemocap_to_lite6.yaml"
    assert saved_path.exists()
    assert summary["source_frame"] == "freemocap"
    assert summary["target_frame"] == "lite6_workspace"
    assert summary["point_count"] == 4
    assert summary["known_pose_name"] == "aruco_triplet"
    assert summary["workspace_description"] == "Lite6 base reference point"
    assert summary["fit_rmse"] == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(loaded.workspace_origin, [0.35, 0.05, 0.12], atol=1e-7)
    assert loaded.known_pose_name == "aruco_triplet"
    assert loaded.workspace_description == "Lite6 base reference point"
    np.testing.assert_allclose(loaded.apply(source_points), target_points, atol=1e-7)
