from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from yaml import YAMLError


DEFAULT_SOURCE_FRAME = "freemocap"
DEFAULT_TARGET_FRAME = "lite6_workspace"


@dataclass(frozen=True)
class RigidTransformCalibration:
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    source_frame: str = DEFAULT_SOURCE_FRAME
    target_frame: str = DEFAULT_TARGET_FRAME
    notes: str | None = None
    fit_rmse: float | None = None
    workspace_origin: np.ndarray | None = None
    workspace_description: str | None = None
    known_pose_name: str | None = None

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_matrix, dtype=float)
        translation = np.asarray(self.translation_vector, dtype=float)
        workspace_origin = None if self.workspace_origin is None else np.asarray(self.workspace_origin, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_matrix must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation_vector must have shape (3,)")
        if workspace_origin is not None and workspace_origin.shape != (3,):
            raise ValueError("workspace_origin must have shape (3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)) or (
            workspace_origin is not None and not np.all(np.isfinite(workspace_origin))
        ):
            raise ValueError("rotation_matrix, translation_vector, and workspace_origin must contain only finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation_matrix must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("rotation_matrix must have determinant 1")
        if not isinstance(self.source_frame, str) or not isinstance(self.target_frame, str):
            raise ValueError("source_frame and target_frame must be strings")
        object.__setattr__(self, "rotation_matrix", rotation)
        object.__setattr__(self, "translation_vector", translation)
        object.__setattr__(self, "workspace_origin", workspace_origin)
        if self.fit_rmse is not None and not np.isfinite(self.fit_rmse):
            raise ValueError("fit_rmse must be finite when provided")
        if self.fit_rmse is not None and float(self.fit_rmse) < 0.0:
            raise ValueError("fit_rmse must be >= 0")

    def apply(self, points: np.ndarray) -> np.ndarray:
        points_array = _coerce_points(points)
        return (points_array @ self.rotation_matrix.T) + self.translation_vector

    def inverse(self) -> "RigidTransformCalibration":
        inverse_rotation = self.rotation_matrix.T
        inverse_translation = -(self.translation_vector @ self.rotation_matrix)
        return RigidTransformCalibration(
            rotation_matrix=inverse_rotation,
            translation_vector=inverse_translation,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            notes=self.notes,
            fit_rmse=self.fit_rmse,
            workspace_origin=None,
            workspace_description="Derived inverse transform",
            known_pose_name=self.known_pose_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "notes": self.notes,
            "fit_rmse": None if self.fit_rmse is None else float(self.fit_rmse),
            "workspace_origin": None if self.workspace_origin is None else self.workspace_origin.tolist(),
            "workspace_description": self.workspace_description,
            "known_pose_name": self.known_pose_name,
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RigidTransformCalibration":
        return cls(
            rotation_matrix=np.asarray(data["rotation_matrix"], dtype=float),
            translation_vector=np.asarray(data["translation_vector"], dtype=float),
            source_frame=data.get("source_frame", DEFAULT_SOURCE_FRAME),
            target_frame=data.get("target_frame", DEFAULT_TARGET_FRAME),
            notes=data.get("notes"),
            fit_rmse=data.get("fit_rmse"),
            workspace_origin=None if data.get("workspace_origin") is None else np.asarray(data.get("workspace_origin"), dtype=float),
            workspace_description=data.get("workspace_description"),
            known_pose_name=data.get("known_pose_name"),
        )


def _validate_calibration_dict(data: dict[str, Any]) -> None:
    required_fields = {"source_frame", "target_frame", "rotation_matrix", "translation_vector"}
    allowed_fields = required_fields | {"notes", "fit_rmse", "workspace_origin", "workspace_description", "known_pose_name"}
    missing_fields = sorted(required_fields - set(data.keys()))
    if missing_fields:
        raise ValueError(f"Calibration schema validation failed: missing required fields {missing_fields}")
    extra_fields = sorted(set(data.keys()) - allowed_fields)
    if extra_fields:
        raise ValueError(f"Calibration schema validation failed: unexpected fields {extra_fields}")
    if not isinstance(data["source_frame"], str) or not isinstance(data["target_frame"], str):
        raise ValueError("Calibration schema validation failed: source_frame and target_frame must be strings")
    rotation = np.asarray(data["rotation_matrix"], dtype=float)
    translation = np.asarray(data["translation_vector"], dtype=float)
    workspace_origin_value = data.get("workspace_origin")
    workspace_origin = None if workspace_origin_value is None else np.asarray(workspace_origin_value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("Calibration schema validation failed: rotation_matrix must have shape (3, 3)")
    if translation.shape != (3,):
        raise ValueError("Calibration schema validation failed: translation_vector must have shape (3,)")
    if workspace_origin is not None and workspace_origin.shape != (3,):
        raise ValueError("Calibration schema validation failed: workspace_origin must have shape (3,)")
    if "fit_rmse" in data and data["fit_rmse"] is not None and float(data["fit_rmse"]) < 0.0:
        raise ValueError("Calibration schema validation failed: fit_rmse must be >= 0")


def _coerce_points(points: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    if not np.all(np.isfinite(points_array)):
        raise ValueError("Points must contain only finite values")
    return points_array


def _validate_point_sets(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = _coerce_points(source_points)
    target = _coerce_points(target_points)
    if source.shape != target.shape:
        raise ValueError("Source and target point sets must have matching shape")
    if source.shape[0] < 3:
        raise ValueError("Rigid transform estimation requires at least 3 points")
    centered = source - source.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("Rigid transform estimation requires at least 3 non-collinear points")
    return source, target


def estimate_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_frame: str = DEFAULT_SOURCE_FRAME,
    target_frame: str = DEFAULT_TARGET_FRAME,
    notes: str | None = None,
    workspace_origin: np.ndarray | None = None,
    workspace_description: str | None = None,
    known_pose_name: str | None = None,
) -> RigidTransformCalibration:
    source, target = _validate_point_sets(source_points, target_points)

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    covariance = source_centered.T @ target_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T

    translation = target_centroid - (source_centroid @ rotation.T)
    calibration = RigidTransformCalibration(
        rotation_matrix=rotation,
        translation_vector=translation,
        source_frame=source_frame,
        target_frame=target_frame,
        notes=notes,
        workspace_origin=workspace_origin,
        workspace_description=workspace_description,
        known_pose_name=known_pose_name,
    )
    residuals = calibration.apply(source) - target
    fit_rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    return RigidTransformCalibration(
        rotation_matrix=rotation,
        translation_vector=translation,
        source_frame=source_frame,
        target_frame=target_frame,
        notes=notes,
        fit_rmse=fit_rmse,
        workspace_origin=workspace_origin,
        workspace_description=workspace_description,
        known_pose_name=known_pose_name,
    )


def save_calibration(calibration: RigidTransformCalibration, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(calibration.to_dict(), sort_keys=False))
    return path


def load_calibration(input_path: str | Path) -> RigidTransformCalibration:
    path = Path(input_path).expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text())
    except YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Calibration file must contain a mapping")
    _validate_calibration_dict(data)
    return RigidTransformCalibration.from_dict(data)
