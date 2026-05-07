from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_DATA_FOLDER_NAME = "output_data"
ANNOTATED_VIDEOS_FOLDER_NAME = "annotated_videos"
SYNCHRONIZED_VIDEOS_FOLDER_NAME = "synchronized_videos"
DATA_3D_NPY_FILE_NAME = "skeleton_3d.npy"
OLD_DATA_3D_NPY_FILE_NAME = "mediaPipeSkel_3d_body_hands_face.npy"


def _normalize_tracker_name(active_tracker: str | None) -> str:
    tracker = (active_tracker or "mediapipe").strip().lower()
    return tracker or "mediapipe"


def _tracker_prefix(active_tracker: str) -> str:
    return active_tracker if active_tracker.endswith("_") else f"{active_tracker}_"


def _normalize_recording_folder_path(path: Path) -> Path:
    if path.name in {OUTPUT_DATA_FOLDER_NAME, ANNOTATED_VIDEOS_FOLDER_NAME, SYNCHRONIZED_VIDEOS_FOLDER_NAME}:
        return path.parent
    return path


def normalize_freemocap_recording_path(input_path: str | Path) -> Path:
    return _normalize_recording_folder_path(Path(input_path).expanduser().resolve())


def _resolve_data_path(input_path: Path, active_tracker: str) -> tuple[Path, str | None]:
    if input_path.suffix == ".npy":
        return input_path, None

    recording_folder = _normalize_recording_folder_path(input_path)
    output_data_path = recording_folder / OUTPUT_DATA_FOLDER_NAME
    preferred_path = output_data_path / f"{_tracker_prefix(active_tracker)}{DATA_3D_NPY_FILE_NAME}"
    if preferred_path.exists():
        return preferred_path, str(recording_folder)

    if active_tracker == "mediapipe":
        legacy_path = output_data_path / OLD_DATA_3D_NPY_FILE_NAME
        if legacy_path.exists():
            return legacy_path, str(recording_folder)

    raise FileNotFoundError(
        f"Could not locate FreeMoCap 3D data for tracker '{active_tracker}' under {recording_folder}"
    )


def load_freemocap_3d_data(input_path: str | Path, active_tracker: str = "mediapipe") -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(input_path).expanduser().resolve()
    tracker = _normalize_tracker_name(active_tracker)
    data_path, recording_folder_path = _resolve_data_path(path, tracker)
    data = np.load(data_path)

    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError(f"Expected 3D landmark array shaped (frames, points, 3), got {data.shape}")

    metadata: dict[str, Any] = {
        "input_path": str(path),
        "source_path": str(data_path),
        "tracker": tracker,
        "frame_count": int(data.shape[0]),
        "point_count": int(data.shape[1]),
    }
    if recording_folder_path is not None:
        metadata["recording_folder_path"] = recording_folder_path

    return data, metadata
