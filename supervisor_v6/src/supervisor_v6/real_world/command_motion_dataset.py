from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class Episode:
    episode_id: int
    command: np.ndarray
    joint_states: np.ndarray
    images: Dict[str, List[np.ndarray]]
    timestamps: List[float]


class CommandMotionDataset:
    """HDF5 dataset storing (command, joint_states, images) per episode."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def store_episode(self, command, joint_states, synced_frames, episode_id: int):
        import h5py  # type: ignore
        import cv2  # type: ignore

        with h5py.File(self.path, "a") as f:
            grp = f.require_group(f"episode_{episode_id:05d}")
            if "command" in grp:
                del grp["command"]
            if "joint_states" in grp:
                del grp["joint_states"]
            grp.create_dataset("command", data=command)
            grp.create_dataset("joint_states", data=joint_states)
            for t, sf in enumerate(synced_frames):
                for cam_id, img in sf.frames.items():
                    if img is None:
                        continue
                    key = f"frames/{cam_id}/t{t:04d}"
                    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        grp.create_dataset(key, data=enc)

    def load_episode(self, episode_id: int) -> Episode:
        import h5py  # type: ignore
        import cv2  # type: ignore

        with h5py.File(self.path, "r") as f:
            grp = f[f"episode_{episode_id:05d}"]
            cmd = grp["command"][:]
            js = grp["joint_states"][:]
            images: Dict[str, List[np.ndarray]] = {}
            ts: List[float] = []
            if "frames" in grp:
                for cam_id in grp["frames"].keys():
                    frames: List[np.ndarray] = []
                    for key in sorted(grp[f"frames/{cam_id}"].keys()):
                        enc = grp[f"frames/{cam_id}/{key}"][:]
                        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                        frames.append(img)
                    images[cam_id] = frames
            return Episode(episode_id=episode_id, command=cmd, joint_states=js, images=images, timestamps=ts)

    def __len__(self) -> int:
        import h5py  # type: ignore

        with h5py.File(self.path, "r") as f:
            return len(f.keys())

    def joint_limit_stats(self) -> dict:
        import h5py  # type: ignore

        all_js = []
        with h5py.File(self.path, "r") as f:
            for key in f.keys():
                all_js.append(f[key]["joint_states"][:])
        stacked = np.concatenate(all_js, axis=0)
        return {
            "min": stacked.min(axis=0).tolist(),
            "max": stacked.max(axis=0).tolist(),
            "mean": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0).tolist(),
        }
