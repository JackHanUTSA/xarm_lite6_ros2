from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import threading
import time

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class CameraConfig:
    cam_id: str
    source: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    K: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None
    T_world: Optional[np.ndarray] = None


@dataclass
class SyncedFrame:
    timestamp: float
    frames: Dict[str, np.ndarray]
    depth_maps: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)


class CameraManager:
    """Discover, calibrate, and synchronously read all server cameras.

    Scaffold version: USB (/dev/video*) + optional RTSP URLs.
    """

    def __init__(self, rtsp_urls: Optional[List[str]] = None, calib_dir: str = "./calib"):
        self.calib_dir = Path(calib_dir)
        self.rtsp_extras = rtsp_urls or []
        self.configs: Dict[str, CameraConfig] = {}
        self.captures: Dict[str, "cv2.VideoCapture"] = {}
        self._lock = threading.Lock()
        if cv2 is None:
            raise RuntimeError("opencv-python(-headless) is required for CameraManager")
        self._discover()

    def _discover(self):
        usb_devs = sorted(Path("/dev").glob("video*"))
        for dev in usb_devs:
            cid = f"usb_{dev.name}"
            cap = cv2.VideoCapture(str(dev))
            if cap.isOpened():
                self.configs[cid] = CameraConfig(cam_id=cid, source=str(dev))
                self.captures[cid] = cap
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                cap.release()

        for url in self.rtsp_extras:
            cid = "rtsp_" + url.split("//")[-1].replace("/", "_").replace(":", "_")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                self.configs[cid] = CameraConfig(cam_id=cid, source=url)
                self.captures[cid] = cap
            else:
                cap.release()

        self._load_calibration()

    def _load_calibration(self):
        import json

        for cid, cfg in self.configs.items():
            path = self.calib_dir / f"{cid}.json"
            if path.exists():
                d = json.loads(path.read_text())
                cfg.K = np.array(d.get("K")) if d.get("K") is not None else None
                cfg.dist = np.array(d.get("dist")) if d.get("dist") is not None else None
                if "T_world" in d:
                    cfg.T_world = np.array(d["T_world"])

    def grab_synced(self) -> SyncedFrame:
        with self._lock:
            ts = time.monotonic()
            frames: Dict[str, np.ndarray] = {}
            for cid, cap in self.captures.items():
                ok, img = cap.read()
                if not ok:
                    continue
                cfg = self.configs[cid]
                if cfg.K is not None and cfg.dist is not None:
                    img = cv2.undistort(img, cfg.K, cfg.dist)
                frames[cid] = img
            return SyncedFrame(timestamp=ts, frames=frames)

    def release(self):
        for cap in self.captures.values():
            cap.release()
