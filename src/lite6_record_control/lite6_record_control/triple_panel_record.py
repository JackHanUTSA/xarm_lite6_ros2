#!/usr/bin/env python3
"""Record a 3-panel monitoring video:
- /dev/video0 (M9)
- /dev/video2 (C920)
- RealSense color via ROS2 topic (default: /camera/camera/color/image_raw)

Optionally highlights moving foreground (robot arm) with a simple motion mask.

This is intended for monitoring robot motion runs. It does not command the robot.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None


@dataclass
class PanelConfig:
    name: str
    width: int = 640
    height: int = 480


class RealSenseSubscriber(Node):
    def __init__(self, topic: str):
        super().__init__("realsense_color_sub")
        self.topic = topic
        self.bridge = CvBridge() if CvBridge is not None else None
        self.lock = threading.Lock()
        self.last_bgr: Optional[np.ndarray] = None
        self.last_stamp = 0.0

        qos = rclpy.qos.QoSProfile(depth=1)
        qos.reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE
        self.create_subscription(Image, topic, self.cb, qos)

    def cb(self, msg: Image):
        try:
            if self.bridge is None:
                enc = msg.encoding.lower()
                if enc not in ("rgb8", "bgr8"):
                    return
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                if enc == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.last_bgr = img
                self.last_stamp = time.time()
        except Exception:
            # best-effort; ignore
            return

    def get_last(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.last_bgr is None:
                return None
            return self.last_bgr.copy()


def draw_motion_outline(frame_bgr: np.ndarray, mog2) -> np.ndarray:
    mask = mog2.apply(frame_bgr)
    # clean up
    mask = cv2.medianBlur(mask, 5)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame_bgr.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area < 800:  # ignore small noise
            continue
        cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
    return out


def start_ffmpeg_encoder(out_path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Videos" / "robot_monitor" / f"robot-3panel-{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--duration", type=float, default=10.0, help="seconds")
    ap.add_argument("--dev0", default="/dev/video0")
    ap.add_argument("--dev1", default="/dev/video2")
    ap.add_argument("--rs_topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--panel_h", type=int, default=720)
    ap.add_argument("--panel_w", type=int, default=640)
    ap.add_argument("--highlight", action="store_true", help="highlight moving foreground (robot arm) with motion outline")
    args = ap.parse_args()

    out_path = Path(args.out)

    # Open webcams
    cap0 = cv2.VideoCapture(args.dev0, cv2.CAP_V4L2)
    cap1 = cv2.VideoCapture(args.dev1, cv2.CAP_V4L2)
    for cap in (cap0, cap1):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap0.isOpened():
        print(f"ERROR: cannot open {args.dev0}", file=sys.stderr)
        return 2
    if not cap1.isOpened():
        print(f"ERROR: cannot open {args.dev1}", file=sys.stderr)
        return 2

    # ROS2 RealSense subscriber
    rclpy.init(args=None)
    rs_node = RealSenseSubscriber(args.rs_topic)
    exec_ = rclpy.executors.SingleThreadedExecutor()
    exec_.add_node(rs_node)

    stop_evt = threading.Event()

    def spin():
        try:
            while rclpy.ok() and not stop_evt.is_set():
                exec_.spin_once(timeout_sec=0.05)
        except Exception:
            # Likely shutdown race; exit quietly
            return

    th = threading.Thread(target=spin, daemon=True)
    th.start()

    # Wait for first RealSense frame
    t0 = time.time()
    while time.time() - t0 < 5.0:
        fr = rs_node.get_last()
        if fr is not None:
            break
        time.sleep(0.05)
    if rs_node.get_last() is None:
        print(f"ERROR: no frames received on {args.rs_topic}. Is realsense2_camera running?", file=sys.stderr)
        stop_evt.set()
        rs_node.destroy_node()
        rclpy.shutdown()
        return 2

    # Motion highlighters
    mog0 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    mog1 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    W = args.panel_w * 3
    H = args.panel_h

    proc = start_ffmpeg_encoder(out_path, W, H, args.fps)
    assert proc.stdin is not None

    def handle_sigint(sig, frame):
        stop_evt.set()

    signal.signal(signal.SIGINT, handle_sigint)

    n_frames = int(args.duration * args.fps)
    dt = 1.0 / args.fps

    for i in range(n_frames):
        if stop_evt.is_set():
            break
        t_frame = time.time()

        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        frs = rs_node.get_last()

        if not ok0 or f0 is None:
            print("WARN: webcam0 frame drop", file=sys.stderr)
            f0 = np.zeros((480, 640, 3), dtype=np.uint8)
        if not ok1 or f1 is None:
            print("WARN: webcam1 frame drop", file=sys.stderr)
            f1 = np.zeros((480, 640, 3), dtype=np.uint8)
        if frs is None:
            frs = np.zeros((480, 640, 3), dtype=np.uint8)

        # Resize to panel size
        f0 = cv2.resize(f0, (args.panel_w, args.panel_h), interpolation=cv2.INTER_AREA)
        f1 = cv2.resize(f1, (args.panel_w, args.panel_h), interpolation=cv2.INTER_AREA)
        frs = cv2.resize(frs, (args.panel_w, args.panel_h), interpolation=cv2.INTER_AREA)

        if args.highlight:
            f0 = draw_motion_outline(f0, mog0)
            f1 = draw_motion_outline(f1, mog1)
            frs = draw_motion_outline(frs, mog2)

        # Labels
        cv2.putText(f0, "M9 (/dev/video0)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(f1, "C920 (/dev/video2)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frs, "RealSense (ROS2 color)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        stacked = np.hstack([f0, f1, frs])
        proc.stdin.write(stacked.tobytes())

        # pace
        elapsed = time.time() - t_frame
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Cleanup
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait(timeout=30)

    cap0.release()
    cap1.release()

    stop_evt.set()
    try:
        th.join(timeout=1.0)
    except Exception:
        pass
    try:
        rs_node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    print(f"VIDEO:{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
