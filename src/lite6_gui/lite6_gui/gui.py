#!/usr/bin/env python3
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtWidgets

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

DEFAULT_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_CMD_TOPIC = "/lite6_motion/joint_command"
DEFAULT_STATUS_TOPIC = "/lite6_motion/status"
DEFAULT_PREPARE_SERVICE = "/lite6_motion/prepare_robot"
DEFAULT_HOME_SERVICE = "/lite6_motion/go_home"


def load_preview_summary(preview_path: str) -> dict:
    from lite6_imitation_bridge.preview_targets import summarize_preview_artifact

    return summarize_preview_artifact(Path(preview_path).expanduser().resolve())


def format_preview_summary_label(summary: dict) -> str:
    frame_range = summary.get("frame_index_range")
    range_text = "none" if frame_range is None else f"{frame_range[0]}-{frame_range[1]}"
    return (
        "Preview only | "
        f"frames={summary.get('frame_count', 0)} | "
        f"valid={summary.get('valid_frame_count', 0)} | "
        f"invalid={summary.get('invalid_frame_count', 0)} | "
        f"clamped={summary.get('clamped_frame_count', 0)} | "
        f"range={range_text}"
    )


@dataclass
class MotionSnapshot:
    stamp: float = 0.0
    ready: bool = False
    reason: str = "waiting for status"
    enabled: bool = False
    has_error: bool = False
    mode: int = -1
    state: int = -1
    joint_names: List[str] = field(default_factory=lambda: DEFAULT_JOINTS.copy())
    joint_positions: List[float] = field(default_factory=lambda: [0.0] * 6)
    last_command: str = "none"


class RosBackend(Node):
    def __init__(self, cmd_topic: str, status_topic: str, prepare_service: str, home_service: str):
        super().__init__("lite6_gui")
        self.cmd_topic = cmd_topic
        self.status_topic = status_topic
        self.prepare_service = prepare_service
        self.home_service = home_service

        self._lock = threading.Lock()
        self._snap = MotionSnapshot()

        self.pub = self.create_publisher(JointState, cmd_topic, 10)
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.create_subscription(String, status_topic, self._on_status, qos)
        self.prepare_cli = self.create_client(Trigger, prepare_service)
        self.home_cli = self.create_client(Trigger, home_service)

    def _on_status(self, msg: String):
        try:
            payload = json.loads(msg.data)
            with self._lock:
                self._snap = MotionSnapshot(
                    stamp=time.time(),
                    ready=bool(payload.get("ready", False)),
                    reason=str(payload.get("reason", "unknown")),
                    enabled=bool(payload.get("enabled", False)),
                    has_error=bool(payload.get("has_error", False)),
                    mode=int(payload.get("mode", -1)),
                    state=int(payload.get("state", -1)),
                    joint_names=list(payload.get("joint_names") or DEFAULT_JOINTS),
                    joint_positions=list(payload.get("joint_positions") or [0.0] * 6),
                    last_command=str(payload.get("last_command", "none")),
                )
        except Exception:
            return

    def get_snapshot(self) -> MotionSnapshot:
        with self._lock:
            return self._snap

    def send_joint_command(self, joint_names: List[str], positions: List[float]):
        msg = JointState()
        msg.name = list(joint_names)
        msg.position = list(positions)
        for _ in range(5):
            self.pub.publish(msg)

    def call_trigger(self, client) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=1.0):
            return False, "service unavailable"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            return False, "service call failed"
        return bool(result.success), str(result.message)


class RosThread(QtCore.QThread):
    snapshot_signal = QtCore.pyqtSignal(object)

    def __init__(self, cmd_topic: str, status_topic: str, prepare_service: str, home_service: str, parent=None):
        super().__init__(parent)
        self.cmd_topic = cmd_topic
        self.status_topic = status_topic
        self.prepare_service = prepare_service
        self.home_service = home_service
        self.node: Optional[RosBackend] = None
        self._stop = threading.Event()

    def run(self):
        rclpy.init(args=None)
        self.node = RosBackend(self.cmd_topic, self.status_topic, self.prepare_service, self.home_service)
        exec_ = rclpy.executors.SingleThreadedExecutor()
        exec_.add_node(self.node)
        try:
            while rclpy.ok() and not self._stop.is_set():
                exec_.spin_once(timeout_sec=0.05)
                if self.node is not None:
                    self.snapshot_signal.emit(self.node.get_snapshot())
        except Exception:
            pass
        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass

    def stop(self):
        self._stop.set()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lite6 ROS2 GUI (lite6_motion)")
        self._last_snapshot = MotionSnapshot()

        self.enable_cb = QtWidgets.QCheckBox("ENABLE MOTION")
        self.enable_cb.setChecked(False)

        self.cmd_topic = QtWidgets.QLineEdit(DEFAULT_CMD_TOPIC)
        self.status_topic = QtWidgets.QLineEdit(DEFAULT_STATUS_TOPIC)
        self.prepare_service = QtWidgets.QLineEdit(DEFAULT_PREPARE_SERVICE)
        self.home_service = QtWidgets.QLineEdit(DEFAULT_HOME_SERVICE)
        self.joint_names = QtWidgets.QLineEdit(",".join(DEFAULT_JOINTS))
        self.preview_path = QtWidgets.QLineEdit()

        self.status = QtWidgets.QLabel("ROS: starting…")
        self.preview_status = QtWidgets.QLabel("Preview: no target artifact loaded")

        self.sliders: List[QtWidgets.QSlider] = []
        self.spinboxes: List[QtWidgets.QDoubleSpinBox] = []

        top = QtWidgets.QGridLayout()
        row = 0
        top.addWidget(QtWidgets.QLabel("Joint command topic"), row, 0)
        top.addWidget(self.cmd_topic, row, 1, 1, 3)
        row += 1
        top.addWidget(QtWidgets.QLabel("Status topic"), row, 0)
        top.addWidget(self.status_topic, row, 1, 1, 3)
        row += 1
        top.addWidget(QtWidgets.QLabel("Prepare service"), row, 0)
        top.addWidget(self.prepare_service, row, 1, 1, 3)
        row += 1
        top.addWidget(QtWidgets.QLabel("Home service"), row, 0)
        top.addWidget(self.home_service, row, 1, 1, 3)
        row += 1
        top.addWidget(QtWidgets.QLabel("Joint names (csv)"), row, 0)
        top.addWidget(self.joint_names, row, 1, 1, 3)
        row += 1
        top.addWidget(QtWidgets.QLabel("Preview artifact"), row, 0)
        top.addWidget(self.preview_path, row, 1, 1, 3)
        row += 1
        top.addWidget(self.enable_cb, row, 2)

        joints_box = QtWidgets.QGroupBox("Joint Targets (rad)")
        joints_layout = QtWidgets.QGridLayout()
        for i in range(6):
            lab = QtWidgets.QLabel(f"J{i+1}")
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(-628, 628)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-6.28, 6.28)
            sp.setSingleStep(0.01)
            sp.setDecimals(3)

            def mk_sync(sl, spin):
                sl.valueChanged.connect(lambda v: spin.setValue(v / 100.0))
                spin.valueChanged.connect(lambda v: sl.setValue(int(v * 100.0)))

            mk_sync(s, sp)
            self.sliders.append(s)
            self.spinboxes.append(sp)
            joints_layout.addWidget(lab, i, 0)
            joints_layout.addWidget(s, i, 1)
            joints_layout.addWidget(sp, i, 2)
        joints_box.setLayout(joints_layout)

        self.sync_btn = QtWidgets.QPushButton("SYNC FROM STATUS")
        self.sync_btn.clicked.connect(self.on_sync_from_status)
        self.prepare_btn = QtWidgets.QPushButton("PREPARE ROBOT")
        self.prepare_btn.clicked.connect(self.on_prepare)
        self.home_btn = QtWidgets.QPushButton("GO HOME")
        self.home_btn.clicked.connect(self.on_home)
        self.preview_btn = QtWidgets.QPushButton("LOAD PREVIEW SUMMARY")
        self.preview_btn.clicked.connect(self.on_load_preview)
        self.exec_btn = QtWidgets.QPushButton("EXECUTE (Publish joint target)")
        self.exec_btn.clicked.connect(self.on_execute)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.sync_btn)
        btn_row.addWidget(self.prepare_btn)
        btn_row.addWidget(self.home_btn)
        btn_row.addWidget(self.exec_btn)

        v = QtWidgets.QVBoxLayout()
        v.addLayout(top)
        v.addWidget(joints_box)
        v.addLayout(btn_row)
        v.addWidget(self.preview_status)
        v.addWidget(self.status)
        self.setLayout(v)

        self.ros_thread = RosThread(
            self.cmd_topic.text(),
            self.status_topic.text(),
            self.prepare_service.text(),
            self.home_service.text(),
        )
        self.ros_thread.snapshot_signal.connect(self.on_snapshot)
        self.ros_thread.start()

    def closeEvent(self, event):
        try:
            self.ros_thread.stop()
            self.ros_thread.wait(1000)
        except Exception:
            pass
        event.accept()

    def on_snapshot(self, snap: MotionSnapshot):
        self._last_snapshot = snap
        age = time.time() - (snap.stamp or 0.0)
        self.status.setText(
            f"lite6_motion | ready={snap.ready} | reason={snap.reason} | age={age:.2f}s | last={snap.last_command}"
        )

    def on_sync_from_status(self):
        positions = list(self._last_snapshot.joint_positions)
        if len(positions) < 6:
            QtWidgets.QMessageBox.warning(self, "Status", "No complete joint snapshot available yet.")
            return
        for spin, value in zip(self.spinboxes, positions[:6]):
            spin.setValue(float(value))

    def on_load_preview(self):
        preview_path = self.preview_path.text().strip()
        if not preview_path:
            QtWidgets.QMessageBox.warning(self, "Preview", "Enter a lite6_targets.jsonl or lite6_targets_report.json path.")
            return
        try:
            summary = load_preview_summary(preview_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Preview", f"Failed to load preview summary: {exc}")
            return
        self.preview_status.setText(format_preview_summary_label(summary))

    def on_prepare(self):
        node = self.ros_thread.node
        if node is None:
            QtWidgets.QMessageBox.critical(self, "ROS", "ROS node not ready")
            return
        ok, message = node.call_trigger(node.prepare_cli)
        if ok:
            QtWidgets.QMessageBox.information(self, "Prepare", message)
        else:
            QtWidgets.QMessageBox.critical(self, "Prepare", message)

    def on_home(self):
        if not self.enable_cb.isChecked():
            QtWidgets.QMessageBox.warning(self, "Safety", "Motion is disabled. Check ENABLE MOTION first.")
            return
        ret = QtWidgets.QMessageBox.question(
            self,
            "Confirm home motion",
            "Send go_home through lite6_motion? Make sure the workspace is clear.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return
        node = self.ros_thread.node
        if node is None:
            QtWidgets.QMessageBox.critical(self, "ROS", "ROS node not ready")
            return
        ok, message = node.call_trigger(node.home_cli)
        if ok:
            QtWidgets.QMessageBox.information(self, "Home", message)
        else:
            QtWidgets.QMessageBox.critical(self, "Home", message)

    def on_execute(self):
        if not self.enable_cb.isChecked():
            QtWidgets.QMessageBox.warning(self, "Safety", "Motion is disabled. Check ENABLE MOTION first.")
            return
        ret = QtWidgets.QMessageBox.question(
            self,
            "Confirm motion",
            "Publish joint target to lite6_motion? Make sure the workspace is clear.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return

        joint_names = [s.strip() for s in self.joint_names.text().split(",") if s.strip()]
        if len(joint_names) != 6:
            QtWidgets.QMessageBox.critical(self, "Config", "Joint names must have 6 entries.")
            return

        positions = [sp.value() for sp in self.spinboxes]
        node = self.ros_thread.node
        if node is None:
            QtWidgets.QMessageBox.critical(self, "ROS", "ROS node not ready")
            return

        node.send_joint_command(joint_names, positions)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(920, 620)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
