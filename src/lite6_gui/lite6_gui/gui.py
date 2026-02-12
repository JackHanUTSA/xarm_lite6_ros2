#!/usr/bin/env python3
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

DEFAULT_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_CMD_TOPIC = "/joint_trajectory_controller/joint_trajectory"
DEFAULT_JS_TOPIC = "/joint_states"


@dataclass
class RobotSnapshot:
    stamp: float = 0.0
    joint_pos: Optional[Dict[str, float]] = None


class RosBackend(Node):
    def __init__(self, cmd_topic: str, js_topic: str):
        super().__init__("lite6_gui")
        self.cmd_topic = cmd_topic
        self.js_topic = js_topic

        self._lock = threading.Lock()
        self._snap = RobotSnapshot()

        self.pub = self.create_publisher(JointTrajectory, cmd_topic, 10)

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.create_subscription(JointState, js_topic, self._on_joint_state, qos)

    def _on_joint_state(self, msg: JointState):
        try:
            d = {n: p for n, p in zip(msg.name, msg.position)}
            with self._lock:
                self._snap = RobotSnapshot(stamp=time.time(), joint_pos=d)
        except Exception:
            return

    def get_snapshot(self) -> RobotSnapshot:
        with self._lock:
            return self._snap

    def send_trajectory(self, joint_names: List[str], positions: List[float], duration_sec: float):
        jt = JointTrajectory()
        jt.joint_names = list(joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start.sec = int(duration_sec)
        pt.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        jt.points = [pt]
        self.pub.publish(jt)


class RosThread(QtCore.QThread):
    snapshot_signal = QtCore.pyqtSignal(object)

    def __init__(self, cmd_topic: str, js_topic: str, parent=None):
        super().__init__(parent)
        self.cmd_topic = cmd_topic
        self.js_topic = js_topic
        self.node: Optional[RosBackend] = None
        self._stop = threading.Event()

    def run(self):
        rclpy.init(args=None)
        self.node = RosBackend(self.cmd_topic, self.js_topic)
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
            rclpy.shutdown()
        except Exception:
            pass

    def stop(self):
        self._stop.set()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lite6 ROS2 GUI (JointTrajectory)")

        self.enable_cb = QtWidgets.QCheckBox("ENABLE MOTION")
        self.enable_cb.setChecked(False)

        self.cmd_topic = QtWidgets.QLineEdit(DEFAULT_CMD_TOPIC)
        self.js_topic = QtWidgets.QLineEdit(DEFAULT_JS_TOPIC)
        self.joint_names = QtWidgets.QLineEdit(",".join(DEFAULT_JOINTS))

        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(0.1, 30.0)
        self.duration.setSingleStep(0.1)
        self.duration.setValue(2.0)
        self.duration.setSuffix(" s")

        self.status = QtWidgets.QLabel("ROS: starting…")

        self.sliders: List[QtWidgets.QSlider] = []
        self.spinboxes: List[QtWidgets.QDoubleSpinBox] = []

        top = QtWidgets.QGridLayout()
        row = 0
        top.addWidget(QtWidgets.QLabel("Command topic"), row, 0)
        top.addWidget(self.cmd_topic, row, 1, 1, 3)
        row += 1
        top.addWidget(QtWidgets.QLabel("JointState topic"), row, 0)
        top.addWidget(self.js_topic, row, 1, 1, 3)
        row += 1
        top.addWidget(QtWidgets.QLabel("Joint names (csv)"), row, 0)
        top.addWidget(self.joint_names, row, 1, 1, 3)
        row += 1

        top.addWidget(QtWidgets.QLabel("Move duration"), row, 0)
        top.addWidget(self.duration, row, 1)
        top.addWidget(self.enable_cb, row, 2)
        row += 1

        joints_box = QtWidgets.QGroupBox("Joint Targets (rad)")
        joints_layout = QtWidgets.QGridLayout()
        for i in range(6):
            lab = QtWidgets.QLabel(f"J{i+1}")
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(-628, 628)  # -6.28 .. +6.28
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

        self.exec_btn = QtWidgets.QPushButton("EXECUTE (Go to target)")
        self.exec_btn.clicked.connect(self.on_execute)

        v = QtWidgets.QVBoxLayout()
        v.addLayout(top)
        v.addWidget(joints_box)
        v.addWidget(self.exec_btn)
        v.addWidget(self.status)
        self.setLayout(v)

        self.ros_thread = RosThread(self.cmd_topic.text(), self.js_topic.text())
        self.ros_thread.snapshot_signal.connect(self.on_snapshot)
        self.ros_thread.start()

    def closeEvent(self, event):
        try:
            self.ros_thread.stop()
            self.ros_thread.wait(1000)
        except Exception:
            pass
        event.accept()

    def on_snapshot(self, snap: RobotSnapshot):
        age = time.time() - (snap.stamp or 0.0)
        if snap.joint_pos:
            self.status.setText(f"ROS OK | joint_states age={age:.2f}s | cmd_topic={self.cmd_topic.text()}")
        else:
            self.status.setText(f"Waiting for {self.js_topic.text()} … | cmd_topic={self.cmd_topic.text()}")

    def on_execute(self):
        if not self.enable_cb.isChecked():
            QtWidgets.QMessageBox.warning(self, "Safety", "Motion is disabled. Check ENABLE MOTION first.")
            return

        ret = QtWidgets.QMessageBox.question(
            self,
            "Confirm motion",
            "Send a JointTrajectory command? Make sure the workspace is clear.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return

        joint_names = [s.strip() for s in self.joint_names.text().split(",") if s.strip()]
        if len(joint_names) != 6:
            QtWidgets.QMessageBox.critical(self, "Config", "Joint names must have 6 entries.")
            return

        positions = [sp.value() for sp in self.spinboxes]
        dur = float(self.duration.value())

        node = self.ros_thread.node
        if node is None:
            QtWidgets.QMessageBox.critical(self, "ROS", "ROS node not ready")
            return

        node.send_trajectory(joint_names, positions, dur)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(860, 560)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
