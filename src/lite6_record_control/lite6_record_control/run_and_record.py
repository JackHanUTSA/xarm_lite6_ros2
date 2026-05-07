import json
import os
import signal
import subprocess
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger


class RunAndRecord(Node):
    def __init__(self):
        super().__init__('run_and_record')

        self.declare_parameter('cam_left', '/dev/video2')
        self.declare_parameter('cam_right', '/dev/video0')
        self.declare_parameter('fps', 15)
        self.declare_parameter('size', '1280x720')
        self.declare_parameter('crf', 23)
        self.declare_parameter('preset', 'veryfast')
        self.declare_parameter('out_dir', os.path.expanduser('~/Videos/robot_monitor'))

        self.declare_parameter('move_wait_sec', 3.0)
        self.declare_parameter('j1_target', 0.6)
        self.declare_parameter('prepare_service', '/lite6_motion/prepare_robot')
        self.declare_parameter('home_service', '/lite6_motion/go_home')
        self.declare_parameter('status_topic', '/lite6_motion/status')
        self.declare_parameter('joint_command_topic', '/lite6_motion/joint_command')

        self.cam_left = str(self.get_parameter('cam_left').value)
        self.cam_right = str(self.get_parameter('cam_right').value)
        self.fps = int(self.get_parameter('fps').value)
        self.size = str(self.get_parameter('size').value)
        self.crf = int(self.get_parameter('crf').value)
        self.preset = str(self.get_parameter('preset').value)
        self.out_dir = str(self.get_parameter('out_dir').value)

        self.move_wait_sec = float(self.get_parameter('move_wait_sec').value)
        self.j1_target = float(self.get_parameter('j1_target').value)

        self.prepare_service = str(self.get_parameter('prepare_service').value)
        self.home_service = str(self.get_parameter('home_service').value)
        self.status_topic = str(self.get_parameter('status_topic').value)
        self.joint_command_topic = str(self.get_parameter('joint_command_topic').value)

        os.makedirs(self.out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.out_path = os.path.join(self.out_dir, f'robot-{ts}.mp4')
        self.ff = None
        self.status_payload = None

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.create_subscription(String, self.status_topic, self._on_status, qos)
        self.cmd_pub = self.create_publisher(JointState, self.joint_command_topic, 10)
        self.prepare_cli = self.create_client(Trigger, self.prepare_service)
        self.home_cli = self.create_client(Trigger, self.home_service)

    def _on_status(self, msg: String):
        try:
            self.status_payload = json.loads(msg.data)
        except Exception:
            self.status_payload = None

    def wait_for_status(self, timeout_sec=2.0):
        end = time.time() + timeout_sec
        while time.time() < end and self.status_payload is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.status_payload is None:
            raise RuntimeError('no lite6_motion status received')
        return self.status_payload

    def call_trigger(self, client):
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f'service unavailable: {client.srv_name}')
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            raise RuntimeError(f'service call failed: {client.srv_name}')
        if not result.success:
            raise RuntimeError(result.message)
        return result.message

    def start_recording(self):
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-thread_queue_size', '512',
            '-f', 'video4linux2', '-framerate', str(self.fps), '-video_size', self.size, '-i', self.cam_left,
            '-thread_queue_size', '512',
            '-f', 'video4linux2', '-framerate', str(self.fps), '-video_size', self.size, '-i', self.cam_right,
            '-filter_complex', '[0:v]scale=640:720[l];[1:v]scale=640:720[r];[l][r]hstack=inputs=2',
            '-c:v', 'libx264', '-preset', self.preset, '-crf', str(self.crf),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            self.out_path,
        ]
        self.get_logger().info(f'Starting recording -> {self.out_path}')
        self.ff = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)

    def stop_recording(self):
        if not self.ff:
            return
        self.get_logger().info('Stopping recording...')
        try:
            self.ff.send_signal(signal.SIGINT)
            self.ff.wait(timeout=10)
        except Exception:
            try:
                self.ff.kill()
            except Exception:
                pass
        self.ff = None

    def publish_joint_target(self, angles):
        msg = JointState()
        msg.name = [f'joint{i}' for i in range(1, 7)]
        msg.position = [float(a) for a in angles]
        for _ in range(5):
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self):
        try:
            self.call_trigger(self.prepare_cli)
            status = self.wait_for_status()
            start = list(status.get('joint_positions') or [0.0] * 6)
            if len(start) < 6:
                raise RuntimeError('lite6_motion status missing complete joint state')

            self.start_recording()
            self.get_logger().info(f'Moving base joint to {self.j1_target} rad and back through lite6_motion')

            target = start[:6]
            target[0] = self.j1_target
            self.publish_joint_target(target)
            time.sleep(self.move_wait_sec)

            self.publish_joint_target(start[:6])
            time.sleep(self.move_wait_sec)

            try:
                self.call_trigger(self.home_cli)
            except Exception as exc:
                self.get_logger().warning(f'go_home failed at end: {exc}')
        finally:
            self.stop_recording()

        self.get_logger().info(f'DONE video: {self.out_path}')
        return self.out_path


def main():
    rclpy.init()
    node = RunAndRecord()
    try:
        out = node.run()
        print(f'VIDEO:{out}')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
