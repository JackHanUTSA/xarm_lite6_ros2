import os
import time
import signal
import subprocess
from datetime import datetime

import rclpy
from rclpy.node import Node

from xarm_msgs.srv import Call, SetInt16, SetInt16ById, MoveJoint


class RunAndRecord(Node):
    def ros2_service_call(self, cmd: str):
        # Run via shell to avoid rclpy executor invalid-handle issues
        import subprocess
        full = ["bash", "-lc", cmd]
        p = subprocess.run(full, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Command failed ({p.returncode}): {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
        return p.stdout

    def __init__(self):
        super().__init__('run_and_record')

        # Recording params
        self.declare_parameter('cam_left', '/dev/video2')   # C920
        self.declare_parameter('cam_right', '/dev/video0')  # M9
        self.declare_parameter('fps', 15)
        self.declare_parameter('size', '1280x720')
        self.declare_parameter('crf', 23)
        self.declare_parameter('preset', 'veryfast')
        self.declare_parameter('out_dir', os.path.expanduser('~/Videos/robot_monitor'))

        # Motion params
        self.declare_parameter('speed', 0.3)   # radians/sec-ish per vendor; keep conservative
        self.declare_parameter('acc', 1.0)
        self.declare_parameter('timeout', 60.0)

        # Default motion: base joint sweep (radians)
        self.declare_parameter('j1_target', 0.6)  # ~34 deg

        self.cam_left = str(self.get_parameter('cam_left').value)
        self.cam_right = str(self.get_parameter('cam_right').value)
        self.fps = int(self.get_parameter('fps').value)
        self.size = str(self.get_parameter('size').value)
        self.crf = int(self.get_parameter('crf').value)
        self.preset = str(self.get_parameter('preset').value)
        self.out_dir = str(self.get_parameter('out_dir').value)

        self.speed = float(self.get_parameter('speed').value)
        self.acc = float(self.get_parameter('acc').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.j1_target = float(self.get_parameter('j1_target').value)

        os.makedirs(self.out_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.out_path = os.path.join(self.out_dir, f'robot-{ts}.mp4')
        self.ff = None

    def _wait_srv(self, cli, name, timeout=10.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service not available: {name}')
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
        time.sleep(0.8)  # let buffers fill

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

    def arm_ready(self):
        # best-effort clear + enable + mode/state
        self.ros2_service_call(
            'cd ~/ws_xarm && source /opt/ros/humble/setup.bash && source ~/ws_xarm/install/setup.bash && ' +
            'ros2 service call /ufactory/clean_error xarm_msgs/srv/Call "{}"'
        )
        try:
            self.ros2_service_call(
                'cd ~/ws_xarm && source /opt/ros/humble/setup.bash && source ~/ws_xarm/install/setup.bash && ' +
                'ros2 service call /ufactory/clean_warn xarm_msgs/srv/Call \"{}\"'
            )
        except Exception:
            pass
        self.ros2_service_call(
            'cd ~/ws_xarm && source /opt/ros/humble/setup.bash && source ~/ws_xarm/install/setup.bash && ' +
            'ros2 service call /ufactory/motion_enable xarm_msgs/srv/SetInt16ById "{id: 8, data: 1}"'
        )
        self.ros2_service_call(
            'cd ~/ws_xarm && source /opt/ros/humble/setup.bash && source ~/ws_xarm/install/setup.bash && ' +
            'ros2 service call /ufactory/set_mode xarm_msgs/srv/SetInt16 "{data: 0}"'
        )
        self.ros2_service_call(
            'cd ~/ws_xarm && source /opt/ros/humble/setup.bash && source ~/ws_xarm/install/setup.bash && ' +
            'ros2 service call /ufactory/set_state xarm_msgs/srv/SetInt16 "{data: 0}"'
        )

    def move_abs(self, angles):
        # angles in radians, absolute
        angles_str = '[' + ', '.join(f'{float(a):.6f}' for a in angles) + ']'
        cmd = (
            'cd ~/ws_xarm && source /opt/ros/humble/setup.bash && source ~/ws_xarm/install/setup.bash && ' +
            'ros2 service call /ufactory/set_servo_angle xarm_msgs/srv/MoveJoint ' +
            f"\"{{angles: {angles_str}, speed: {self.speed}, acc: {self.acc}, mvtime: 0.0, wait: true, timeout: {self.timeout}, radius: -1.0, relative: false}}\""
        )
        out = self.ros2_service_call(cmd)
        if 'ret=0' not in out and 'ret: 0' not in out:
            raise RuntimeError('MoveJoint did not report success')

    def run(self):

        try:
            self.arm_ready()
            self.start_recording()

            # Ensure starting at zero
            self.move_abs([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            self.get_logger().info(f'Moving base joint to {self.j1_target} rad and back')
            self.move_abs([self.j1_target, 0.0, 0.0, 0.0, 0.0, 0.0])
            # Always return to zero at end
            self.move_abs([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        finally:
            self.stop_recording()

        self.get_logger().info(f'DONE video: {self.out_path}')
        return self.out_path


def main():
    rclpy.init()
    node = RunAndRecord()
    try:
        out = node.run()
        # print a machine-friendly marker
        print(f'VIDEO:{out}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
