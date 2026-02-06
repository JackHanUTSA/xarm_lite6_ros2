import os
import time
import signal
import subprocess
from datetime import datetime

import rclpy
from rclpy.node import Node

from xarm_msgs.srv import Call, SetInt16, SetInt16ById, MoveJoint


class RunAndRecord(Node):
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

        # Services
        self.cli_clean_error = self.create_client(Call, '/ufactory/clean_error')
        self.cli_clean_warn = self.create_client(Call, '/ufactory/clean_warn')
        self.cli_motion_enable = self.create_client(SetInt16ById, '/ufactory/motion_enable')
        self.cli_set_mode = self.create_client(SetInt16, '/ufactory/set_mode')
        self.cli_set_state = self.create_client(SetInt16, '/ufactory/set_state')
        self.cli_move_joint = self.create_client(MoveJoint, '/ufactory/set_servo_angle')

        self.ff = None

    def _wait_srv(self, cli, name, timeout=10.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service not available: {name}')

    def _call(self, cli, req, name):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.timeout)
        if not fut.done():
            raise RuntimeError(f'timeout calling {name}')
        return fut.result()

    def start_recording(self):
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-thread_queue_size', '512',
            '-f', 'video4linux2', '-framerate', str(self.fps), '-video_size', self.size, '-i', self.cam_left,
            '-thread_queue_size', '512',
            '-f', 'video4linux2', '-framerate', str(self.fps), '-video_size', self.size, '-i', self.cam_right,
            '-filter_complex', '[0:v][1:v]hstack=inputs=2',
            '-c:v', 'libx264', '-preset', self.preset, '-crf', str(self.crf),
            '-pix_fmt', 'yuv420p',
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
        self._wait_srv(self.cli_clean_error, 'clean_error')
        self._wait_srv(self.cli_motion_enable, 'motion_enable')
        self._wait_srv(self.cli_set_mode, 'set_mode')
        self._wait_srv(self.cli_set_state, 'set_state')
        self._wait_srv(self.cli_move_joint, 'set_servo_angle')

        self._call(self.cli_clean_error, Call.Request(), 'clean_error')
        # clean_warn may not always succeed; ignore failures
        try:
            self._wait_srv(self.cli_clean_warn, 'clean_warn', timeout=2.0)
            self._call(self.cli_clean_warn, Call.Request(), 'clean_warn')
        except Exception:
            pass

        me = SetInt16ById.Request()
        me.id = 8
        me.data = 1
        self._call(self.cli_motion_enable, me, 'motion_enable')

        m = SetInt16.Request(); m.data = 0
        self._call(self.cli_set_mode, m, 'set_mode')

        s = SetInt16.Request(); s.data = 0
        self._call(self.cli_set_state, s, 'set_state')

    def move_abs(self, angles):
        req = MoveJoint.Request()
        req.angles = [float(a) for a in angles]
        req.speed = float(self.speed)
        req.acc = float(self.acc)
        req.mvtime = 0.0
        req.wait = True
        req.timeout = float(self.timeout)
        req.radius = -1.0
        req.relative = False
        res = self._call(self.cli_move_joint, req, 'set_servo_angle')
        if res.ret != 0:
            raise RuntimeError(f'move failed ret={res.ret} msg={res.message!r}')

    def run(self):
        try:
            self.arm_ready()
            self.start_recording()

            # motion sequence: 0 -> target -> 0 (joint1 only)
            self.get_logger().info(f'Moving base joint to {self.j1_target} rad and back')
            self.move_abs([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.move_abs([self.j1_target, 0.0, 0.0, 0.0, 0.0, 0.0])
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
