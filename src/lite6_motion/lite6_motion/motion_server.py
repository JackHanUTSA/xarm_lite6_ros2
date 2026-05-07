import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import Call, MoveJoint, SetInt16, SetInt16ById

from .robot_client import Lite6RobotClient
from .safety_gate import MotionLimits, RobotHealth, SafetyGate


class Lite6MotionServer(Node):
    def __init__(self):
        super().__init__('lite6_motion_server')
        self.declare_parameter('joint_states_topic', '/ufactory/joint_states')
        self.declare_parameter('robot_states_topic', '/ufactory/robot_states')
        self.declare_parameter('status_topic', '/lite6_motion/status')
        self.declare_parameter('joint_command_topic', '/lite6_motion/joint_command')
        self.declare_parameter('joint_position_limit', 3.14)
        self.declare_parameter('stale_state_timeout_sec', 0.5)
        self.declare_parameter('default_speed', 0.25)
        self.declare_parameter('default_acc', 0.5)
        self.declare_parameter('default_timeout', 15.0)

        limits = MotionLimits(
            joint_position_limit=float(self.get_parameter('joint_position_limit').value),
            stale_state_timeout_sec=float(self.get_parameter('stale_state_timeout_sec').value),
        )
        self.safety_gate = SafetyGate(limits)
        self.robot_client = Lite6RobotClient(
            default_speed=float(self.get_parameter('default_speed').value),
            default_acc=float(self.get_parameter('default_acc').value),
            default_timeout=float(self.get_parameter('default_timeout').value),
        )

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = QoSDurabilityPolicy.VOLATILE

        self._joint_names: list[str] = []
        self._joint_positions: list[float] = []
        self._joint_state_stamp: Optional[float] = None
        self._robot_enabled = False
        self._robot_has_error = False
        self._robot_mode = -1
        self._robot_state = -1
        self._last_command_summary = 'none'
        self._stop_requested = False

        self.create_subscription(JointState, self.get_parameter('joint_states_topic').value, self._on_joint_state, qos)
        self.create_subscription(RobotMsg, self.get_parameter('robot_states_topic').value, self._on_robot_state, qos)
        self.create_subscription(JointState, self.get_parameter('joint_command_topic').value, self._on_joint_command, 10)

        self.status_pub = self.create_publisher(String, self.get_parameter('status_topic').value, 10)
        self.prepare_srv = self.create_service(Trigger, '/lite6_motion/prepare_robot', self._handle_prepare)
        self.home_srv = self.create_service(Trigger, '/lite6_motion/go_home', self._handle_home)
        self.stop_srv = self.create_service(Trigger, '/lite6_motion/stop', self._handle_stop)

        self.clean_error_cli = self.create_client(Call, '/ufactory/clean_error')
        self.clean_warn_cli = self.create_client(Call, '/ufactory/clean_warn')
        self.motion_enable_cli = self.create_client(SetInt16ById, '/ufactory/motion_enable')
        self.set_mode_cli = self.create_client(SetInt16, '/ufactory/set_mode')
        self.set_state_cli = self.create_client(SetInt16, '/ufactory/set_state')
        self.move_joint_cli = self.create_client(MoveJoint, '/ufactory/set_servo_angle')

        self.create_timer(0.25, self._publish_status)

    def _on_joint_state(self, msg: JointState):
        self._joint_names = list(msg.name)
        self._joint_positions = [float(value) for value in msg.position]
        self._joint_state_stamp = time.time()

    def _on_robot_state(self, msg: RobotMsg):
        try:
            self._robot_enabled = bool(msg.motor_brake_states and all(state == 0 for state in msg.motor_brake_states))
        except Exception:
            self._robot_enabled = False
        self._robot_has_error = bool(getattr(msg, 'err', 0))
        self._robot_mode = int(getattr(msg, 'mode', -1))
        self._robot_state = int(getattr(msg, 'state', -1))

    def _current_health(self) -> RobotHealth:
        age = 1e9 if self._joint_state_stamp is None else max(0.0, time.time() - self._joint_state_stamp)
        return RobotHealth(
            enabled=self._robot_enabled,
            has_error=self._robot_has_error,
            mode=self._robot_mode,
            state=self._robot_state,
            joint_names=self._joint_names,
            joint_positions=self._joint_positions,
            last_state_age_sec=age,
        )

    def _publish_status(self):
        status = self.safety_gate.build_status(self._current_health())
        status['last_command'] = self._last_command_summary
        status['stop_requested'] = self._stop_requested
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(message)

    def _call_triggerish_service(self, client, request):
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f'service unavailable: {client.srv_name}')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            raise RuntimeError(f'service call failed: {client.srv_name}')
        return future.result()

    def _handle_prepare(self, _request, response):
        try:
            for call in self.robot_client.prepare_sequence():
                if call.service_name.endswith('clean_error'):
                    req = Call.Request()
                    self._call_triggerish_service(self.clean_error_cli, req)
                elif call.service_name.endswith('clean_warn'):
                    req = Call.Request()
                    self._call_triggerish_service(self.clean_warn_cli, req)
                elif call.service_name.endswith('motion_enable'):
                    req = SetInt16ById.Request()
                    req.id = int(call.payload['id'])
                    req.data = int(call.payload['data'])
                    self._call_triggerish_service(self.motion_enable_cli, req)
                elif call.service_name.endswith('set_mode'):
                    req = SetInt16.Request()
                    req.data = int(call.payload['data'])
                    self._call_triggerish_service(self.set_mode_cli, req)
                elif call.service_name.endswith('set_state'):
                    req = SetInt16.Request()
                    req.data = int(call.payload['data'])
                    self._call_triggerish_service(self.set_state_cli, req)
            self._stop_requested = False
            self._last_command_summary = 'prepare_robot'
            response.success = True
            response.message = 'robot prepared'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _send_move(self, payload: dict):
        req = MoveJoint.Request()
        req.angles = payload['angles']
        req.speed = float(payload['speed'])
        req.acc = float(payload['acc'])
        req.mvtime = float(payload['mvtime'])
        req.wait = bool(payload['wait'])
        req.timeout = float(payload['timeout'])
        req.radius = float(payload['radius'])
        req.relative = bool(payload['relative'])
        return self._call_triggerish_service(self.move_joint_cli, req)

    def _handle_stop(self, _request, response):
        self._stop_requested = True
        self._last_command_summary = 'stop'
        response.success = True
        response.message = 'future motion commands blocked until prepare_robot'
        return response

    def _on_joint_command(self, msg: JointState):
        if self._stop_requested:
            self.get_logger().warning('rejected joint command: stop requested')
            self._last_command_summary = 'rejected:stop requested'
            return
        targets = [float(value) for value in msg.position]
        allowed, reason = self.safety_gate.can_execute_motion(self._current_health(), targets)
        if not allowed:
            self.get_logger().warning(f'rejected joint command: {reason}')
            self._last_command_summary = f'rejected:{reason}'
            return
        try:
            payload = self.robot_client.build_move_joint_payload(targets)
            self._send_move(payload)
            self._last_command_summary = f'move:{payload["angles"]}'
        except Exception as exc:
            self._last_command_summary = f'error:{exc}'
            self.get_logger().error(str(exc))

    def _handle_home(self, _request, response):
        if self._stop_requested:
            response.success = False
            response.message = 'stop requested'
            return response
        targets = self.robot_client.build_home_payload()['angles']
        allowed, reason = self.safety_gate.can_execute_motion(self._current_health(), targets)
        if not allowed:
            response.success = False
            response.message = reason
            return response
        try:
            self._send_move(self.robot_client.build_home_payload())
            self._last_command_summary = 'go_home'
            response.success = True
            response.message = 'home motion requested'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response


def main():
    rclpy.init()
    node = Lite6MotionServer()
    try:
        rclpy.spin(node)
    except Exception as exc:
        if exc.__class__.__name__ != 'ExternalShutdownException':
            raise
    finally:
        node.destroy_node()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass
