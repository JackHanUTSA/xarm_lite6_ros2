#!/usr/bin/env zsh
set -eo pipefail
set +u

source /opt/ros/humble/setup.zsh
if [[ -f "$HOME/ws_xarm/install/setup.zsh" ]]; then
  source "$HOME/ws_xarm/install/setup.zsh"
fi
set +u

typeset -r MOTION_STATUS_TOPIC=/lite6_motion/status
typeset -r MOTION_COMMAND_TOPIC=/lite6_motion/joint_command
typeset -r PREPARE_SRV=/lite6_motion/prepare_robot
typeset -r GO_HOME_SRV=/lite6_motion/go_home
typeset -r STOP_SRV=/lite6_motion/stop

die() { print -r -- "ERROR: $*" >&2; exit 2 }

confirm_motion() {
  if [[ "${YES_MOVE:-}" == "1" ]]; then
    return 0
  fi
  if [[ -t 0 ]]; then
    print -r -- "MOTION REQUESTED. Ensure workspace is clear and E-stop reachable."
    vared -p "Type YES to proceed: " -c ans
    [[ "$ans" == "YES" ]] || die "motion aborted"
  else
    die "motion blocked (non-interactive). Set YES_MOVE=1 to override."
  fi
}

motion_status_json() {
  python3 - <<'PY'
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import String

class OneShot(Node):
    def __init__(self):
        super().__init__('lite6_status_cli')
        self.payload = None
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.create_subscription(String, '/lite6_motion/status', self.cb, qos)
    def cb(self, msg):
        self.payload = msg.data

rclpy.init()
node = OneShot()
try:
    end = node.get_clock().now().nanoseconds + int(2.0e9)
    while rclpy.ok() and node.payload is None and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.payload is None:
        raise SystemExit('no lite6_motion status received')
    obj = json.loads(node.payload)
    print(json.dumps(obj))
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
}

status() {
  print -r -- "== lite6_motion/status =="
  motion_status_json | python3 -m json.tool
}

enable() {
  print -r -- "Preparing robot through lite6_motion"
  ros2 service call "$PREPARE_SRV" std_srvs/srv/Trigger "{}"
}

clean() {
  enable
}

angles() {
  motion_status_json | python3 - <<'PY'
import json, sys
obj = json.load(sys.stdin)
print('joint_names:', obj.get('joint_names'))
print('joint_positions:', obj.get('joint_positions'))
PY
}

publish_joint_target() {
  local angles_str=${1:-}
  [[ -n "$angles_str" ]] || die "publish_joint_target requires 6 angles string"
  local -a arr
  arr=(${=angles_str})
  (( ${#arr} == 6 )) || die "need exactly 6 angles (got ${#arr})"
  python3 - "$angles_str" <<'PY'
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

angles = [float(x) for x in sys.argv[1].split()]

rclpy.init()
node = Node('lite6_joint_command_cli')
pub = node.create_publisher(JointState, '/lite6_motion/joint_command', 10)
msg = JointState()
msg.name = [f'joint{i}' for i in range(1, 7)]
msg.position = angles
deadline = time.time() + 1.0
while time.time() < deadline and pub.get_subscription_count() == 0:
    rclpy.spin_once(node, timeout_sec=0.05)
if pub.get_subscription_count() == 0:
    raise SystemExit('no subscribers on /lite6_motion/joint_command')
pub.publish(msg)
rclpy.spin_once(node, timeout_sec=0.05)
node.destroy_node()
rclpy.shutdown()
PY
}

move_pose() {
  local angles_str=${1:-}
  [[ -n "$angles_str" ]] || die "move_pose requires 6 angles string, e.g. '0 -0.5 0.8 0 0 0'"
  confirm_motion
  enable >/dev/null
  print -r -- "Publishing absolute joint target through lite6_motion: [$angles_str]"
  publish_joint_target "$angles_str"
}

go_home() {
  confirm_motion
  ros2 service call "$GO_HOME_SRV" std_srvs/srv/Trigger "{}"
}

preview_demo() {
  local source_path=${1:-}
  [[ -n "$source_path" ]] || die "preview_demo requires path to lite6_targets.jsonl"
  ros2 run lite6_imitation_bridge execute_lite6_targets preview_demo "$source_path"
}

execute_demo() {
  local source_path=${1:-}
  [[ -n "$source_path" ]] || die "execute_demo requires path to lite6_targets.jsonl"
  confirm_motion
  enable >/dev/null
  ros2 run lite6_imitation_bridge execute_lite6_targets execute_demo "$source_path" --confirm
}

stop() {
  ros2 service call "$STOP_SRV" std_srvs/srv/Trigger "{}"
}

tiny_test() {
  confirm_motion
  enable >/dev/null

  python3 - <<'PY'
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class TinyTest(Node):
    def __init__(self):
        super().__init__('lite6_tiny_test_cli')
        self.status_payload = None
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.create_subscription(String, '/lite6_motion/status', self._on_status, qos)
        self.pub = self.create_publisher(JointState, '/lite6_motion/joint_command', 10)
    def _on_status(self, msg):
        self.status_payload = json.loads(msg.data)
    def wait_status(self, timeout=2.0):
        end = time.time() + timeout
        while rclpy.ok() and time.time() < end and self.status_payload is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.status_payload
    def send(self, positions):
        msg = JointState()
        msg.name = [f'joint{i}' for i in range(1, 7)]
        msg.position = positions
        deadline = time.time() + 1.0
        while time.time() < deadline and self.pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.pub.get_subscription_count() == 0:
            raise SystemExit('no subscribers on /lite6_motion/joint_command')
        self.pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)

rclpy.init()
node = TinyTest()
try:
    status = node.wait_status()
    if status is None:
        raise SystemExit('no lite6_motion status received')
    start = list(status.get('joint_positions') or [0.0] * 6)
    if len(start) < 6:
        raise SystemExit('lite6_motion status missing complete joint state')
    print('START', start)
    for sign in (+1, -1):
        target = start[:6]
        target[1] = start[1] + sign * 0.10
        print('TARGET', target)
        node.send(target)
        time.sleep(2.0)
    node.send(start[:6])
    time.sleep(2.0)
    print('END', start[:6])
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
}

record_3panel_yolo() {
  local dur=${1:-20}
  print -r -- "Recording 3-panel YOLO for ${dur}s (no motion)"
  ros2 run lite6_record_control triple_panel_record \
    --duration "$dur" --fps 10 --yolo --yolo_conf 0.25
}

help() {
  print -r -- "Lite6 CLI via lite6_motion"
  print -r -- "Usage:"
  print -r -- "  ./scripts/lite6_cli.zsh status"
  print -r -- "  ./scripts/lite6_cli.zsh enable"
  print -r -- "  ./scripts/lite6_cli.zsh angles"
  print -r -- "  ./scripts/lite6_cli.zsh tiny_test"
  print -r -- "  ./scripts/lite6_cli.zsh preview_demo path/to/lite6_targets.jsonl"
  print -r -- "  YES_MOVE=1 ./scripts/lite6_cli.zsh execute_demo path/to/lite6_targets.jsonl"
  print -r -- "  ./scripts/lite6_cli.zsh stop"
  print -r -- "  YES_MOVE=1 ./scripts/lite6_cli.zsh move_pose '0 -0.5 0.8 0 0 0'"
  print -r -- "  ./scripts/lite6_cli.zsh go_home"
  print -r -- "  ./scripts/lite6_cli.zsh record_3panel_yolo 20"
}

cmd=${1:-help}
shift || true
case "$cmd" in
  status) status "$@";;
  enable) enable;;
  clean) clean;;
  angles) angles;;
  move_pose) move_pose "$@";;
  tiny_test) tiny_test;;
  preview_demo) preview_demo "$@";;
  execute_demo) execute_demo "$@";;
  stop) stop;;
  go_home) go_home;;
  record_3panel_yolo) record_3panel_yolo "$@";;
  help|--help|-h) help;;
  *) die "unknown command: $cmd";;
esac
