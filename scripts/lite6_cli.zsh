#!/usr/bin/env zsh
set -eo pipefail

# Avoid nounset (ROS setup scripts reference unset vars)
set +u

# Lite6 CLI helpers (zsh)
# Usage:
#   ./scripts/lite6_cli.zsh status
#   ./scripts/lite6_cli.zsh enable
#   ./scripts/lite6_cli.zsh angles
#   ./scripts/lite6_cli.zsh tiny_test
#   ./scripts/lite6_cli.zsh move_pose "0 -0.5 0.8 0 0 0"   (radians)
#   ./scripts/lite6_cli.zsh record_3panel_yolo 20
#
# Notes:
# - Uses xarm ROS2 services under /ufactory/*
# - Motion commands require explicit confirmation.

source /opt/ros/humble/setup.zsh
if [[ -f "$HOME/ws_xarm/install/setup.zsh" ]]; then
  source "$HOME/ws_xarm/install/setup.zsh"
fi

# Avoid nounset issues with ROS setup scripts in some shells
set +u

typeset -r MOTION_ENABLE_SRV=/ufactory/motion_enable
typeset -r SET_MODE_SRV=/ufactory/set_mode
typeset -r SET_STATE_SRV=/ufactory/set_state
typeset -r CLEAN_ERR_SRV=/ufactory/clean_error
typeset -r CLEAN_WARN_SRV=/ufactory/clean_warn
typeset -r GET_ANGLES_SRV=/ufactory/get_servo_angle
typeset -r SET_ANGLES_SRV=/ufactory/set_servo_angle

typeset -r ROBOT_STATES_TOPIC=/ufactory/robot_states

die() { print -r -- "ERROR: $*" >&2; exit 2 }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

confirm_motion() {
  # Require either interactive confirmation or env var.
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

ros2_topic_one() {
  # Print a few lines and exit; BrokenPipe is fine.
  timeout 2 ros2 topic echo "$1" 2>/dev/null | head -n "${2:-60}" || true
}

status() {
  print -r -- "== /ufactory/robot_states (sample) =="
  ros2_topic_one "$ROBOT_STATES_TOPIC" 60

  print -r -- "== services present =="
  ros2 service list | egrep '^/ufactory/' | head -n 50 || true
}

enable() {
  print -r -- "Enabling motion + setting mode/state (no motion)"
  ros2 service call "$MOTION_ENABLE_SRV" xarm_msgs/srv/SetInt16 "{data: 1}" >/dev/null
  ros2 service call "$SET_MODE_SRV" xarm_msgs/srv/SetInt16 "{data: 0}" >/dev/null
  ros2 service call "$SET_STATE_SRV" xarm_msgs/srv/SetInt16 "{data: 0}" >/dev/null
  print -r -- "OK"
}

clean() {
  print -r -- "Cleaning warn/error (no motion)"
  ros2 service call "$CLEAN_WARN_SRV" xarm_msgs/srv/SetInt16 "{data: 1}" >/dev/null || true
  ros2 service call "$CLEAN_ERR_SRV"  xarm_msgs/srv/SetInt16 "{data: 1}" >/dev/null || true
  print -r -- "OK"
}

angles() {
  print -r -- "== get_servo_angle =="
  ros2 service call "$GET_ANGLES_SRV" xarm_msgs/srv/GetFloat32List "{}" | sed -n '1,120p'
}

move_pose() {
  local angles_str=${1:-}
  [[ -n "$angles_str" ]] || die "move_pose requires 6 angles string, e.g. '0 -0.5 0.8 0 0 0'"

  # Parse into YAML list
  local -a arr
  arr=(${=angles_str})
  (( ${#arr} == 6 )) || die "need exactly 6 angles (got ${#arr})"

  confirm_motion
  enable

  local yaml_list
  yaml_list=$(printf "%s, %s, %s, %s, %s, %s" $arr[1] $arr[2] $arr[3] $arr[4] $arr[5] $arr[6])

  print -r -- "Calling set_servo_angle (wait=true): [$yaml_list]"
  ros2 service call "$SET_ANGLES_SRV" xarm_msgs/srv/SetServoAngle \
    "{angles: [$yaml_list], speed: 0.25, mvacc: 0.5, wait: true, radius: -1.0}"
}

tiny_test() {
  # Tiny joint2 +/-0.10 rad and return using current robot_states as start.
  confirm_motion
  enable

  python3 - <<'PY'
import os, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from xarm_msgs.msg import RobotMsg
import subprocess

class OneShot(Node):
    def __init__(self, topic='/ufactory/robot_states'):
        super().__init__('one_shot_robot_states')
        self.msg=None
        qos=QoSProfile(depth=1)
        qos.reliability=QoSReliabilityPolicy.BEST_EFFORT
        qos.durability=QoSDurabilityPolicy.VOLATILE
        self.create_subscription(RobotMsg, topic, self.cb, qos)
    def cb(self, m):
        if self.msg is None:
            self.msg=m

def read_angles(timeout_sec=3.0):
    rclpy.init()
    n=OneShot()
    t0=time.time()
    while rclpy.ok() and n.msg is None and time.time()-t0<timeout_sec:
        rclpy.spin_once(n, timeout_sec=0.1)
    if n.msg is None:
        raise RuntimeError('timeout reading robot_states')
    ang=list(n.msg.angle)
    n.destroy_node()
    rclpy.shutdown()
    return ang

def servo(angles, wait=True):
    req = "{angles: [%s], speed: 0.25, mvacc: 0.5, wait: %s, radius: -1.0}" % (
        ", ".join(f"{v:.6f}" for v in angles),
        'true' if wait else 'false'
    )
    subprocess.check_call(['ros2','service','call','/ufactory/set_servo_angle','xarm_msgs/srv/SetServoAngle', req])

start=read_angles()
print('START', start)
for sign in (+1, -1):
    tgt=start[:]
    tgt[1]=start[1] + sign*0.10
    servo(tgt, wait=True)
servo(start, wait=True)
end=read_angles()
print('END', end)
PY
}

record_3panel_yolo() {
  local dur=${1:-20}
  need_cmd ros2

  print -r -- "Recording 3-panel YOLO for ${dur}s (no motion)"
  ros2 run lite6_record_control triple_panel_record \
    --duration "$dur" --fps 10 --yolo --yolo_conf 0.25
}

help() {
  sed -n '1,120p' "$0"
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
  record_3panel_yolo) record_3panel_yolo "$@";;
  help|--help|-h) help;;
  *) die "unknown command: $cmd";;
esac
