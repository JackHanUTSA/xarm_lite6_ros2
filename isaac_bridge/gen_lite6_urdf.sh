#!/usr/bin/env bash
set -eo pipefail

WS=~/ws_xarm
OUT=${1:-$WS/isaac_bridge/lite6.urdf}

set +u
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u

XACRO=$(command -v xacro || true)
if [ -z "$XACRO" ]; then
  echo "xacro not found; install ros-humble-xacro" >&2
  exit 1
fi

XACRO_FILE="$WS/src/xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro"

"$XACRO" "$XACRO_FILE" \
  robot_type:=lite \
  dof:=6 \
  add_gripper:=false \
  add_vacuum_gripper:=false \
  add_bio_gripper:=false \
  limited:=true \
  > "$OUT"

echo "WROTE_URDF:$OUT"
