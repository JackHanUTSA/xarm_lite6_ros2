#!/usr/bin/env bash
set -e

# Avoid nounset issues with ROS setup scripts
set +u

source /opt/ros/humble/setup.bash
source "$HOME/ws_xarm/install/setup.bash"

# Start GUI (optionally override topics via env)
CMD_TOPIC=${CMD_TOPIC:-/joint_trajectory_controller/joint_trajectory}
JS_TOPIC=${JS_TOPIC:-/joint_states}

exec ros2 launch lite6_gui lite6_gui.launch.py cmd_topic:=$CMD_TOPIC js_topic:=$JS_TOPIC
