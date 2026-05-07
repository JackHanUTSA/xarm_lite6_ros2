# epuck2_sim_real_control

New project scaffold for e-puck2 sim-to-real control in ROS 2.

Goal:
- start with a clean project for sim-first control experiments
- keep simulation and real-robot interfaces aligned
- make it easy to add safety checks before real deployment

Current scaffold includes:
- a ROS 2 Python package under `src/epuck2_sim_real_control`
- a YAML session manifest for sim/real configuration
- a `project_info` node/executable that loads and publishes the current mode manifest
- a launch file for quick bring-up sanity checks

Main files:
- `epuck2_sim_real_control/session_manifest.py`
- `epuck2_sim_real_control/project_info.py`
- `config/default_session.yaml`
- `launch/epuck2_sim_real_control.launch.py`

Quick start:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
colcon build --packages-select epuck2_sim_real_control
source ~/ws_xarm/install/setup.zsh
ros2 run epuck2_sim_real_control project_info
```

Launch with explicit mode:

```zsh
ros2 launch epuck2_sim_real_control epuck2_sim_real_control.launch.py mode:=sim
```

Next good steps:
1. add the actual simulator adapter (for example Webots or Gazebo)
2. add the real robot adapter / driver integration
3. define one shared observation/action interface
4. add stop, watchdog, and confirmation gates for real mode
5. add logging and replay for sim-to-real comparison
