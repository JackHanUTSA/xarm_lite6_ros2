# epuck2 sim-real control

Project overview
- Workspace root: `~/ws_xarm`
- ROS package: `~/ws_xarm/src/epuck2_sim_real_control`
- Primary purpose: build a clean sim-first control stack for e-puck2 that can later switch to real hardware with explicit safety gates.

Initial architecture
1. Session manifest
   - Central YAML file for choosing sim vs real mode.
   - Stores backend names, topic contract, and operator-confirmation requirements.
2. Info/bring-up node
   - Loads the manifest.
   - Publishes the active configuration on `/epuck2/control_mode`.
   - Gives a simple first sanity check that the package and launch path work.
3. Future adapters
   - `sim_adapter.py` for Webots/Gazebo integration.
   - `real_adapter.py` for physical e-puck2 driver integration.
   - `policy_bridge.py` for shared observation/action APIs.
   - `safety_gate.py` for stop, watchdog, and motion gating.

Recommended next implementation steps
- define the exact simulator backend
- define the exact real robot ROS interface
- lock down observation/action message schemas
- add teleop or policy command path
- add record/replay evaluation between sim and real
