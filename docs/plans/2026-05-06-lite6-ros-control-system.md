# Lite6 ROS Control System Implementation Plan

> For Hermes: Use subagent-driven-development if executing this plan. Keep motion-gated changes separate from perception/recording changes.

Goal: build a safe, testable ROS 2 control system for the Lite6 arm that supports manual jogging, scripted motions, planner-based moves, camera-assisted recording, and future policy/RL integration.

Architecture: use the existing xArm ROS 2 driver and MoveIt stack as the hardware/control foundation, then add a thin Lite6 application layer in `ws_xarm/src` with explicit safety gating. Keep low-level actuation in vendor ROS interfaces, keep operator workflows in small ROS nodes/services/actions, and keep vision/recording separate from motion execution.

Tech stack: ROS 2 Humble, xarm_ros2, rclpy, xarm_msgs services, JointTrajectory, MoveIt realmove launch, ffmpeg, PyQt GUI.

Existing useful pieces already present:
- Driver launch: `/home/r91/ros2_ws/src/xarm_ros2/xarm_api/launch/lite6_driver.launch.py`
- MoveIt real robot launch: `/home/r91/ros2_ws/src/xarm_ros2/xarm_planner/launch/lite6_planner_realmove.launch.py`
- Simple trajectory publisher: `/home/r91/ws_xarm/src/lite6_control_demo/lite6_control_demo/send_joint_traj.py`
- GUI prototype: `/home/r91/ws_xarm/src/lite6_gui/lite6_gui/gui.py`
- CLI safety helpers: `/home/r91/ws_xarm/scripts/lite6_cli.zsh`
- Motion + recording prototype: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/run_and_record.py`

Recommended system decomposition:
1. Bringup layer
   - Robot driver, joint states, robot state, enable/clean/mode/state services.
2. Motion service layer
   - Safe wrapper node around `/ufactory/*` services and optional JointTrajectory publishing.
3. Planning layer
   - MoveIt-based planner for goal poses/joint targets.
4. Operator layer
   - GUI, CLI, and scripted routines calling the motion layer.
5. Perception/recording layer
   - Cameras, synchronized capture, annotations, robot-state stamping.
6. Integration layer
   - Future Isaac/RL bridge should call the same motion abstraction, not bypass safety policy.

---

## Phase 0: System contract

Define the control contract before changing code.

Control modes to support:
- `monitor_only`: read joint states, no motion.
- `service_motion`: direct xArm service motion for small deterministic moves.
- `trajectory_motion`: JointTrajectory commands for controller-driven moves.
- `planner_motion`: MoveIt plans for collision-aware target execution.
- `scripted_routine`: prerecorded task sequence using one of the above.
- `policy_motion`: future external agent interface, guarded by same safety checks.

Safety invariants:
- no motion unless explicit enable flag is set in the node making the request
- every motion command must have timeout, speed, acceleration, and stop path
- motion interface must verify robot state is healthy before execute
- GUI/CLI must require operator confirmation for non-trivial moves
- recording and perception may run even when motion is disabled

Acceptance criteria:
- operator can launch one command to bring up Lite6 ROS control stack
- operator can inspect state without moving
- operator can send a tiny safe test move
- operator can execute a planned joint or Cartesian move
- operator can record cameras while robot moves
- logs capture issued command, timestamps, and resulting joint state

---

## Phase 1: Standardize bringup

Objective: make one repeatable ROS bringup path for the real Lite6.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_bringup/`
- Create: `/home/r91/ws_xarm/src/lite6_bringup/launch/lite6_real_system.launch.py`
- Create: `/home/r91/ws_xarm/src/lite6_bringup/config/lite6_topics.yaml`
- Reference: `/home/r91/ros2_ws/src/xarm_ros2/xarm_api/launch/lite6_driver.launch.py`
- Reference: `/home/r91/ros2_ws/src/xarm_ros2/xarm_planner/launch/lite6_planner_realmove.launch.py`

Design:
- include vendor driver launch
- optionally include planner launch through a launch arg `with_moveit`
- expose `robot_ip`, `hw_ns`, `cmd_topic`, `joint_states_topic`
- publish a consistent namespace contract for all local Lite6 nodes

Expected interface contract:
- joint states topic: `/ufactory/joint_states` or standardized remap to `/joint_states`
- state topic: `/ufactory/robot_states`
- xArm services remain under `/ufactory/*`
- planner interface brought up only when requested

Verification commands:
- `source /opt/ros/humble/setup.zsh && source ~/ws_xarm/install/setup.zsh && ros2 launch lite6_bringup lite6_real_system.launch.py robot_ip:=<IP>`
- `ros2 topic list | egrep 'joint_states|robot_states'`
- `ros2 service list | egrep '^/ufactory/'`

---

## Phase 2: Build one safe motion API node

Objective: centralize all direct motion through one ROS node instead of letting each app call vendor services ad hoc.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_motion/lite6_motion/safety_gate.py`
- Create: `/home/r91/ws_xarm/src/lite6_motion/lite6_motion/robot_client.py`
- Create: `/home/r91/ws_xarm/src/lite6_motion/lite6_motion/motion_server.py`
- Create: `/home/r91/ws_xarm/src/lite6_motion/package.xml`
- Create: `/home/r91/ws_xarm/src/lite6_motion/setup.py`
- Create: `/home/r91/ws_xarm/src/lite6_motion/resource/lite6_motion`
- Test: `/home/r91/ws_xarm/src/lite6_motion/test/`
- Reference: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/run_and_record.py`
- Reference: `/home/r91/ws_xarm/scripts/lite6_cli.zsh`

Node responsibilities:
- subscribe to joint state and robot state
- wrap `clean_error`, `clean_warn`, `motion_enable`, `set_mode`, `set_state`, `set_servo_angle`
- expose higher-level ROS services or actions:
  - `prepare_robot`
  - `move_joints_absolute`
  - `move_joints_relative`
  - `stop_motion`
  - `go_home`
- reject commands if stale state, error state, missing state, or motion disabled
- log requested command and final observed state

Recommended ROS API:
- topic `/lite6_motion/status` with current readiness, stale flags, last command, last result
- service `/lite6_motion/prepare_robot`
- service `/lite6_motion/go_home`
- action `/lite6_motion/move_joints`
- optional action `/lite6_motion/move_pose`

Why this matters:
- removes duplicated shell-based service calls
- gives GUI, CLI, recorders, and future policies a single control surface
- makes safety checks reusable and testable

Verification:
- call `prepare_robot` and ensure it does not move
- call tiny relative move with a tiny delta and low speed
- inspect `/lite6_motion/status`

---

## Phase 3: Keep direct trajectory control, but behind the API

Objective: preserve JointTrajectory support without making it the first-class user entry point.

Files:
- Modify: `/home/r91/ws_xarm/src/lite6_control_demo/lite6_control_demo/send_joint_traj.py`
- Create: `/home/r91/ws_xarm/src/lite6_motion/lite6_motion/trajectory_adapter.py`

Design:
- continue supporting the controller topic when available
- move trajectory publish logic into a reusable adapter
- validate that the trajectory controller joint order matches configured Lite6 joint names
- expose this as one execution backend inside `motion_server.py`

Decision rule:
- use direct service motion for tiny deterministic service-level tests
- use trajectory backend for simple joint target execution when controller path is stable
- use MoveIt planner for workspace-aware target moves

Verification:
- command one small move through trajectory backend
- confirm observed joint positions track requested target

---

## Phase 4: Add MoveIt-backed planner execution

Objective: support safe goal moves beyond trivial joint nudges.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_planning_bridge/lite6_planning_bridge/planner_client.py`
- Create: `/home/r91/ws_xarm/src/lite6_planning_bridge/lite6_planning_bridge/pose_move_server.py`
- Create: `/home/r91/ws_xarm/src/lite6_planning_bridge/package.xml`
- Create: `/home/r91/ws_xarm/src/lite6_planning_bridge/setup.py`
- Reference: `/home/r91/ros2_ws/src/xarm_ros2/xarm_planner/launch/lite6_planner_realmove.launch.py`

Design:
- when `with_moveit:=true`, planner bridge offers pose/joint goal interfaces
- planner bridge asks MoveIt for a plan first, then requires explicit execute
- planner bridge reports planning failure separately from execution failure
- planner results should be available to GUI and CLI as dry-run previews

Recommended APIs:
- action `/lite6_planner/move_to_joint_goal`
- action `/lite6_planner/move_to_pose_goal`
- service `/lite6_planner/plan_only_pose_goal`

Verification:
- launch MoveIt realmove stack
- issue a plan-only request to a reachable pose
- review success/failure and only then execute

---

## Phase 5: Upgrade operator interfaces

Objective: make all user-facing controls call the new motion/planning APIs.

Files:
- Modify: `/home/r91/ws_xarm/src/lite6_gui/lite6_gui/gui.py`
- Modify: `/home/r91/ws_xarm/scripts/lite6_cli.zsh`
- Create: `/home/r91/ws_xarm/src/lite6_ops/lite6_ops/routine_runner.py`
- Create: `/home/r91/ws_xarm/src/lite6_ops/lite6_ops/home_and_test.py`

GUI upgrades:
- show robot health, mode, state age, and motion-enabled status
- add explicit backend selector: service / trajectory / planner
- add plan-only button for planner goals
- add home button
- add emergency software stop button
- disable execute buttons until state is fresh and robot prepared

CLI upgrades:
- keep current zsh ergonomics
- make commands call new ROS services/actions instead of vendor services directly where possible
- retain `YES_MOVE=1` safety gate
- add `plan_pose`, `go_home`, `stop`, `prepare`, `backend_status`

Routine runner:
- execute saved JSON/YAML routines with metadata and motion limits
- support named routines like `tiny_test`, `sweep_base`, `home_pose`

---

## Phase 6: Integrate recording and perception cleanly

Objective: motion and recording should be synchronized but decoupled.

Files:
- Modify: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/run_and_record.py`
- Create: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/recording_session.py`
- Create: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/robot_state_logger.py`

Design:
- recording node should not call vendor motion directly
- recording node calls motion server to prepare + execute selected routine
- save alongside video:
  - command metadata
  - robot state samples
  - joint states timeline
  - camera timestamps
  - launch configuration used
- keep camera device configuration externalized in YAML

Output layout recommendation:
- `~/Videos/robot_monitor/<session_id>/video_main.mp4`
- `~/Videos/robot_monitor/<session_id>/robot_states.jsonl`
- `~/Videos/robot_monitor/<session_id>/joint_states.jsonl`
- `~/Videos/robot_monitor/<session_id>/command.json`
- `~/Videos/robot_monitor/<session_id>/session.yaml`

---

## Phase 7: External policy / Isaac / RL integration

Objective: ensure future autonomous control uses the same ROS safety boundary.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_policy_bridge/lite6_policy_bridge/policy_command_server.py`
- Create: `/home/r91/ws_xarm/src/lite6_policy_bridge/lite6_policy_bridge/policy_rate_limiter.py`
- Reference: `/home/r91/ws_xarm/isaac_bridge/lite6_ros2_bridge.py`
- Reference: `/home/r91/ws_xarm/rl/lite6_rpc_env.py`

Design:
- policy bridge converts external actions into approved ROS motion requests
- rate-limit outgoing commands
- clamp joint deltas and workspace targets
- require heartbeat from policy source; stop robot on heartbeat loss
- allow monitor-only mode for vision/RL debugging without motion

Important rule:
- RL or Isaac code must never talk straight to `/ufactory/set_servo_angle`
- all such commands should flow through `lite6_motion`

---

## Phase 8: Testing and verification ladder

Objective: verify safely from read-only to real motion.

Test ladder:
1. static ROS graph test
   - all topics/services visible
2. read-only state freshness test
   - verify joint states update rate and staleness detection
3. prepare-only test
   - clean/enable/mode/state path succeeds with no movement
4. tiny joint move test
   - one small safe delta and return
5. trajectory backend test
   - one joint target via controller
6. planner plan-only test
   - reachable pose plan generated
7. planner execution test
   - operator-confirmed execution of small move
8. run-and-record integration test
   - motion plus synchronized video and state logs

Recommended commands:
- `cd ~/ws_xarm && source /opt/ros/humble/setup.zsh && source ~/ws_xarm/install/setup.zsh`
- `./scripts/lite6_cli.zsh status`
- `./scripts/lite6_cli.zsh enable`
- `./scripts/lite6_cli.zsh tiny_test`
- `ros2 launch lite6_gui lite6_gui.launch.py`

---

## Minimum viable build order

If you want the fastest useful system, do this first:
1. create `lite6_bringup`
2. create `lite6_motion` safety wrapper
3. repoint CLI to `lite6_motion`
4. repoint GUI to `lite6_motion`
5. update `run_and_record.py` to call `lite6_motion`
6. add planner bridge after that

That gives you one ROS-native control surface quickly, with safety and future extensibility.

---

## Recommended immediate next implementation

Start with this exact scope:
- create `lite6_motion` package
- implement `prepare_robot`, `move_joints_absolute`, `go_home`, `status`
- update `lite6_cli.zsh tiny_test` to call `lite6_motion`
- update GUI execute button to call `lite6_motion`

That is the smallest change that turns the current collection of scripts into a coherent ROS control system.
