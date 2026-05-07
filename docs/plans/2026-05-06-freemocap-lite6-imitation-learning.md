# FreeMoCap + Lite6 Imitation Learning Implementation Plan

> For Hermes: Use subagent-driven-development if executing this plan. Keep motion-producing work behind the existing `lite6_motion` safety gate.

Goal: integrate FreeMoCap and the Lite6 ROS control stack so human arm motion can be captured, retargeted into robot-friendly trajectories, logged as demonstrations, and used to train policies that respect Lite6 dynamics.

Architecture: split the system into four layers: capture, retargeting, demonstration dataset building, and policy execution. FreeMoCap remains the human-pose front end; `lite6_motion` remains the only robot actuation interface; a new imitation bridge converts human arm kinematics into Lite6 targets; and a learning layer consumes those demonstrations to train and evaluate robot policies.

Tech stack: FreeMoCap, skellycam, ROS 2 Humble, rclpy, numpy, scipy, xarm ROS services/messages, existing `lite6_motion` package, optional PyTorch/JAX for imitation and dynamics models.

Existing project anchors:
- FreeMoCap CLI entrypoint: `/home/r91/freemocap/freemocap/__main__.py`
- FreeMoCap direct camera CLI: `/home/r91/freemocap/freemocap/direct_camera.py`
- FreeMoCap headless processing: `/home/r91/freemocap/freemocap/core_processes/process_motion_capture_videos/process_recording_headless.py`
- FreeMoCap recording path model: `/home/r91/freemocap/freemocap/data_layer/recording_models/recording_info_model.py`
- Lite6 motion gateway: `/home/r91/ws_xarm/src/lite6_motion/lite6_motion/motion_server.py`
- Lite6 recording prototype: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/run_and_record.py`
- Lite6 ROS GUI: `/home/r91/ws_xarm/src/lite6_gui/lite6_gui/gui.py`
- Lite6 CLI helpers: `/home/r91/ws_xarm/scripts/lite6_cli.zsh`

Key human landmarks available from MediaPipe/FreeMoCap:
- `right_shoulder`
- `right_elbow`
- `right_wrist`
- `right_index`
- `right_thumb`
- symmetric left-arm landmarks for mirrored teleop

Safety rule:
- No human-to-robot motion path may call `/ufactory/set_servo_angle` directly.
- All live execution must flow through `lite6_motion`.

---

## System target behavior

End-state workflow:
1. Launch FreeMoCap with a known camera preset.
2. Record human arm motion performing a task demonstration.
3. Process recording headlessly to obtain 3D arm landmarks.
4. Convert human arm motion into robot-arm features in a robot reference frame.
5. Generate a retargeted Lite6 trajectory or action sequence.
6. Preview trajectory offline.
7. Execute trajectory on Lite6 only through `lite6_motion`.
8. Log robot observations, issued commands, and resulting state.
9. Add paired human/robot demonstrations to a training dataset.
10. Train imitation and/or dynamics-aware policies from that dataset.

Acceptance criteria:
- one CLI command records a FreeMoCap session with chosen camera preset
- one CLI command processes a recording and exports human-arm features
- one CLI command retargets a processed session into a Lite6 trajectory file
- one CLI command previews or dry-runs the retargeted trajectory
- one CLI command executes the retargeted trajectory via `lite6_motion`
- training data folder contains aligned human pose, robot targets, robot state, and metadata

---

## Phase 1: Stabilize FreeMoCap capture presets for robot-learning sessions

Objective: make capture reproducible so demonstrations are consistent enough for retargeting and learning.

Files:
- Modify: `/home/r91/freemocap/freemocap/direct_camera.py`
- Modify: `/home/r91/freemocap/scripts/freemocap-direct.zsh`
- Create: `/home/r91/freemocap/freemocap/camera_presets.py`
- Create: `/home/r91/freemocap/freemocap/session_manifest.py`

Tasks:
- add named presets such as `c920_robot_demo`, `m9_workspace`, `dual_robot_demo`
- record the chosen preset, camera IDs, resolution, fps, fourcc, and host metadata in a session manifest
- add flags to disable auto-detect and enforce selected cameras only
- write per-session JSON metadata beside recordings

Output contract:
- each recording session gets a `session_manifest.json`
- manifest stores camera IDs, preset name, capture settings, hostname, and start time

Verification:
- run the direct CLI with `--camera-id 2`
- verify camera selection and manifest creation
- verify identical preset settings can be replayed later

---

## Phase 2: Add headless human-arm feature extraction

Objective: convert processed FreeMoCap output into a compact human-arm motion representation.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/freemocap_loader.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/human_arm_features.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/export_human_arm_demo.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/setup.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/package.xml`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/resource/lite6_imitation_bridge`
- Test: `/home/r91/ws_xarm/src/lite6_imitation_bridge/test/test_human_arm_features.py`

Design:
- load processed FreeMoCap `data_3d_npy_file_path` using the path conventions from `recording_info_model.py`
- extract right-arm and/or left-arm landmarks by name
- compute derived features per frame:
  - shoulder-to-elbow vector
  - elbow-to-wrist vector
  - upper-arm length
  - forearm length
  - elbow bend angle
  - wrist direction vector
  - torso-relative frame
- export to `human_arm_demo.jsonl` or `human_arm_demo.npz`

Output layout recommendation:
- `<recording>/output_data/human_arm_demo.jsonl`
- `<recording>/output_data/human_arm_demo.npz`
- `<recording>/output_data/human_arm_demo_metadata.json`

Verification:
- unit-test a synthetic arm pose sequence
- run on one processed recording and confirm features exist for each frame

---

## Phase 3: Add camera-to-robot calibration and reference-frame alignment

Objective: put human motion into a frame the robot can use consistently.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/robot_frame_calibration.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/calibrate_human_to_robot_frame.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/config/calibration_schema.yaml`
- Test: `/home/r91/ws_xarm/src/lite6_imitation_bridge/test/test_robot_frame_calibration.py`

Design:
- define a robot workspace frame with origin near the Lite6 base or task surface
- solve a rigid transform from FreeMoCap space to robot space
- store calibration using a human-readable YAML/TOML file
- allow calibration from a simple known-pose procedure

Recommended calibration procedure:
- collect one short session with a calibration wand or known fiducials near the robot workspace
- estimate transform from FreeMoCap coordinates to the robot base frame
- save to `~/ws_xarm/calibration/freemocap_to_lite6.yaml`

Verification:
- transformed wrist trajectory lies in expected workspace ranges
- calibration file reloads and reproduces the same transform

---

## Phase 4: Add human-to-Lite6 retargeting

Objective: convert human arm features into Lite6-friendly joint or end-effector targets.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/retargeter.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/lite6_kinematics_proxy.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/export_lite6_targets.py`
- Test: `/home/r91/ws_xarm/src/lite6_imitation_bridge/test/test_retargeter.py`

Retargeting strategy:
- start simple with pose-to-pose geometric retargeting
- map human shoulder/elbow/wrist motion to Lite6 end-effector target plus elbow configuration heuristics
- clamp to Lite6 reachable workspace and joint limits
- emit confidence and rejection reasons per frame

Recommended initial representation:
- target endpoint position
- target approach direction
- optional gripper/open-close intent derived from thumb-index distance or external labels

Output artifacts:
- `lite6_targets.jsonl`
- `lite6_targets_preview.npz`
- `lite6_targets_report.json`

Verification:
- no target outside workspace bounds
- no target stream with impossible jumps between adjacent frames
- sample trajectory can be visualized before execution

---

## Phase 5: Add preview + dry-run against the Lite6 control stack

Objective: make retargeted motion inspectable before any live execution.

Files:
- Modify: `/home/r91/ws_xarm/src/lite6_gui/lite6_gui/gui.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/preview_targets.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/publish_joint_command_preview.py`

Design:
- preview a retargeted trajectory without moving hardware
- overlay target statistics in GUI or CLI
- publish preview frames/status on ROS topics for operator inspection
- require explicit operator confirmation before execution mode changes from dry-run to live

Recommended ROS topics:
- `/lite6_imitation/status`
- `/lite6_imitation/preview_targets`
- `/lite6_imitation/live_targets`

Verification:
- preview mode runs without `lite6_motion` issuing any moves
- operator sees counts, frame range, and target validity summary

---

## Phase 6: Add execution bridge through `lite6_motion`

Objective: execute retargeted motion only through the existing safe robot control API.

Files:
- Modify: `/home/r91/ws_xarm/src/lite6_motion/lite6_motion/motion_server.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/execute_lite6_targets.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/trajectory_scheduler.py`
- Modify: `/home/r91/ws_xarm/scripts/lite6_cli.zsh`

Design:
- add a higher-level execution path that consumes `lite6_targets.jsonl`
- schedule target frames at a bounded rate
- verify freshness and health on every step using `lite6_motion`
- stop immediately on state loss, robot error, or target invalidation

Suggested CLI commands:
- `./scripts/lite6_cli.zsh execute_demo <path/to/lite6_targets.jsonl>`
- `./scripts/lite6_cli.zsh preview_demo <path/to/lite6_targets.jsonl>`
- `./scripts/lite6_cli.zsh stop`

Verification:
- execute tiny 5-frame safe demo only
- robot stops cleanly if the bridge loses state

---

## Phase 7: Build demonstration dataset logging

Objective: log human and robot trajectories together for imitation and dynamics learning.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/demo_dataset_writer.py`
- Modify: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/run_and_record.py`
- Create: `/home/r91/ws_xarm/src/lite6_record_control/lite6_record_control/log_demo_episode.py`

Dataset fields per episode:
- recording/session metadata
- FreeMoCap 3D landmarks
- derived human-arm features
- robot target sequence
- actual robot joint states
- actual robot execution timestamps
- success/failure labels
- task ID and optional natural-language label

Recommended dataset layout:
- `~/ws_xarm/datasets/lite6_imitation/<episode_id>/human_arm_demo.npz`
- `~/ws_xarm/datasets/lite6_imitation/<episode_id>/lite6_targets.jsonl`
- `~/ws_xarm/datasets/lite6_imitation/<episode_id>/robot_joint_states.jsonl`
- `~/ws_xarm/datasets/lite6_imitation/<episode_id>/episode_metadata.yaml`

Verification:
- each demo episode contains both human and robot data
- timestamps can be aligned post hoc without ambiguity

---

## Phase 8: Add learning pipeline with robot dynamics awareness

Objective: let the robot learn from demonstrations while respecting its own dynamics.

Files:
- Create: `/home/r91/ws_xarm/rl/scripts/train_lite6_imitation.py`
- Create: `/home/r91/ws_xarm/rl/scripts/eval_lite6_imitation.py`
- Create: `/home/r91/ws_xarm/rl/imitation/lite6_demo_dataset.py`
- Create: `/home/r91/ws_xarm/rl/imitation/human_to_robot_policy.py`
- Create: `/home/r91/ws_xarm/rl/imitation/lite6_dynamics_model.py`
- Create: `/home/r91/ws_xarm/rl/imitation/rollout_bridge.py`

Recommended learning curriculum:
1. behavior cloning from retargeted targets
2. action smoothing and rate regularization
3. residual correction model against actual Lite6 tracking error
4. optional dynamics model predicting next joint state / end-effector state
5. optional RL fine-tuning in simulation using demonstration initialization

Important rule:
- the first trainable policy should predict robot-friendly targets, not raw human joint angles
- dynamics-awareness should be introduced as regularization or residual correction, not by bypassing safety limits

Verification:
- offline validation loss decreases on held-out demonstrations
- policy rollouts remain within target rate and workspace bounds
- sim rollouts show smoother tracking than naive geometric retargeting alone

---

## Phase 9: Add online teleoperation / co-training mode

Objective: enable low-rate live imitation once offline replay is stable.

Files:
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/live_freemocap_to_lite6.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/rate_limiter.py`
- Create: `/home/r91/ws_xarm/src/lite6_imitation_bridge/lite6_imitation_bridge/intent_filter.py`

Design:
- consume live or near-live human-arm targets at low rate
- smooth and downsample before robot execution
- require hold-to-enable or explicit operator arm-disarm switch
- log all online sessions as demonstrations for continual learning

Verification:
- robot remains stable when operator freezes
- bridge halts cleanly on dropped tracking or stale human input

---

## Minimum viable build order

Fastest useful path:
1. stabilize FreeMoCap presets and manifests
2. create `lite6_imitation_bridge`
3. export human-arm features from processed recordings
4. add offline retargeting to `lite6_targets.jsonl`
5. preview and dry-run targets
6. execute via `lite6_motion`
7. log paired datasets
8. train behavior-cloning baseline

---

## Recommended immediate next implementation

Start here:
- create `lite6_imitation_bridge`
- implement `freemocap_loader.py`
- implement `human_arm_features.py`
- implement `export_human_arm_demo.py`
- add tests for synthetic shoulder/elbow/wrist trajectories

That is the smallest next step that materially moves the project from camera integration toward actual human-arm-to-robot learning.
