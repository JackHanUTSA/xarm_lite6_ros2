# Lite6 Imitation Workflow: How to Use It

This file explains the current end-to-end workflow for the FreeMoCap -> Lite6 imitation stack in this workspace.

Current phases implemented:
- Phase 2: export human arm features from FreeMoCap
- Phase 3: calibrate FreeMoCap space into Lite6 workspace
- Phase 4: export conservative Lite6 target records
- Phase 5: dry-run preview and operator inspection
- Phase 6: safe execution through `lite6_motion`

Important safety model:
- FreeMoCap does not command the robot directly.
- All real robot motion goes through `lite6_motion`.
- Preview tools never publish to the live motion topic.
- Execution is blocked unless the target file contains valid `joint_positions` records.

## 1) Prerequisites

Build the workspace:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
colcon build --symlink-install
source ~/ws_xarm/install/setup.zsh
```

Available ROS executables:
- `ros2 run lite6_imitation_bridge export_human_arm_demo`
- `ros2 run lite6_imitation_bridge calibrate_human_to_robot_frame`
- `ros2 run lite6_imitation_bridge export_lite6_targets`
- `ros2 run lite6_imitation_bridge preview_targets`
- `ros2 run lite6_imitation_bridge publish_joint_command_preview`
- `ros2 run lite6_imitation_bridge execute_lite6_targets`
- `ros2 run lite6_motion motion_server`
- `ros2 run lite6_gui lite6_gui`

## 2) Files produced by each stage

After Phase 2:
- `human_arm_demo.jsonl`
- `human_arm_demo.npz`
- `human_arm_demo_metadata.json`

After Phase 4:
- `lite6_targets.jsonl`
- `lite6_targets_preview.npz`
- `lite6_targets_report.json`

Default calibration output:
- `~/ws_xarm/calibration/freemocap_to_lite6.yaml`

## 3) Typical workflow

The normal operator flow is:
1. Export human arm features from a FreeMoCap recording.
2. Optionally create or update a calibration file.
3. Export Lite6 retargeted targets.
4. Preview the targets offline.
5. Inspect them in CLI and/or GUI.
6. Only if you have a file with valid `joint_positions`, execute it through `lite6_motion`.

## 4) Step-by-step usage

### Step A - Export human arm features from FreeMoCap

Input can be either:
- a FreeMoCap recording folder, or
- a direct `.npy` landmarks file

Example:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
source ~/ws_xarm/install/setup.zsh

ros2 run lite6_imitation_bridge export_human_arm_demo \
  /path/to/freemocap_recording \
  --arm-side right
```

Optional flags:
- `--arm-side left|right` (default: `right`)
- `--output-dir /custom/output/dir`
- `--active-tracker mediapipe`

Notes:
- Only the `mediapipe` tracker is currently supported for this export path.
- If `--output-dir` is omitted, outputs are written into the recording's `output_data` directory, or next to the `.npy` file.

### Step B - Create calibration (optional but recommended)

Calibration aligns FreeMoCap coordinates with the Lite6 workspace.

Command:

```zsh
ros2 run lite6_imitation_bridge calibrate_human_to_robot_frame \
  /path/to/calibration_manifest.json
```

Optional:

```zsh
ros2 run lite6_imitation_bridge calibrate_human_to_robot_frame \
  /path/to/calibration_manifest.json \
  --output-path ~/ws_xarm/calibration/freemocap_to_lite6.yaml
```

The calibration manifest must be a JSON object containing at least:
- `source_points`
- `target_points`

It may also include:
- `source_frame`
- `target_frame`
- `notes`
- `known_pose_name`
- `workspace_origin`
- `workspace_description`

Example manifest:

```json
{
  "source_points": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
  "target_points": [[0.2, 0.0, 0.2], [0.3, 0.0, 0.2], [0.2, 0.1, 0.2]],
  "source_frame": "freemocap",
  "target_frame": "lite6_workspace",
  "known_pose_name": "desk_reference",
  "workspace_description": "front-right tabletop"
}
```

### Step C - Export Lite6 target records

This converts human-arm features into conservative Lite6-oriented target records.

Without calibration:

```zsh
ros2 run lite6_imitation_bridge export_lite6_targets \
  /path/to/freemocap_recording \
  --arm-side right
```

With calibration:

```zsh
ros2 run lite6_imitation_bridge export_lite6_targets \
  /path/to/freemocap_recording \
  --arm-side right \
  --calibration-path ~/ws_xarm/calibration/freemocap_to_lite6.yaml
```

Optional flags:
- `--output-dir /custom/output/dir`
- `--active-tracker mediapipe`
- `--max-step-distance 0.25`

What this writes:
- `lite6_targets.jsonl`
- `lite6_targets_preview.npz`
- `lite6_targets_report.json`

Important:
- These exported records are valid for preview.
- Real execution currently requires per-frame `joint_positions` values to exist in the JSONL records.
- If the records only contain Cartesian-style fields such as `target_position`, execution will be refused safely.

## 5) Preview the targets safely

### CLI preview summary

Use this to summarize a target artifact without commanding the robot:

```zsh
ros2 run lite6_imitation_bridge preview_targets \
  /path/to/lite6_targets.jsonl
```

You can also point it at the report file:

```zsh
ros2 run lite6_imitation_bridge preview_targets \
  /path/to/lite6_targets_report.json
```

The summary includes:
- frame count
- valid/invalid count
- clamped count
- frame range
- confidence statistics
- rejection reason counts

### Publish preview messages to ROS only

This is still preview-only and does not touch the live motion topic:

```zsh
ros2 run lite6_imitation_bridge publish_joint_command_preview \
  /path/to/lite6_targets.jsonl \
  --ros-publish
```

Preview/status topics used:
- `/lite6_imitation/status`
- `/lite6_imitation/preview_targets`

Preview flow never publishes to:
- `/lite6_motion/joint_command`
- `/lite6_imitation/live_targets`

## 6) Start the safe motion server

Before any real motion, the `lite6_motion` server must be running.

In one terminal:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
source ~/ws_xarm/install/setup.zsh
ros2 run lite6_motion motion_server
```

Useful status check in another terminal:

```zsh
cd ~/ws_xarm
./scripts/lite6_cli.zsh status
```

## 7) Use the zsh helper CLI

Wrapper commands:

```zsh
cd ~/ws_xarm
./scripts/lite6_cli.zsh status
./scripts/lite6_cli.zsh enable
./scripts/lite6_cli.zsh angles
./scripts/lite6_cli.zsh preview_demo /path/to/lite6_targets.jsonl
./scripts/lite6_cli.zsh stop
```

Real motion commands require explicit confirmation.

For non-interactive execution:

```zsh
YES_MOVE=1 ./scripts/lite6_cli.zsh execute_demo /path/to/file.jsonl
```

Other helper commands:
- `./scripts/lite6_cli.zsh tiny_test`
- `YES_MOVE=1 ./scripts/lite6_cli.zsh move_pose '0 -0.5 0.8 0 0 0'`
- `./scripts/lite6_cli.zsh go_home`
- `./scripts/lite6_cli.zsh record_3panel_yolo 20`

## 8) Execute a demo safely

Execution is only supported when the JSONL file contains:
- `valid: true`
- `joint_positions: [j1, j2, j3, j4, j5, j6]`

If a record is invalid, or if `joint_positions` is missing, execution will be refused.

### Direct ROS command

```zsh
ros2 run lite6_imitation_bridge execute_lite6_targets \
  execute_demo /path/to/file.jsonl \
  --rate-hz 2.0 \
  --confirm
```

### zsh helper command

```zsh
cd ~/ws_xarm
YES_MOVE=1 ./scripts/lite6_cli.zsh execute_demo /path/to/file.jsonl
```

What execution does:
- loads executable joint targets from the JSONL file
- schedules them at bounded rate
- rechecks motion status before each publish
- rechecks stop state after sleep and before publish
- publishes only to `/lite6_motion/joint_command`
- requests stop automatically if execution becomes unsafe

What execution does not do:
- it does not infer IK automatically from `target_position`
- it does not bypass `lite6_motion`
- it does not ignore invalid frames

## 9) Stop behavior

You can request an immediate stop with either command:

```zsh
./scripts/lite6_cli.zsh stop
```

or

```zsh
ros2 run lite6_imitation_bridge execute_lite6_targets stop
```

Current stop semantics:
- future joint command execution is blocked
- `go_home` is blocked
- stop remains active until `prepare_robot` is called again

To clear stop and re-arm motion:

```zsh
./scripts/lite6_cli.zsh enable
```

## 10) GUI preview inspection

You can inspect preview artifacts in the GUI:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
source ~/ws_xarm/install/setup.zsh
ros2 run lite6_gui lite6_gui
```

In the GUI:
1. Enter the path to `lite6_targets.jsonl` or `lite6_targets_report.json` in the `Preview artifact` field.
2. Click `LOAD PREVIEW SUMMARY`.
3. Inspect the preview summary label before doing any motion.

The GUI preview summary is inspection-only.
It does not execute the target sequence by itself.

## 11) Recommended operator workflow

For a normal session, use this order:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
source ~/ws_xarm/install/setup.zsh

# 1) export features
ros2 run lite6_imitation_bridge export_human_arm_demo /path/to/freemocap_recording --arm-side right

# 2) export retargeted Lite6 preview artifacts
ros2 run lite6_imitation_bridge export_lite6_targets /path/to/freemocap_recording --arm-side right --calibration-path ~/ws_xarm/calibration/freemocap_to_lite6.yaml

# 3) preview only
ros2 run lite6_imitation_bridge preview_targets /path/to/output/lite6_targets.jsonl

# 4) optional ROS preview publish
ros2 run lite6_imitation_bridge publish_joint_command_preview /path/to/output/lite6_targets.jsonl --ros-publish

# 5) if and only if you have a joint_positions-capable file, start motion server and execute
ros2 run lite6_motion motion_server
```

Then in another terminal:

```zsh
cd ~/ws_xarm
./scripts/lite6_cli.zsh status
./scripts/lite6_cli.zsh enable
YES_MOVE=1 ./scripts/lite6_cli.zsh execute_demo /path/to/joint_position_demo.jsonl
```

## 12) Current limitations

Important current limitation:
- The Phase 6 executor accepts only files that already contain per-frame `joint_positions`.
- A plain Phase 4 `lite6_targets.jsonl` may still be previewable but not executable.
- If the file contains only `target_position`, `approach_direction`, `elbow_configuration`, etc., the executor will reject it on purpose.

This is intentional safety behavior.
Nothing is silently converted into robot motion.

## 13) Quick troubleshooting

If `preview_targets` works but execution fails:
- inspect the JSONL file for `joint_positions`
- check for any invalid frames
- check `./scripts/lite6_cli.zsh status`
- verify `lite6_motion` is running
- verify the robot was prepared with `./scripts/lite6_cli.zsh enable`
- verify stop was not latched; if it was, call `enable` again

If `execute_demo` says there are no subscribers:
- make sure `ros2 run lite6_motion motion_server` is running

If motion is refused:
- check `/lite6_motion/status`
- check stale state / robot not enabled / robot error / stop requested

## 14) Fastest minimal commands

Preview only:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
source ~/ws_xarm/install/setup.zsh
ros2 run lite6_imitation_bridge preview_targets /path/to/lite6_targets.jsonl
```

Execute only, if file already contains `joint_positions`:

Terminal 1:

```zsh
cd ~/ws_xarm
source /opt/ros/humble/setup.zsh
source ~/ws_xarm/install/setup.zsh
ros2 run lite6_motion motion_server
```

Terminal 2:

```zsh
cd ~/ws_xarm
YES_MOVE=1 ./scripts/lite6_cli.zsh enable
YES_MOVE=1 ./scripts/lite6_cli.zsh execute_demo /path/to/joint_position_demo.jsonl
```
