# Lite6 CLI (zsh)

Quick control helpers for the UFactory xArm Lite6 ROS2 stack.

## Usage

```zsh
cd ~/ws_xarm
./scripts/lite6_cli.zsh status
./scripts/lite6_cli.zsh enable
./scripts/lite6_cli.zsh angles

# Motion commands require confirmation
./scripts/lite6_cli.zsh tiny_test
YES_MOVE=1 ./scripts/lite6_cli.zsh move_pose "0 -0.5 0.8 0 0 0"

# Record video (no motion)
./scripts/lite6_cli.zsh record_3panel_yolo 20
```
