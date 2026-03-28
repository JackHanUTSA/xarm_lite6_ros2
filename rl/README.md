# RL (DreamerV3) Plan

- Env: Lite6 reach in Isaac Sim (**headless by default**).
- Action: joint position deltas (6)
- Obs (simple): q(6) + ee_pos(3) + target_pos(3)
- Episode length: 200
- Bounds default (m): x 0.20..0.45, y -0.20..0.20, z 0.12..0.40

## GUI toggle (show Isaac Sim window)

The Isaac worker is **headless by default** for performance.

To show the Isaac Sim window (for real-time viewing), set:

```zsh
export LITE6_GUI=1
```

Example:

```zsh
cd ~/ws_xarm/rl
LITE6_GUI=1 ./scripts/run_supervised_lite6.zsh
```

Notes:
- Requires a display (local desktop/VNC/X-forwarding).
- Do **not** run V3 and V4 at the same time on the same port.
