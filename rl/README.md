# RL (DreamerV3) Plan

- Env: Lite6 reach in Isaac Sim (headless).
- Action: joint position deltas (6)
- Obs (simple): q(6) + ee_pos(3) + target_pos(3)
- Episode length: 200
- Bounds default (m): x 0.20..0.45, y -0.20..0.20, z 0.12..0.40
