# Move robot-arm training to the **swarmlab** workstation

This doc is a **runbook** for migrating the entire robot arm training experiment from the current workstation to the other node/workstation ("swarmlab", OpenClaw node name: `swarm-lab-026`).

## Goal
- Copy **code + environment setup + checkpoints + logs + generated videos** to swarmlab
- Start training on swarmlab using the **same launch command**
- (Optional) stop training on the current workstation after swarmlab is confirmed running

## Preconditions
1. You have shell access on both machines.
2. The OpenClaw node on swarmlab is running and connected.
3. You know the current training launch command(s).

---

## 0) Identify the current run
On the **current** workstation:

- Repo root:
  - `~/ws_xarm`
- Logs (example):
  - `~/ws_xarm/rl/logdir/`
- Common subfolders to migrate (if they exist):
  - `~/ws_xarm/rl/logdir/episodes/` (videos)
  - `~/ws_xarm/rl/logdir/checkpoints/` (or whatever your ckpt path is)
  - `~/ws_xarm/rl/logdir/wandb/` (if using W&B offline)

If you have a different checkpoint dir, write it down now.

---

## 1) Bring swarmlab online in OpenClaw
On **swarmlab**:

1. Start the OpenClaw node service/agent (whatever your setup uses).
2. Verify it is connected:
   - `openclaw status`

In OpenClaw UI, it should show as **connected**:
- Node: `swarm-lab-026`

---

## 2) Prepare the target directory on swarmlab
On **swarmlab** (recommended):

```sh
mkdir -p ~/ws_xarm
mkdir -p ~/ws_xarm/rl/logdir
```

If you want to keep multiple runs, you can instead target something like:
- `~/ws_xarm_runs/<run_id>/...`

---

## 3) Sync the repo (code)
### Option A: git (preferred)
On swarmlab:

```sh
cd ~
git clone <YOUR_REPO_URL> ws_xarm
cd ~/ws_xarm
git checkout <BRANCH_OR_COMMIT>
```

### Option B: rsync the working tree
From current workstation → swarmlab:

```sh
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  ~/ws_xarm/  <USER>@<SWARMLAB_HOST>:~/ws_xarm/
```

---

## 4) Recreate the Python environment on swarmlab
Pick the method you actually use:

### Conda
```sh
conda env create -f environment.yml
conda activate <ENV_NAME>
```

### venv
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Docker
- Pull/build the same image/tag
- Ensure GPU runtime (`--gpus all`) matches

Notes:
- Ensure CUDA / drivers match what your training stack expects.
- If you rely on system libs (mujoco, ffmpeg, etc.), install them on swarmlab too.

---

## 5) Sync checkpoints + logs + videos
From current workstation → swarmlab:

```sh
rsync -av \
  ~/ws_xarm/rl/logdir/  <USER>@<SWARMLAB_HOST>:~/ws_xarm/rl/logdir/
```

If you want to reduce transfer size, you can sync only what matters:
- checkpoints
- latest logs
- episodes/videos

---

## 6) Start training on swarmlab
Run the **same launch command** you use today (examples below). Use `tmux`/`screen` or a service so it survives disconnects.

### Example: tmux
```sh
tmux new -s xarm_train
cd ~/ws_xarm
# activate env here
python dreamerv3_lite6_main.py --config ...
```

If you also have a supervised sidecar process:
```sh
tmux new -s xarm_supervised
cd ~/ws_xarm
# activate env here
python supervised_train_lite6.py --config ...
```

---

## 7) Validate it’s running
On swarmlab:

```sh
pgrep -af dreamerv3_lite6_main\.py || true
pgrep -af supervised_train_lite6\.py || true
nvidia-smi
```

Confirm:
- exactly one PID for each intended process
- GPU memory and utilization are non-zero
- new entries appear in `~/ws_xarm/rl/logdir/scores.jsonl`

---

## 8) (Optional) Stop training on the old workstation
Only after swarmlab is confirmed healthy.

If launched via tmux:
- attach and stop cleanly, or kill the session

If launched directly:
```sh
pkill -f dreamerv3_lite6_main\.py
pkill -f supervised_train_lite6\.py
```

---

## Operational tips
- **Avoid double-running** against the same robot hardware unless you *intend* two controllers.
- If the robot arm is physically connected to only one workstation, ensure:
  - USB/ethernet device is reachable from swarmlab
  - any ROS/driver endpoints are updated
- If you log videos, confirm ffmpeg is installed on swarmlab.

---

## Fill-in checklist (to make this repeatable)
- [ ] swarmlab host/IP: `______________`
- [ ] ssh user: `______________`
- [ ] repo URL or sync method: `______________`
- [ ] env name / setup steps: `______________`
- [ ] training launch command: `______________`
- [ ] supervised launch command (if any): `______________`
- [ ] checkpoint dir (if not under logdir): `______________`
