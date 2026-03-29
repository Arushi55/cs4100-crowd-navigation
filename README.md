# cs4100-crowd-navigation

Pygame-based crowd navigation sandbox with two main pieces:

- an interactive simulator for manually watching scenarios and pedestrian behavior
- a Gymnasium RL environment for training DQN or PPO agents on the same maps

The current workflow is centered around the airport scenario, grouped pedestrian traffic, and PPO training with frame-stacked observations.

## Quick setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the simulator

From the repository root:

```bash
python src/main.py
```

Useful flags:

- `--scenario {home|airport|shopping_center}`
- `--seed <int>`
- `--random-seed`
- `--random-world`
- `--pedestrians <int>`
- `--scenario-config-dir <path>`

Examples:

```bash
python src/main.py --scenario airport --pedestrians 24 --seed 42
python src/main.py --scenario home --random-seed --random-world
```

Controls:

- `WASD` or arrow keys move the robot in manual mode
- `1`, `2`, `3` switch scenarios
- close the pygame window to exit

## RL training workflow (DQN)

We removed stable-baselines3 and now train with a lightweight PyTorch DQN.

Main entrypoint (wraps `src/dqn.py`):

```bash
python src/train.py \
  --scenario airport \
  --vary-pedestrians \
  --pedestrians-min 12 \
  --pedestrians-max 24 \
  --ped-speed-min 0.95 \
  --ped-speed-max 1.15 \
  --frame-stack 4 \
  --total-steps 200000 \
  --buffer-size 200000 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --target-update 2000 \
  --output-dir training_output/dqn
```

What gets saved under `output-dir`:

- `dqn.pt` (model weights)
- `run_config.json` (run metadata)

### Evaluate a DQN

```bash
python src/evaluate.py --model training_output/dqn/dqn.pt --scenario airport --episodes 10 --fps 20
```

### Benchmark across crowd sizes/speeds

```bash
python src/benchmark.py training_output/dqn/dqn.pt
```

- `--gamma`
  - reward discount factor
  - recent runs used `0.995`
- `--ent-coef`
  - exploration pressure for PPO
  - recent runs used `0.01` or `0.008`
- `--eval-freq`
  - how often the script runs the eval callback
- `--checkpoint-freq`
  - how often intermediate checkpoints are saved
- `--resume`
  - path to a `.zip` model for continued training
- `--output-dir`
  - where run artifacts are written

## Training modes

There are three main training modes in `src/train.py`.

### 1. Fixed single-scenario training

```bash
python src/train.py --algo ppo --scenario airport --pedestrians 24
```

Use this when you want one fixed density.

### 2. Single-scenario variable crowd training

```bash
python src/train.py \
  --algo ppo \
  --scenario airport \
  --vary-pedestrians \
  --pedestrians-min 16 \
  --pedestrians-max 30
```

Use this when you want one map but different crowd sizes each episode. This is the most useful mode for current airport work.

### 3. Multi-scenario curriculum-style training

```bash
python src/train.py --algo ppo --multi-scenario
```

This samples across `airport`, `home`, and `shopping_center`, along with random crowd count and speed.

## Evaluate a trained model

Main entrypoint:

```bash
python src/evaluate.py --model training_output/airport_ppo_run/best_model/best_model.zip
```

### Batch evaluation

```bash
python src/evaluate.py \
  --model training_output/airport_ppo_run/best_model/best_model.zip \
  --scenario airport \
  --pedestrians 30 \
  --episodes 10 \
  --max-steps 1000 \
  --no-render
```

### Visual evaluation

```bash
python src/evaluate.py \
  --model training_output/airport_ppo_run/best_model/best_model.zip \
  --scenario airport \
  --pedestrians 30 \
  --episodes 5 \
  --fps 20
```

`evaluate.py` automatically reads `run_config.json` when possible, so it can recover the right algorithm and frame-stack settings from the saved run.

Reported metrics include:

- success rate
- average reward
- average steps
- collisions
- near misses
- personal-space intrusions
- pedestrian slowdown
- blocking pressure

## Benchmark a model

For a broader sweep across pedestrian counts and speeds:

```bash
python src/benchmark.py training_output/airport_ppo_run/best_model/best_model.zip
```

This prints summary tables for:

- scaling pedestrian count
- scaling pedestrian speed
- combined stress tests

## Current environment notes

The current RL environment in `src/crowd_env.py` includes:

- goal-touch success, meaning the robot succeeds when it physically touches the goal area
- ray-based sensing
- frame-stacked observations during training when enabled
- nearest visible pedestrian motion features
- reward shaping for progress, collisions, personal-space violations, crowd pressure, blocking, slowdown, and wall interactions
- smoothed control to reduce overly twitchy steering

The airport scenario now uses grouped edge-to-edge pedestrian traffic intended to feel more like families moving through an airport concourse.

## Project structure

```text
src/main.py                              # Interactive simulator
src/crowd_env.py                         # Gymnasium crowd-navigation environment
src/train.py                             # RL training entrypoint
src/evaluate.py                          # Model evaluation + pygame viewer
src/benchmark.py                         # Batch benchmarking across counts/speeds
src/multi_env.py                         # Variable-crowd and multi-scenario env wrappers
src/wrappers.py                          # Observation frame stacking
src/constants.py                         # Shared dimensions/constants
src/agent/behaviors.py                   # Robot control policies for simulator mode
src/agent/sensor.py                      # Ray sensor and visualization
src/environment/robot.py                 # Robot movement and collision handling
src/environment/pedestrian.py            # Pedestrian state, pathing, waypoint steering
src/environment/behaviors.py             # Pedestrian behavior models, including family groups
src/environment/pathfinding.py           # A* navigation grid
src/environment/scenarios.py             # Scenario loading and pedestrian generation
src/environment/scenario_configs/*.json  # Scenario configs
PROJECT.md                               # Extended project reference and roadmap
```
