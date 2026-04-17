# cs4100-crowd-navigation

Pygame-based crowd navigation sandbox with two main pieces:

- an interactive simulator for manually watching scenarios and pedestrian behavior
- a Gymnasium RL environment for training DQN agents on the same maps

The primary workflow is now the custom PyTorch DQN pipeline (`src/dqn.py`) with frame-stacked observations.

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
- `--mode {manual|potential_field|naive|random}`
- `--scenario-config-dir <path>`

Examples:

```bash
python src/main.py --scenario airport --pedestrians 24 --seed 42
python src/main.py --scenario home --random-seed --random-world
python src/main.py --scenario airport --pedestrians 24 --mode manual
```

Default simulator mode is `potential_field`. Use `--mode manual` for keyboard control.

Controls:

- `WASD` or arrow keys move the robot in `manual` mode
- `1`, `2`, `3` switch scenarios
- `Q` or `Esc` exits
- closing the pygame window also exits

## RL training workflow (DQN-first)

Main training entrypoint:

```bash
python src/dqn.py
```

### Recommended airport command

```bash
python src/dqn.py \
  --scenario airport \
  --vary-pedestrians \
  --pedestrians-min 12 --pedestrians-max 24 \
  --ped-speed-min 0.95 --ped-speed-max 1.15 \
  --frame-stack 4 \
  --total-steps 200000 \
  --buffer-size 200000 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --target-update 2000 \
  --output-dir training_output/dqn
```

### What the DQN script saves

Each run writes:

- `training_output/dqn/dqn.pt`
- `training_output/dqn/run_config.json`

The script periodically overwrites `dqn.pt` as checkpoints and writes the final weights to the same path.

## Training parameters that matter most

These are the most commonly tuned DQN flags:

- `--scenario`
- `--pedestrians`
- `--vary-pedestrians`
- `--pedestrians-min`, `--pedestrians-max`
- `--ped-speed-min`, `--ped-speed-max`
- `--frame-stack`
- `--total-steps`
- `--max-steps`
- `--learning-rate`
- `--batch-size`
- `--buffer-size`
- `--target-update`
- `--eps-start`, `--eps-end`, `--eps-decay-steps`
- `--hidden-sizes`, `--hidden-activation`
- `--dueling-dqn`, `--double-dqn`
- `--prioritized`, `--priority-alpha`, `--priority-beta-start`, `--priority-beta-frames`
- `--n-step`
- `--output-dir`

## Training modes

### 1. Fixed single-scenario training

```bash
python src/dqn.py --scenario airport --pedestrians 24
```

### 2. Single-scenario variable crowd training

```bash
python src/dqn.py \
  --scenario airport \
  --vary-pedestrians \
  --pedestrians-min 16 \
  --pedestrians-max 30
```

### 3. Multi-scenario training

```bash
python src/dqn.py --multi-scenario
```

## Evaluate a trained model

Main entrypoint:

```bash
python src/evaluate.py --model training_output/dqn/dqn.pt
```

### Batch evaluation

```bash
python src/evaluate.py --model training_output/dqn/dqn.pt --scenario airport --episodes 10 --fps 20 --no-render
```

### Visual evaluation

```bash
python src/evaluate.py --model training_output/dqn/dqn.pt --scenario airport --episodes 10 --fps 20
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
python src/benchmark.py training_output/dqn/dqn.pt
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
src/dqn.py                               # Custom PyTorch DQN trainer
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
