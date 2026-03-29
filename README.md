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

## RL training workflow

The main training entrypoint is:

```bash
python src/train.py
```

The current recommended setup is PPO with frame stacking on a single scenario while varying pedestrian count across episodes.

### Recommended airport command

This is the best starting point for the current repo:

```bash
python src/train.py \
  --algo ppo \
  --scenario airport \
  --vary-pedestrians \
  --pedestrians-min 16 \
  --pedestrians-max 30 \
  --ped-speed-min 0.95 \
  --ped-speed-max 1.15 \
  --frame-stack 4 \
  --num-envs 4 \
  --total-timesteps 120000 \
  --max-episode-steps 1000 \
  --learning-rate 3e-4 \
  --batch-size 256 \
  --n-steps 1024 \
  --gamma 0.995 \
  --ent-coef 0.01 \
  --eval-freq 10000 \
  --checkpoint-freq 25000 \
  --output-dir training_output/airport_ppo_run
```

### What the training script saves

Each run writes a folder under `training_output/` containing:

- `best_model/best_model.zip`
- `final_model.zip`
- `checkpoints/`
- `logs/`
- `run_config.json`

`best_model.zip` is usually the model you want to evaluate or visualize first.

### Resume training

To continue from an existing checkpoint:

```bash
python src/train.py \
  --algo ppo \
  --scenario airport \
  --vary-pedestrians \
  --pedestrians-min 16 \
  --pedestrians-max 30 \
  --frame-stack 4 \
  --num-envs 4 \
  --total-timesteps 80000 \
  --resume training_output/airport_ppo_run/best_model/best_model.zip \
  --output-dir training_output/airport_ppo_run_resume
```

## Training parameters that matter most

These are the ones the team will probably tweak most often:

- `--algo`
  - `ppo` is the default and the recommended option now
  - `dqn` still works, but is generally weaker on the current dynamic crowd setup
- `--scenario`
  - choose the map to train on
- `--pedestrians`
  - fixed pedestrian count for single-density training
- `--vary-pedestrians`
  - randomizes pedestrian count every episode for one scenario
- `--pedestrians-min`, `--pedestrians-max`
  - crowd range when varying pedestrians
- `--ped-speed-min`, `--ped-speed-max`
  - speed multiplier range applied each episode in variable-crowd environments
- `--frame-stack`
  - how many consecutive observations are stacked together
  - current recommended value is `4`
- `--num-envs`
  - number of parallel PPO envs
  - current recommended value is `4`
- `--total-timesteps`
  - total training budget
- `--max-episode-steps`
  - timeout horizon before an episode truncates
- `--learning-rate`
  - PPO default in the script is `1e-4`, but our stronger recent airport runs used `3e-4` or `2.5e-4`
- `--batch-size`
  - typical PPO values we used were `256`
- `--n-steps`
  - PPO rollout length per env before update
  - typical value: `1024`
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
