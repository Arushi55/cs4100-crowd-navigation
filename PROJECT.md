# CS4100 Crowd Navigation Project Reference

## 1) Project Overview

### Problem Statement
As robots become more common in public and semi-public spaces (hospitals, airports, hallways, evacuation routes), crowd navigation is a core adoption barrier. The key challenge is balancing:

- **Task efficiency**: reaching the destination quickly and reliably
- **Social safety/comfort**: avoiding collisions and minimizing intrusions into personal space

This project studies the trade-off explicitly:

> Is it worth accepting a social penalty (getting too close to people) to increase destination success and speed?

### Core Goal
Build and evaluate a crowd-navigation policy in partially observable simulation environments, measuring performance with consistent metrics:

- time-to-goal (steps-to-goal)
- collisions/overlaps
- proxemic intrusions (personal-space violations)
- near misses
- pedestrian slowdown caused by the robot
- aggregate reward

---

## 2) Current Repository Snapshot

- **GitHub repo**: `https://github.com/Arushi55/cs4100-crowd-navigation`
- **Default branch**: `main`
- **Current structure**:

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
src/environment/scenario_configs/*.json  # Scenario configs (airport, home, shopping_center)
```

### What Is Implemented

1. **Three scenarios**: `airport`, `home`, `shopping_center`, each with distinct layouts, spawn points, and pedestrian flow patterns.

2. **Custom PyTorch DQN** (`src/dqn.py`):
   - Dueling DQN architecture
   - Double DQN target computation
   - Prioritized experience replay (PER)
   - N-step returns
   - Frame-stacked observations (configurable stack size)
   - Configurable epsilon schedules (linear, exponential, cosine, constant)
   - Configurable hidden sizes, activations, optimizers, and loss functions

3. **Observation space** (per frame):
   - Robot and goal positions (normalized)
   - 36-ray full-circle sensor (distance + hit type per ray)
   - Nearest 6 visible pedestrians: relative position, velocity, distance

4. **Reward shaping**:
   - Progress toward goal (distance delta)
   - Goal contact (+1000 terminal)
   - Timeout penalty
   - Collision, personal-space, near-miss, caution-zone penalties
   - Crowd pressure and crowd approach penalties
   - Wall proximity and wall scrape penalties
   - Turn/smoothness penalty
   - Near-goal penalty blending (social penalties scale down as agent approaches goal)
   - No-progress stagnation penalty

5. **Training modes**:
   - Fixed single scenario with fixed pedestrian count
   - Single scenario with variable pedestrian count and speed each episode
   - Multi-scenario (randomizes across all scenarios each episode)

6. **Evaluation and benchmarking** (`src/evaluate.py`, `src/benchmark.py`):
   - Per-episode metrics: success, steps, reward, collisions, near misses, intrusions, slowdown, blocking, wall contacts, path efficiency
   - Benchmark sweeps across pedestrian counts and speeds

---

## 3) Commit-History Milestones

1. Initial project scaffold and pygame startup
2. Refactor into separate classes/modules
3. Add penalties for proximity/collision
4. Add social-force pedestrian behavior
5. Add goal detection, multiple control modes, and episode metrics
6. Structural reorganization into current module layout
7. Gymnasium environment (`crowd_env.py`) with ray sensor and frame stacking
8. Custom PyTorch DQN with prioritized replay, dueling/double DQN, n-step returns
9. Multi-scenario and variable-pedestrian environment wrappers
10. Improved pedestrian behavior (family groups, flow patterns)
11. Reward refinements: near-goal penalty blending, no-progress penalty, social shaping tuning

---

## 4) Research Context and References

### Papers Mentioned

1. **Intention-aware interaction graph (Liu et al., 2022)**
   - Uses attention-based interaction graphs to predict trajectories and reduce path intrusions.

2. **DS-RNN (Liu et al., ICRA 2021)**
   - Decentralized structured RNN for robot crowd navigation in partially observable dense crowds.

3. **Improved DS-RNN (Zhang & Feng, 2023)**
   - Adds coarse local maps to model human-human interactions better and improve safety.

### Tools and Additional References

- **RVO2**: `https://github.com/snape/RVO2`
  - Useful for velocity-obstacle-based baseline comparisons.

- **Proxemic distance definitions**:
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC7918518/`
  - Grounds intrusion metrics in established intimate/personal/social/public distance zones.

---

## 5) Evaluation Framing

Each policy is evaluated under identical scenarios with fixed random seeds:

- **Success rate**: percentage of episodes reaching goal before timeout
- **Efficiency**: steps to goal
- **Safety**: collision count
- **Social compliance**: near-miss count, personal-space intrusion count/duration, pedestrian slowdown caused
- **Path efficiency**: actual path length / shortest path length

### Objective Template

Use a tunable composite score for ablation sweeps over social cost weights:

`score = w_goal * goal_reached - w_time * steps - w_collision * overlaps - w_intrusion * intrusion_time`

---

## 6) Scenario Descriptions

### airport
Edge-to-edge family-group pedestrian flow across a large concourse with obstacles. Intended to simulate families moving through a busy terminal.

### home
Residential layout with smaller rooms and tighter corridors. Variable pedestrian count (12–50) at moderate speed.

### shopping_center
Open floor plan with scattered obstacles and bidirectional pedestrian traffic.

---

## 7) Current Next Steps

### Research / Evaluation
1. **Run multi-seed evaluation** across all three scenarios at varying pedestrian counts (0, 12, 30, 50+) to produce reportable results.
2. **Ablation on social-penalty weights** — sweep `blocking_penalty_scale`, `personal_space_penalty`, `crowd_approach_scale` to quantify the efficiency/safety trade-off.
3. **Compare training modes** — fixed vs. variable-crowd vs. multi-scenario on generalization.
4. **Add hand-crafted baselines** for comparison:
   - Naive goal-seeking (already in simulator)
   - Potential field (already in simulator)
   - Consider RVO/ORCA integration as a stronger baseline

### Engineering
5. **Save per-episode metrics to CSV** during `evaluate.py` runs (flag already exists: `--save-metrics`) and build a summary analysis script.
6. **Multi-seed training sweeps** — train N seeds per config, report mean ± std.

---

## 8) Risks and Mitigations

1. **Metric drift / inconsistent definitions**
   - Mitigation: formal metric schema; use `evaluate.py` consistently across all comparisons.

2. **Non-reproducible experiments**
   - Mitigation: `run_config.json` is saved with every training run; always pass `--seed`.

3. **Overfitting to one crowd pattern**
   - Mitigation: use `--vary-pedestrians` or `--multi-scenario` during training; evaluate out-of-distribution.

4. **Performance claims without statistical support**
   - Mitigation: run multiple seeds and report mean + variance.

---

## 9) Project Tracker: Individual Contributions (Commit-Graph Based)

### Identity Normalization Note

One contributor appears under two git author names with the same email:

- `phisomni-edu <dharma.ar@northeastern.edu>`
- `phisomni <dharma.ar@northeastern.edu>`

For project tracking, these are treated as the same contributor.

### Contribution Summary by Contributor

1. **efsmert** (`samiareski05@gmail.com`)
   - Initial project setup and pygame simulation with baseline crowd/room/start-end behavior
   - Repository hygiene (.gitignore)

2. **Arushi Aggarwal** (`arushi3.14@outlook.com`)
   - Refactored simulation into separate classes/modules
   - Added penalty logic to scoring behavior
   - Gymnasium environment, DQN training pipeline, evaluation and benchmark scripts
   - Reward shaping, multi-scenario wrappers, pedestrian improvements

3. **Dharma (phisomni / phisomni-edu)** (`dharma.ar@northeastern.edu`)
   - Goal detection and multi-mode control behavior
   - Autonomous behavior set and episode metrics support
   - Structural reorganization to current module layout (`src/agent`, `src/environment`, `src/main.py`)

4. **Lisa W** (`wan.lis@northeastern.edu`)
   - Social-force pedestrian behavior integration

### Tracker Maintenance Rule (Going Forward)

At each milestone, append:

- contributor name
- commit hashes merged in milestone
- short description of net contribution

This keeps individual-contribution evidence synchronized with the evolving codebase.
