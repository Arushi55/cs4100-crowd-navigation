# CS4100 Crowd Navigation Project Reference

## 1) Project Overview

### Problem Statement
As robots become more common in public and semi-public spaces (hospitals, airports, hallways, evacuation routes), crowd navigation is a core adoption barrier. The key challenge is balancing:

- **Task efficiency**: reaching the destination quickly and reliably
- **Social safety/comfort**: avoiding collisions and minimizing intrusions into personal space

This project studies the trade-off explicitly:

> Is it worth accepting a social penalty (getting too close to people) to increase destination success and speed?

### Core Goal
Build and compare multiple crowd-navigation policies in partially observable simulation environments, then evaluate performance with consistent metrics:

- time-to-goal (or steps-to-goal)
- collisions/overlaps
- proxemic intrusions (personal-space violations)
- aggregate reward/penalty

---

## 2) Current Repository Snapshot

- **GitHub repo**: `https://github.com/Arushi55/cs4100-crowd-navigation`
- **Default branch**: `main`
- **Current structure**:
  - `src/main.py` (pygame simulation loop, penalties, episodes)
  - `src/agent/behaviors.py` (manual, naive, random, potential-field policies)
  - `src/environment/pedestrian.py` (social-force-like pedestrian dynamics)
  - `src/environment/robot.py` (robot state and movement)
  - `src/constants.py` (simulation dimensions)
  - `tests/test_smoke.py` (very minimal smoke test)

### Important Current-State Notes

1. **Simulation is functional and already supports multiple control modes**:
   - `MANUAL`
   - `NAIVE` (goal-seeking only)
   - `RANDOM`
   - `POTENTIAL_FIELD` (goal attraction + pedestrian repulsion)

2. **Penalty shaping is present**:
   - Near-distance penalty for entering a close radius
   - Tiered overlap penalties based on collision depth

3. **Episode tracking exists**:
   - Per-episode penalty and step counts
   - Running averages printed to console

4. **Pedestrian behavior includes social-force components**:
   - self-driving force toward goal
   - pedestrian-pedestrian repulsion
   - optional wall repulsion terms (currently set so wall force has no effect)

5. **Docs/setup mismatch exists**:
   - `README.md` references package paths/extras (`crowd_navigation`, `pyproject.toml`, `requirements-dev.txt`) that are not currently present in the repository snapshot.

---

## 3) Commit-History Milestones (Observed)

Recent commits indicate this progression:

1. Initial project scaffold and pygame startup
2. Refactor into separate classes/modules
3. Add penalties for proximity/collision
4. Add social-force pedestrians
5. Add goal detection, multiple control modes, and episode metrics
6. Structural reorganization and mode documentation

This is a solid base for moving into proper experiment design and reproducible benchmarking.

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
  - Can ground intrusion metrics in established distance zones.

---

## 5) Proposed Evaluation Framing

To answer the central trade-off question, evaluate each policy under identical scenarios with fixed random seeds:

- **Success rate**: percentage of episodes reaching goal before timeout
- **Efficiency**: steps/time to goal
- **Safety**: collision count and overlap severity
- **Social compliance**: proxemic intrusion count/duration
- **Composite score**: weighted objective to explore trade-offs

### Recommended Objective Template

Use a tunable objective so you can run ablations over social cost weights:

`score = w_goal * goal_reached - w_time * steps - w_collision * overlaps - w_intrusion * intrusion_time`

Run sweeps over `w_intrusion`/`w_collision` to quantify when social penalties meaningfully improve or hurt mission success.

---

## 6) Scenario Plan (Controlled Environments)

Start with a small suite of reproducible scenarios:

1. **Hallway Flow**
   - Bidirectional pedestrian traffic through a corridor.

2. **Bottleneck / Doorway**
   - Dense crossing region where robot must choose between waiting and squeezing.

3. **Intersection**
   - Perpendicular streams of pedestrians.

4. **Sparse-to-Dense Transition**
   - Robot starts in open space and enters denser crowd zone.

For each scenario, define:
- map geometry
- pedestrian spawn/goal distributions
- episode timeout
- random seed protocol

---

## 7) Policy/Baseline Roadmap

Near-term baselines:

1. **Manual** (human-control reference)
2. **Naive goal-seeking**
3. **Random**
4. **Potential field** (already implemented)
5. **RVO/ORCA-style baseline** (planned; external integration)

Mid-term learning baselines:

6. **Simple RL baseline** (e.g., PPO with partial observations)
7. **Structured recurrent policy** (DS-RNN-inspired architecture)

---

## 8) Immediate Next To-Dos (Priority Ordered)

### P0 (Do Next)

1. **Fix and align documentation to actual repo**
   - Update `README.md` commands/paths to match current source layout.
   - Add a short "how to run experiments" section once logging exists.

2. **Add reproducible experiment configuration**
   - Centralize parameters (seed, pedestrian count, penalty weights, timeout, mode).
   - Ensure deterministic seeding for `random` (and `numpy` later if added).

3. **Implement structured metrics logging**
   - Save per-episode metrics to CSV/JSON instead of only printing.
   - Include mode, scenario, seed, success, steps, penalties, collisions, intrusions.

4. **Create scenario abstraction**
   - Replace hardcoded spawn/goal logic with named scenario configs.
   - Allow running N episodes per scenario/mode from CLI flags.

5. **Add stronger tests**
   - Keep smoke test, then add unit tests for:
     - penalty calculation tiers
     - goal detection/reset behavior
     - deterministic episode setup with fixed seeds

### P1 (Next After P0)

6. **Define proxemic zones explicitly**
   - Add intimate/personal/social/public thresholds in constants.
   - Track intrusion count and cumulative intrusion time by zone.

7. **Refactor reward/penalty into a dedicated module**
   - Keep environment dynamics and scoring separated for easier ablations.

8. **Build a lightweight analysis notebook/script**
   - Compare policies across scenarios with summary tables and plots.

### P2 (Research Expansion)

9. **Integrate RVO2/ORCA baseline**
10. **Introduce partial observability sensor model**
11. **Add RL training pipeline and compare against hand-crafted baselines**
12. **Run sensitivity studies on social-penalty weights**

---

## 9) Suggested Team Workflow

- Use short feature branches per milestone (`metrics-logging`, `scenario-config`, etc.).
- Require one reproducibility check before merge:
  - same seed, same config, stable aggregate metrics within tolerance.
- Track all experiment runs with:
  - git commit hash
  - scenario id
  - control mode
  - seed
  - metric outputs

---

## 10) Risks and Mitigations

1. **Metric drift / inconsistent definitions**
   - Mitigation: formal metric schema and tests for metric calculations.

2. **Non-reproducible experiments**
   - Mitigation: mandatory seeding and config snapshots in result files.

3. **Overfitting to one crowd pattern**
   - Mitigation: evaluate across multiple scenario families and densities.

4. **Performance claims without statistical support**
   - Mitigation: run multiple seeds and report mean + variance/confidence intervals.

---

## 11) Definition of "Good Progress" (Short-Term)

You are in a strong position if, within the next milestone, you can:

- run at least 3 scenarios x 4 policies x 10 seeds automatically
- produce a results file for each run
- generate one summary comparison table
- answer the key question with initial evidence:
  - how much goal efficiency improves or degrades as social penalties are increased

---

## 12) Open Questions to Finalize as a Team

1. What exact penalty weights define your default objective?
2. What episode timeout defines failure?
3. How many pedestrians and what density levels are in the first benchmark suite?
4. Which comparison is your minimum viable "reportable result" for class milestones?

Answering these will let you lock a v1 benchmark protocol and move from coding to measurable research iterations.

---

## 13) Project Tracker: Individual Contributions (Commit-Graph Based)

This section satisfies the requirement to show individual contributions via commit history.

### Identity Normalization Note

One contributor appears under two git author names with the same email:

- `phisomni-edu <dharma.ar@northeastern.edu>`
- `phisomni <dharma.ar@northeastern.edu>`

For project tracking, these are treated as the same contributor.

### Contribution Summary by Contributor

1. **efsmert** (`samiareski05@gmail.com`) - 3 commits
   - Initial project setup and first commit
   - Brought up pygame simulation with baseline crowd/room/start-end behavior
   - Added repository hygiene updates (`.gitignore`)
   - Representative commits: `050b8ee`, `021e1d5`, `3ea89f1`

2. **Arushi Aggarwal** (`arushi3.14@outlook.com`) - 2 commits
   - Refactored simulation into separate classes/modules
   - Added penalty logic to scoring behavior
   - Representative commits: `ea436aa`, `6f8d416`

3. **Dharma (phisomni / phisomni-edu)** (`dharma.ar@northeastern.edu`) - 5 commits total (incl. merge)
   - Added goal detection and multi-mode control behavior
   - Added autonomous behavior set and episode metrics support
   - Led structural reorganization to current module layout (`src/agent`, `src/environment`, `src/main.py`)
   - Added control-mode documentation in main loop
   - Representative commits: `0f1d7a0`, `4710b6d`, `d590db8`, `a5d13d1`, `d9e2454` (merge)

4. **Lisa W** (`wan.lis@northeastern.edu`) - 1 commit
   - Added social-force pedestrian behavior integration
   - Representative commit: `284cc15`

### Scope Mapping (Who Owned What)

- **Core bootstrapping + initial runnable sim**: efsmert
- **Refactor + penalty model**: Arushi Aggarwal
- **Behavior modes + goal/episode metrics + structural reorg**: Dharma
- **Social-force pedestrian dynamics**: Lisa W

### Tracker Maintenance Rule (Going Forward)

At each milestone, append:

- contributor name
- commit hashes merged in milestone
- short description of net contribution

This keeps individual-contribution evidence synchronized with the evolving codebase and avoids reconstructing it at report time.
