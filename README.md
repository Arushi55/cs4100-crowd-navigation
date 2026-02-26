# cs4100-crowd-navigation

Pygame-based sandbox for crowd-aware robot navigation experiments.

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

## Run the simulation

From the repository root:

```bash
python src/main.py
```

### Scenario and seed flags

You can now control deterministic vs random worlds using CLI flags (or env vars):

```bash
python src/main.py --scenario airport --seed 42
python src/main.py --scenario home --random-seed --random-world
```

Useful flags:

- `--scenario {home|airport|shopping_center}`
- `--seed <int>` deterministic numpy seed (default is fixed)
- `--random-seed` use entropy-based random seed
- `--random-world` randomize obstacle layout each episode
- `--pedestrians <int>` set pedestrian count
- `--scenario-config-dir <path>` custom JSON scenario directory

Equivalent env vars:

- `CROWD_SIM_SCENARIO`
- `CROWD_SIM_PEDESTRIANS`
- `CROWD_SIM_RANDOM_SEED` (`1` to enable)
- `CROWD_SIM_RANDOM_WORLD` (`1` to enable)
- `CROWD_SIM_SCENARIO_DIR`

## Control modes

Set the mode in `src/main.py`:

```python
MODE = ControlMode.POTENTIAL_FIELD
```

Available modes in `src/agent/behaviors.py`:

- `ControlMode.MANUAL`
- `ControlMode.NAIVE`
- `ControlMode.RANDOM`
- `ControlMode.POTENTIAL_FIELD`

## Controls

- `WASD` or arrow keys to move the robot (manual mode)
- `1` switch to Home scenario
- `2` switch to Airport scenario
- `3` switch to Shopping Center scenario
- Close the pygame window to exit

## Current metrics in simulation

Per episode, the simulation tracks:

- total penalty
- total steps
- running average penalty
- running average steps

Penalties currently include:

- near-proximity penalty
- tiered overlap/collision penalties

## Run tests

```bash
pytest -q
```

## Project structure

```text
src/main.py                    # Simulation entrypoint and episode loop
src/constants.py               # Shared dimensions/constants
src/agent/behaviors.py         # Robot control policies
src/environment/robot.py       # Robot model and movement constraints
src/environment/pedestrian.py  # Pedestrian dynamics (social-force style)
src/environment/scenarios.py   # Scenario loader + random world generation
src/environment/scenario_configs/*.json  # Map configs
tests/test_smoke.py            # Basic smoke test
PROJECT.md                     # Extended project reference and roadmap
```
