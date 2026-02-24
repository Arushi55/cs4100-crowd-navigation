# cs4100-crowd-navigation

Pygame-based sandbox for crowd-aware robot navigation experiments.

## Quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run the simulation:

```bash
crowd-sim
```

You can also run it directly:

```bash
python -m crowd_navigation.main
```

Controls:
- `WASD` or arrow keys to move the robot
- Close the window to exit

## Optional extras for this proposal

Install all extras (analysis + RL + crowd + dev):

```bash
pip install -r requirements-dev.txt
```

Or install only what you need:

```bash
pip install -e ".[analysis]"
pip install -e ".[rl]"
pip install -e ".[crowd]"
pip install -e ".[dev]"
```

What each extra is for:
- `analysis`: metrics and plots (time-to-goal, collisions, personal-space intrusions)
- `rl`: reinforcement learning baselines in partially observable settings
- `crowd`: optional velocity-obstacle style crowd-flow tooling (`pyRVO`)
- `dev`: testing and linting

## Project structure

```text
src/crowd_navigation/main.py   # Pygame simulation entrypoint
tests/test_smoke.py            # Basic test scaffold
pyproject.toml                 # Package + dependencies
```
