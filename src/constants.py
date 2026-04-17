"""Shared simulation constants"""

import math

WIDTH = 960
HEIGHT = 640

# Simulation timing and world scaling.
# At default calibration, 1 px/step numerically corresponds to 1 m/s.
SIM_FPS = 60
SIM_SECONDS_PER_STEP = 1.0 / SIM_FPS
WORLD_METERS_PER_PIXEL = 1.0 / SIM_FPS
PIXELS_PER_METER = 1.0 / WORLD_METERS_PER_PIXEL

REFERENCE_ENV_DIAGONAL = math.hypot(WIDTH, HEIGHT)


def speed_mps_to_px_per_step(speed_mps):
    """Convert m/s to simulation px/step."""
    return speed_mps * PIXELS_PER_METER * SIM_SECONDS_PER_STEP


def speed_px_per_step_to_mps(speed_px_per_step):
    """Convert simulation px/step to m/s."""
    return speed_px_per_step * WORLD_METERS_PER_PIXEL / SIM_SECONDS_PER_STEP
