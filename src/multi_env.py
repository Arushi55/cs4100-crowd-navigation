"""Wrapper environment that randomizes scenario, pedestrian count, and speed each episode."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from crowd_env import CrowdNavEnv


class VariablePedestrianEnv(gym.Env):
    """
    Wraps CrowdNavEnv for a single fixed scenario while randomizing pedestrian
    count and speed on every reset.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        scenario_id: str = "airport",
        ped_count_range: tuple[int, int] = (6, 20),
        speed_range: tuple[float, float] = (1.0, 1.0),
        max_steps: int = 1200,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.scenario_id = scenario_id
        self.ped_count_range = ped_count_range
        self.speed_range = speed_range
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self._env = CrowdNavEnv(
            scenario_id=self.scenario_id,
            num_pedestrians=ped_count_range[0],
            max_steps=max_steps,
            seed=seed,
            render_mode=render_mode,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n_peds = int(
            self.rng.integers(self.ped_count_range[0], self.ped_count_range[1] + 1)
        )
        speed_mult = float(self.rng.uniform(self.speed_range[0], self.speed_range[1]))

        if self._env is not None:
            self._env.close()

        self._env = CrowdNavEnv(
            scenario_id=self.scenario_id,
            num_pedestrians=n_peds,
            max_steps=self.max_steps,
            render_mode=self.render_mode,
        )
        self._env.rng = self.rng

        obs, info = self._env.reset()

        for ped in self._env.pedestrians:
            ped.desired_speed *= speed_mult
            ped.max_speed *= speed_mult

        info["scenario"] = self.scenario_id
        info["n_peds"] = n_peds
        info["speed_mult"] = speed_mult
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()

    @property
    def pedestrians(self):
        return self._env.pedestrians

    @property
    def robot(self):
        return self._env.robot

    @property
    def scenario(self):
        return self._env.scenario


class MultiScenarioEnv(gym.Env):
    """
    Wraps CrowdNavEnv and randomizes scenario, pedestrian count,
    and pedestrian speed on every reset.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        scenarios: list[str] | None = None,
        ped_count_range: tuple[int, int] = (6, 20),
        speed_range: tuple[float, float] = (1.0, 2.5),
        max_steps: int = 1200,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.scenarios = scenarios or ["airport", "home", "shopping_center"]
        self.ped_count_range = ped_count_range
        self.speed_range = speed_range
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # Create a reference env to get spaces
        self._env = CrowdNavEnv(
            scenario_id=self.scenarios[0],
            num_pedestrians=ped_count_range[0],
            max_steps=max_steps,
            seed=seed,
            render_mode=render_mode,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Randomize configuration
        scenario = self.rng.choice(self.scenarios)
        n_peds = int(self.rng.integers(self.ped_count_range[0], self.ped_count_range[1] + 1))
        speed_mult = float(self.rng.uniform(self.speed_range[0], self.speed_range[1]))

        # Close old env and create new one with randomized config
        if self._env is not None:
            self._env.close()

        self._env = CrowdNavEnv(
            scenario_id=scenario,
            num_pedestrians=n_peds,
            max_steps=self.max_steps,
            render_mode=self.render_mode,
        )
        self._env.rng = self.rng

        obs, info = self._env.reset()

        # Apply speed multiplier
        for ped in self._env.pedestrians:
            ped.desired_speed *= speed_mult
            ped.max_speed *= speed_mult

        info["scenario"] = scenario
        info["n_peds"] = n_peds
        info["speed_mult"] = speed_mult
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()

    @property
    def pedestrians(self):
        return self._env.pedestrians

    @property
    def robot(self):
        return self._env.robot

    @property
    def scenario(self):
        return self._env.scenario
