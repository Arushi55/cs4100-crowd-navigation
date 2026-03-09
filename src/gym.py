"""Gymnasium environment for crowd navigation."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

from constants import HEIGHT, WIDTH
from environment.pedestrian import Pedestrian
from environment.robot import Robot
from environment.scenarios import (
    SCENARIO_CONFIG_DIR,
    Scenario,
    build_scenario,
    load_scenario_templates,
    random_pedestrian_route,
)

from environment.sensor import RaySensor, draw_rays

ACTION_VECTORS = [
    (0, -1),   # 0: up
    (1, -1),   # 1: up-right
    (1, 0),    # 2: right
    (1, 1),    # 3: down-right
    (0, 1),    # 4: down
    (-1, 1),   # 5: down-left
    (-1, 0),   # 6: left
    (-1, -1),  # 7: up-left
    (0, 0),    # 8: stay
]


class CrowdNavEnv(gym.Env):
    """
    crowd nav env
    
    observation: [robot_x, robot_y, goal_x, goal_y, ped1_x, ped1_y, ped1_vx, ped1_vy, ...]
    actions: 0-7 for 8 directions, 8 for stay
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        scenario_id: str = "airport",
        num_pedestrians: int = 12,
        max_observed_pedestrians: int = 6,
        max_steps: int = 1000,
        seed: int | None = None,
        random_world: bool = False,
        render_mode: str | None = None,
        scenario_config_dir: Path = SCENARIO_CONFIG_DIR,
    ):
        super().__init__()
        
        self.scenario_id = scenario_id
        self.num_pedestrians = num_pedestrians
        self.max_observed_pedestrians = max_observed_pedestrians
        self.max_steps = max_steps
        self.random_world = random_world
        self.render_mode = render_mode
        
        self.templates = load_scenario_templates(scenario_config_dir)
        if scenario_id not in self.templates:
            raise ValueError(f"Unknown scenario '{scenario_id}'")
        self.template = self.templates[scenario_id]

        self.ray_sensor = RaySensor(
            num_rays=36,          # 10° resolution
            max_range=150.0,      # how far the robot can "see"
            fov_degrees=360.0,    # full circle
        )
        
        self.rng = np.random.default_rng(seed)
        
        self.action_space = gym.spaces.Discrete(9)
        
        obs_dim = 4 + (self.ray_sensor.num_rays * 2)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
                
        # reward params
        self.goal_radius = 20.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1
        self.collision_penalty = -10.0
        self.personal_space_radius = 48.0
        self.personal_space_penalty = -0.5
        
        # state
        self.scenario: Scenario | None = None
        self.robot: Robot | None = None
        self.pedestrians: list[Pedestrian] = []
        self.goal_pos: tuple[float, float] | None = None
        self.current_step = 0
        
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # build scenario and spawn entities
        self.scenario = build_scenario(
            self.template, self.rng, randomize_world=self.random_world
        )
        self.robot = Robot(
            x=self.scenario.robot_start[0],
            y=self.scenario.robot_start[1],
        )
        self.pedestrians = self._generate_pedestrians()
        self.goal_pos = self.scenario.robot_goal
        self.current_step = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: int):
        assert self.robot is not None
        assert self.scenario is not None
        
        self.current_step += 1
        
        dx, dy = ACTION_VECTORS[action]
        if dx != 0 or dy != 0:
            move = pygame.Vector2(dx, dy).normalize() * self.robot.speed
            self.robot.move_with_obstacles(move, self.scenario.obstacles)
        
        for ped in self.pedestrians:
            ped.update(self.pedestrians, self.scenario.obstacles, rng=self.rng)
        self._reassign_reached_goals()
        
        reward = self._compute_reward()
        
        dist_to_goal = np.hypot(
            self.robot.x - self.goal_pos[0],
            self.robot.y - self.goal_pos[1],
        )
        terminated = dist_to_goal < self.goal_radius
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            reward += self.goal_reward
        
        observation = self._get_observation()
        info = {
            "distance_to_goal": dist_to_goal,
            "steps": self.current_step,
        }
        
        return observation, reward, terminated, truncated, info

    def _generate_pedestrians(self) -> list[Pedestrian]:
        peds = []
        for _ in range(self.num_pedestrians):
            (sx, sy), (gx, gy) = random_pedestrian_route(self.scenario, self.rng)
            peds.append(Pedestrian(x=sx, y=sy, vx=0.0, vy=0.0, goal_x=gx, goal_y=gy))
        return peds

    def _reassign_reached_goals(self):
        for ped in self.pedestrians:
            if ped.has_reached_goal():
                (sx, sy), (gx, gy) = random_pedestrian_route(self.scenario, self.rng)
                ped.x, ped.y = sx, sy
                ped.vx, ped.vy = 0.0, 0.0
                ped.goal_x, ped.goal_y = gx, gy

    def _get_observation(self) -> np.ndarray:
        ray_obs = self.ray_sensor.cast_rays_flat(
            self.robot.x, self.robot.y,
            self.pedestrians, self.scenario.obstacles
        )
        
        base_obs = np.array([
            self.robot.x / WIDTH,
            self.robot.y / HEIGHT,
            self.goal_pos[0] / WIDTH,
            self.goal_pos[1] / HEIGHT,
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, ray_obs])

    def _compute_reward(self) -> float:
        """compute step reward (goal reward added separately in step())."""
        reward = self.step_penalty
        
        for ped in self.pedestrians:
            dist = np.hypot(self.robot.x - ped.x, self.robot.y - ped.y)
            overlap_dist = self.robot.radius + ped.radius
            
            if dist < overlap_dist:
                reward += self.collision_penalty
            elif dist < self.personal_space_radius:
                reward += self.personal_space_penalty
        
        return reward

    def render(self):
        if self.render_mode is None:
            return None
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Crowd Navigation")
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            else:
                self.screen = pygame.Surface((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        
        # colors
        bg_color = (245, 247, 240)
        obstacle_color = (165, 170, 185)
        goal_color = (245, 130, 40)
        
        self.screen.fill(bg_color)
        
        # draw obstacles
        for obstacle in self.scenario.obstacles:
            pygame.draw.rect(self.screen, obstacle_color, obstacle, border_radius=4)
        
        # draw goal
        pygame.draw.circle(
            self.screen, goal_color, 
            (int(self.goal_pos[0]), int(self.goal_pos[1])), 16
        )
        
        # draw pedestrians and robot
        for ped in self.pedestrians:
            ped.draw(self.screen)
        self.robot.draw(self.screen)

        endpoints = self.ray_sensor.get_ray_endpoints(
            self.robot.x, self.robot.y,
            self.pedestrians, self.scenario.obstacles
        )
        draw_rays(self.screen, endpoints)
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        ) if self.render_mode == "rgb_array" else None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


# Quick test
if __name__ == "__main__":
    env = CrowdNavEnv(render_mode="human", num_pedestrians=8)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    for _ in range(500):
        action = env.action_space.sample()  # Random policy
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended: reward={reward:.1f}, steps={info['steps']}")
            obs, info = env.reset()
    
    env.close()