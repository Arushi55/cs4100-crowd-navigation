"""Gymnasium environment for crowd navigation."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

from constants import HEIGHT, WIDTH
from environment.pedestrian import Pedestrian
from environment.robot import Robot
from environment.pathfinding import NavGrid
from environment.scenarios import (
    SCENARIO_CONFIG_DIR,
    Scenario,
    build_scenario,
    generate_pedestrian_population,
    load_scenario_templates,
    respawn_family_group_members,
    random_pedestrian_route,
)

from agent.sensor import RaySensor, draw_rays

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
    Crowd navigation environment with robot/goal state, ray features,
    and short-range pedestrian motion context.

    Base observation:
    [robot_x, robot_y, goal_x, goal_y, ray_distance_0, ray_type_0, ...,
     ped_rel_x, ped_rel_y, ped_rel_vx, ped_rel_vy, ped_dist, ...]
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
        
        self.ped_context_dim = self.max_observed_pedestrians * 5
        obs_dim = 4 + (self.ray_sensor.num_rays * 2) + self.ped_context_dim
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
                
        # reward params
        self.goal_visual_radius = 16.0
        self.goal_reward = 300.0
        self.step_penalty = -0.02
        self.collision_penalty = -18.0
        self.personal_space_radius = 52.0
        self.personal_space_penalty = -2.35
        self.near_miss_radius = 38.0
        self.near_miss_penalty = -1.15
        self.caution_radius = 80.0
        self.caution_penalty = -0.42
        self.wall_penalty_radius = 22.0
        self.wall_penalty_scale = -0.44
        self.wall_scrape_penalty = -1.1
        self.wall_approach_radius = 56.0
        self.wall_approach_scale = -0.6
        self.action_smoothing = 0.35
        self.turn_penalty_scale = -0.18
        self.progress_scale = 3.0          # reward for moving closer to goal
        self.blocking_radius = 84.0
        self.blocking_path_width = 26.0
        self.blocking_penalty_scale = -2.25
        self.slowdown_penalty_radius = 82.0
        self.slowdown_penalty_scale = -1.85
        self.crowd_pressure_radius = 92.0
        self.crowd_pressure_scale = -0.95
        self.crowd_approach_radius = 112.0
        self.crowd_approach_scale = -0.65
        self.timeout_penalty = -45.0       # penalty for running out of steps
        
        # state
        self.scenario: Scenario | None = None
        self.nav_grid: NavGrid | None = None
        self.robot: Robot | None = None
        self.pedestrians: list[Pedestrian] = []
        self.goal_pos: tuple[float, float] | None = None
        self.current_step = 0
        self._prev_dist_to_goal = 0.0      # for progress-based shaping
        self._episode_near_misses = 0
        self._episode_personal_space_intrusions = 0
        self._episode_pedestrian_slowdown = 0.0
        self._episode_blocking_pressure = 0.0
        self._episode_wall_contacts = 0
        self._last_blocked_axes = 0
        self._last_move = np.zeros(2, dtype=np.float32)
        self._last_actual_move = np.zeros(2, dtype=np.float32)
        self._last_turn_penalty = 0.0

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # build scenario and spawn entities
        self.scenario = build_scenario(
            self.template, self.rng, randomize_world=self.random_world
        )
        self.nav_grid = NavGrid(WIDTH, HEIGHT, self.scenario.obstacles)
        self.robot = Robot(
            x=self.scenario.robot_start[0],
            y=self.scenario.robot_start[1],
        )
        self.pedestrians = self._generate_pedestrians()
        self.goal_pos = self.scenario.robot_goal
        self.current_step = 0
        self._prev_dist_to_goal = np.hypot(
            self.robot.x - self.goal_pos[0],
            self.robot.y - self.goal_pos[1],
        )
        self._episode_near_misses = 0
        self._episode_personal_space_intrusions = 0
        self._episode_pedestrian_slowdown = 0.0
        self._episode_blocking_pressure = 0.0
        self._episode_wall_contacts = 0
        self._last_blocked_axes = 0
        self._last_move = np.zeros(2, dtype=np.float32)
        self._last_actual_move = np.zeros(2, dtype=np.float32)
        self._last_turn_penalty = 0.0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: int):
        assert self.robot is not None
        assert self.scenario is not None
        
        self.current_step += 1
        
        dx, dy = ACTION_VECTORS[action]
        if dx != 0 or dy != 0:
            desired_move = pygame.Vector2(dx, dy).normalize() * self.robot.speed
            self._last_move = self._smoothed_action_move(
                np.array([desired_move.x, desired_move.y], dtype=np.float32)
            )
            self._last_turn_penalty = self._turn_penalty(self._last_move)
            old_x, old_y = self.robot.x, self.robot.y
            self._last_blocked_axes = self.robot.move_with_obstacles(
                pygame.Vector2(float(self._last_move[0]), float(self._last_move[1])),
                self.scenario.obstacles,
            )
            self._last_actual_move = np.array(
                [self.robot.x - old_x, self.robot.y - old_y],
                dtype=np.float32,
            )
        else:
            self._last_move = np.zeros(2, dtype=np.float32)
            self._last_actual_move = np.zeros(2, dtype=np.float32)
            self._last_blocked_axes = 0
            self._last_turn_penalty = 0.0
        
        for ped in self.pedestrians:
            ped.update(self.pedestrians, self.scenario.obstacles, rng=self.rng)
        self._reassign_reached_goals()
        
        dist_to_goal = np.hypot(
            self.robot.x - self.goal_pos[0],
            self.robot.y - self.goal_pos[1],
        )
        
        reward = self._compute_reward(dist_to_goal)
        
        terminated = dist_to_goal <= self._goal_contact_radius()
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            reward += self.goal_reward
        elif truncated:
            reward += self.timeout_penalty

        social_metrics = self._social_metrics_snapshot()
        self._episode_near_misses += social_metrics["near_misses"]
        self._episode_personal_space_intrusions += social_metrics["personal_space_intrusions"]
        self._episode_pedestrian_slowdown += social_metrics["pedestrian_slowdown"]
        self._episode_blocking_pressure += social_metrics["blocking_pressure"]
        self._episode_wall_contacts += self._last_blocked_axes
        
        self._prev_dist_to_goal = dist_to_goal
        
        observation = self._get_observation()
        info = {
            "distance_to_goal": dist_to_goal,
            "steps": self.current_step,
            "collisions": self._count_collisions(),
            "is_success": terminated,
            "near_misses": social_metrics["near_misses"],
            "personal_space_intrusions": social_metrics["personal_space_intrusions"],
            "pedestrian_slowdown": social_metrics["pedestrian_slowdown"],
            "blocking_pressure": social_metrics["blocking_pressure"],
            "wall_contacts": self._last_blocked_axes,
            "episode_near_misses": self._episode_near_misses,
            "episode_personal_space_intrusions": self._episode_personal_space_intrusions,
            "episode_pedestrian_slowdown": self._episode_pedestrian_slowdown,
            "episode_blocking_pressure": self._episode_blocking_pressure,
            "episode_wall_contacts": self._episode_wall_contacts,
        }
        
        return observation, reward, terminated, truncated, info

    def _generate_pedestrians(self) -> list[Pedestrian]:
        return generate_pedestrian_population(
            self.scenario,
            self.template,
            self.nav_grid,
            self.rng,
            count=self.num_pedestrians,
        )

    def _reassign_reached_goals(self):
        respawned_groups: set[int] = set()
        for ped in self.pedestrians:
            if ped.has_reached_goal():
                if ped.group_id is not None and ped.group_id not in respawned_groups:
                    group_members = [
                        member for member in self.pedestrians if member.group_id == ped.group_id
                    ]
                    respawn_family_group_members(
                        group_members,
                        self.nav_grid,
                        self.rng,
                        obstacles=self.scenario.obstacles,
                    )
                    respawned_groups.add(ped.group_id)
                    continue
                # Use Pedestrian.respawn to avoid obstacle collisions on respawn
                ped.respawn(self.rng, obstacles=self.scenario.obstacles, nav_grid=self.nav_grid)

    def _get_observation(self) -> np.ndarray:
        ray_obs = self.ray_sensor.cast_rays_flat(
            self.robot.x, self.robot.y,
            self.pedestrians, self.scenario.obstacles
        )
        ped_context = self._pedestrian_context_features()
        
        base_obs = np.array([
            self.robot.x / WIDTH,
            self.robot.y / HEIGHT,
            self.goal_pos[0] / WIDTH,
            self.goal_pos[1] / HEIGHT,
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, ray_obs, ped_context])

    def _pedestrian_context_features(self) -> np.ndarray:
        visible = self.ray_sensor.get_visible_pedestrians(
            self.robot.x,
            self.robot.y,
            self.pedestrians,
            self.scenario.obstacles,
        )
        visible.sort(key=lambda ped: np.hypot(self.robot.x - ped.x, self.robot.y - ped.y))

        max_rel_dist = self.ray_sensor.max_range
        max_rel_speed = 4.0
        default_entry = np.array([0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32)
        features: list[np.ndarray] = []

        for ped in visible[:self.max_observed_pedestrians]:
            rel_x = ped.x - self.robot.x
            rel_y = ped.y - self.robot.y
            rel_vx = ped.vx
            rel_vy = ped.vy
            dist = np.hypot(rel_x, rel_y)
            features.append(np.array([
                self._encode_signed(rel_x, max_rel_dist),
                self._encode_signed(rel_y, max_rel_dist),
                self._encode_signed(rel_vx, max_rel_speed),
                self._encode_signed(rel_vy, max_rel_speed),
                min(1.0, dist / max_rel_dist),
            ], dtype=np.float32))

        while len(features) < self.max_observed_pedestrians:
            features.append(default_entry.copy())

        return np.concatenate(features, dtype=np.float32)

    @staticmethod
    def _encode_signed(value: float, scale: float) -> float:
        if scale <= 1e-6:
            return 0.5
        normalized = np.clip(value / scale, -1.0, 1.0)
        return float((normalized + 1.0) * 0.5)

    def _smoothed_action_move(self, desired_move: np.ndarray) -> np.ndarray:
        desired_norm = np.hypot(desired_move[0], desired_move[1])
        if desired_norm < 1e-6:
            return np.zeros(2, dtype=np.float32)

        desired_dir = desired_move / desired_norm
        prev_norm = np.hypot(self._last_actual_move[0], self._last_actual_move[1])
        if prev_norm < 1e-6:
            return desired_dir.astype(np.float32) * self.robot.speed

        prev_dir = self._last_actual_move / prev_norm
        blended_dir = (
            (1.0 - self.action_smoothing) * desired_dir
            + self.action_smoothing * prev_dir
        )
        blended_norm = np.hypot(blended_dir[0], blended_dir[1])
        if blended_norm < 1e-6:
            blended_dir = desired_dir
        else:
            blended_dir = blended_dir / blended_norm
        return blended_dir.astype(np.float32) * self.robot.speed

    def _turn_penalty(self, move: np.ndarray) -> float:
        move_norm = np.hypot(move[0], move[1])
        prev_norm = np.hypot(self._last_actual_move[0], self._last_actual_move[1])
        if move_norm < 1e-6 or prev_norm < 1e-6:
            return 0.0

        move_dir = move / move_norm
        prev_dir = self._last_actual_move / prev_norm
        alignment = float(np.clip(np.dot(move_dir, prev_dir), -1.0, 1.0))
        if alignment >= 0.85:
            return 0.0

        sharpness = 0.5 * (1.0 - alignment)
        return self.turn_penalty_scale * (sharpness ** 2)

    def _compute_reward(self, dist_to_goal: float) -> float:
        """compute step reward (goal reward added separately in step())."""
        reward = self.step_penalty
        
        # progress-based shaping: reward for getting closer, penalize moving away
        progress = self._prev_dist_to_goal - dist_to_goal
        reward += progress * self.progress_scale
        
        for ped in self.pedestrians:
            dist = np.hypot(self.robot.x - ped.x, self.robot.y - ped.y)
            overlap_dist = self.robot.radius + ped.radius
            
            if dist < overlap_dist:
                reward += self.collision_penalty
            elif dist < self.personal_space_radius:
                closeness = 1.0 - (dist / self.personal_space_radius)
                reward += self.personal_space_penalty * (1.0 + closeness)
                reward += self._near_miss_penalty(dist, overlap_dist)
            elif dist < self.caution_radius:
                caution_scale = 1.0 - (
                    (dist - self.personal_space_radius)
                    / (self.caution_radius - self.personal_space_radius)
                )
                reward += self.caution_penalty * max(0.0, caution_scale)
            blocking_penalty = self._blocking_penalty(ped, dist)
            reward += blocking_penalty
            reward += self._pedestrian_slowdown_penalty(ped, dist, blocking_penalty)

        reward += self._crowd_pressure_penalty()
        reward += self._crowd_approach_penalty()
        reward += self._wall_approach_penalty()
        reward += self._last_turn_penalty

        nearest_wall = self._nearest_wall_distance()
        if nearest_wall < self.wall_penalty_radius:
            wall_closeness = 1.0 - (nearest_wall / self.wall_penalty_radius)
            reward += self.wall_penalty_scale * (wall_closeness ** 2)
        if self._last_blocked_axes:
            reward += self.wall_scrape_penalty * self._last_blocked_axes

        return reward

    def _blocking_penalty(self, ped: Pedestrian, dist: float) -> float:
        if dist >= self.blocking_radius:
            return 0.0

        tx, ty = ped.get_steering_target()
        dir_x = tx - ped.x
        dir_y = ty - ped.y
        dir_norm = np.hypot(dir_x, dir_y)
        if dir_norm < 1e-6:
            return 0.0

        dir_x /= dir_norm
        dir_y /= dir_norm

        rel_x = self.robot.x - ped.x
        rel_y = self.robot.y - ped.y
        forward_dist = rel_x * dir_x + rel_y * dir_y
        if forward_dist <= 0.0 or forward_dist >= self.blocking_radius:
            return 0.0

        lateral_dist = abs(rel_x * dir_y - rel_y * dir_x)
        if lateral_dist >= self.blocking_path_width:
            return 0.0

        forward_scale = 1.0 - (forward_dist / self.blocking_radius)
        lateral_scale = 1.0 - (lateral_dist / self.blocking_path_width)
        return self.blocking_penalty_scale * forward_scale * lateral_scale

    def _goal_contact_radius(self) -> float:
        return self.goal_visual_radius + self.robot.radius

    def _near_miss_penalty(self, dist: float, overlap_dist: float) -> float:
        if dist <= overlap_dist or dist >= self.near_miss_radius:
            return 0.0
        span = max(1e-6, self.near_miss_radius - overlap_dist)
        closeness = 1.0 - ((dist - overlap_dist) / span)
        return self.near_miss_penalty * (closeness ** 2)

    def _pedestrian_slowdown_penalty(
        self,
        ped: Pedestrian,
        dist: float,
        blocking_penalty: float,
    ) -> float:
        if dist >= self.slowdown_penalty_radius:
            return 0.0

        desired_speed = max(0.2, ped.desired_speed)
        current_speed = np.hypot(ped.vx, ped.vy)
        slowdown = max(0.0, 1.0 - (current_speed / desired_speed))
        if slowdown <= 0.0:
            return 0.0

        proximity = 1.0 - (dist / self.slowdown_penalty_radius)
        blocking_strength = 0.0
        if self.blocking_penalty_scale != 0.0:
            blocking_strength = min(
                1.0,
                max(0.0, -blocking_penalty / abs(self.blocking_penalty_scale)),
            )
        pressure = max(proximity * 0.35, blocking_strength) * slowdown
        return self.slowdown_penalty_scale * pressure

    def _social_metrics_snapshot(self) -> dict[str, float]:
        near_misses = 0
        personal_space_intrusions = 0
        pedestrian_slowdown = 0.0
        blocking_pressure = 0.0

        for ped in self.pedestrians:
            dist = np.hypot(self.robot.x - ped.x, self.robot.y - ped.y)
            overlap_dist = self.robot.radius + ped.radius
            if overlap_dist <= dist < self.near_miss_radius:
                near_misses += 1
            if dist < self.personal_space_radius:
                personal_space_intrusions += 1

            blocking_penalty = self._blocking_penalty(ped, dist)
            if blocking_penalty < 0.0 and self.blocking_penalty_scale != 0.0:
                blocking_pressure += min(
                    1.0,
                    -blocking_penalty / abs(self.blocking_penalty_scale),
                )

            desired_speed = max(0.2, ped.desired_speed)
            current_speed = np.hypot(ped.vx, ped.vy)
            slowdown = max(0.0, 1.0 - (current_speed / desired_speed))
            if dist < self.slowdown_penalty_radius:
                proximity = 1.0 - (dist / self.slowdown_penalty_radius)
                pedestrian_slowdown += slowdown * proximity

        return {
            "near_misses": near_misses,
            "personal_space_intrusions": personal_space_intrusions,
            "pedestrian_slowdown": pedestrian_slowdown,
            "blocking_pressure": blocking_pressure,
        }

    def _crowd_pressure_penalty(self) -> float:
        local_pressure = 0.0
        close_neighbors = 0
        for ped in self.pedestrians:
            dist = np.hypot(self.robot.x - ped.x, self.robot.y - ped.y)
            if dist >= self.crowd_pressure_radius:
                continue
            close_neighbors += 1
            local_pressure += 1.0 - (dist / self.crowd_pressure_radius)

        if close_neighbors <= 1:
            return 0.0

        # Penalize entering dense clumps much more than passing one isolated pedestrian.
        return self.crowd_pressure_scale * local_pressure * (1.0 + 0.25 * (close_neighbors - 1))

    def _crowd_approach_penalty(self) -> float:
        move_norm = np.hypot(self._last_move[0], self._last_move[1])
        if move_norm < 1e-6:
            return 0.0

        move_dir = self._last_move / move_norm
        pressure = 0.0
        forward_neighbors = 0

        for ped in self.pedestrians:
            rel_x = ped.x - self.robot.x
            rel_y = ped.y - self.robot.y
            dist = np.hypot(rel_x, rel_y)
            if dist < 1e-6 or dist >= self.crowd_approach_radius:
                continue

            rel_dir = np.array([rel_x / dist, rel_y / dist], dtype=np.float32)
            alignment = float(np.dot(move_dir, rel_dir))
            if alignment <= 0.15:
                continue

            proximity = 1.0 - (dist / self.crowd_approach_radius)
            pressure += proximity * (alignment ** 2)
            forward_neighbors += 1

        if forward_neighbors == 0:
            return 0.0

        return self.crowd_approach_scale * pressure * (1.0 + 0.18 * (forward_neighbors - 1))

    def _wall_approach_penalty(self) -> float:
        move_norm = np.hypot(self._last_move[0], self._last_move[1])
        if move_norm < 1e-6:
            return 0.0

        move_dir = self._last_move / move_norm
        pressure = 0.0
        forward_hits = 0

        for angle in self.ray_sensor.angles:
            ray_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            alignment = float(np.dot(move_dir, ray_dir))
            if alignment <= 0.45:
                continue

            dist, hit_type = self.ray_sensor._cast_single_ray(
                self.robot.x,
                self.robot.y,
                angle,
                self.pedestrians,
                self.scenario.obstacles,
            )
            if hit_type not in (1, 3) or dist >= self.wall_approach_radius:
                continue

            closeness = 1.0 - (dist / self.wall_approach_radius)
            pressure += closeness * (alignment ** 2)
            forward_hits += 1

        if forward_hits == 0:
            return 0.0

        return self.wall_approach_scale * pressure * (1.0 + 0.12 * (forward_hits - 1))

    def _nearest_wall_distance(self) -> float:
        boundary_dist = min(
            self.robot.x - self.robot.radius,
            WIDTH - self.robot.radius - self.robot.x,
            self.robot.y - self.robot.radius,
            HEIGHT - self.robot.radius - self.robot.y,
        )
        obstacle_dist = float("inf")
        for rect in self.scenario.obstacles:
            closest_x = max(rect.left, min(self.robot.x, rect.right))
            closest_y = max(rect.top, min(self.robot.y, rect.bottom))
            obstacle_dist = min(
                obstacle_dist,
                np.hypot(self.robot.x - closest_x, self.robot.y - closest_y) - self.robot.radius,
            )
        return max(0.0, min(boundary_dist, obstacle_dist))

    def _count_collisions(self) -> int:
        """Count how many pedestrians the robot is currently colliding with."""
        count = 0
        for ped in self.pedestrians:
            dist = np.hypot(self.robot.x - ped.x, self.robot.y - ped.y)
            if dist < self.robot.radius + ped.radius:
                count += 1
        return count

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
            (int(self.goal_pos[0]), int(self.goal_pos[1])), int(self.goal_visual_radius)
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
