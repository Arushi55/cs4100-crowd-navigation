from dataclasses import dataclass, field
import math
import pygame
import numpy as np
from typing import TYPE_CHECKING

from constants import HEIGHT, WIDTH, SIM_SECONDS_PER_STEP, WORLD_METERS_PER_PIXEL

if TYPE_CHECKING:
    from environment.behaviors import PedestrianBehavior
    from environment.pathfinding import NavGrid

PEDESTRIAN_COLOR = (10, 155, 110)
GOAL_COLOR = (255, 200, 0)

@dataclass
class Pedestrian:
    x: float
    y: float
    vx: float
    vy: float
    goal_x: float
    goal_y: float
    radius: int = 10
    behavior: "PedestrianBehavior | None" = field(default=None)
    goal_region_indices: list[int] | None = field(default=None)
    group_id: int | None = field(default=None)
    spawn_edge: str | None = field(default=None)
    goal_edge: str | None = field(default=None)
    group_spawn_spread: float = field(default=28.0)
    group_goal_spread: float = field(default=48.0)

    # Pathfinding waypoints (computed by set_goal)
    _waypoints: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _waypoint_idx: int = field(default=0, repr=False)

    desired_speed: float = 1.5
    relaxation_time: float = 18.0
    max_speed: float = 3.0
    max_acceleration_mps2: float = 3.0
    max_turn_rate_deg_per_sec: float = 240.0

    ped_A: float = 8.0
    ped_B: float = 8.0

    wall_A: float = 5.0
    wall_B: float = 8.0

    obstacle_A: float = 10.0
    obstacle_B: float = 10.0
    obstacle_range: float = 40.0

    def desired_speed_step(self) -> float:
        """Desired cruising speed in px/step."""
        return self.desired_speed * SIM_SECONDS_PER_STEP / WORLD_METERS_PER_PIXEL

    def max_speed_step(self) -> float:
        """Maximum speed in px/step."""
        return self.max_speed * SIM_SECONDS_PER_STEP / WORLD_METERS_PER_PIXEL

    def max_delta_v_step(self) -> float:
        """Maximum change in velocity (px/step) allowed per simulation step."""
        return (
            self.max_acceleration_mps2
            * SIM_SECONDS_PER_STEP
            * SIM_SECONDS_PER_STEP
            / WORLD_METERS_PER_PIXEL
        )

    def max_turn_step_radians(self) -> float:
        """Maximum heading change in radians per simulation step."""
        return math.radians(self.max_turn_rate_deg_per_sec) * SIM_SECONDS_PER_STEP

    def set_goal(
        self,
        gx: float,
        gy: float,
        nav_grid: "NavGrid | None" = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Set the goal and compute a path using A* if a NavGrid is provided."""
        self.goal_x = gx
        self.goal_y = gy
        if nav_grid is not None:
            raw_path = nav_grid.find_path((self.x, self.y), (gx, gy))
            # Add tiny random offsets so pedestrians don't follow identical lines
            if rng is not None and len(raw_path) > 1:
                jittered: list[tuple[float, float]] = []
                for i, (wx, wy) in enumerate(raw_path):
                    if i < len(raw_path) - 1:  # don't jitter the final goal
                        wx += float(rng.uniform(-4, 4))
                        wy += float(rng.uniform(-4, 4))
                    jittered.append((wx, wy))
                self._waypoints = jittered
            else:
                self._waypoints = list(raw_path)
        else:
            self._waypoints = [(gx, gy)]
        self._waypoint_idx = 0

    def get_steering_target(self) -> tuple[float, float]:
        """Return the current waypoint to steer toward."""
        if not self._waypoints:
            return self.goal_x, self.goal_y

        # Advance past reached waypoints
        while self._waypoint_idx < len(self._waypoints) - 1:
            wx, wy = self._waypoints[self._waypoint_idx]
            if math.hypot(wx - self.x, wy - self.y) < 18.0:
                self._waypoint_idx += 1
            else:
                break

        return self._waypoints[self._waypoint_idx]

    def _self_driving_force(self) -> tuple[float, float]:
        """Social-force drive toward current waypoint (not final goal)."""
        tx, ty = self.get_steering_target()
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0
        ex, ey = dx / dist, dy / dist
        desired_speed = self.desired_speed_step()
        fx = (desired_speed * ex - self.vx) / self.relaxation_time
        fy = (desired_speed * ey - self.vy) / self.relaxation_time
        return fx, fy

    def _pedestrian_repulsion(self, others: list["Pedestrian"]) -> tuple[float, float]:
        fx, fy = 0.0, 0.0
        for other in others:
            if other is self:
                continue
            if other.y > HEIGHT + other.radius:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                dist = 1e-6
            nx, ny = dx / dist, dy / dist
            r_ij = self.radius + other.radius
            magnitude = self.ped_A * math.exp((r_ij - dist) / self.ped_B)
            fx += magnitude * nx
            fy += magnitude * ny
        return fx, fy

    def _wall_repulsion(self) -> tuple[float, float]:
        fx, fy = 0.0, 0.0
        walls = [
            (self.x,            1.0,  0.0),   # left
            (WIDTH - self.x,   -1.0,  0.0),   # right
            (self.y,            0.0,  1.0),   # top
            (HEIGHT - self.y,   0.0, -1.0),   # bottom
        ]
        for dist, nx, ny in walls:
            if dist < 1e-6:
                dist = 1e-6
            magnitude = self.wall_A * math.exp((self.radius - dist) / self.wall_B)
            fx += magnitude * nx
            fy += magnitude * ny
        return fx, fy

    def _obstacle_repulsion(
        self, obstacles: list[pygame.Rect] | None
    ) -> tuple[float, float]:
        """Repulsive force from nearby obstacles (walls, furniture, etc.)."""
        fx, fy = 0.0, 0.0
        if not obstacles:
            return fx, fy
        for rect in obstacles:
            # Closest point on the rect to the pedestrian centre
            closest_x = max(rect.left, min(self.x, rect.right))
            closest_y = max(rect.top, min(self.y, rect.bottom))
            dx = self.x - closest_x
            dy = self.y - closest_y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                # Inside the obstacle — push away from centre
                cx = rect.centerx
                cy = rect.centery
                dx = self.x - cx
                dy = self.y - cy
                dist = math.hypot(dx, dy)
                if dist < 1e-6:
                    dx, dy, dist = 1.0, 0.0, 1.0
            if dist < self.obstacle_range:
                nx, ny = dx / dist, dy / dist
                magnitude = self.obstacle_A * math.exp(
                    (self.radius - dist) / self.obstacle_B
                )
                fx += magnitude * nx
                fy += magnitude * ny
        return fx, fy
    
    def respawn(self, rng: np.random.Generator) -> None:
        self.x = float(rng.uniform(self.radius, WIDTH - self.radius))
        self.y = HEIGHT + self.radius
        self.vx = 0.0
        self.vy = 0.0
        self.goal_x = float(rng.uniform(self.radius, WIDTH - self.radius))

    def update(
        self,
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if self.behavior is not None:
            self.behavior.update(self, others, obstacles, rng)
        else:
            # Fallback to social force if no behavior assigned
            from environment.behaviors import SocialForceBehavior
            default_behavior = SocialForceBehavior()
            default_behavior.update(self, others, obstacles, rng)

    def has_reached_goal(self, threshold: float = 15.0) -> bool:
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y) < threshold

    def _would_collide(self, x: float, y: float, obstacles: list[pygame.Rect]) -> bool:
        hitbox = pygame.Rect(
            int(x - self.radius),
            int(y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )
        return any(hitbox.colliderect(obstacle) for obstacle in obstacles)

    def draw(self, surface: pygame.Surface) -> None:
        # Draw waypoint path (faint)
        if self._waypoints and len(self._waypoints) > 1:
            pts = [(int(self.x), int(self.y))]
            for wp in self._waypoints[self._waypoint_idx:]:
                pts.append((int(wp[0]), int(wp[1])))
            if len(pts) >= 2:
                pygame.draw.lines(surface, (180, 220, 180), False, pts, 1)

        pygame.draw.circle(surface, GOAL_COLOR, (int(self.goal_x), int(self.goal_y)), 4)
        pygame.draw.line(
            surface, (255, 255, 255),
            (int(self.x), int(self.y)),
            (int(self.x + self.vx * 8), int(self.y + self.vy * 8)), 2
        )
        pygame.draw.circle(surface, PEDESTRIAN_COLOR, (int(self.x), int(self.y)), self.radius)
