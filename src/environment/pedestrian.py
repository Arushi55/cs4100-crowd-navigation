from __future__ import annotations

from dataclasses import dataclass, field
import math
import pygame
import numpy as np
from typing import TYPE_CHECKING

from constants import HEIGHT, WIDTH

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
    relaxation_time: float = 30.0
    max_speed: float = 3.0

    ped_A: float = 8.0
    ped_B: float = 8.0

    wall_A = 2.0
    wall_B = 14.0

    obstacle_A = 3.0
    obstacle_B = 14.0
    obstacle_range = 24.0

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
            if rng is not None and len(raw_path) > 2:
                jittered: list[tuple[float, float]] = []
                for i, (wx, wy) in enumerate(raw_path):
                    # Never jitter start or final goal
                    if 0 < i < len(raw_path) - 1:
                        wx += float(rng.uniform(-1.5, 1.5))
                        wy += float(rng.uniform(-1.5, 1.5))
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

        while self._waypoint_idx < len(self._waypoints) - 1:
            wx, wy = self._waypoints[self._waypoint_idx]
            dist_curr = math.hypot(wx - self.x, wy - self.y)

            nx, ny = self._waypoints[self._waypoint_idx + 1]
            dist_next = math.hypot(nx - self.x, ny - self.y)

            # Advance if close enough OR current waypoint has become worse than next
            if dist_curr < 20.0 or dist_next + 4.0 < dist_curr:
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
        fx = (self.desired_speed * ex - self.vx) / self.relaxation_time
        fy = (self.desired_speed * ey - self.vy) / self.relaxation_time
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
    
    def _obstacle_slide_force(self, obstacles: list[pygame.Rect] | None) -> tuple[float, float]:
        if not obstacles:
            return 0.0, 0.0

        tx, ty = self.get_steering_target()
        desired_dx = tx - self.x
        desired_dy = ty - self.y
        desired_dist = math.hypot(desired_dx, desired_dy)
        if desired_dist < 1e-6:
            return 0.0, 0.0

        desired_dx /= desired_dist
        desired_dy /= desired_dist

        best_force = (0.0, 0.0)
        best_dist = float("inf")

        for rect in obstacles:
            closest_x = max(rect.left, min(self.x, rect.right))
            closest_y = max(rect.top, min(self.y, rect.bottom))
            dx = self.x - closest_x
            dy = self.y - closest_y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue

            if dist < self.radius + 10.0 and dist < best_dist:
                best_dist = dist
                nx, ny = dx / dist, dy / dist

                # Tangent direction
                tx1, ty1 = -ny, nx
                tx2, ty2 = ny, -nx

                # Pick tangent that best matches goal direction
                dot1 = tx1 * desired_dx + ty1 * desired_dy
                dot2 = tx2 * desired_dx + ty2 * desired_dy
                sx, sy = (tx1, ty1) if dot1 > dot2 else (tx2, ty2)

                strength = max(0.0, 1.0 - dist / (self.radius + 10.0))
                best_force = (sx * 0.25 * strength, sy * 0.25 * strength)

        return best_force
    
    def _find_free_position(
    self,
    rng: np.random.Generator,
    obstacles: list[pygame.Rect] | None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    max_tries: int = 200,
) -> tuple[float, float]:
        x_min = self.radius if x_min is None else x_min
        x_max = WIDTH - self.radius if x_max is None else x_max
        y_min = self.radius if y_min is None else y_min
        y_max = HEIGHT - self.radius if y_max is None else y_max

        for _ in range(max_tries):
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))

            if obstacles is None or not self._would_collide(x, y, obstacles):
                return x, y

        # Fallback: scan a coarse grid if random retries fail
        step = max(2 * self.radius, 8)
        for yy in range(int(y_min), int(y_max) + 1, step):
            for xx in range(int(x_min), int(x_max) + 1, step):
                if obstacles is None or not self._would_collide(float(xx), float(yy), obstacles):
                    return float(xx), float(yy)

        # Absolute fallback
        return float(self.radius), float(self.radius)

    def respawn(
        self,
        rng: np.random.Generator,
        obstacles: list[pygame.Rect] | None = None,
        nav_grid: "NavGrid | None" = None,
    ) -> None:
        self.x, self.y = self._find_free_position(
            rng,
            obstacles,
            x_min=self.radius,
            x_max=WIDTH - self.radius,
            y_min=self.radius,
            y_max=HEIGHT - self.radius,
        )
        self.vx = 0.0
        self.vy = 0.0

        gx, gy = self._find_free_position(
            rng,
            obstacles,
            x_min=self.radius,
            x_max=WIDTH - self.radius,
            y_min=self.radius,
            y_max=max(self.radius, HEIGHT * 0.25),
        )
        self.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)

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
