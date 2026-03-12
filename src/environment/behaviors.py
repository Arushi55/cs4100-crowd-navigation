"""Pedestrian behavior strategies for different scenarios."""

from abc import ABC, abstractmethod
import math
import pygame
import numpy as np

from constants import HEIGHT, WIDTH


class PedestrianBehavior(ABC):
    """Base class for pedestrian movement behaviors."""

    @abstractmethod
    def update(
        self,
        pedestrian: "Pedestrian",
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Update pedestrian velocity and position based on behavior."""
        pass


class SocialForceBehavior(PedestrianBehavior):
    """Classic social force model (current default behavior)."""

    def update(
        self,
        pedestrian: "Pedestrian",
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        from environment.pedestrian import Pedestrian

        f1x, f1y = pedestrian._self_driving_force()
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        f3x, f3y = pedestrian._wall_repulsion()

        pedestrian.vx += f1x + f2x + f3x
        pedestrian.vy += f1y + f2y + f3y

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        if speed > pedestrian.max_speed:
            pedestrian.vx = pedestrian.vx / speed * pedestrian.max_speed
            pedestrian.vy = pedestrian.vy / speed * pedestrian.max_speed

        _apply_movement(pedestrian, obstacles, rng)


class StationaryBehavior(PedestrianBehavior):
    """Mostly stationary with infrequent goal updates."""

    def __init__(self, movement_probability: float = 0.01):
        self.movement_probability = movement_probability
        self.stationary_frame_count = 0

    def update(
        self,
        pedestrian: "Pedestrian",
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        # Slowly decay velocity toward zero
        pedestrian.vx *= 0.95
        pedestrian.vy *= 0.95

        # Occasionally move toward goal with weak force
        if rng and rng.uniform() < self.movement_probability:
            dx = pedestrian.goal_x - pedestrian.x
            dy = pedestrian.goal_y - pedestrian.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                ex, ey = dx / dist, dy / dist
                pedestrian.vx = ex * 0.3
                pedestrian.vy = ey * 0.3

        # Weak repulsion from walls
        f3x, f3y = pedestrian._wall_repulsion()
        pedestrian.vx += f3x * 0.3
        pedestrian.vy += f3y * 0.3

        _apply_movement(pedestrian, obstacles, rng)


class RandomWalkerBehavior(PedestrianBehavior):
    """Move toward random goals across entire screen at increased speed."""

    def __init__(self, speed_multiplier: float = 2.0):
        self.speed_multiplier = speed_multiplier

    def update(
        self,
        pedestrian: "Pedestrian",
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        # Move toward current goal with increased speed
        dx = pedestrian.goal_x - pedestrian.x
        dy = pedestrian.goal_y - pedestrian.y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ex, ey = dx / dist, dy / dist
            desired_speed = pedestrian.desired_speed * self.speed_multiplier
            fx = (desired_speed * ex - pedestrian.vx) / pedestrian.relaxation_time
            fy = (desired_speed * ey - pedestrian.vy) / pedestrian.relaxation_time
        else:
            fx, fy = 0.0, 0.0

        # Weak pedestrian avoidance to avoid getting stuck
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        pedestrian.vx += fx + f2x * 0.3
        pedestrian.vy += fy + f2y * 0.3

        # Weak wall repulsion for boundary awareness
        f3x, f3y = pedestrian._wall_repulsion()
        pedestrian.vx += f3x * 0.3
        pedestrian.vy += f3y * 0.3

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        if speed > pedestrian.max_speed:
            pedestrian.vx = pedestrian.vx / speed * pedestrian.max_speed
            pedestrian.vy = pedestrian.vy / speed * pedestrian.max_speed

        _apply_movement(pedestrian, obstacles, rng)


class ClumpBehavior(PedestrianBehavior):
    """Move together in groups toward goal."""

    def __init__(self, clump_radius: float = 120.0):
        self.clump_radius = clump_radius

    def update(
        self,
        pedestrian: "Pedestrian",
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        # Self-driving force toward goal (reduced)
        dx = pedestrian.goal_x - pedestrian.x
        dy = pedestrian.goal_y - pedestrian.y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ex, ey = dx / dist, dy / dist
            fx = (pedestrian.desired_speed * ex * 0.5 - pedestrian.vx) / pedestrian.relaxation_time
            fy = (pedestrian.desired_speed * ey * 0.5 - pedestrian.vy) / pedestrian.relaxation_time
        else:
            fx, fy = 0.0, 0.0

        # Cohesion: attract toward center of mass of nearby pedestrians
        f_cohesion_x, f_cohesion_y = 0.0, 0.0
        nearby_count = 0
        center_x, center_y = 0.0, 0.0
        
        for other in others:
            if other is pedestrian:
                continue
            dx = other.x - pedestrian.x
            dy = other.y - pedestrian.y
            dist = math.hypot(dx, dy)
            if dist < self.clump_radius and dist > 1e-6:
                nearby_count += 1
                center_x += other.x
                center_y += other.y
        
        if nearby_count > 0:
            center_x /= nearby_count
            center_y /= nearby_count
            dx = center_x - pedestrian.x
            dy = center_y - pedestrian.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                f_cohesion_x = (dx / dist) * 0.5
                f_cohesion_y = (dy / dist) * 0.5

        # Separation: weak repulsion to avoid overlap
        f_sep_x, f_sep_y = 0.0, 0.0
        for other in others:
            if other is pedestrian:
                continue
            dx = pedestrian.x - other.x
            dy = pedestrian.y - other.y
            dist = math.hypot(dx, dy)
            if dist < self.clump_radius and dist > 1e-6:
                nx, ny = dx / dist, dy / dist
                magnitude = pedestrian.ped_A * 0.2 * math.exp((pedestrian.radius + other.radius - dist) / pedestrian.ped_B)
                f_sep_x += magnitude * nx
                f_sep_y += magnitude * ny

        f3x, f3y = pedestrian._wall_repulsion()

        pedestrian.vx += fx + f_cohesion_x + f_sep_x + f3x * 0.5
        pedestrian.vy += fy + f_cohesion_y + f_sep_y + f3y * 0.5

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        if speed > pedestrian.max_speed:
            pedestrian.vx = pedestrian.vx / speed * pedestrian.max_speed
            pedestrian.vy = pedestrian.vy / speed * pedestrian.max_speed

        _apply_movement(pedestrian, obstacles, rng)


class ZigzagBehavior(PedestrianBehavior):
    """Zigzag across space frequently, changing direction."""

    def __init__(self, direction_change_frames: int = 30):
        self.direction_change_frames = direction_change_frames
        self.frame_count = 0

    def update(
        self,
        pedestrian: "Pedestrian",
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.frame_count += 1

        # Change direction periodically
        if self.frame_count % self.direction_change_frames == 0:
            dx = pedestrian.goal_x - pedestrian.x
            dy = pedestrian.goal_y - pedestrian.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                # Rotate direction slightly for zigzag
                angle = math.atan2(dy, dx)
                if rng:
                    angle += float(rng.uniform(-0.6, 0.6))
                pedestrian.vx = math.cos(angle) * pedestrian.desired_speed
                pedestrian.vy = math.sin(angle) * pedestrian.desired_speed

        # Weak pedestrian avoidance
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        pedestrian.vx += f2x * 0.3
        pedestrian.vy += f2y * 0.3

        # Wall repulsion
        f3x, f3y = pedestrian._wall_repulsion()
        pedestrian.vx += f3x
        pedestrian.vy += f3y

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        if speed > pedestrian.max_speed:
            pedestrian.vx = pedestrian.vx / speed * pedestrian.max_speed
            pedestrian.vy = pedestrian.vy / speed * pedestrian.max_speed

        _apply_movement(pedestrian, obstacles, rng)


def _apply_movement(
    pedestrian: "Pedestrian",
    obstacles: list[pygame.Rect] | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Apply velocity to position with collision handling."""
    old_x, old_y = pedestrian.x, pedestrian.y
    proposed_x = pedestrian.x + pedestrian.vx
    proposed_y = pedestrian.y + pedestrian.vy

    if obstacles and pedestrian._would_collide(proposed_x, proposed_y, obstacles):
        if not pedestrian._would_collide(proposed_x, old_y, obstacles):
            pedestrian.x = proposed_x
        elif not pedestrian._would_collide(old_x, proposed_y, obstacles):
            pedestrian.y = proposed_y
        else:
            pedestrian.vx *= -0.4
            pedestrian.vy *= -0.4
            if rng is not None:
                pedestrian.vx += float(rng.uniform(-0.2, 0.2))
                pedestrian.vy += float(rng.uniform(-0.2, 0.2))
            pedestrian.x = old_x + pedestrian.vx
            pedestrian.y = old_y + pedestrian.vy
    else:
        pedestrian.x = proposed_x
        pedestrian.y = proposed_y

    pedestrian.x = max(pedestrian.radius, min(WIDTH - pedestrian.radius, pedestrian.x))
    pedestrian.y = max(pedestrian.radius, min(HEIGHT - pedestrian.radius, pedestrian.y))
