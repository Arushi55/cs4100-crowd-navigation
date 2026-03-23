"""Pedestrian behavior strategies for different scenarios."""

from __future__ import annotations

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
        f1x, f1y = pedestrian._self_driving_force()
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        f3x, f3y = pedestrian._wall_repulsion()
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        f5x, f5y = pedestrian._obstacle_slide_force(obstacles)

        pedestrian.vx += f1x + f2x + f3x + f4x + f5x
        pedestrian.vy += f1y + f2y + f3y + f4y + f5y

        # Damping kills oscillation near walls/corners
        pedestrian.vx *= 0.90
        pedestrian.vy *= 0.90

        # Tiny deadzone removes visible micro-jitter
        if abs(pedestrian.vx) < 0.02:
            pedestrian.vx = 0.0
        if abs(pedestrian.vy) < 0.02:
            pedestrian.vy = 0.0

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

        # Occasionally move toward current waypoint with weak force
        if rng and rng.uniform() < self.movement_probability:
            tx, ty = pedestrian.get_steering_target()
            dx = tx - pedestrian.x
            dy = ty - pedestrian.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                ex, ey = dx / dist, dy / dist
                pedestrian.vx = ex * 0.3
                pedestrian.vy = ey * 0.3

        # Weak repulsion from walls and obstacles
        f3x, f3y = pedestrian._wall_repulsion()
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        f5x, f5y = pedestrian._obstacle_slide_force(obstacles)
        pedestrian.vx += f3x * 0.3 + f4x * 0.5 + f5x * 0.3
        pedestrian.vy += f3y * 0.3 + f4y * 0.5 + f5y * 0.3

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
        # Move toward current waypoint with increased speed
        tx, ty = pedestrian.get_steering_target()
        dx = tx - pedestrian.x
        dy = ty - pedestrian.y
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

        # Wall + obstacle repulsion
        f3x, f3y = pedestrian._wall_repulsion()
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        f5x, f5y = pedestrian._obstacle_slide_force(obstacles)
        pedestrian.vx += f3x * 0.3 + f4x + f5x * 0.4
        pedestrian.vy += f3y * 0.3 + f4y + f5y * 0.4

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        if speed > pedestrian.max_speed:
            pedestrian.vx = pedestrian.vx / speed * pedestrian.max_speed
            pedestrian.vy = pedestrian.vy / speed * pedestrian.max_speed

        _apply_movement(pedestrian, obstacles, rng)


class FamilyGroupBehavior(PedestrianBehavior):
    """Keeps a small group moving together toward a shared edge goal."""

    def __init__(
        self,
        cohesion_strength: float = 0.9,
        alignment_strength: float = 0.18,
        separation_strength: float = 0.28,
        wander_strength: float = 0.2,
        wander_interval: int = 35,
    ):
        self.cohesion_strength = cohesion_strength
        self.alignment_strength = alignment_strength
        self.separation_strength = separation_strength
        self.wander_strength = wander_strength
        self.wander_interval = wander_interval
        self.frame_count = 0
        self._wander_angle = 0.0

    def update(
        self,
        pedestrian: "Pedestrian",
        others: list["Pedestrian"],
        obstacles: list[pygame.Rect] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.frame_count += 1
        if rng is not None and (
            self.frame_count == 1 or self.frame_count % self.wander_interval == 0
        ):
            self._wander_angle = float(rng.uniform(0.0, 2.0 * math.pi))

        group_members = [
            other
            for other in others
            if other is not pedestrian and other.group_id == pedestrian.group_id
        ]

        f_goal_x, f_goal_y = pedestrian._self_driving_force()
        f_cohesion_x, f_cohesion_y = 0.0, 0.0
        f_align_x, f_align_y = 0.0, 0.0
        f_group_sep_x, f_group_sep_y = 0.0, 0.0

        if group_members:
            center_x = sum(member.x for member in group_members) / len(group_members)
            center_y = sum(member.y for member in group_members) / len(group_members)
            to_center_x = center_x - pedestrian.x
            to_center_y = center_y - pedestrian.y
            center_dist = math.hypot(to_center_x, to_center_y)
            if center_dist > 1e-6:
                f_cohesion_x = (
                    to_center_x / center_dist * self.cohesion_strength
                )
                f_cohesion_y = (
                    to_center_y / center_dist * self.cohesion_strength
                )

            avg_vx = sum(member.vx for member in group_members) / len(group_members)
            avg_vy = sum(member.vy for member in group_members) / len(group_members)
            f_align_x = (avg_vx - pedestrian.vx) * self.alignment_strength
            f_align_y = (avg_vy - pedestrian.vy) * self.alignment_strength

            for member in group_members:
                dx = pedestrian.x - member.x
                dy = pedestrian.y - member.y
                dist = math.hypot(dx, dy)
                comfortable_dist = pedestrian.radius * 2.8
                if 1e-6 < dist < comfortable_dist:
                    scale = (comfortable_dist - dist) / comfortable_dist
                    f_group_sep_x += dx / dist * scale * self.separation_strength
                    f_group_sep_y += dy / dist * scale * self.separation_strength

        f_ped_x, f_ped_y = pedestrian._pedestrian_repulsion(others)
        f_wall_x, f_wall_y = pedestrian._wall_repulsion()
        f_obstacle_x, f_obstacle_y = pedestrian._obstacle_repulsion(obstacles)
        f_slide_x, f_slide_y = pedestrian._obstacle_slide_force(obstacles)
        f_wander_x = math.cos(self._wander_angle) * self.wander_strength
        f_wander_y = math.sin(self._wander_angle) * self.wander_strength

        pedestrian.vx += (
            f_goal_x * 1.15
            + f_cohesion_x
            + f_align_x
            + f_group_sep_x
            + f_ped_x * 0.22
            + f_wall_x * 0.55
            + f_obstacle_x
            + f_slide_x * 0.3
            + f_wander_x
        )
        pedestrian.vy += (
            f_goal_y * 1.15
            + f_cohesion_y
            + f_align_y
            + f_group_sep_y
            + f_ped_y * 0.22
            + f_wall_y * 0.55
            + f_obstacle_y
            + f_slide_y * 0.3
            + f_wander_y
        )

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        group_max_speed = pedestrian.max_speed * 1.08
        if speed > group_max_speed:
            pedestrian.vx = pedestrian.vx / speed * group_max_speed
            pedestrian.vy = pedestrian.vy / speed * group_max_speed

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
        # Self-driving force toward current waypoint (reduced)
        tx, ty = pedestrian.get_steering_target()
        dx = tx - pedestrian.x
        dy = ty - pedestrian.y
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
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        f5x, f5y = pedestrian._obstacle_slide_force(obstacles)

        pedestrian.vx += fx + f_cohesion_x + f_sep_x + f3x * 0.5 + f4x + f5x * 0.3
        pedestrian.vy += fy + f_cohesion_y + f_sep_y + f3y * 0.5 + f4y + f5y * 0.3

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

        # Change direction periodically — aim toward current waypoint with zigzag offset
        if self.frame_count % self.direction_change_frames == 0:
            tx, ty = pedestrian.get_steering_target()
            dx = tx - pedestrian.x
            dy = ty - pedestrian.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                angle = math.atan2(dy, dx)
                if rng:
                    angle += float(rng.uniform(-0.6, 0.6))
                pedestrian.vx = math.cos(angle) * pedestrian.desired_speed
                pedestrian.vy = math.sin(angle) * pedestrian.desired_speed

        # Pedestrian avoidance + obstacle repulsion
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        f5x, f5y = pedestrian._obstacle_slide_force(obstacles)
        pedestrian.vx += f2x * 0.3 + f4x + f5x * 0.4
        pedestrian.vy += f2y * 0.3 + f4y + f5y * 0.4

        # Wall repulsion
        f3x, f3y = pedestrian._wall_repulsion()
        pedestrian.vx += f3x
        pedestrian.vy += f3y

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        if speed > pedestrian.max_speed:
            pedestrian.vx = pedestrian.vx / speed * pedestrian.max_speed
            pedestrian.vy = pedestrian.vy / speed * pedestrian.max_speed

        _apply_movement(pedestrian, obstacles, rng)


# Track how long each pedestrian has been stuck (keyed by id)
_stuck_counters: dict[int, tuple[float, float, int]] = {}
_STUCK_THRESHOLD_FRAMES = 40    # frames before considered stuck
_STUCK_MOVE_THRESHOLD = 2.0     # px of movement to consider "not stuck"


def _apply_movement(
    pedestrian: "Pedestrian",
    obstacles: list[pygame.Rect] | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Apply velocity to position with collision handling and stuck recovery."""
    old_x, old_y = pedestrian.x, pedestrian.y
    proposed_x = pedestrian.x + pedestrian.vx
    proposed_y = pedestrian.y + pedestrian.vy

    moved = False

    if obstacles:
        hit_full = pedestrian._would_collide(proposed_x, proposed_y, obstacles)

        if not hit_full:
            pedestrian.x = proposed_x
            pedestrian.y = proposed_y
            moved = True
        else:
            # Smooth sliding: try diagonal, then individual axes
            can_move_x = not pedestrian._would_collide(proposed_x, old_y, obstacles)
            can_move_y = not pedestrian._would_collide(old_x, proposed_y, obstacles)

            if can_move_x and can_move_y:
                # Both axes possible: move on both with reduced velocity damping
                pedestrian.x = proposed_x
                pedestrian.y = proposed_y
                pedestrian.vx *= 0.75
                pedestrian.vy *= 0.75
                moved = True
            elif can_move_x:
                # Can only slide along X
                pedestrian.x = proposed_x
                pedestrian.vy = 0.0
                pedestrian.vx *= 0.85
                moved = True
            elif can_move_y:
                # Can only slide along Y
                pedestrian.y = proposed_y
                pedestrian.vx = 0.0
                pedestrian.vy *= 0.85
                moved = True

            # Fully blocked: gentle damping instead of aggressive stop
            if not moved:
                pedestrian.vx *= 0.5
                pedestrian.vy *= 0.5
    else:
        pedestrian.x = proposed_x
        pedestrian.y = proposed_y
        moved = True

    pedestrian.x = max(pedestrian.radius, min(WIDTH - pedestrian.radius, pedestrian.x))
    pedestrian.y = max(pedestrian.radius, min(HEIGHT - pedestrian.radius, pedestrian.y))

    # ---- Stuck detection / recovery ----
    pid = id(pedestrian)
    if pid in _stuck_counters:
        sx, sy, count = _stuck_counters[pid]
        dist_moved = math.hypot(pedestrian.x - sx, pedestrian.y - sy)

        if dist_moved < _STUCK_MOVE_THRESHOLD:
            count += 1
        else:
            count = 0
            sx, sy = pedestrian.x, pedestrian.y

        _stuck_counters[pid] = (sx, sy, count)

        if count >= _STUCK_THRESHOLD_FRAMES:
            # Re-aim toward the steering target with noise and gentler speed
            tx, ty = pedestrian.get_steering_target()
            dx = tx - pedestrian.x
            dy = ty - pedestrian.y
            dist = math.hypot(dx, dy)

            if dist > 1e-6:
                dx /= dist
                dy /= dist
                if rng is not None:
                    angle_noise = float(rng.uniform(-0.5, 0.5))
                    ca = math.cos(angle_noise)
                    sa = math.sin(angle_noise)
                    ndx = dx * ca - dy * sa
                    ndy = dx * sa + dy * ca
                    dx, dy = ndx, ndy

                # Use desired_speed * 1.0 (not 1.2) for smoother recovery
                kick = pedestrian.desired_speed
                pedestrian.vx = dx * kick
                pedestrian.vy = dy * kick
            else:
                pedestrian.vx = 0.0
                pedestrian.vy = 0.0

            _stuck_counters[pid] = (pedestrian.x, pedestrian.y, 0)
    else:
        _stuck_counters[pid] = (pedestrian.x, pedestrian.y, 0)