from abc import ABC, abstractmethod
import math

from constants import HEIGHT, WIDTH, SIM_SECONDS_PER_STEP


class PedestrianBehavior(ABC):
    @abstractmethod
    def update(self, pedestrian, others, obstacles = None, rng = None):
        """Update pedestrian velocity and position based on behavior."""
        pass


class SocialForceBehavior(PedestrianBehavior):
    """Classic social force model (current default behavior)."""

    def update(self, pedestrian, others, obstacles = None, rng = None):
        f1x, f1y = pedestrian._self_driving_force()
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        f3x, f3y = pedestrian._wall_repulsion()
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        f5x, f5y = _ttc_avoidance_force(pedestrian, others)

        pedestrian.vx += f1x + f2x + f3x + f4x + f5x
        pedestrian.vy += f1y + f2y + f3y + f4y + f5y

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        max_speed = pedestrian.max_speed_step()
        if speed > max_speed:
            pedestrian.vx = pedestrian.vx / speed * max_speed
            pedestrian.vy = pedestrian.vy / speed * max_speed

        _apply_movement(pedestrian, obstacles, rng)


class StationaryBehavior(PedestrianBehavior):
    """Mostly stationary with infrequent goal updates."""

    def __init__(self, movement_probability = 0.08):
        self.movement_probability = movement_probability
        self.stationary_frame_count = 0

    def update(self, pedestrian, others, obstacles = None, rng = None):
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
        f5x, f5y = _ttc_avoidance_force(pedestrian, others)
        pedestrian.vx += f3x * 0.3 + f4x * 0.5 + f5x * 0.35
        pedestrian.vy += f3y * 0.3 + f4y * 0.5 + f5y * 0.35

        _apply_movement(pedestrian, obstacles, rng)


class RandomWalkerBehavior(PedestrianBehavior):
    """Move toward random goals across entire screen at increased speed."""

    def __init__(self, speed_multiplier = 2.0):
        self.speed_multiplier = speed_multiplier

    def update(self, pedestrian, others, obstacles = None, rng = None):
        # Move toward current waypoint with increased speed
        tx, ty = pedestrian.get_steering_target()
        dx = tx - pedestrian.x
        dy = ty - pedestrian.y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ex, ey = dx / dist, dy / dist
            desired_speed = pedestrian.desired_speed_step() * self.speed_multiplier
            fx = (desired_speed * ex - pedestrian.vx) / pedestrian.relaxation_time
            fy = (desired_speed * ey - pedestrian.vy) / pedestrian.relaxation_time
        else:
            fx, fy = 0.0, 0.0

        # Weak pedestrian avoidance to avoid getting stuck
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        f5x, f5y = _ttc_avoidance_force(pedestrian, others)
        pedestrian.vx += fx + f2x * 0.3 + f5x * 0.6
        pedestrian.vy += fy + f2y * 0.3 + f5y * 0.6

        # Wall + obstacle repulsion
        f3x, f3y = pedestrian._wall_repulsion()
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        pedestrian.vx += f3x * 0.3 + f4x
        pedestrian.vy += f3y * 0.3 + f4y

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        max_speed = pedestrian.max_speed_step()
        if speed > max_speed:
            pedestrian.vx = pedestrian.vx / speed * max_speed
            pedestrian.vy = pedestrian.vy / speed * max_speed

        _apply_movement(pedestrian, obstacles, rng)


class FamilyGroupBehavior(PedestrianBehavior):
    """Keeps a small group moving together toward a shared edge goal."""

    def __init__(self, cohesion_strength = 0.9, alignment_strength = 0.18, separation_strength = 0.28, wander_strength = 0.2, wander_interval = 35):
        self.cohesion_strength = cohesion_strength
        self.alignment_strength = alignment_strength
        self.separation_strength = separation_strength
        self.wander_strength = wander_strength
        self.wander_interval = wander_interval
        self.frame_count = 0
        self._wander_angle = 0.0

    def update(self, pedestrian, others, obstacles = None, rng = None):
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
        f_ttc_x, f_ttc_y = _ttc_avoidance_force(pedestrian, others)
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
            + f_ttc_x * 0.55
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
            + f_ttc_y * 0.55
            + f_wander_y
        )

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        group_max_speed = pedestrian.max_speed_step() * 1.08
        if speed > group_max_speed:
            pedestrian.vx = pedestrian.vx / speed * group_max_speed
            pedestrian.vy = pedestrian.vy / speed * group_max_speed

        _apply_movement(pedestrian, obstacles, rng)


class ClumpBehavior(PedestrianBehavior):
    """Move together in groups toward goal."""

    def __init__(self, clump_radius = 120.0):
        self.clump_radius = clump_radius

    def update(self, pedestrian, others, obstacles = None, rng = None):
        # Self-driving force toward current waypoint (reduced)
        tx, ty = pedestrian.get_steering_target()
        dx = tx - pedestrian.x
        dy = ty - pedestrian.y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ex, ey = dx / dist, dy / dist
            desired_speed = pedestrian.desired_speed_step()
            fx = (desired_speed * ex * 0.5 - pedestrian.vx) / pedestrian.relaxation_time
            fy = (desired_speed * ey * 0.5 - pedestrian.vy) / pedestrian.relaxation_time
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

        f5x, f5y = _ttc_avoidance_force(pedestrian, others)
        pedestrian.vx += fx + f_cohesion_x + f_sep_x + f3x * 0.5 + f4x + f5x * 0.45
        pedestrian.vy += fy + f_cohesion_y + f_sep_y + f3y * 0.5 + f4y + f5y * 0.45

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        max_speed = pedestrian.max_speed_step()
        if speed > max_speed:
            pedestrian.vx = pedestrian.vx / speed * max_speed
            pedestrian.vy = pedestrian.vy / speed * max_speed

        _apply_movement(pedestrian, obstacles, rng)


class ZigzagBehavior(PedestrianBehavior):
    """Zigzag across space frequently, changing direction."""

    def __init__(self, direction_change_frames = 30):
        self.direction_change_frames = direction_change_frames
        self.frame_count = 0
        self._target_offset = 0.0
        self._current_offset = 0.0
        self._max_offset = 0.38
        self._velocity_blend = 0.22

    def update(self, pedestrian, others, obstacles = None, rng = None):
        self.frame_count += 1

        # Smoothly drift around the goal heading instead of abrupt heading resets.
        tx, ty = pedestrian.get_steering_target()
        dx = tx - pedestrian.x
        dy = ty - pedestrian.y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            base_angle = math.atan2(dy, dx)
            if rng is not None and (
                self.frame_count == 1
                or self.frame_count % self.direction_change_frames == 0
            ):
                self._target_offset = float(rng.uniform(-self._max_offset, self._max_offset))

            smoothing = max(0.08, min(0.28, 2.0 / max(1, self.direction_change_frames)))
            self._current_offset += (self._target_offset - self._current_offset) * smoothing
            desired_angle = base_angle + self._current_offset
            desired_speed = pedestrian.desired_speed_step()
            desired_vx = math.cos(desired_angle) * desired_speed
            desired_vy = math.sin(desired_angle) * desired_speed
            pedestrian.vx = (
                pedestrian.vx * (1.0 - self._velocity_blend)
                + desired_vx * self._velocity_blend
            )
            pedestrian.vy = (
                pedestrian.vy * (1.0 - self._velocity_blend)
                + desired_vy * self._velocity_blend
            )

        # Pedestrian avoidance + obstacle repulsion
        f2x, f2y = pedestrian._pedestrian_repulsion(others)
        f4x, f4y = pedestrian._obstacle_repulsion(obstacles)
        f5x, f5y = _ttc_avoidance_force(pedestrian, others)
        pedestrian.vx += f2x * 0.3 + f4x + f5x * 0.55
        pedestrian.vy += f2y * 0.3 + f4y + f5y * 0.55

        # Wall repulsion
        f3x, f3y = pedestrian._wall_repulsion()
        pedestrian.vx += f3x
        pedestrian.vy += f3y

        speed = math.hypot(pedestrian.vx, pedestrian.vy)
        max_speed = pedestrian.max_speed_step()
        if speed > max_speed:
            pedestrian.vx = pedestrian.vx / speed * max_speed
            pedestrian.vy = pedestrian.vy / speed * max_speed

        _apply_movement(pedestrian, obstacles, rng)


_TTC_HORIZON_SECONDS = 1.0
_TTC_COMFORT_MARGIN_PX = 8.0
_TTC_FORCE_SCALE = 1.2

# Track how long each pedestrian has been stuck (keyed by id)
_stuck_counters = {}
_velocity_history = {}
_VELOCITY_SMOOTHING = 0.14      # mild blend with last frame to reduce jitter
_STUCK_THRESHOLD_FRAMES = 75    # frames before considered stuck
_STUCK_MOVE_THRESHOLD = 2.0     # px of movement to consider "not stuck"
_STUCK_KICK_COOLDOWN = 30       # frames before another random kick is allowed


def _ttc_avoidance_force(pedestrian, others):
    """
    Anticipatory side-step force using short-horizon time-to-collision (TTC).
    Keeps trajectories smoother than pure distance-based repulsion.
    """
    fx, fy = 0.0, 0.0
    horizon_steps = max(8.0, _TTC_HORIZON_SECONDS / SIM_SECONDS_PER_STEP)

    for other in others:
        if other is pedestrian:
            continue
        if other.y > HEIGHT + other.radius:
            continue

        rel_x = other.x - pedestrian.x
        rel_y = other.y - pedestrian.y
        rel_vx = other.vx - pedestrian.vx
        rel_vy = other.vy - pedestrian.vy
        rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
        if rel_speed_sq < 1e-6:
            continue

        closing = -(rel_x * rel_vx + rel_y * rel_vy)
        if closing <= 0.0:
            continue

        ttc_steps = closing / rel_speed_sq
        if ttc_steps <= 0.0 or ttc_steps > horizon_steps:
            continue

        closest_x = rel_x + rel_vx * ttc_steps
        closest_y = rel_y + rel_vy * ttc_steps
        closest_dist = math.hypot(closest_x, closest_y)
        comfort = pedestrian.radius + other.radius + _TTC_COMFORT_MARGIN_PX
        risk = max(0.0, 1.0 - (closest_dist / max(comfort, 1e-6)))
        if risk <= 0.0:
            continue

        # Prefer lateral avoidance relative to closing direction.
        side_x = -rel_vy
        side_y = rel_vx
        side_norm = math.hypot(side_x, side_y)
        if side_norm > 1e-6:
            side_x /= side_norm
            side_y /= side_norm
            if side_x * closest_x + side_y * closest_y > 0.0:
                side_x *= -1.0
                side_y *= -1.0
        else:
            away_norm = max(closest_dist, 1e-6)
            side_x = -closest_x / away_norm
            side_y = -closest_y / away_norm

        away_norm = max(closest_dist, 1e-6)
        away_x = -closest_x / away_norm
        away_y = -closest_y / away_norm
        urgency = 1.0 - (ttc_steps / horizon_steps)
        magnitude = _TTC_FORCE_SCALE * risk * urgency * pedestrian.desired_speed_step()
        fx += side_x * magnitude + away_x * magnitude * 0.35
        fy += side_y * magnitude + away_y * magnitude * 0.35

    return fx, fy


def _apply_velocity_limits(pedestrian, prev_vx, prev_vy):
    """Limit acceleration and turn rate to reduce oscillatory motion."""
    # 1) Acceleration cap (limit delta-v magnitude per step).
    dvx = pedestrian.vx - prev_vx
    dvy = pedestrian.vy - prev_vy
    dv_mag = math.hypot(dvx, dvy)
    max_dv = max(1e-6, pedestrian.max_delta_v_step())
    if dv_mag > max_dv:
        scale = max_dv / dv_mag
        pedestrian.vx = prev_vx + dvx * scale
        pedestrian.vy = prev_vy + dvy * scale

    # 2) Turn-rate cap (limit heading change per step).
    prev_speed = math.hypot(prev_vx, prev_vy)
    curr_speed = math.hypot(pedestrian.vx, pedestrian.vy)
    max_turn = pedestrian.max_turn_step_radians()
    if prev_speed > 1e-6 and curr_speed > 1e-6 and max_turn > 1e-6:
        prev_heading = math.atan2(prev_vy, prev_vx)
        curr_heading = math.atan2(pedestrian.vy, pedestrian.vx)
        delta = math.atan2(
            math.sin(curr_heading - prev_heading),
            math.cos(curr_heading - prev_heading),
        )
        if abs(delta) > max_turn:
            clamped_heading = prev_heading + math.copysign(max_turn, delta)
            pedestrian.vx = math.cos(clamped_heading) * curr_speed
            pedestrian.vy = math.sin(clamped_heading) * curr_speed

    # 3) Speed cap.
    speed = math.hypot(pedestrian.vx, pedestrian.vy)
    max_speed = pedestrian.max_speed_step()
    if speed > max_speed:
        pedestrian.vx = pedestrian.vx / speed * max_speed
        pedestrian.vy = pedestrian.vy / speed * max_speed


def _apply_movement(pedestrian, obstacles = None, rng = None):
    """Apply velocity to position with collision handling and stuck recovery."""
    pid = id(pedestrian)
    prev_vx, prev_vy = _velocity_history.get(pid, (pedestrian.vx, pedestrian.vy))
    _apply_velocity_limits(pedestrian, prev_vx, prev_vy)
    pedestrian.vx = (
        pedestrian.vx * (1.0 - _VELOCITY_SMOOTHING)
        + prev_vx * _VELOCITY_SMOOTHING
    )
    pedestrian.vy = (
        pedestrian.vy * (1.0 - _VELOCITY_SMOOTHING)
        + prev_vy * _VELOCITY_SMOOTHING
    )

    old_x, old_y = pedestrian.x, pedestrian.y
    proposed_x = pedestrian.x + pedestrian.vx
    proposed_y = pedestrian.y + pedestrian.vy

    if obstacles and pedestrian._would_collide(proposed_x, proposed_y, obstacles):
        # Try sliding along each axis independently
        if not pedestrian._would_collide(proposed_x, old_y, obstacles):
            pedestrian.x = proposed_x
        elif not pedestrian._would_collide(old_x, proposed_y, obstacles):
            pedestrian.y = proposed_y
        else:
            # Fully blocked: reverse with stronger kick
            pedestrian.vx *= -0.35
            pedestrian.vy *= -0.35
            if rng is not None:
                pedestrian.vx += float(rng.uniform(-0.25, 0.25))
                pedestrian.vy += float(rng.uniform(-0.25, 0.25))
    else:
        pedestrian.x = proposed_x
        pedestrian.y = proposed_y

    # Clamp to screen
    pedestrian.x = max(pedestrian.radius, min(WIDTH - pedestrian.radius, pedestrian.x))
    pedestrian.y = max(pedestrian.radius, min(HEIGHT - pedestrian.radius, pedestrian.y))
    _velocity_history[pid] = (pedestrian.vx, pedestrian.vy)

    # ---- Stuck detection / recovery ----
    if pid in _stuck_counters:
        sx, sy, count, cooldown = _stuck_counters[pid]
        if cooldown > 0:
            cooldown -= 1
        moved = math.hypot(pedestrian.x - sx, pedestrian.y - sy)
        if moved < _STUCK_MOVE_THRESHOLD:
            count += 1
        else:
            count = 0
            sx, sy = pedestrian.x, pedestrian.y
        _stuck_counters[pid] = (sx, sy, count, cooldown)

        if count >= _STUCK_THRESHOLD_FRAMES and cooldown == 0:
            # Give a strong random kick to break free
            if rng is not None:
                angle = float(rng.uniform(0, 2 * math.pi))
                kick = pedestrian.desired_speed_step() * 0.9
                pedestrian.vx = math.cos(angle) * kick
                pedestrian.vy = math.sin(angle) * kick
            _velocity_history[pid] = (pedestrian.vx, pedestrian.vy)
            _stuck_counters[pid] = (pedestrian.x, pedestrian.y, 0, _STUCK_KICK_COOLDOWN)
    else:
        _stuck_counters[pid] = (pedestrian.x, pedestrian.y, 0, 0)
