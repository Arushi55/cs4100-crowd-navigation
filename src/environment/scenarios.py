
from dataclasses import dataclass
import json
from pathlib import Path
import pygame
import numpy as np

from constants import HEIGHT, WIDTH
from environment.behaviors import (
    SocialForceBehavior,
    StationaryBehavior,
    RandomWalkerBehavior,
    FamilyGroupBehavior,
    ClumpBehavior,
    ZigzagBehavior,
)
from environment.pedestrian import Pedestrian


# Initial spawn-region -> goal-region preferences for more purposeful first trips.
_INITIAL_SPAWN_TO_GOAL_WEIGHTS = {
    "home": [
        [0.45, 0.10, 0.30, 0.15],
        [0.55, 0.10, 0.10, 0.25],
        [0.15, 0.50, 0.15, 0.20],
    ],
    "shopping_center": [
        [0.12, 0.18, 0.48, 0.22],
        [0.30, 0.28, 0.22, 0.20],
        [0.50, 0.10, 0.20, 0.20],
    ],
    "airport": [
        [0.52, 0.08, 0.30, 0.10],
        [0.08, 0.52, 0.10, 0.30],
    ],
}

_INITIAL_SPEED_SCALE_BY_SCENARIO = {
    "home": (0.86, 1.10),
    "shopping_center": (0.82, 1.16),
    "airport": (0.88, 1.20),
}

_INITIAL_SPEED_FRACTION_BY_SCENARIO = {
    "home": (0.18, 0.38),
    "shopping_center": (0.22, 0.48),
    "airport": (0.25, 0.52),
}

_INITIAL_HEADING_JITTER_BY_SCENARIO = {
    "home": 0.22,
    "shopping_center": 0.18,
    "airport": 0.12,
}


SCENARIO_CONFIG_DIR = Path(__file__).with_name("scenario_configs")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    name: str
    robot_start: tuple[float, float]
    robot_goal: tuple[float, float]
    obstacles: list[pygame.Rect]
    pedestrian_spawn_regions: list[pygame.Rect]
    pedestrian_goal_regions: list[pygame.Rect]


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    name: str
    robot_start: tuple[float, float]
    robot_goal: tuple[float, float]
    obstacles: list[tuple[int, int, int, int]]
    pedestrian_spawn_regions: list[tuple[int, int, int, int]]
    pedestrian_goal_regions: list[tuple[int, int, int, int]]
    obstacle_jitter_px: int
    obstacle_size_jitter_px: int
    extra_obstacle_range: tuple[int, int]
    random_obstacle_size_range: tuple[int, int]
    pedestrian_behaviors: list[dict[str, int | str]]  # e.g., [{"type": "stationary", "count": 11}]
    pedestrian_groups: list[dict[str, int | float | str]]


def _tuple4(raw):
    return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])


def _point_hits_obstacles(x, y, obstacles, clearance_radius):
    if not obstacles:
        return False
    hitbox = pygame.Rect(
        int(x - clearance_radius),
        int(y - clearance_radius),
        clearance_radius * 2,
        clearance_radius * 2,
    )
    return any(hitbox.colliderect(obstacle) for obstacle in obstacles)


def _build_behavior(behavior_spec):
    """
    Factory function to create behaviors from scenario config specs.
    
    Returns:
        Tuple of (behavior, goal_region_indices)
    """
    behavior_type = str(behavior_spec.get("type", "social_force"))
    goal_region_indices = behavior_spec.get("goal_region_indices")
    if goal_region_indices is not None:
        goal_region_indices = [int(x) for x in goal_region_indices]
    
    if behavior_type == "stationary":
        movement_prob = float(behavior_spec.get("movement_probability", 0.08))
        return StationaryBehavior(movement_probability=movement_prob), goal_region_indices
    elif behavior_type == "random_walker":
        speed_mult = float(behavior_spec.get("speed_multiplier", 2.0))
        return RandomWalkerBehavior(speed_multiplier=speed_mult), goal_region_indices
    elif behavior_type == "family_group":
        cohesion = float(behavior_spec.get("cohesion_strength", 0.9))
        alignment = float(behavior_spec.get("alignment_strength", 0.18))
        separation = float(behavior_spec.get("separation_strength", 0.28))
        wander = float(behavior_spec.get("wander_strength", 0.2))
        interval = int(behavior_spec.get("wander_interval", 35))
        return FamilyGroupBehavior(
            cohesion_strength=cohesion,
            alignment_strength=alignment,
            separation_strength=separation,
            wander_strength=wander,
            wander_interval=interval,
        ), goal_region_indices
    elif behavior_type == "clump":
        clump_radius = float(behavior_spec.get("clump_radius", 60.0))
        return ClumpBehavior(clump_radius=clump_radius), goal_region_indices
    elif behavior_type == "zigzag":
        direction_frames = int(behavior_spec.get("direction_change_frames", 30))
        return ZigzagBehavior(direction_change_frames=direction_frames), goal_region_indices
    else:  # "social_force" or default
        return SocialForceBehavior(), goal_region_indices


def load_scenario_templates(config_dir = SCENARIO_CONFIG_DIR):
    templates = {}
    for path in sorted(config_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        rid = raw["id"]
        rand = raw.get("randomization", {})
        templates[rid] = ScenarioTemplate(
            scenario_id=rid,
            name=raw["name"],
            robot_start=(float(raw["robot_start"][0]), float(raw["robot_start"][1])),
            robot_goal=(float(raw["robot_goal"][0]), float(raw["robot_goal"][1])),
            obstacles=[_tuple4(rect) for rect in raw["obstacles"]],
            pedestrian_spawn_regions=[_tuple4(rect) for rect in raw["pedestrian_spawn_regions"]],
            pedestrian_goal_regions=[_tuple4(rect) for rect in raw["pedestrian_goal_regions"]],
            obstacle_jitter_px=int(rand.get("obstacle_jitter_px", 0)),
            obstacle_size_jitter_px=int(rand.get("obstacle_size_jitter_px", 0)),
            extra_obstacle_range=(
                int(rand.get("extra_obstacle_min", 0)),
                int(rand.get("extra_obstacle_max", 0)),
            ),
            random_obstacle_size_range=(
                int(rand.get("random_obstacle_min_size", 35)),
                int(rand.get("random_obstacle_max_size", 110)),
            ),
            pedestrian_behaviors=raw.get("pedestrian_behaviors", [{"type": "social_force", "count": 12}]),
            pedestrian_groups=raw.get("pedestrian_groups", []),
        )
    if not templates:
        raise ValueError(f"No scenario config files found in {config_dir}")
    return templates


def _rect_to_safe_bounds(x, y, w, h):
    w = max(20, min(w, WIDTH - 20))
    h = max(20, min(h, HEIGHT - 20))
    x = max(0, min(x, WIDTH - w))
    y = max(0, min(y, HEIGHT - h))
    return pygame.Rect(x, y, w, h)


def _randomized_obstacle(rect, rng, position_jitter, size_jitter):
    x, y, w, h = rect
    if position_jitter > 0:
        x += int(rng.integers(-position_jitter, position_jitter + 1))
        y += int(rng.integers(-position_jitter, position_jitter + 1))
    if size_jitter > 0:
        w += int(rng.integers(-size_jitter, size_jitter + 1))
        h += int(rng.integers(-size_jitter, size_jitter + 1))
    return _rect_to_safe_bounds(x, y, w, h)


def _circle_hits_rect(center, radius, rect):
    cx, cy = center
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius


def _sample_extra_obstacles(template, obstacles, rng):
    out = list(obstacles)
    extra_min, extra_max = template.extra_obstacle_range
    if extra_max <= 0:
        return out
    count = int(rng.integers(extra_min, extra_max + 1))
    min_size, max_size = template.random_obstacle_size_range
    for _ in range(count):
        for _attempt in range(30):
            w = int(rng.integers(min_size, max_size + 1))
            h = int(rng.integers(min_size, max_size + 1))
            x = int(rng.integers(0, max(1, WIDTH - w)))
            y = int(rng.integers(0, max(1, HEIGHT - h)))
            rect = _rect_to_safe_bounds(x, y, w, h)
            if _circle_hits_rect(template.robot_start, 45, rect):
                continue
            if _circle_hits_rect(template.robot_goal, 45, rect):
                continue
            if any(rect.colliderect(existing.inflate(20, 20)) for existing in out):
                continue
            out.append(rect)
            break
    return out


def build_scenario(template, rng, randomize_world = False):
    if randomize_world:
        obstacles = [
            _randomized_obstacle(
                rect,
                rng,
                position_jitter=template.obstacle_jitter_px,
                size_jitter=template.obstacle_size_jitter_px,
            )
            for rect in template.obstacles
        ]
        obstacles = _sample_extra_obstacles(template, obstacles, rng)
    else:
        obstacles = [_rect_to_safe_bounds(*rect) for rect in template.obstacles]

    return Scenario(
        scenario_id=template.scenario_id,
        name=template.name,
        robot_start=template.robot_start,
        robot_goal=template.robot_goal,
        obstacles=obstacles,
        pedestrian_spawn_regions=[pygame.Rect(*rect) for rect in template.pedestrian_spawn_regions],
        pedestrian_goal_regions=[pygame.Rect(*rect) for rect in template.pedestrian_goal_regions],
    )


def random_point_in_region(region, rng, margin = 12, obstacles = None, clearance_radius = 10, max_attempts = 80):
    left = max(0.0, float(region.left + margin))
    right = min(float(WIDTH), float(region.right - margin))
    top = max(0.0, float(region.top + margin))
    bottom = min(float(HEIGHT), float(region.bottom - margin))

    if right <= left:
        left = float(region.centerx)
        right = left + 1.0
    if bottom <= top:
        top = float(region.centery)
        bottom = top + 1.0

    for _ in range(max_attempts):
        x = float(rng.uniform(left, right))
        y = float(rng.uniform(top, bottom))
        if not _point_hits_obstacles(x, y, obstacles, clearance_radius):
            return x, y

    # Fallback: deterministic scan for a valid point within the region.
    scan_step = max(4, clearance_radius // 2)
    for py in range(int(top), int(bottom) + 1, scan_step):
        for px in range(int(left), int(right) + 1, scan_step):
            x = float(px)
            y = float(py)
            if not _point_hits_obstacles(x, y, obstacles, clearance_radius):
                return x, y

    # Last resort (region may be fully blocked): return center.
    return float(region.centerx), float(region.centery)


def random_pedestrian_route(scenario, rng, goal_region_indices = None):
    """
    Pick a random spawn and goal for a pedestrian.
    
    Args:
        scenario: The scenario with spawn/goal regions
        rng: Random number generator
        goal_region_indices: If provided, only pick goals from these region indices.
                            If None, all goal regions are available.
    """
    spawn_region = scenario.pedestrian_spawn_regions[int(rng.integers(0, len(scenario.pedestrian_spawn_regions)))]
    
    if goal_region_indices is not None and len(goal_region_indices) > 0:
        # Pick from specified goal regions only
        chosen_idx = int(rng.integers(0, len(goal_region_indices)))
        region_idx = goal_region_indices[chosen_idx]
        goal_region = scenario.pedestrian_goal_regions[region_idx]
    else:
        # Pick from all available goal regions
        goal_region = scenario.pedestrian_goal_regions[int(rng.integers(0, len(scenario.pedestrian_goal_regions)))]
    
    spawn = random_point_in_region(
        spawn_region,
        rng,
        obstacles=scenario.obstacles,
    )
    goal = random_point_in_region(
        goal_region,
        rng,
        obstacles=scenario.obstacles,
    )
    return spawn, goal


def _sample_initial_goal_region_index(scenario, rng, spawn_region_idx, goal_region_indices = None):
    allowed = (
        list(goal_region_indices)
        if goal_region_indices is not None and len(goal_region_indices) > 0
        else list(range(len(scenario.pedestrian_goal_regions)))
    )
    if not allowed:
        return int(rng.integers(0, len(scenario.pedestrian_goal_regions)))
    if len(allowed) == 1:
        return int(allowed[0])

    matrix = _INITIAL_SPAWN_TO_GOAL_WEIGHTS.get(scenario.scenario_id)
    if matrix is not None and 0 <= spawn_region_idx < len(matrix):
        row = matrix[spawn_region_idx]
        raw_weights = []
        for idx in allowed:
            weight = float(row[idx]) if 0 <= idx < len(row) else 0.0
            raw_weights.append(max(0.0, weight))
        weight_sum = float(sum(raw_weights))
        if weight_sum > 1e-9:
            probs = np.array(raw_weights, dtype=np.float64) / weight_sum
            choice_idx = int(rng.choice(len(allowed), p=probs))
            return int(allowed[choice_idx])

    # Fallback: uniform among allowed indices.
    choice_idx = int(rng.integers(0, len(allowed)))
    return int(allowed[choice_idx])


def _sample_initial_pedestrian_route(scenario, rng, goal_region_indices = None):
    spawn_region_idx = int(rng.integers(0, len(scenario.pedestrian_spawn_regions)))
    spawn_region = scenario.pedestrian_spawn_regions[spawn_region_idx]
    goal_region_idx = _sample_initial_goal_region_index(
        scenario,
        rng,
        spawn_region_idx,
        goal_region_indices,
    )
    goal_region = scenario.pedestrian_goal_regions[goal_region_idx]

    spawn = random_point_in_region(
        spawn_region,
        rng,
        obstacles=scenario.obstacles,
    )
    goal = random_point_in_region(
        goal_region,
        rng,
        obstacles=scenario.obstacles,
    )
    return spawn, goal


def _apply_initial_non_group_profile(ped, scenario_id, rng):
    lo, hi = _INITIAL_SPEED_SCALE_BY_SCENARIO.get(scenario_id, (0.86, 1.14))
    speed_scale = float(rng.uniform(lo, hi))
    ped.desired_speed *= speed_scale
    ped.max_speed = max(ped.desired_speed + 0.4, ped.max_speed * speed_scale)
    ped.relaxation_time *= float(rng.uniform(0.90, 1.16))


def _seed_initial_velocity(ped, scenario_id, rng):
    tx, ty = ped.get_steering_target()
    dx = tx - ped.x
    dy = ty - ped.y
    dist = float(np.hypot(dx, dy))
    if dist < 1e-6:
        ped.vx = 0.0
        ped.vy = 0.0
        return

    base_heading = float(np.arctan2(dy, dx))
    jitter = _INITIAL_HEADING_JITTER_BY_SCENARIO.get(scenario_id, 0.16)
    heading = base_heading + float(rng.uniform(-jitter, jitter))
    lo, hi = _INITIAL_SPEED_FRACTION_BY_SCENARIO.get(scenario_id, (0.20, 0.45))
    speed_fraction = float(rng.uniform(lo, hi))
    speed = ped.desired_speed_step() * speed_fraction
    ped.vx = float(np.cos(heading) * speed)
    ped.vy = float(np.sin(heading) * speed)


def _clamp_world_point(x, y, padding = 14):
    return (
        float(max(padding, min(WIDTH - padding, x))),
        float(max(padding, min(HEIGHT - padding, y))),
    )


def _sample_edge_point(edge, rng, padding = 18):
    axis_padding = padding * 2
    if edge == "left":
        return float(padding), float(rng.uniform(axis_padding, HEIGHT - axis_padding))
    if edge == "right":
        return float(WIDTH - padding), float(rng.uniform(axis_padding, HEIGHT - axis_padding))
    if edge == "top":
        return float(rng.uniform(axis_padding, WIDTH - axis_padding)), float(padding)
    if edge == "bottom":
        return float(rng.uniform(axis_padding, WIDTH - axis_padding)), float(HEIGHT - padding)
    raise ValueError(f"Unsupported edge '{edge}'")


def _sample_edge_point_safe(edge, rng, obstacles, padding = 18, clearance_radius = 10, max_attempts = 40):
    for _ in range(max_attempts):
        x, y = _sample_edge_point(edge, rng, padding=padding)
        if not _point_hits_obstacles(x, y, obstacles, clearance_radius):
            return x, y
    return _sample_edge_point(edge, rng, padding=padding)


def _sample_group_member_positions(anchor, size, rng, spread, obstacles = None, clearance_radius = 10, max_attempts = 60):
    points = []
    for _ in range(size):
        chosen = None
        for _attempt in range(max_attempts):
            px = anchor[0] + float(rng.uniform(-spread, spread))
            py = anchor[1] + float(rng.uniform(-spread, spread))
            x, y = _clamp_world_point(px, py)
            if not _point_hits_obstacles(x, y, obstacles, clearance_radius):
                chosen = (x, y)
                break
        if chosen is None:
            chosen = _clamp_world_point(anchor[0], anchor[1])
        points.append(chosen)
    return points


def _normalize_group_specs(raw_specs):
    if not raw_specs:
        return None
    weights = np.array(
        [float(spec.get("weight", 1.0)) for spec in raw_specs],
        dtype=np.float64,
    )
    total = weights.sum()
    if total <= 0:
        return None
    return raw_specs, weights / total


def _scaled_behavior_counts(behavior_specs, target_count):
    """
    Scale per-behavior template counts to match the requested pedestrian total.

    This preserves the relative behavior mix from scenario JSON while allowing
    runtime flags (e.g., --pedestrians) to control total population size.
    """
    if target_count <= 0 or not behavior_specs:
        return [0 for _ in behavior_specs]

    base_counts = np.array(
        [max(0, int(spec.get("count", 0))) for spec in behavior_specs],
        dtype=np.float64,
    )
    total_base = float(base_counts.sum())
    if total_base <= 0:
        # If template counts are missing/invalid, fall back to uniform mix.
        base_counts = np.ones(len(behavior_specs), dtype=np.float64)
        total_base = float(len(behavior_specs))

    scaled_float = (base_counts / total_base) * float(target_count)
    scaled = np.floor(scaled_float).astype(int)
    remaining = int(target_count - int(scaled.sum()))
    if remaining <= 0:
        return scaled.tolist()

    # Largest-remainder allocation to hit exact target_count.
    fractional = scaled_float - scaled
    order = np.argsort(-fractional)
    for idx in order[:remaining]:
        scaled[int(idx)] += 1
    return scaled.tolist()


def _generate_family_groups(scenario, template, nav_grid, rng, target_count):
    normalized = _normalize_group_specs(template.pedestrian_groups)
    if normalized is None or target_count <= 0:
        return []

    group_specs, probabilities = normalized
    pedestrians = []
    next_group_id = 0

    while len(pedestrians) < target_count:
        spec_idx = int(rng.choice(len(group_specs), p=probabilities))
        spec = group_specs[spec_idx]
        size_min = int(spec.get("size_min", 2))
        size_max = int(spec.get("size_max", size_min))
        group_size = int(rng.integers(size_min, size_max + 1))
        remaining = target_count - len(pedestrians)
        if remaining <= 0:
            break
        # Clamp to remaining slots so requested count is exact even for odd totals.
        group_size = min(group_size, remaining)

        spawn_edge = str(spec.get("spawn_edge", "left"))
        goal_edge = str(spec.get("goal_edge", "right"))
        spawn_anchor = _sample_edge_point(spawn_edge, rng)
        goal_anchor = _sample_edge_point(goal_edge, rng)
        spawn_spread = float(spec.get("spawn_spread", 28.0))
        goal_spread = float(spec.get("goal_spread", 22.0))
        behavior_spec = {
            "type": "family_group",
            "cohesion_strength": float(spec.get("cohesion_strength", 0.9)),
            "alignment_strength": float(spec.get("alignment_strength", 0.18)),
            "separation_strength": float(spec.get("separation_strength", 0.28)),
            "wander_strength": float(spec.get("wander_strength", 0.2)),
            "wander_interval": int(spec.get("wander_interval", 35)),
        }
        member_positions = _sample_group_member_positions(
            spawn_anchor,
            group_size,
            rng,
            spread=spawn_spread,
            obstacles=scenario.obstacles,
        )
        raw_goal = _clamp_world_point(
            goal_anchor[0] + float(rng.uniform(-goal_spread, goal_spread)),
            goal_anchor[1] + float(rng.uniform(-goal_spread, goal_spread)),
        )
        goal_region = pygame.Rect(
            int(max(0, raw_goal[0] - goal_spread)),
            int(max(0, raw_goal[1] - goal_spread)),
            int(goal_spread * 2),
            int(goal_spread * 2),
        )
        group_goal = random_point_in_region(
            goal_region,
            rng,
            margin=0,
            obstacles=scenario.obstacles,
        )

        for sx, sy in member_positions:
            gx, gy = group_goal
            behavior, goal_region_indices = _build_behavior(behavior_spec)
            ped = Pedestrian(
                x=sx,
                y=sy,
                vx=0.0,
                vy=0.0,
                goal_x=gx,
                goal_y=gy,
                behavior=behavior,
                goal_region_indices=goal_region_indices,
                group_id=next_group_id,
                spawn_edge=spawn_edge,
                goal_edge=goal_edge,
                group_spawn_spread=spawn_spread,
                group_goal_spread=goal_spread,
            )
            ped.desired_speed = float(spec.get("desired_speed", ped.desired_speed))
            ped.max_speed = float(spec.get("max_speed", ped.max_speed))
            ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
            _seed_initial_velocity(ped, scenario.scenario_id, rng)
            pedestrians.append(ped)

        next_group_id += 1

    return pedestrians


def generate_pedestrian_population(scenario, template, nav_grid, rng, count = 12):
    if template.pedestrian_groups:
        return _generate_family_groups(scenario, template, nav_grid, rng, count)

    pedestrians = []
    if template.pedestrian_behaviors:
        behavior_counts = _scaled_behavior_counts(template.pedestrian_behaviors, count)
        for behavior_spec, behavior_count in zip(template.pedestrian_behaviors, behavior_counts):
            for _ in range(behavior_count):
                behavior, goal_region_indices = _build_behavior(behavior_spec)
                (sx, sy), (gx, gy) = _sample_initial_pedestrian_route(
                    scenario,
                    rng,
                    goal_region_indices,
                )
                ped = Pedestrian(
                    x=sx,
                    y=sy,
                    vx=0.0,
                    vy=0.0,
                    goal_x=gx,
                    goal_y=gy,
                    behavior=behavior,
                    goal_region_indices=goal_region_indices,
                )
                _apply_initial_non_group_profile(ped, scenario.scenario_id, rng)
                ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
                _seed_initial_velocity(ped, scenario.scenario_id, rng)
                pedestrians.append(ped)
        return pedestrians

    for _ in range(count):
        (sx, sy), (gx, gy) = _sample_initial_pedestrian_route(scenario, rng)
        ped = Pedestrian(
            x=sx,
            y=sy,
            vx=0.0,
            vy=0.0,
            goal_x=gx,
            goal_y=gy,
        )
        _apply_initial_non_group_profile(ped, scenario.scenario_id, rng)
        ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
        _seed_initial_velocity(ped, scenario.scenario_id, rng)
        pedestrians.append(ped)
    return pedestrians


def respawn_family_group_members(members, scenario, nav_grid, rng):
    if not members:
        return

    template_member = members[0]
    if template_member.spawn_edge is None or template_member.goal_edge is None:
        return

    obstacle_list = scenario.obstacles
    spawn_anchor = _sample_edge_point_safe(template_member.spawn_edge, rng, obstacle_list)
    goal_anchor = _sample_edge_point_safe(template_member.goal_edge, rng, obstacle_list)
    positions = _sample_group_member_positions(
        spawn_anchor,
        len(members),
        rng,
        spread=template_member.group_spawn_spread,
        obstacles=obstacle_list,
    )
    group_goal = None
    for _ in range(40):
        candidate_goal = _clamp_world_point(
            goal_anchor[0] + float(
                rng.uniform(-template_member.group_goal_spread, template_member.group_goal_spread)
            ),
            goal_anchor[1] + float(
                rng.uniform(-template_member.group_goal_spread, template_member.group_goal_spread)
            ),
        )
        if not _point_hits_obstacles(candidate_goal[0], candidate_goal[1], obstacle_list, 10):
            group_goal = candidate_goal
            break
    if group_goal is None:
        raw_goal = _clamp_world_point(goal_anchor[0], goal_anchor[1])
        goal_region = pygame.Rect(
            int(max(0, raw_goal[0] - template_member.group_goal_spread)),
            int(max(0, raw_goal[1] - template_member.group_goal_spread)),
            int(template_member.group_goal_spread * 2),
            int(template_member.group_goal_spread * 2),
        )
        group_goal = random_point_in_region(
            goal_region,
            rng,
            margin=0,
            obstacles=obstacle_list,
        )

    for ped, (sx, sy) in zip(members, positions):
        if _point_hits_obstacles(sx, sy, obstacle_list, ped.radius):
            safe_pos = None
            for _ in range(40):
                cx, cy = _clamp_world_point(
                    spawn_anchor[0] + float(
                        rng.uniform(-template_member.group_spawn_spread, template_member.group_spawn_spread)
                    ),
                    spawn_anchor[1] + float(
                        rng.uniform(-template_member.group_spawn_spread, template_member.group_spawn_spread)
                    ),
                )
                if not _point_hits_obstacles(cx, cy, obstacle_list, ped.radius):
                    safe_pos = (cx, cy)
                    break
            if safe_pos is None:
                spawn_region = scenario.pedestrian_spawn_regions[
                    int(rng.integers(0, len(scenario.pedestrian_spawn_regions)))
                ]
                safe_pos = random_point_in_region(spawn_region, rng, obstacles=obstacle_list)
            sx, sy = safe_pos

        gx, gy = group_goal
        ped.x = sx
        ped.y = sy
        ped.vx = 0.0
        ped.vy = 0.0
        ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
