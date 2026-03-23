from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pygame
import numpy as np

from constants import HEIGHT, WIDTH
from environment.behaviors import (
    PedestrianBehavior,
    SocialForceBehavior,
    StationaryBehavior,
    RandomWalkerBehavior,
    FamilyGroupBehavior,
    ClumpBehavior,
    ZigzagBehavior,
)
from environment.pedestrian import Pedestrian
from environment.pathfinding import NavGrid


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


def _tuple4(raw: list[int]) -> tuple[int, int, int, int]:
    return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])


def _build_behavior(behavior_spec: dict[str, int | str]) -> tuple[PedestrianBehavior, list[int] | None]:
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
        movement_prob = float(behavior_spec.get("movement_probability", 0.01))
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


def load_scenario_templates(config_dir: Path = SCENARIO_CONFIG_DIR) -> dict[str, ScenarioTemplate]:
    templates: dict[str, ScenarioTemplate] = {}
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


def _rect_to_safe_bounds(x: int, y: int, w: int, h: int) -> pygame.Rect:
    w = max(20, min(w, WIDTH - 20))
    h = max(20, min(h, HEIGHT - 20))
    x = max(0, min(x, WIDTH - w))
    y = max(0, min(y, HEIGHT - h))
    return pygame.Rect(x, y, w, h)


def _randomized_obstacle(
    rect: tuple[int, int, int, int],
    rng: np.random.Generator,
    position_jitter: int,
    size_jitter: int,
) -> pygame.Rect:
    x, y, w, h = rect
    if position_jitter > 0:
        x += int(rng.integers(-position_jitter, position_jitter + 1))
        y += int(rng.integers(-position_jitter, position_jitter + 1))
    if size_jitter > 0:
        w += int(rng.integers(-size_jitter, size_jitter + 1))
        h += int(rng.integers(-size_jitter, size_jitter + 1))
    return _rect_to_safe_bounds(x, y, w, h)


def _circle_hits_rect(center: tuple[float, float], radius: int, rect: pygame.Rect) -> bool:
    cx, cy = center
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius


def _sample_extra_obstacles(
    template: ScenarioTemplate,
    obstacles: list[pygame.Rect],
    rng: np.random.Generator,
) -> list[pygame.Rect]:
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


def build_scenario(
    template: ScenarioTemplate,
    rng: np.random.Generator,
    randomize_world: bool = False,
) -> Scenario:
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


def random_point_in_region(
    region: pygame.Rect,
    rng: np.random.Generator,
    obstacles: list[pygame.Rect] | None = None,
    margin: int = 12,
    max_attempts: int = 40,
) -> tuple[float, float]:
    left = region.left + margin
    right = region.right - margin
    top = region.top + margin
    bottom = region.bottom - margin

    for _ in range(max_attempts):
        x = float(rng.uniform(left, right))
        y = float(rng.uniform(top, bottom))
        if not obstacles:
            return x, y

        if not any(
            _circle_hits_rect((x, y), 10, obs)
            for obs in obstacles
        ):
            return x, y

    # Second pass: deterministic grid fallback to avoid obstacles
    step = max(6, margin)
    for yy in range(int(top), int(bottom) + 1, step):
        for xx in range(int(left), int(right) + 1, step):
            if not obstacles or not any(
                _circle_hits_rect((float(xx), float(yy)), 10, obs)
                for obs in obstacles
            ):
                return float(xx), float(yy)

    # As a safe final fallback, return a valid point within region anyway
    return float(left), float(top)


def random_pedestrian_route(
    scenario: Scenario,
    rng: np.random.Generator,
    goal_region_indices: list[int] | None = None,
    obstacles: list[pygame.Rect] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Pick a random spawn and goal for a pedestrian.
    
    Args:
        scenario: The scenario with spawn/goal regions
        rng: Random number generator
        goal_region_indices: If provided, only pick goals from these region indices.
                            If None, all goal regions are available.
        obstacles: Optional obstacle list to avoid during spawn.
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
    
    spawn = random_point_in_region(spawn_region, rng, obstacles=obstacles)
    goal = random_point_in_region(goal_region, rng, obstacles=obstacles)
    return spawn, goal


def _clamp_world_point(x: float, y: float, padding: int = 14) -> tuple[float, float]:
    return (
        float(max(padding, min(WIDTH - padding, x))),
        float(max(padding, min(HEIGHT - padding, y))),
    )


def _sample_edge_point(
    edge: str,
    rng: np.random.Generator,
    padding: int = 18,
) -> tuple[float, float]:
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


def _sample_edge_point_safe(
    edge: str,
    rng: np.random.Generator,
    obstacles: list[pygame.Rect] | None,
    padding: int = 18,
    max_attempts: int = 40,
) -> tuple[float, float]:
    if not obstacles:
        return _sample_edge_point(edge, rng, padding)

    for _ in range(max_attempts):
        x, y = _sample_edge_point(edge, rng, padding)
        if not any(_circle_hits_rect((x, y), 10, obs) for obs in obstacles):
            return x, y

    # Fallback to non-safe if we cannot find in a reasonable number of tries
    return _sample_edge_point(edge, rng, padding)


def _sample_group_member_positions(
    anchor: tuple[float, float],
    size: int,
    rng: np.random.Generator,
    spread: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for _ in range(size):
        px = anchor[0] + float(rng.uniform(-spread, spread))
        py = anchor[1] + float(rng.uniform(-spread, spread))
        points.append(_clamp_world_point(px, py))
    return points


def _normalize_group_specs(
    raw_specs: list[dict[str, int | float | str]],
) -> tuple[list[dict[str, int | float | str]], np.ndarray] | None:
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


def _generate_family_groups(
    scenario: Scenario,
    template: ScenarioTemplate,
    nav_grid: NavGrid,
    rng: np.random.Generator,
    target_count: int,
) -> list[Pedestrian]:
    normalized = _normalize_group_specs(template.pedestrian_groups)
    if normalized is None or target_count <= 0:
        return []

    group_specs, probabilities = normalized
    pedestrians: list[Pedestrian] = []
    next_group_id = 0

    while len(pedestrians) < target_count:
        spec_idx = int(rng.choice(len(group_specs), p=probabilities))
        spec = group_specs[spec_idx]
        size_min = int(spec.get("size_min", 2))
        size_max = int(spec.get("size_max", size_min))
        group_size = int(rng.integers(size_min, size_max + 1))
        remaining = target_count - len(pedestrians)
        if remaining < group_size and remaining >= 2:
            group_size = remaining

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
        )
        group_goal = _clamp_world_point(
            goal_anchor[0] + float(rng.uniform(-goal_spread, goal_spread)),
            goal_anchor[1] + float(rng.uniform(-goal_spread, goal_spread)),
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
            pedestrians.append(ped)

        next_group_id += 1

    return pedestrians


def generate_pedestrian_population(
    scenario: Scenario,
    template: ScenarioTemplate,
    nav_grid: NavGrid,
    rng: np.random.Generator,
    count: int = 12,
) -> list[Pedestrian]:
    if template.pedestrian_groups:
        return _generate_family_groups(scenario, template, nav_grid, rng, count)

    pedestrians: list[Pedestrian] = []
    if template.pedestrian_behaviors:
        for behavior_spec in template.pedestrian_behaviors:
            behavior_count = int(behavior_spec.get("count", 0))
            for _ in range(behavior_count):
                behavior, goal_region_indices = _build_behavior(behavior_spec)
                (sx, sy), (gx, gy) = random_pedestrian_route(
                    scenario,
                    rng,
                    goal_region_indices,
                    obstacles=scenario.obstacles,
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
                ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
                pedestrians.append(ped)
        return pedestrians

    for _ in range(count):
        (sx, sy), (gx, gy) = random_pedestrian_route(
            scenario,
            rng,
            obstacles=scenario.obstacles,
        )
        ped = Pedestrian(
            x=sx,
            y=sy,
            vx=0.0,
            vy=0.0,
            goal_x=gx,
            goal_y=gy,
        )
        ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
        pedestrians.append(ped)
    return pedestrians


def respawn_family_group_members(
    members: list[Pedestrian],
    nav_grid: NavGrid,
    rng: np.random.Generator,
    obstacles: list[pygame.Rect] | None = None,
) -> None:
    if not members:
        return

    template_member = members[0]
    if template_member.spawn_edge is None or template_member.goal_edge is None:
        return

    spawn_anchor = _sample_edge_point_safe(template_member.spawn_edge, rng, obstacles)
    goal_anchor = _sample_edge_point_safe(template_member.goal_edge, rng, obstacles)

    positions = _sample_group_member_positions(
        spawn_anchor,
        len(members),
        rng,
        spread=template_member.group_spawn_spread,
    )

    # Ensure group goal is not inside an obstacle
    group_goal = None
    for _attempt in range(40):
        candidate_goal = _clamp_world_point(
            goal_anchor[0] + float(rng.uniform(-template_member.group_goal_spread, template_member.group_goal_spread)),
            goal_anchor[1] + float(rng.uniform(-template_member.group_goal_spread, template_member.group_goal_spread)),
        )
        if not any(_circle_hits_rect(candidate_goal, 10, obs) for obs in nav_grid.obstacles):
            group_goal = candidate_goal
            break

    if group_goal is None:
        group_goal = _clamp_world_point(
            goal_anchor[0] + float(rng.uniform(-template_member.group_goal_spread, template_member.group_goal_spread)),
            goal_anchor[1] + float(rng.uniform(-template_member.group_goal_spread, template_member.group_goal_spread)),
        )

    for ped, (sx, sy) in zip(members, positions):
        # Ensure individual spawn positions are obstacle-free
        safe_position_found = False
        for _attempt in range(40):
            candidate_x, candidate_y = sx, sy
            if any(_circle_hits_rect((candidate_x, candidate_y), ped.radius, obs) for obs in nav_grid.obstacles):
                candidate_x, candidate_y = _clamp_world_point(
                    template_member.group_spawn_spread + float(rng.uniform(-template_member.group_spawn_spread, template_member.group_spawn_spread)),
                    template_member.group_spawn_spread + float(rng.uniform(-template_member.group_spawn_spread, template_member.group_spawn_spread)),
                )
                continue
            sx, sy = candidate_x, candidate_y
            safe_position_found = True
            break

        if not safe_position_found:
            # fallback to Pedestrian.respawn (solid obstacle avoidance path)
            ped.respawn(rng, obstacles=nav_grid.obstacles, nav_grid=nav_grid)
            continue

        ped.x = sx
        ped.y = sy
        ped.vx = 0.0
        ped.vy = 0.0
        gx, gy = group_goal
        ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
