from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pygame
import numpy as np

from constants import HEIGHT, WIDTH


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


def _tuple4(raw: list[int]) -> tuple[int, int, int, int]:
    return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])


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
    margin: int = 12,
) -> tuple[float, float]:
    left = region.left + margin
    right = region.right - margin
    top = region.top + margin
    bottom = region.bottom - margin
    return (
        float(rng.uniform(left, right)),
        float(rng.uniform(top, bottom)),
    )


def random_pedestrian_route(
    scenario: Scenario,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    spawn_region = scenario.pedestrian_spawn_regions[int(rng.integers(0, len(scenario.pedestrian_spawn_regions)))]
    goal_region = scenario.pedestrian_goal_regions[int(rng.integers(0, len(scenario.pedestrian_goal_regions)))]
    spawn = random_point_in_region(spawn_region, rng)
    goal = random_point_in_region(goal_region, rng)
    return spawn, goal
