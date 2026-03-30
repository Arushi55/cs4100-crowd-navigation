from __future__ import annotations

import numpy as np

from constants import HEIGHT, WIDTH
from environment.pathfinding import NavGrid
from environment.pedestrian import Pedestrian
from environment.scenarios import (
    Scenario,
    random_point_in_region,
    random_pedestrian_route,
    respawn_family_group_members,
)

GOAL_DWELL_SECONDS_RANGE = (0.6, 2.0)
GOAL_DWELL_SECONDS_BY_REGION = {
    "home": {
        0: (1.1, 2.6),
        1: (0.4, 1.1),
        2: (1.0, 2.3),
        3: (1.4, 3.0),
    },
    "shopping_center": {
        0: (0.7, 1.5),
        1: (0.5, 1.2),
        2: (1.3, 2.8),
        3: (1.2, 2.4),
    },
    "airport": {
        0: (0.5, 1.2),
        1: (0.5, 1.2),
        2: (0.9, 1.9),
        3: (1.0, 2.1),
    },
}

PERIMETER_FLOW_RATIO_BY_SCENARIO = {
    "home": 0.05,
    "shopping_center": 0.18,
    "airport": 0.55,
}

GOAL_REGION_TRANSITION_WEIGHTS = {
    "home": [
        [0.05, 0.55, 0.20, 0.20],
        [0.50, 0.05, 0.15, 0.30],
        [0.35, 0.15, 0.05, 0.45],
        [0.30, 0.30, 0.30, 0.10],
    ],
    "shopping_center": [
        [0.05, 0.30, 0.45, 0.20],
        [0.35, 0.05, 0.40, 0.20],
        [0.30, 0.25, 0.05, 0.40],
        [0.30, 0.25, 0.40, 0.05],
    ],
    "airport": [
        [0.05, 0.55, 0.25, 0.15],
        [0.55, 0.05, 0.15, 0.25],
        [0.25, 0.15, 0.05, 0.55],
        [0.15, 0.25, 0.55, 0.05],
    ],
}

_SAME_REGION_WEIGHT_SCALE = 0.35


def select_flow_pedestrians(
    pedestrians: list[Pedestrian],
    scenario_id: str,
    rng: np.random.Generator,
) -> set[int]:
    ratio = float(PERIMETER_FLOW_RATIO_BY_SCENARIO.get(scenario_id, 0.15))
    ratio = max(0.0, min(1.0, ratio))
    candidates = [ped for ped in pedestrians if ped.group_id is None]
    if not candidates or ratio <= 0.0:
        return set()
    count = int(round(len(candidates) * ratio))
    count = max(0, min(len(candidates), count))
    if count <= 0:
        return set()
    selected = rng.choice(len(candidates), size=count, replace=False)
    return {id(candidates[int(i)]) for i in np.atleast_1d(selected)}


def goal_region_index_for_point(scenario: Scenario, x: float, y: float) -> int | None:
    for idx, region in enumerate(scenario.pedestrian_goal_regions):
        if region.collidepoint(x, y):
            return idx
    return None


def sample_goal_dwell_frames(
    ped: Pedestrian,
    scenario: Scenario,
    scenario_id: str,
    rng: np.random.Generator,
    sim_fps: int,
) -> int:
    lo_sec, hi_sec = GOAL_DWELL_SECONDS_RANGE
    scenario_overrides = GOAL_DWELL_SECONDS_BY_REGION.get(scenario_id)
    if scenario_overrides:
        region_idx = goal_region_index_for_point(scenario, ped.goal_x, ped.goal_y)
        if region_idx is not None and region_idx in scenario_overrides:
            lo_sec, hi_sec = scenario_overrides[region_idx]
    lo = max(1, int(round(lo_sec * sim_fps)))
    hi = max(lo, int(round(hi_sec * sim_fps)))
    return int(rng.integers(lo, hi + 1))


def sample_next_goal_region_index(
    scenario: Scenario,
    scenario_id: str,
    current_region_idx: int | None,
    allowed_region_indices: list[int],
    rng: np.random.Generator,
) -> int:
    if len(allowed_region_indices) == 1:
        return int(allowed_region_indices[0])

    matrix = GOAL_REGION_TRANSITION_WEIGHTS.get(scenario_id)
    if matrix is not None and current_region_idx is not None:
        if 0 <= current_region_idx < len(matrix):
            row = matrix[current_region_idx]
            if len(row) >= len(scenario.pedestrian_goal_regions):
                raw_weights: list[float] = []
                for idx in allowed_region_indices:
                    w = float(row[idx]) if 0 <= idx < len(row) else 0.0
                    if idx == current_region_idx:
                        w *= _SAME_REGION_WEIGHT_SCALE
                    raw_weights.append(max(0.0, w))
                weight_sum = float(sum(raw_weights))
                if weight_sum > 1e-9:
                    probs = np.array(raw_weights, dtype=np.float64) / weight_sum
                    choice_idx = int(rng.choice(len(allowed_region_indices), p=probs))
                    return int(allowed_region_indices[choice_idx])

    # Fallback: weakly discourage selecting the same region again.
    fallback_weights = []
    for idx in allowed_region_indices:
        if current_region_idx is not None and idx == current_region_idx:
            fallback_weights.append(_SAME_REGION_WEIGHT_SCALE)
        else:
            fallback_weights.append(1.0)
    probs = np.array(fallback_weights, dtype=np.float64)
    probs /= probs.sum()
    choice_idx = int(rng.choice(len(allowed_region_indices), p=probs))
    return int(allowed_region_indices[choice_idx])


def sample_next_goal_point(
    ped: Pedestrian,
    scenario: Scenario,
    scenario_id: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    allowed = (
        list(ped.goal_region_indices)
        if ped.goal_region_indices is not None and len(ped.goal_region_indices) > 0
        else list(range(len(scenario.pedestrian_goal_regions)))
    )
    if not allowed:
        (_, _), (gx, gy) = random_pedestrian_route(scenario, rng, ped.goal_region_indices)
        return gx, gy

    current_region = goal_region_index_for_point(scenario, ped.goal_x, ped.goal_y)
    region_idx = sample_next_goal_region_index(
        scenario,
        scenario_id,
        current_region,
        allowed,
        rng,
    )
    goal_region = scenario.pedestrian_goal_regions[region_idx]
    gx, gy = random_point_in_region(
        goal_region,
        rng,
        obstacles=scenario.obstacles,
    )
    return gx, gy


def sample_perimeter_spawn(
    ped: Pedestrian,
    scenario: Scenario,
    rng: np.random.Generator,
) -> tuple[float, float]:
    margin = ped.radius + 2.0
    obstacles = scenario.obstacles
    for _ in range(40):
        edge = int(rng.integers(0, 4))
        if edge == 0:  # left
            x = margin
            y = float(rng.uniform(margin, HEIGHT - margin))
        elif edge == 1:  # right
            x = WIDTH - margin
            y = float(rng.uniform(margin, HEIGHT - margin))
        elif edge == 2:  # top
            x = float(rng.uniform(margin, WIDTH - margin))
            y = margin
        else:  # bottom
            x = float(rng.uniform(margin, WIDTH - margin))
            y = HEIGHT - margin
        if not obstacles or not ped._would_collide(x, y, obstacles):
            return x, y
    return ped.x, ped.y


def reassign_reached_goals(
    pedestrians: list[Pedestrian],
    scenario: Scenario,
    nav_grid: NavGrid,
    rng: np.random.Generator,
    scenario_id: str,
    goal_dwell_frames: dict[int, int],
    pending_perimeter_respawn: set[int],
    flow_pedestrian_ids: set[int],
    sim_fps: int,
) -> None:
    respawned_groups: set[int] = set()
    for ped in pedestrians:
        pid = id(ped)
        dwell_frames = goal_dwell_frames.get(pid, 0)
        if dwell_frames > 0:
            dwell_frames -= 1
            if dwell_frames <= 0:
                if pid in pending_perimeter_respawn:
                    sx, sy = sample_perimeter_spawn(ped, scenario, rng)
                    ped.x, ped.y = sx, sy
                    pending_perimeter_respawn.discard(pid)
                gx, gy = sample_next_goal_point(ped, scenario, scenario_id, rng)
                ped.vx = 0.0
                ped.vy = 0.0
                ped.set_goal(gx, gy, nav_grid=nav_grid, rng=rng)
                goal_dwell_frames.pop(pid, None)
            else:
                goal_dwell_frames[pid] = dwell_frames
            continue

        if ped.has_reached_goal():
            if ped.group_id is not None and ped.group_id not in respawned_groups:
                group_members = [member for member in pedestrians if member.group_id == ped.group_id]
                respawn_family_group_members(group_members, scenario, nav_grid, rng)
                for member in group_members:
                    member_id = id(member)
                    goal_dwell_frames.pop(member_id, None)
                    pending_perimeter_respawn.discard(member_id)
                respawned_groups.add(ped.group_id)
                continue
            ped.vx = 0.0
            ped.vy = 0.0
            if pid in flow_pedestrian_ids:
                pending_perimeter_respawn.add(pid)
            goal_dwell_frames[pid] = sample_goal_dwell_frames(
                ped,
                scenario,
                scenario_id,
                rng,
                sim_fps=sim_fps,
            )
            if hasattr(ped.behavior, "frame_count"):
                ped.behavior.frame_count = 0
