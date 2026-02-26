"""Minimal pygame simulation loop for crowd-navigation experiments."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import pygame
import numpy as np

from constants import HEIGHT, WIDTH
from environment.pedestrian import Pedestrian
from environment.robot import Robot
from environment.scenarios import (
    SCENARIO_CONFIG_DIR,
    Scenario,
    ScenarioTemplate,
    build_scenario,
    load_scenario_templates,
    random_pedestrian_route,
)
from agent.behaviors import BEHAVIORS, ControlMode

FPS = 60
BACKGROUND_COLOR = (245, 247, 240)
HUD_TEXT_COLOR = (45, 45, 45)
OBSTACLE_COLOR = (165, 170, 185)
SCENARIO_TEXT_COLOR = (70, 70, 70)

CLOSE_RADIUS = 48
NEAR_PENALTY = 0.1
GOAL_RADIUS = 20
OVERLAP_PENALTY_LOW = 0.5
OVERLAP_PENALTY_MID = 1.0
OVERLAP_PENALTY_HIGH = 1.5

# set control mode with enum value
MODE = ControlMode.POTENTIAL_FIELD
DEFAULT_SCENARIO_ID = "airport"
DEFAULT_SEED = 42


def compute_penalty(robot: Robot, pedestrians: list[Pedestrian]) -> float:
    frame_penalty = 0.0

    for ped in pedestrians:
        distance = pygame.Vector2(robot.x - ped.x, robot.y - ped.y).length()
        overlap_distance = robot.radius + ped.radius

        if distance < overlap_distance:
            overlap_ratio = (overlap_distance - distance) / overlap_distance
            if overlap_ratio < 0.33:
                frame_penalty += OVERLAP_PENALTY_LOW
            elif overlap_ratio < 0.66:
                frame_penalty += OVERLAP_PENALTY_MID
            else:
                frame_penalty += OVERLAP_PENALTY_HIGH
        elif distance < CLOSE_RADIUS:
            frame_penalty += NEAR_PENALTY

    return frame_penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crowd navigation simulation")
    parser.add_argument(
        "--scenario",
        type=str,
        default=os.getenv("CROWD_SIM_SCENARIO", DEFAULT_SCENARIO_ID),
        help="Scenario id from scenario config files (home, airport, shopping_center).",
    )
    parser.add_argument(
        "--pedestrians",
        type=int,
        default=int(os.getenv("CROWD_SIM_PEDESTRIANS", "12")),
        help="Number of pedestrians to spawn.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed for numpy random generator.",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        default=os.getenv("CROWD_SIM_RANDOM_SEED", "0") == "1",
        help="Use entropy-based random seed instead of deterministic seed.",
    )
    parser.add_argument(
        "--random-world",
        action="store_true",
        default=os.getenv("CROWD_SIM_RANDOM_WORLD", "0") == "1",
        help="Randomize obstacle layouts per scenario template and episode.",
    )
    parser.add_argument(
        "--scenario-config-dir",
        type=str,
        default=os.getenv("CROWD_SIM_SCENARIO_DIR", str(SCENARIO_CONFIG_DIR)),
        help="Path to scenario JSON config directory.",
    )
    return parser.parse_args()


def init_rng(seed: int | None, random_seed: bool) -> tuple[np.random.Generator, int]:
    if random_seed:
        seed_sequence = np.random.SeedSequence()
        entropy = int(seed_sequence.entropy)
        return np.random.default_rng(seed_sequence), entropy
    chosen_seed = DEFAULT_SEED if seed is None else seed
    return np.random.default_rng(chosen_seed), chosen_seed


def generate_pedestrians(
    scenario: Scenario,
    rng: np.random.Generator,
    count: int = 12,
) -> list[Pedestrian]:
    peds = []
    for _ in range(count):
        (sx, sy), (gx, gy) = random_pedestrian_route(scenario, rng)
        peds.append(
            Pedestrian(
                x = sx,
                y = sy,
                vx = 0.0,
                vy = 0.0,
                goal_x = gx,
                goal_y = gy,
            )
        )
    return peds


def reassign_reached_goals(
    pedestrians: list[Pedestrian],
    scenario: Scenario,
    rng: np.random.Generator,
) -> None:
    for ped in pedestrians:
        if ped.has_reached_goal():
            (sx, sy), (gx, gy) = random_pedestrian_route(scenario, rng)
            ped.x = sx
            ped.y = sy
            ped.vx = 0.0
            ped.vy = 0.0
            ped.goal_x = gx
            ped.goal_y = gy


def draw_scenario(screen: pygame.Surface, scenario: Scenario) -> None:
    for obstacle in scenario.obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle, border_radius=4)


def build_episode_state(
    template: ScenarioTemplate,
    rng: np.random.Generator,
    pedestrian_count: int,
    random_world: bool,
) -> tuple[Scenario, Robot, list[Pedestrian], pygame.Vector2]:
    scenario = build_scenario(template, rng, randomize_world=random_world)
    robot = Robot(x=scenario.robot_start[0], y=scenario.robot_start[1])
    pedestrians = generate_pedestrians(scenario, rng, count=pedestrian_count)
    goal_pos = pygame.Vector2(*scenario.robot_goal)
    return scenario, robot, pedestrians, goal_pos


def run() -> None:
    args = parse_args()
    rng, used_seed = init_rng(seed=args.seed, random_seed=args.random_seed)
    templates = load_scenario_templates(Path(args.scenario_config_dir))
    scenario_ids = list(templates.keys())
    if args.scenario not in templates:
        raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {', '.join(scenario_ids)}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Crowd Navigation Sandbox")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    sub_font = pygame.font.Font(None, 24)

    current_scenario_id = args.scenario
    current_template = templates[current_scenario_id]
    scenario, robot, pedestrians, goal_pos = build_episode_state(
        current_template,
        rng=rng,
        pedestrian_count=args.pedestrians,
        random_world=args.random_world,
    )
    total_penalty = 0.0

    episode = 0
    steps = 0
    total_penalties = []
    total_steps = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    current_scenario_id = "home"
                elif event.key == pygame.K_2:
                    current_scenario_id = "airport"
                elif event.key == pygame.K_3:
                    current_scenario_id = "shopping_center"
                else:
                    continue

                if current_scenario_id not in templates:
                    continue
                current_template = templates[current_scenario_id]
                scenario, robot, pedestrians, goal_pos = build_episode_state(
                    current_template,
                    rng=rng,
                    pedestrian_count=args.pedestrians,
                    random_world=args.random_world,
                )
                total_penalty = 0.0
                steps = 0

        keys = pygame.key.get_pressed()
        move = BEHAVIORS[MODE](robot, goal_pos, pedestrians, keys)
        steps += 1

        if move.length_squared() > 0:
            move = move.normalize() * robot.speed
            robot.move_with_obstacles(move, scenario.obstacles)

        for ped in pedestrians:
            ped.update(pedestrians, scenario.obstacles, rng=rng)
        reassign_reached_goals(pedestrians, scenario, rng=rng)

        total_penalty += compute_penalty(robot, pedestrians)
        distance_to_goal = pygame.Vector2(robot.x - goal_pos.x, robot.y - goal_pos.y).length()
        
        if distance_to_goal < GOAL_RADIUS:
            episode += 1
            total_penalties.append(total_penalty)
            total_steps.append(steps)
            
            avg_penalty = sum(total_penalties) / len(total_penalties)
            avg_steps = sum(total_steps) / len(total_steps)
            
            print(f"Episode {episode}: penalty={total_penalty:.1f}, steps={steps}")
            print(f"  Averages: penalty={avg_penalty:.1f}, steps={avg_steps:.1f}")

            # Reset for next episode
            scenario, robot, pedestrians, goal_pos = build_episode_state(
                current_template,
                rng=rng,
                pedestrian_count=args.pedestrians,
                random_world=args.random_world,
            )
            total_penalty = 0.0
            steps = 0

        screen.fill(BACKGROUND_COLOR)
        draw_scenario(screen, scenario)
        pygame.draw.circle(screen, (245, 130, 40), goal_pos, 16)
        robot.draw(screen)
        for ped in pedestrians:
            ped.draw(screen)

        penalty_label = font.render(f"Penalty: {total_penalty:.1f}", True, HUD_TEXT_COLOR)
        screen.blit(penalty_label, (20, 20))
        scenario_label = sub_font.render(
            f"Scenario: {scenario.name} (1:Home 2:Airport 3:Shopping)",
            True,
            SCENARIO_TEXT_COLOR,
        )
        screen.blit(scenario_label, (20, 52))
        seed_label = sub_font.render(
            f"Seed: {used_seed} | random_world={args.random_world}",
            True,
            SCENARIO_TEXT_COLOR,
        )
        screen.blit(seed_label, (20, 76))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run()