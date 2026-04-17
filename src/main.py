import argparse
import os
from pathlib import Path
import pygame
import numpy as np

from constants import HEIGHT, WIDTH
from environment.robot import Robot
from environment.pathfinding import NavGrid
from environment.scenarios import (
    SCENARIO_CONFIG_DIR,
    build_scenario,
    generate_pedestrian_population,
    load_scenario_templates,
)
from environment.pedestrian_lifecycle import (
    reassign_reached_goals,
    select_flow_pedestrians,
)
from agent.behaviors import BEHAVIORS, ControlMode
from agent.sensor import RaySensor, draw_rays

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

DEFAULT_MODE = "potential_field"
CONTROL_MODE_BY_NAME = {
    "manual": ControlMode.MANUAL,
    "naive": ControlMode.NAIVE,
    "random": ControlMode.RANDOM,
    "potential_field": ControlMode.POTENTIAL_FIELD,
}

SHOW_RAY_TRACING = True

DEFAULT_SCENARIO_ID = "shopping_center"
DEFAULT_SEED = 42


def compute_penalty(robot, pedestrians):
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


def parse_args():
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
        "--mode",
        type=str,
        choices=sorted(CONTROL_MODE_BY_NAME.keys()),
        default=os.getenv("CROWD_SIM_MODE", DEFAULT_MODE),
        help="Robot control mode.",
    )
    parser.add_argument(
        "--scenario-config-dir",
        type=str,
        default=os.getenv("CROWD_SIM_SCENARIO_DIR", str(SCENARIO_CONFIG_DIR)),
        help="Path to scenario JSON config directory.",
    )
    return parser.parse_args()


def init_rng(seed, random_seed):
    if random_seed:
        seed_sequence = np.random.SeedSequence()
        entropy = int(seed_sequence.entropy)
        return np.random.default_rng(seed_sequence), entropy
    chosen_seed = DEFAULT_SEED if seed is None else seed
    return np.random.default_rng(chosen_seed), chosen_seed


def generate_pedestrians(scenario, template, nav_grid, rng, count = 12):
    return generate_pedestrian_population(
        scenario,
        template,
        nav_grid,
        rng,
        count=count,
    )


def draw_scenario(screen, scenario):
    for obstacle in scenario.obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle, border_radius=4)


def build_episode_state(template, rng, pedestrian_count, random_world):
    scenario = build_scenario(template, rng, randomize_world=random_world)
    nav_grid = NavGrid(WIDTH, HEIGHT, scenario.obstacles)
    robot = Robot(x=scenario.robot_start[0], y=scenario.robot_start[1])
    pedestrians = generate_pedestrians(scenario, template, nav_grid, rng, count=pedestrian_count)
    goal_pos = pygame.Vector2(*scenario.robot_goal)
    return scenario, nav_grid, robot, pedestrians, goal_pos


def run():
    args = parse_args()
    control_mode = CONTROL_MODE_BY_NAME[args.mode]
    rng, used_seed = init_rng(seed=args.seed, random_seed=args.random_seed)
    templates = load_scenario_templates(Path(args.scenario_config_dir))
    scenario_ids = list(templates.keys())
    if args.scenario not in templates:
        raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {', '.join(scenario_ids)}")

    if not pygame.display.get_init():
        pygame.display.init()
    if not pygame.font.get_init():
        pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Crowd Navigation Sandbox")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    sub_font = pygame.font.Font(None, 24)

    current_scenario_id = args.scenario
    current_template = templates[current_scenario_id]
    scenario, nav_grid, robot, pedestrians, goal_pos = build_episode_state(
        current_template,
        rng=rng,
        pedestrian_count=args.pedestrians,
        random_world=args.random_world,
    )

    ray_sensor = RaySensor(
        num_rays=36,
        max_range=150.0,
        fov_degrees=360.0,
        screen_width=WIDTH,
        screen_height=HEIGHT,
    ) 

    total_penalty = 0.0
    episode = 0
    steps = 0
    total_penalties = []
    total_steps = []
    goal_dwell_frames = {}
    pending_perimeter_respawn = set()
    flow_pedestrian_ids = select_flow_pedestrians(pedestrians, current_scenario_id, rng)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    continue
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
                scenario, nav_grid, robot, pedestrians, goal_pos = build_episode_state(
                    current_template,
                    rng=rng,
                    pedestrian_count=args.pedestrians,
                    random_world=args.random_world,
                )
                goal_dwell_frames.clear()
                pending_perimeter_respawn.clear()
                flow_pedestrian_ids = select_flow_pedestrians(
                    pedestrians,
                    current_scenario_id,
                    rng,
                )
                total_penalty = 0.0
                steps = 0

        if SHOW_RAY_TRACING:
            visible_pedestrians = ray_sensor.get_visible_pedestrians(
                robot.x, robot.y, pedestrians, scenario.obstacles
            )
        else:
            visible_pedestrians = pedestrians

        keys = pygame.key.get_pressed()
        move = BEHAVIORS[control_mode](robot, goal_pos, visible_pedestrians, keys)
        steps += 1

        if move.length_squared() > 0:
            move = move.normalize() * robot.speed
            robot.move_with_obstacles(move, scenario.obstacles)

        for ped in pedestrians:
            if goal_dwell_frames.get(id(ped), 0) > 0:
                ped.vx *= 0.5
                ped.vy *= 0.5
                continue
            ped.update(pedestrians, scenario.obstacles, rng=rng)
        reassign_reached_goals(
            pedestrians,
            scenario,
            nav_grid,
            rng=rng,
            scenario_id=current_scenario_id,
            goal_dwell_frames=goal_dwell_frames,
            pending_perimeter_respawn=pending_perimeter_respawn,
            flow_pedestrian_ids=flow_pedestrian_ids,
            sim_fps=FPS,
        )

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
            scenario, nav_grid, robot, pedestrians, goal_pos = build_episode_state(
                current_template,
                rng=rng,
                pedestrian_count=args.pedestrians,
                random_world=args.random_world,
            )
            goal_dwell_frames.clear()
            pending_perimeter_respawn.clear()
            flow_pedestrian_ids = select_flow_pedestrians(
                pedestrians,
                current_scenario_id,
                rng,
            )
            total_penalty = 0.0
            steps = 0

        screen.fill(BACKGROUND_COLOR)
        draw_scenario(screen, scenario)
        pygame.draw.circle(screen, (245, 130, 40), goal_pos, 16)
        robot.draw(screen)
        for ped in pedestrians:
            ped.draw(screen)

        if SHOW_RAY_TRACING:
            endpoints = ray_sensor.get_ray_endpoints(
                robot.x, robot.y,
                pedestrians, scenario.obstacles
            )
            draw_rays(screen, endpoints)

        penalty_label = font.render(f"Penalty: {total_penalty:.1f}", True, HUD_TEXT_COLOR)
        screen.blit(penalty_label, (20, 20))
        scenario_label = sub_font.render(
            f"Scenario: {scenario.name} (1:Home 2:Airport 3:Shopping)",
            True,
            SCENARIO_TEXT_COLOR,
        )
        screen.blit(scenario_label, (20, 52))
        seed_label = sub_font.render(
            f"Seed: {used_seed} | mode={args.mode} | random_world={args.random_world}",
            True,
            SCENARIO_TEXT_COLOR,
        )
        screen.blit(seed_label, (20, 76))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run()
