"""Minimal pygame simulation loop for crowd-navigation experiments."""

from __future__ import annotations
import random
import pygame

from constants import HEIGHT, WIDTH
from pedestrian import Pedestrian
from robot import Robot
from behaviors import BEHAVIORS, ControlMode

FPS = 60
BACKGROUND_COLOR = (245, 247, 240)
HUD_TEXT_COLOR = (45, 45, 45)

CLOSE_RADIUS = 48
NEAR_PENALTY = 0.1
GOAL_RADIUS = 20
OVERLAP_PENALTY_LOW = 0.5
OVERLAP_PENALTY_MID = 1.0
OVERLAP_PENALTY_HIGH = 1.5

MODE = ControlMode.POTENTIAL_FIELD


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

def generate_pedestrians(count: int = 12) -> list[Pedestrian]:
    return [
        Pedestrian(
            x=random.randint(40, WIDTH - 40),
            y=random.randint(40, HEIGHT - 40),
            vx=random.choice([-1, 1]) * random.uniform(0.8, 2.2),
            vy=random.choice([-1, 1]) * random.uniform(0.8, 2.2),
        )
        for _ in range(count)
    ]


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Crowd Navigation Sandbox")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    goal_pos = pygame.Vector2(WIDTH - 80, HEIGHT // 2)
    pedestrians = generate_pedestrians()
    robot = Robot()
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

        keys = pygame.key.get_pressed()
        move = BEHAVIORS[MODE](robot, goal_pos, pedestrians, keys)
        steps += 1

        if move.length_squared() > 0:
            move = move.normalize() * robot.speed
            robot.move(move)

        for ped in pedestrians:
            ped.update()

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
            robot.x, robot.y = 80, HEIGHT // 2
            pedestrians = generate_pedestrians()
            total_penalty = 0.0
            steps = 0

        screen.fill(BACKGROUND_COLOR)

        screen.fill(BACKGROUND_COLOR)
        pygame.draw.circle(screen, (245, 130, 40), goal_pos, 16)
        robot.draw(screen)
        for ped in pedestrians:
            ped.draw(screen)

        penalty_label = font.render(f"Penalty: {total_penalty:.1f}", True, HUD_TEXT_COLOR)
        screen.blit(penalty_label, (20, 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run()

