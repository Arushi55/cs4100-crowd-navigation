"""Minimal pygame simulation loop for crowd-navigation experiments."""

from __future__ import annotations
import random
import pygame

from .constants import HEIGHT, WIDTH
from .pedestrian import Pedestrian
from .robot import Robot

FPS = 60
BACKGROUND_COLOR = (245, 247, 240)
HUD_TEXT_COLOR = (45, 45, 45)

CLOSE_RADIUS = 48
NEAR_PENALTY = 0.1
OVERLAP_PENALTY_LOW = 0.5
OVERLAP_PENALTY_MID = 1.0
OVERLAP_PENALTY_HIGH = 1.5


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

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        move = pygame.Vector2(0, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move.y -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move.y += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move.x -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move.x += 1
        if move.length_squared() > 0:
            move = move.normalize() * robot.speed
            robot.move(move)

        for ped in pedestrians:
            ped.update()

        total_penalty += compute_penalty(robot, pedestrians)

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

