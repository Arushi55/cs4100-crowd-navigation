"""Minimal pygame simulation loop for crowd-navigation experiments."""

from __future__ import annotations
import random
import pygame

from .constants import HEIGHT, WIDTH
from .pedestrian import Pedestrian
from .robot import Robot
FPS = 60
BACKGROUND_COLOR = (245, 247, 240)

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

    goal_pos = pygame.Vector2(WIDTH - 80, HEIGHT // 2)
    pedestrians = generate_pedestrians()
    robot = Robot()

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

        screen.fill(BACKGROUND_COLOR)
        pygame.draw.circle(screen, (245, 130, 40), goal_pos, 16)
        robot.draw(screen)
        for ped in pedestrians:
            ped.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run()

