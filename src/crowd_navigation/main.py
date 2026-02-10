"""Minimal pygame simulation loop for crowd-navigation experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pygame

WIDTH = 960
HEIGHT = 640
FPS = 60

ROBOT_COLOR = (45, 90, 255)
PEDESTRIAN_COLOR = (10, 155, 110)
BACKGROUND_COLOR = (245, 247, 240)


@dataclass
class Pedestrian:
    x: float
    y: float
    vx: float
    vy: float
    radius: int = 10

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        if self.x < self.radius or self.x > WIDTH - self.radius:
            self.vx *= -1
        if self.y < self.radius or self.y > HEIGHT - self.radius:
            self.vy *= -1

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, PEDESTRIAN_COLOR, (int(self.x), int(self.y)), self.radius)


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

    robot_pos = pygame.Vector2(80, HEIGHT // 2)
    goal_pos = pygame.Vector2(WIDTH - 80, HEIGHT // 2)
    robot_speed = 3
    robot_radius = 12
    pedestrians = generate_pedestrians()

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
            move = move.normalize() * robot_speed
            robot_pos += move
            robot_pos.x = max(robot_radius, min(WIDTH - robot_radius, robot_pos.x))
            robot_pos.y = max(robot_radius, min(HEIGHT - robot_radius, robot_pos.y))

        for ped in pedestrians:
            ped.update()

        screen.fill(BACKGROUND_COLOR)
        pygame.draw.circle(screen, (245, 130, 40), goal_pos, 16)
        pygame.draw.circle(screen, ROBOT_COLOR, robot_pos, robot_radius)
        for ped in pedestrians:
            ped.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run()

