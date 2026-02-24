from dataclasses import dataclass
import pygame

from constants import HEIGHT, WIDTH

PEDESTRIAN_COLOR = (10, 155, 110)

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
