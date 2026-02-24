from dataclasses import dataclass
import pygame

from constants import HEIGHT, WIDTH

ROBOT_COLOR = (45, 90, 255)

@dataclass
class Robot:
    x: float = 80.0
    y: float =  80.0
    speed: float = 3.0
    radius: int = 12

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(
            surface,
            ROBOT_COLOR,
            (int(self.x), int(self.y)),
            self.radius,
        )

    def move(self, delta: pygame.Vector2) -> None:
        self.x += delta.x
        self.y += delta.y
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))
