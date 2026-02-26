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

    def move_with_obstacles(
        self,
        delta: pygame.Vector2,
        obstacles: list[pygame.Rect],
    ) -> None:
        old_x, old_y = self.x, self.y

        # Resolve x movement first, then y movement to allow sliding behavior.
        self.x += delta.x
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        if self._collides_any(obstacles):
            self.x = old_x

        self.y += delta.y
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))
        if self._collides_any(obstacles):
            self.y = old_y

    def _collides_any(self, obstacles: list[pygame.Rect]) -> bool:
        hitbox = pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )
        return any(hitbox.colliderect(obstacle) for obstacle in obstacles)
