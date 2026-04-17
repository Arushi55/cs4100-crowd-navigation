from dataclasses import dataclass
import pygame

from constants import HEIGHT, WIDTH, speed_mps_to_px_per_step

ROBOT_COLOR = (45, 90, 255)

@dataclass
class Robot:
    x: float = 80.0
    y: float =  80.0
    speed: float = speed_mps_to_px_per_step(3.0)
    radius: int = 12

    def draw(self, surface):
        pygame.draw.circle(
            surface,
            ROBOT_COLOR,
            (int(self.x), int(self.y)),
            self.radius,
        )

    def move(self, delta):
        self.x += delta.x
        self.y += delta.y
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

    def move_with_obstacles(self, delta, obstacles):
        old_x, old_y = self.x, self.y
        blocked_axes = 0

        # Resolve x movement first, then y movement to allow sliding behavior.
        attempted_x = self.x + delta.x
        clamped_x = max(self.radius, min(WIDTH - self.radius, attempted_x))
        blocked_x = abs(clamped_x - attempted_x) > 1e-6
        self.x = clamped_x
        if self._collides_any(obstacles):
            self.x = old_x
            blocked_x = True
        if blocked_x:
            blocked_axes += 1

        attempted_y = self.y + delta.y
        clamped_y = max(self.radius, min(HEIGHT - self.radius, attempted_y))
        blocked_y = abs(clamped_y - attempted_y) > 1e-6
        self.y = clamped_y
        if self._collides_any(obstacles):
            self.y = old_y
            blocked_y = True
        if blocked_y:
            blocked_axes += 1

        return blocked_axes

    def _collides_any(self, obstacles):
        hitbox = pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )
        return any(hitbox.colliderect(obstacle) for obstacle in obstacles)
