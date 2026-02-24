from dataclasses import dataclass
import math
import random
import pygame

from constants import HEIGHT, WIDTH

PEDESTRIAN_COLOR = (10, 155, 110)
GOAL_COLOR = (255, 200, 0)

@dataclass
class Pedestrian:
    x: float
    y: float
    vx: float
    vy: float
    goal_x: float
    goal_y: float
    radius: int = 10

    desired_speed: float = 1.5
    relaxation_time: float = 30.0
    max_speed: float = 3.0

    ped_A: float = 8.0
    ped_B: float = 8.0

    wall_A: float = 0.0
    wall_B: float = 5.0

    def _self_driving_force(self) -> tuple[float, float]:
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0
        ex, ey = dx / dist, dy / dist
        fx = (self.desired_speed * ex - self.vx) / self.relaxation_time
        fy = (self.desired_speed * ey - self.vy) / self.relaxation_time
        return fx, fy

    def _pedestrian_repulsion(self, others: list["Pedestrian"]) -> tuple[float, float]:
        fx, fy = 0.0, 0.0
        for other in others:
            if other is self:
                continue
            if other.y > HEIGHT + other.radius:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                dist = 1e-6
            nx, ny = dx / dist, dy / dist
            r_ij = self.radius + other.radius
            magnitude = self.ped_A * math.exp((r_ij - dist) / self.ped_B)
            fx += magnitude * nx
            fy += magnitude * ny
        return fx, fy

    def _wall_repulsion(self) -> tuple[float, float]:
        fx, fy = 0.0, 0.0
        walls = [
            (self.x,           1.0,  0.0),
            (WIDTH - self.x,  -1.0,  0.0),
        ]
        for dist, nx, ny in walls:
            if dist < 1e-6:
                dist = 1e-6
            magnitude = self.wall_A * math.exp((self.radius - dist) / self.wall_B)
            fx += magnitude * nx
            fy += magnitude * ny
        return fx, fy
    
    def respawn(self) -> None:
        self.x = random.uniform(self.radius, WIDTH - self.radius)
        self.y = HEIGHT + self.radius
        self.vx = 0.0
        self.vy = 0.0
        self.goal_x = random.uniform(self.radius, WIDTH - self.radius)

    def update(self, others: list["Pedestrian"]) -> None:
        f1x, f1y = self._self_driving_force()
        f2x, f2y = self._pedestrian_repulsion(others)
        f3x, f3y = self._wall_repulsion()

        self.vx += f1x + f2x + f3x
        self.vy += f1y + f2y + f3y

        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            self.vx = self.vx / speed * self.max_speed
            self.vy = self.vy / speed * self.max_speed

        self.x += self.vx
        self.y += self.vy

    def has_reached_goal(self, threshold: float = 15.0) -> bool:
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y) < threshold

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, GOAL_COLOR, (int(self.goal_x), int(self.goal_y)), 4)
        pygame.draw.line(
            surface, (255, 255, 255),
            (int(self.x), int(self.y)),
            (int(self.x + self.vx * 8), int(self.y + self.vy * 8)), 2
        )
        pygame.draw.circle(surface, PEDESTRIAN_COLOR, (int(self.x), int(self.y)), self.radius)
