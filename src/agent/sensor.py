"""Ray-based sensor for partial observability."""

import math
import numpy as np
import pygame

HIT_NOTHING = 0
HIT_OBSTACLE = 1
HIT_PEDESTRIAN = 2
HIT_WALL = 3


class RaySensor:
    """
    casts rays from robot position and reports distances/type
    
    observation per ray: [distance (normalized), hit_type (one-hot or int)]
    """
    
    def __init__(
        self,
        num_rays: int = 36,
        max_range: float = 150.0,
        fov_degrees: float = 360.0,
        screen_width: int = 960,
        screen_height: int = 640,
    ):
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov = math.radians(fov_degrees)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # precompute ray angles
        if fov_degrees >= 360.0:
            self.angles = [2 * math.pi * i / num_rays for i in range(num_rays)]
        else:
            # partial FOV centered on forward direction
            half_fov = self.fov / 2
            self.angles = [
                -half_fov + (self.fov * i / (num_rays - 1))
                for i in range(num_rays)
            ]
    
    def cast_rays(
        self,
        robot_x: float,
        robot_y: float,
        pedestrians: list,
        obstacles: list[pygame.Rect],
    ) -> np.ndarray:
        """
        cast all rays and return observations.
        
        returns: array of shape (num_rays, 2) with [normalized_dist, hit_type]
        """
        results = []
        
        for angle in self.angles:
            dist, hit_type = self._cast_single_ray(
                robot_x, robot_y, angle, pedestrians, obstacles
            )
            normalized_dist = dist / self.max_range
            results.append([normalized_dist, hit_type / 3.0])
        
        return np.array(results, dtype=np.float32)
    
    def cast_rays_flat(
        self,
        robot_x: float,
        robot_y: float,
        pedestrians: list,
        obstacles: list[pygame.Rect],
    ) -> np.ndarray:
        """returns flattened observation vector."""
        return self.cast_rays(robot_x, robot_y, pedestrians, obstacles).flatten()
    
    def get_visible_pedestrians(
        self,
        robot_x: float,
        robot_y: float,
        pedestrians: list,
        obstacles: list[pygame.Rect],
    ) -> list:
        """return pedestrians that are hit by at least one ray."""
        visible_ids = set()
        visible = []
        
        for angle in self.angles:
            dist, hit_type = self._cast_single_ray(
                robot_x, robot_y, angle, pedestrians, obstacles
            )
            if hit_type == HIT_PEDESTRIAN:
                end_x = robot_x + math.cos(angle) * dist
                end_y = robot_y + math.sin(angle) * dist
                for ped in pedestrians:
                    if math.hypot(end_x - ped.x, end_y - ped.y) <= ped.radius + 2:
                        visible_ids.add(id(ped))
                        visible.append(ped)
                        break
        
        return list(visible)

    def _cast_single_ray(
        self,
        ox: float,
        oy: float,
        angle: float,
        pedestrians: list,
        obstacles: list[pygame.Rect],
    ) -> tuple[float, int]:
        """cast one ray, return (distance, hit_type)."""
        
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        closest_dist = self.max_range
        closest_type = HIT_NOTHING
        
        # check walls (screen boundaries)
        wall_dist = self._ray_vs_walls(ox, oy, dx, dy)
        if wall_dist < closest_dist:
            closest_dist = wall_dist
            closest_type = HIT_WALL
        
        # check obstacles
        for rect in obstacles:
            dist = self._ray_vs_rect(ox, oy, dx, dy, rect)
            if dist is not None and dist < closest_dist:
                closest_dist = dist
                closest_type = HIT_OBSTACLE
        
        # check pedestrians
        for ped in pedestrians:
            dist = self._ray_vs_circle(ox, oy, dx, dy, ped.x, ped.y, ped.radius)
            if dist is not None and dist < closest_dist:
                closest_dist = dist
                closest_type = HIT_PEDESTRIAN
        
        return closest_dist, closest_type
    
    def _ray_vs_walls(self, ox: float, oy: float, dx: float, dy: float) -> float:
        """distance to screen boundary."""
        dists = []
        
        # left wall (x = 0)
        if dx < 0:
            dists.append(-ox / dx)
        # right wall (x = width)
        if dx > 0:
            dists.append((self.screen_width - ox) / dx)
        # top wall (y = 0)
        if dy < 0:
            dists.append(-oy / dy)
        # bottom wall (y = height)
        if dy > 0:
            dists.append((self.screen_height - oy) / dy)
        
        return min(dists) if dists else self.max_range
    
    def _ray_vs_rect(
        self,
        ox: float, oy: float,
        dx: float, dy: float,
        rect: pygame.Rect,
    ) -> float | None:
        """ray-rectangle intersection using slab method."""
        
        # avoid division by zero
        if abs(dx) < 1e-9:
            dx = 1e-9 if dx >= 0 else -1e-9
        if abs(dy) < 1e-9:
            dy = 1e-9 if dy >= 0 else -1e-9
        
        t1 = (rect.left - ox) / dx
        t2 = (rect.right - ox) / dx
        t3 = (rect.top - oy) / dy
        t4 = (rect.bottom - oy) / dy
        
        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))
        
        # no intersection
        if tmax < 0 or tmin > tmax:
            return None
        
        # return nearest positive intersection
        t = tmin if tmin >= 0 else tmax
        if t < 0 or t > self.max_range:
            return None
        
        return t
    
    def _ray_vs_circle(
        self,
        ox: float, oy: float,
        dx: float, dy: float,
        cx: float, cy: float,
        radius: float,
    ) -> float | None:
        """ray-circle intersection."""
        
        # vector from ray origin to circle center
        fx = ox - cx
        fy = oy - cy
        
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        if t1 >= 0 and t1 <= self.max_range:
            return t1
        if t2 >= 0 and t2 <= self.max_range:
            return t2
        
        return None
    
    def get_ray_endpoints(
        self,
        robot_x: float,
        robot_y: float,
        pedestrians: list,
        obstacles: list[pygame.Rect],
    ) -> list[tuple[float, float, float, float, int]]:
        """
        for vis: returns list of (start_x, start_y, end_x, end_y, hit_type)
        """
        endpoints = []
        
        for angle in self.angles:
            dist, hit_type = self._cast_single_ray(
                robot_x, robot_y, angle, pedestrians, obstacles
            )
            end_x = robot_x + math.cos(angle) * dist
            end_y = robot_y + math.sin(angle) * dist
            endpoints.append((robot_x, robot_y, end_x, end_y, hit_type))
        
        return endpoints


def draw_rays(
    surface: pygame.Surface,
    endpoints: list[tuple[float, float, float, float, int]],
    alpha: int = 80,
):
    """draw rays on a pygame surface for debugging"""
    colors = {
        HIT_NOTHING: (100, 100, 100),
        HIT_OBSTACLE: (200, 100, 50),
        HIT_PEDESTRIAN: (255, 50, 50),
        HIT_WALL: (50, 50, 200),
    }
    
    for x1, y1, x2, y2, hit_type in endpoints:
        color = colors.get(hit_type, (100, 100, 100))
        pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), 1)
        pygame.draw.circle(surface, color, (int(x2), int(y2)), 3)