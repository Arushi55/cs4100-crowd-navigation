"""A* pathfinding on a discretized navigation grid."""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
import pygame

CELL_SIZE = 10  # px per grid cell


class NavGrid:
    """
    Discretized grid for A* pathfinding.

    Obstacles are inflated by the pedestrian radius so the path
    stays comfortably clear of walls.
    """

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: list[pygame.Rect],
        inflate: int = 14,
    ):
        self.cell_size = CELL_SIZE
        self.cols = width // CELL_SIZE + 1
        self.rows = height // CELL_SIZE + 1
        self.width = width
        self.height = height

        # Build occupancy grid (True = blocked)
        self.blocked = np.zeros((self.rows, self.cols), dtype=bool)
        for rect in obstacles:
            inflated = rect.inflate(inflate * 2, inflate * 2)
            c_min = max(0, inflated.left // CELL_SIZE)
            c_max = min(self.cols, inflated.right // CELL_SIZE + 1)
            r_min = max(0, inflated.top // CELL_SIZE)
            r_max = min(self.rows, inflated.bottom // CELL_SIZE + 1)
            self.blocked[r_min:r_max, c_min:c_max] = True

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        c = max(0, min(self.cols - 1, int(x / self.cell_size)))
        r = max(0, min(self.rows - 1, int(y / self.cell_size)))
        return r, c

    def grid_to_world(self, r: int, c: int) -> tuple[float, float]:
        return (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size

    # ------------------------------------------------------------------
    # A* pathfinding
    # ------------------------------------------------------------------

    def find_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Return a list of world-coordinate waypoints from *start* to *goal*."""
        sr, sc = self.world_to_grid(*start)
        gr, gc = self.world_to_grid(*goal)

        # If start or goal sits inside an obstacle, snap to nearest free cell
        sr, sc = self._nearest_free(sr, sc)
        gr, gc = self._nearest_free(gr, gc)

        if (sr, sc) == (gr, gc):
            return [goal]

        # A* with 8-directional movement
        SQRT2 = math.sqrt(2)
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, SQRT2),
            (-1, 1, SQRT2),
            (1, -1, SQRT2),
            (1, 1, SQRT2),
        ]

        open_set: list[tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, sr, sc))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {(sr, sc): 0.0}

        while open_set:
            _, cr, cc = heapq.heappop(open_set)

            if (cr, cc) == (gr, gc):
                # Reconstruct and smooth
                raw: list[tuple[float, float]] = []
                node: tuple[int, int] = (gr, gc)
                while node in came_from:
                    raw.append(self.grid_to_world(*node))
                    node = came_from[node]
                raw.reverse()
                raw.append(goal)  # end with exact goal position
                return self._smooth_path(raw)

            for dr, dc, cost in neighbors:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.blocked[nr, nc]:
                    continue
                # Prevent corner-cutting through diagonal obstacles
                if dr != 0 and dc != 0:
                    if self.blocked[cr + dr, cc] or self.blocked[cr, cc + dc]:
                        continue

                new_g = g_score[(cr, cc)] + cost
                if new_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = new_g
                    h = math.hypot(nr - gr, nc - gc)
                    heapq.heappush(open_set, (new_g + h, nr, nc))
                    came_from[(nr, nc)] = (cr, cc)

        # No path found — fall back to direct line
        return [goal]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nearest_free(self, r: int, c: int) -> tuple[int, int]:
        """BFS for nearest unblocked cell."""
        if not self.blocked[r, c]:
            return r, c
        visited: set[tuple[int, int]] = {(r, c)}
        queue: deque[tuple[int, int]] = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                    if not self.blocked[nr, nc]:
                        return nr, nc
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return r, c

    def _smooth_path(
        self, path: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Remove redundant waypoints using line-of-sight checks."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Skip as far ahead as line-of-sight allows
            farthest = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                if self._line_of_sight(path[i], path[j]):
                    farthest = j
                    break
            smoothed.append(path[farthest])
            i = farthest

        return smoothed

    def _line_of_sight(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> bool:
        """Check if the straight line between two world points is clear."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        steps = max(1, int(dist / (self.cell_size * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            x = a[0] + dx * t
            y = a[1] + dy * t
            r, c = self.world_to_grid(x, y)
            if self.blocked[r, c]:
                return False
        return True
