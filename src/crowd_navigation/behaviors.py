import random
from enum import Enum, auto
import pygame

class ControlMode(Enum):
    MANUAL = auto()
    NAIVE = auto()
    RANDOM = auto()
    POTENTIAL_FIELD = auto()


def get_manual_move(robot, goal_pos, pedestrians, keys):
    """Keyboard-controlled movement."""
    move = pygame.Vector2(0, 0)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        move.y -= 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        move.y += 1
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        move.x -= 1
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        move.x += 1
    return move


def get_naive_move(robot, goal_pos, pedestrians, keys):
    """Move directly toward goal, ignoring pedestrians."""
    direction = pygame.Vector2(goal_pos.x - robot.x, goal_pos.y - robot.y)
    if direction.length_squared() > 0:
        return direction.normalize()
    return pygame.Vector2(0, 0)


def get_random_move(robot, goal_pos, pedestrians, keys):
    """Random movement in any direction."""
    return pygame.Vector2(
        random.choice([-1, 0, 1]),
        random.choice([-1, 0, 1]),
    )


def get_potential_field_move(robot, goal_pos, pedestrians, keys):
    """Potential field: attracted to goal, repelled by pedestrians."""
    ATTRACT_STRENGTH = 1.0
    REPEL_STRENGTH = 50.0
    REPEL_RADIUS = 60.0

    to_goal = pygame.Vector2(goal_pos.x - robot.x, goal_pos.y - robot.y)
    if to_goal.length() > 0:
        attract = to_goal.normalize() * ATTRACT_STRENGTH
    else:
        attract = pygame.Vector2(0, 0)

    repel = pygame.Vector2(0, 0)
    for ped in pedestrians:
        to_robot = pygame.Vector2(robot.x - ped.x, robot.y - ped.y)
        dist = to_robot.length()
        if 0 < dist < REPEL_RADIUS:
            strength = REPEL_STRENGTH / (dist * dist)
            repel += to_robot.normalize() * strength

    return attract + repel


BEHAVIORS = {
    ControlMode.MANUAL: get_manual_move,
    ControlMode.NAIVE: get_naive_move,
    ControlMode.RANDOM: get_random_move,
    ControlMode.POTENTIAL_FIELD: get_potential_field_move,
}