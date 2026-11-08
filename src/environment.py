"""Grid world environment with radiation hotspots and obstacle walls."""

from __future__ import annotations

import math
from typing import Tuple, List, Optional
import numpy as np

GRID_SHAPE = (50, 50)
START_CELL = (2, 2)
GOAL_CELL = (47, 47)

# 8 Actions: N, NE, E, SE, S, SW, W, NW
ACTIONS = [
    (-1, 0),   # 0: N
    (-1, 1),   # 1: NE
    (0, 1),    # 2: E
    (1, 1),    # 3: SE
    (1, 0),    # 4: S
    (1, -1),   # 5: SW
    (0, -1),   # 6: W
    (-1, -1),  # 7: NW
]
ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_LENGTHS = [math.sqrt(dr * dr + dc * dc) for dr, dc in ACTIONS]

# Inclusive wall rectangles: (r_min, r_max, c_min, c_max)
WALL_RECTS = [
    (22, 27, 18, 34),
    (34, 44, 36, 38),
    (8, 14, 26, 28),
]

# Radiation sources: (row, col), I0
SOURCES = [
    ((18, 20), 150.0),
    ((32, 30), 200.0),
    ((25, 12), 120.0),
]


def create_wall_mask(shape: Tuple[int, int] = GRID_SHAPE) -> np.ndarray:
    """Build a boolean mask of walls (True = wall, False = free)."""
    mask = np.zeros(shape, dtype=bool)
    for r_min, r_max, c_min, c_max in WALL_RECTS:
        mask[r_min : r_max + 1, c_min : c_max + 1] = True
    return mask


def compute_radiation_field(
    shape: Tuple[int, int] = GRID_SHAPE,
    sources: Optional[List[Tuple[Tuple[int, int], float]]] = None,
) -> np.ndarray:
    """Vectorized calculation of static radiation intensity across the grid.
    
    Formula: I(r, c) = sum_s I0_s / ((r - sr)^2 + (c - sc)^2 + 2.0)
    """
    if sources is None:
        sources = SOURCES
    r_coords, c_coords = np.indices(shape)
    field = np.zeros(shape, dtype=np.float64)
    for (sr, sc), i0 in sources:
        dist_sq = (r_coords - sr) ** 2 + (c_coords - sc) ** 2
        field += i0 / (dist_sq + 2.0)
    return field


class RadiationEnv:
    """2D grid world environment for radiation-aware navigation."""

    def __init__(
        self,
        start: Tuple[int, int] = START_CELL,
        goal: Tuple[int, int] = GOAL_CELL,
        max_steps: int = 300,
    ) -> None:
        self.shape = GRID_SHAPE
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.walls = create_wall_mask(self.shape)
        self.radiation = compute_radiation_field(self.shape, SOURCES)
        self.state = self.start
        self.steps = 0

    def is_collision(self, r: int, c: int, dr: int, dc: int) -> bool:
        """Check if moving by (dr, dc) from (r, c) causes a collision.
        
        Collision occurs if:
        1. Target cell (r+dr, c+dc) is outside the grid bounds.
        2. Target cell is a wall.
        3. Diagonal move cuts a wall corner (either orthogonal adjacent is wall).
        """
        nr, nc = r + dr, c + dc
        # Out of bounds
        if not (0 <= nr < self.shape[0] and 0 <= nc < self.shape[1]):
            return True
        # Target is wall
        if self.walls[nr, nc]:
            return True
        # Corner-cutting check for diagonal moves
        if abs(dr) == 1 and abs(dc) == 1:
            if self.walls[r + dr, c] or self.walls[r, c + dc]:
                return True
        return False

    def step(
        self, action: int
    ) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Execute action from current state.
        
        Returns:
            (next_state, reward, done, info)
        """
        dr, dc = ACTIONS[action]
        length = ACTION_LENGTHS[action]
        r, c = self.state
        self.steps += 1

        if self.is_collision(r, c, dr, dc):
            # Robot stays at current position on collision; terminal
            reward = -1000.0
            done = True
            info = {
                "outcome": "collision",
                "dose": 0.0,
                "step_length": length,
                "steps": self.steps,
            }
            return self.state, reward, done, info

        nr, nc = r + dr, c + dc
        self.state = (nr, nc)
        dose = self.radiation[nr, nc] * length

        if self.state == self.goal:
            reward = 1000.0
            done = True
            info = {
                "outcome": "goal",
                "dose": dose,
                "step_length": length,
                "steps": self.steps,
            }
            return self.state, reward, done, info

        # Step penalty including radiation cost
        reward = -length * (1.0 + 2.5 * self.radiation[nr, nc])
        timeout = self.steps >= self.max_steps
        done = timeout
        info = {
            "outcome": "timeout" if timeout else "step",
            "dose": dose,
            "step_length": length,
            "steps": self.steps,
        }
        return self.state, reward, done, info

    def reset(self, start: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Reset environment to start state."""
        self.state = self.start if start is None else start
        self.steps = 0
        return self.state
