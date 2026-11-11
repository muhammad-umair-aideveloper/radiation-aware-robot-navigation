"""Seeded procedural layout generator with connectivity and physical validation.

Generates 50x50 navigation environments containing random wall obstacles (10-25%
wall fraction) and point radiation sources, satisfying 8-connected free space
reachability between start and goal without corner-cutting collisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.environment import (
    GRID_SHAPE,
    START_CELL,
    GOAL_CELL,
    RadiationEnv,
    compute_radiation_field,
)


@dataclass
class Layout:
    """Specification of a procedural grid environment layout."""

    seed: int
    shape: Tuple[int, int]
    walls: np.ndarray
    wall_fraction: float
    sources: List[Tuple[Tuple[int, int], float]]
    start: Tuple[int, int]
    goal: Tuple[int, int]

    def to_env(self, max_steps: int = 300) -> RadiationEnv:
        """Create a RadiationEnv instance configured with this layout."""
        return RadiationEnv(
            start=self.start,
            goal=self.goal,
            max_steps=max_steps,
            walls=self.walls,
            sources=self.sources,
        )


def check_connectivity(
    walls: np.ndarray,
    start: Tuple[int, int] = START_CELL,
    goal: Tuple[int, int] = GOAL_CELL,
) -> bool:
    """Verify that start and goal are free and all free cells form a connected component.

    Uses 8-connected transitions respecting the environment's no-corner-cutting rule.
    """
    shape = walls.shape
    if walls[start] or walls[goal]:
        return False

    visited = np.zeros(shape, dtype=bool)
    visited[start] = True
    stack = [start]

    actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while stack:
        r, c = stack.pop()
        for dr, dc in actions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < shape[0] and 0 <= nc < shape[1] and not walls[nr, nc] and not visited[nr, nc]:
                # Diagonal move: check corner cutting
                if abs(dr) == 1 and abs(dc) == 1:
                    if walls[r + dr, c] or walls[r, c + dc]:
                        continue
                visited[nr, nc] = True
                stack.append((nr, nc))

    # Goal must be reachable, and free space must be fully connected
    total_free = np.sum(~walls)
    reachable_free = np.sum(visited)
    return bool(visited[goal] and (reachable_free == total_free))


def generate_random_layout(
    seed: int,
    shape: Tuple[int, int] = GRID_SHAPE,
    start: Tuple[int, int] = START_CELL,
    goal: Tuple[int, int] = GOAL_CELL,
    min_wall_frac: float = 0.10,
    max_wall_frac: float = 0.25,
    num_sources: int = 3,
    min_i0: float = 120.0,
    max_i0: float = 200.0,
    max_attempts: int = 1000,
) -> Layout:
    """Deterministically generate a valid procedural layout for a given seed.

    Args:
        seed: Deterministic random seed.
        shape: Grid dimensions (H, W).
        start: Starting (row, col) coordinates.
        goal: Target (row, col) coordinates.
        min_wall_frac: Minimum wall area percentage.
        max_wall_frac: Maximum wall area percentage.
        num_sources: Number of radiation point sources.
        min_i0: Minimum radiation intensity I0.
        max_i0: Maximum radiation intensity I0.
        max_attempts: Maximum rejection sampling attempts.

    Returns:
        Layout instance with verified walls, connectivity, and sources.
    """
    rng = np.random.RandomState(seed)

    for attempt in range(max_attempts):
        walls = np.zeros(shape, dtype=bool)
        n_rects = rng.randint(4, 8)  # 4 to 7 rectangles

        for _ in range(n_rects):
            h = rng.randint(3, 13)   # 3 to 12 height
            w = rng.randint(3, 13)   # 3 to 12 width
            r = rng.randint(0, shape[0] - h + 1)
            c = rng.randint(0, shape[1] - w + 1)
            walls[r : r + h, c : c + w] = True

        if walls[start] or walls[goal]:
            continue

        wall_frac = float(np.mean(walls))
        if not (min_wall_frac <= wall_frac <= max_wall_frac):
            continue

        if not check_connectivity(walls, start=start, goal=goal):
            continue

        # Place radiation sources uniformly on free cells (excluding start & goal)
        free_coords = [
            (int(r), int(c))
            for r, c in zip(*np.where(~walls))
            if (r, c) != start and (r, c) != goal
        ]
        if len(free_coords) < num_sources:
            continue

        chosen_indices = rng.choice(len(free_coords), size=num_sources, replace=False)
        sources: List[Tuple[Tuple[int, int], float]] = []
        for idx in chosen_indices:
            pos = free_coords[idx]
            i0 = float(rng.uniform(min_i0, max_i0))
            sources.append((pos, i0))

        return Layout(
            seed=seed,
            shape=shape,
            walls=walls,
            wall_fraction=wall_frac,
            sources=sources,
            start=start,
            goal=goal,
        )

    raise RuntimeError(
        f"Failed to generate valid layout for seed {seed} after {max_attempts} attempts."
    )

