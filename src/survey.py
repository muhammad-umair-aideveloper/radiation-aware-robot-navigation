"""Survey waypoint selection for the S12 online navigation study."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence, Tuple

import numpy as np

GridCell = Tuple[int, int]


def reachable_free_cells(
    start: GridCell,
    free_mask: np.ndarray,
) -> set[GridCell]:
    """Return free cells reachable from start using 8-connected movement."""
    mask = np.asarray(free_mask, dtype=bool)
    if not mask[ start ]:
        return set()

    reached = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                next_cell = (row + dr, col + dc)
                if (
                    0 <= next_cell[0] < mask.shape[0]
                    and 0 <= next_cell[1] < mask.shape[1]
                    and mask[next_cell]
                    and next_cell not in reached
                    and not (
                        abs(dr) == 1
                        and abs(dc) == 1
                        and (
                            mask[row, col + dc] is False
                            or mask[row + dr, col] is False
                        )
                    )
                ):
                    reached.add(next_cell)
                    queue.append(next_cell)
    return reached


def survey_waypoints(
    start: GridCell,
    free_mask: np.ndarray,
    spacing: int = 12,
) -> list[GridCell]:
    """Select and visit coarse-grid waypoints reachable from start."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    mask = np.asarray(free_mask, dtype=bool)
    reachable = reachable_free_cells(start, mask)
    if not reachable:
        return []

    candidates = [
        (row, col)
        for row in range(6, mask.shape[0], spacing)
        for col in range(6, mask.shape[1], spacing)
        if mask[row, col] and (row, col) in reachable and (row, col) != start
    ]

    ordered: list[GridCell] = []
    current = start
    remaining = set(candidates)
    while remaining:
        next_cell = min(
            remaining,
            key=lambda cell: (
                (cell[0] - current[0]) ** 2 + (cell[1] - current[1]) ** 2,
                cell[0],
                cell[1],
            ),
        )
        ordered.append(next_cell)
        remaining.remove(next_cell)
        current = next_cell
    return ordered
