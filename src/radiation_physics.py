"""Radiation physics module: inverse-square law intensity and dose accumulation.

Physical model
--------------
Point-source intensity with softening factor epsilon to avoid singularities:

    I(p) = I0 / (||p_robot - p_source||_2^2 + epsilon)

For N sources, total intensity is the superposition:

    I_total(p) = sum_s  I0_s / (||p - p_s||_2^2 + epsilon)

Cumulative dose for a trajectory step of geometric length L:

    delta_D = I_total(p_new) * L

The softened denominator prevents division-by-zero when the robot is exactly
at a source location (d=0), yielding I_max = I0 / epsilon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# Default softening factor (cells^2).  Matches the value used throughout the
# project: denominator = dist_sq + EPSILON.
EPSILON: float = 2.0


@dataclass(frozen=True)
class PointSource:
    """A single radiation point source on the grid."""

    position: Tuple[int, int]   # (row, col)
    intensity: float            # I0  (arbitrary units)


def point_source_intensity(
    robot_pos: Tuple[float, float],
    source: PointSource,
    epsilon: float = EPSILON,
) -> float:
    """Return the radiation intensity at *robot_pos* from one point source.

    Args:
        robot_pos: (row, col) position of the robot (floats for sub-cell use).
        source:    PointSource with position and I0.
        epsilon:   Softening factor (> 0) preventing division by zero.

    Returns:
        Scalar intensity I = I0 / (d^2 + epsilon).
    """
    dr = robot_pos[0] - source.position[0]
    dc = robot_pos[1] - source.position[1]
    dist_sq = dr * dr + dc * dc
    return source.intensity / (dist_sq + epsilon)


def total_intensity(
    robot_pos: Tuple[float, float],
    sources: Sequence[PointSource],
    epsilon: float = EPSILON,
) -> float:
    """Return superimposed intensity from all point sources at *robot_pos*.

    I_total = sum_s  I0_s / (||p - p_s||^2 + epsilon)

    Guaranteed finite for any robot_pos (epsilon > 0 prevents division by zero).
    """
    return sum(
        point_source_intensity(robot_pos, s, epsilon) for s in sources
    )


def vectorized_intensity_field(
    shape: Tuple[int, int],
    sources: Sequence[PointSource],
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Compute the full-grid intensity field in a single vectorized pass.

    Returns a float64 array of shape *shape* where each cell value equals
    total_intensity((r, c), sources, epsilon).
    """
    r_idx, c_idx = np.indices(shape)
    field = np.zeros(shape, dtype=np.float64)
    for src in sources:
        dist_sq = (r_idx - src.position[0]) ** 2 + (c_idx - src.position[1]) ** 2
        field += src.intensity / (dist_sq + epsilon)
    return field


def step_dose(
    new_cell: Tuple[int, int],
    step_length: float,
    intensity_field: np.ndarray,
) -> float:
    """Radiation dose absorbed in one navigation step.

    Args:
        new_cell:        (row, col) of the cell the robot moves *into*.
        step_length:     Euclidean length of the step (1 for cardinal,
                         sqrt(2) for diagonal).
        intensity_field: Precomputed intensity array (shape = grid shape).

    Returns:
        delta_D = I(new_cell) * step_length  (arbitrary units).
    """
    return float(intensity_field[new_cell]) * step_length


def trajectory_dose(
    path: List[Tuple[int, int]],
    intensity_field: np.ndarray,
) -> float:
    """Cumulative dose absorbed along an entire path.

    Sums delta_D = I(cell_{t+1}) * ||cell_{t+1} - cell_t||_2 for each step.

    Args:
        path:            Ordered list of (row, col) waypoints including start.
        intensity_field: Precomputed intensity array.

    Returns:
        Total absorbed dose (arbitrary units).
    """
    import math

    total = 0.0
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        length = math.sqrt(dr * dr + dc * dc)
        total += step_dose(path[i + 1], length, intensity_field)
    return total

