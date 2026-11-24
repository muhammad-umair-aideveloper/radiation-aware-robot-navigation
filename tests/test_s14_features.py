"""Unit tests for S14 layout feature computation module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.s14_features import (
    compute_corridor_intensity_share,
    compute_mean_shortest_path_length,
    compute_mean_source_to_path_distance,
    compute_min_source_distance,
    compute_wall_fraction,
)


def test_compute_wall_fraction() -> None:
    walls = np.zeros((10, 10), dtype=bool)
    assert compute_wall_fraction(walls) == 0.0

    walls[:5, :] = True
    assert compute_wall_fraction(walls) == 0.5

    walls[:, :] = True
    assert compute_wall_fraction(walls) == 1.0


def test_compute_min_source_distance_known_and_equal() -> None:
    # Collinear equidistant sources: dist(s1, s2) == 4, dist(s2, s3) == 4, dist(s1, s3) == 8
    sources = [(0, 0), (0, 4), (0, 8)]
    assert pytest.approx(compute_min_source_distance(sources)) == 4.0

    # Equilateral triangle (all pairwise distances equal)
    # Side length 6.0: (0, 0), (0, 6), (3 * sqrt(3), 3)
    s1 = (0.0, 0.0)
    s2 = (0.0, 6.0)
    s3 = (3.0 * math.sqrt(3.0), 3.0)
    assert pytest.approx(compute_min_source_distance([s1, s2, s3])) == 6.0

    # Edge case: single source or empty
    assert compute_min_source_distance([(5, 5)]) == 0.0
    assert compute_min_source_distance([]) == 0.0


def test_compute_mean_source_to_path_distance_straight_line() -> None:
    # Straight line horizontal path along row 0: (0, 0) to (0, 9)
    path = [(0, c) for c in range(10)]
    # Sources:
    # s1 at (0, 3) -> distance to path is 0.0
    # s2 at (4, 3) -> distance to path point (0, 3) is 4.0
    # s3 at (8, 3) -> distance to path point (0, 3) is 8.0
    sources = [(0, 3), (4, 3), (8, 3)]
    # Mean distance for this path = (0 + 4 + 8) / 3 = 4.0
    res = compute_mean_source_to_path_distance(sources, [path])
    assert pytest.approx(res) == 4.0

    # Two identical paths
    res2 = compute_mean_source_to_path_distance(sources, [path, path])
    assert pytest.approx(res2) == 4.0

    # Empty paths
    assert compute_mean_source_to_path_distance(sources, []) == 0.0


def test_compute_corridor_intensity_share_known_field() -> None:
    # 10x10 grid with uniform radiation 1.0 everywhere on free cells
    radiation = np.ones((10, 10), dtype=np.float64)
    walls = np.zeros((10, 10), dtype=bool)

    # Path along row 0 from (0, 0) to (0, 9)
    path = [(0, c) for c in range(10)]

    # Cells within radius 5.0 of path:
    # For point (r, c), min distance to path [(0, 0)..(0, 9)] is r (since c is in [0..9])
    # So cells with r <= 5 have distance <= 5.0. That is rows 0, 1, 2, 3, 4, 5 (6 rows x 10 cols = 60 cells).
    # Total cells = 100. All cells have radiation 1.0.
    # Share = 60 / 100 = 0.60
    share = compute_corridor_intensity_share(radiation, walls, [path], radius=5.0)
    assert pytest.approx(share) == 0.60

    # Edge cases
    assert compute_corridor_intensity_share(radiation, walls, []) == 0.0
    all_walls = np.ones((10, 10), dtype=bool)
    assert compute_corridor_intensity_share(radiation, all_walls, [path]) == 0.0


def test_compute_mean_shortest_path_length_known_straight_and_diagonal() -> None:
    # Straight path of 9 orthogonal steps: length = 9.0
    straight_path = [(0, c) for c in range(10)]
    assert pytest.approx(compute_mean_shortest_path_length([straight_path])) == 9.0

    # Diagonal path of 5 diagonal steps: length = 5 * sqrt(2)
    diag_path = [(i, i) for i in range(6)]
    assert pytest.approx(compute_mean_shortest_path_length([diag_path])) == 5.0 * math.sqrt(2.0)

    # Combined average
    expected_mean = (9.0 + 5.0 * math.sqrt(2.0)) / 2.0
    assert pytest.approx(compute_mean_shortest_path_length([straight_path, diag_path])) == expected_mean

    # Empty
    assert compute_mean_shortest_path_length([]) == 0.0
