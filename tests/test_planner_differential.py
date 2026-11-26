"""Randomized differential checks for the A* planners."""

from __future__ import annotations

import heapq
import math

import numpy as np

from src.environment import RadiationEnv
from src.planners import astar_search

_MOVES = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)


def _reference_cost(
    walls: np.ndarray,
    radiation: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    aware: bool,
    radiation_weight: float,
) -> float:
    """Independent Dijkstra reference, including diagonal corner blocking."""
    distances = {start: 0.0}
    frontier = [(0.0, start)]
    height, width = walls.shape
    while frontier:
        distance, (row, column) = heapq.heappop(frontier)
        if distance != distances[(row, column)]:
            continue
        if (row, column) == goal:
            return distance
        for dr, dc in _MOVES:
            nr, nc = row + dr, column + dc
            if not (0 <= nr < height and 0 <= nc < width) or walls[nr, nc]:
                continue
            if dr and dc and (walls[row + dr, column] or walls[row, column + dc]):
                continue
            length = math.sqrt(2.0) if dr and dc else 1.0
            step_cost = length
            if aware:
                step_cost *= 1.0 + radiation_weight * radiation[nr, nc]
            candidate = distance + step_cost
            if candidate < distances.get((nr, nc), math.inf):
                distances[(nr, nc)] = candidate
                heapq.heappush(frontier, (candidate, (nr, nc)))
    return math.inf


def _path_cost(path, radiation: np.ndarray, radiation_weight: float) -> float:
    total = 0.0
    for (row, column), (nr, nc) in zip(path, path[1:]):
        length = math.sqrt(2.0) if row != nr and column != nc else 1.0
        total += length * (1.0 + radiation_weight * radiation[nr, nc])
    return total


def _assert_valid_path(path, walls: np.ndarray) -> None:
    for (row, column), (nr, nc) in zip(path, path[1:]):
        assert not walls[nr, nc]
        assert max(abs(nr - row), abs(nc - column)) == 1
        if row != nr and column != nc:
            assert not walls[row, nc]
            assert not walls[nr, column]


def test_randomized_astar_matches_independent_dijkstra() -> None:
    cases = 300
    radiation_weight = 2.5
    for seed in range(cases):
        rng = np.random.default_rng(seed)
        height, width = (int(value) for value in rng.integers(3, 10, size=2))
        walls = rng.random((height, width)) < rng.uniform(0.0, 0.42)
        start = (0, 0)
        goal = (height - 1, width - 1)
        walls[start] = False
        walls[goal] = False
        # Include explicit separating barriers to exercise unreachable endpoints.
        if seed % 7 == 0 and width > 1:
            walls[:, width // 2] = True
            walls[start] = False
            walls[goal] = False
        radiation = rng.uniform(0.0, 3.0, size=(height, width))
        for _ in range(max(1, height // 2)):
            radiation[tuple(rng.integers((height, width)))] += rng.uniform(5.0, 20.0)

        env = RadiationEnv(start=start, goal=goal, walls=walls, radiation=radiation)
        env.shape = walls.shape
        expected_aware = _reference_cost(
            walls, radiation, start, goal, True, radiation_weight
        )
        expected_shortest = _reference_cost(
            walls, radiation, start, goal, False, radiation_weight
        )

        for mode, expected in (("aware", expected_aware), ("shortest", expected_shortest)):
            path, _, steps, _ = astar_search(
                start, goal, mode=mode, env=env, lambda_rad=radiation_weight
            )
            if math.isinf(expected):
                assert path is None
                assert math.isinf(steps)
                continue

            assert path is not None
            _assert_valid_path(path, walls)
            actual = (
                _path_cost(path, radiation, radiation_weight)
                if mode == "aware"
                else sum(
                    math.sqrt(2.0) if row != nr and column != nc else 1.0
                    for (row, column), (nr, nc) in zip(path, path[1:])
                )
            )
            assert abs(actual - expected) <= 1e-9
            if mode == "shortest":
                assert math.isclose(actual, expected, abs_tol=1e-9)
