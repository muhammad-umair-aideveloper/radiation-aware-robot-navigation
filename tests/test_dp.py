"""Tests for exact Dynamic Programming / Dijkstra reference solver.

Validates that Dijkstra optimal costs match A* aware on evaluation starts (tol 1e-9)
and verify exact ground truth on small grids.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, START_CELL, GOAL_CELL
from src.planners import astar_search
from src.dp_reference import dijkstra_search
from main import generate_evaluation_starts


def test_dijkstra_empty_grid() -> None:
    """Dijkstra on an empty small grid returns the exact octile path length and cost."""
    env = RadiationEnv(start=(0, 0), goal=(3, 4), max_steps=100)
    env.shape = (10, 10)
    env.walls = np.zeros((10, 10), dtype=bool)
    env.radiation = np.zeros((10, 10), dtype=np.float64)

    path, cost, dose, steps, expanded = dijkstra_search((0, 0), (3, 4), lambda_rad=0.0, env=env)
    assert path is not None, "Dijkstra failed to find path on empty grid"

    expected_length = 3 * math.sqrt(2.0) + 1 * 1.0
    assert math.isclose(cost, expected_length, rel_tol=1e-9), (
        f"Expected cost {expected_length}, got {cost}"
    )
    assert steps == 4, f"Expected 4 steps, got {steps}"
    print("PASS: test_dijkstra_empty_grid")


def test_dijkstra_matches_astar_canonical() -> None:
    """Optimal cost of Dijkstra equals A* aware on canonical (2,2) -> (47,47) route."""
    env = RadiationEnv()
    path_a, _, _, _ = astar_search(START_CELL, GOAL_CELL, mode="aware", lambda_rad=2.5, env=env)
    path_d, cost_d, _, _, _ = dijkstra_search(START_CELL, GOAL_CELL, lambda_rad=2.5, env=env)

    assert path_a is not None and path_d is not None

    cost_a = 0.0
    for i in range(len(path_a) - 1):
        dr = path_a[i + 1][0] - path_a[i][0]
        dc = path_a[i + 1][1] - path_a[i][1]
        L = math.sqrt(dr * dr + dc * dc)
        cost_a += L * (1.0 + 2.5 * env.radiation[path_a[i + 1]])

    assert math.isclose(cost_a, cost_d, abs_tol=1e-9), (
        f"A* cost {cost_a} != Dijkstra cost {cost_d}"
    )
    print("PASS: test_dijkstra_matches_astar_canonical")


def test_dijkstra_matches_astar_50_starts() -> None:
    """Optimal cost of Dijkstra equals A* aware across all 50 fixed starts (tol 1e-9)."""
    env = RadiationEnv()
    eval_starts = generate_evaluation_starts(env, num_starts=50, seed=123)

    for idx, s in enumerate(eval_starts):
        path_a, _, _, _ = astar_search(s, env.goal, mode="aware", lambda_rad=2.5, env=env)
        path_d, cost_d, _, _, _ = dijkstra_search(s, env.goal, lambda_rad=2.5, env=env)

        assert path_a is not None and path_d is not None, f"Failed on start {s}"

        cost_a = 0.0
        for i in range(len(path_a) - 1):
            dr = path_a[i + 1][0] - path_a[i][0]
            dc = path_a[i + 1][1] - path_a[i][1]
            L = math.sqrt(dr * dr + dc * dc)
            cost_a += L * (1.0 + 2.5 * env.radiation[path_a[i + 1]])

        diff = abs(cost_a - cost_d)
        assert math.isclose(cost_a, cost_d, abs_tol=1e-9), (
            f"Start {idx} ({s}): A* cost {cost_a} != DP cost {cost_d} (diff={diff})"
        )

    print("PASS: test_dijkstra_matches_astar_50_starts (all 50 starts matched within 1e-9)")


def run_all_tests() -> None:
    print("Running Dijkstra reference tests...")
    test_dijkstra_empty_grid()
    test_dijkstra_matches_astar_canonical()
    test_dijkstra_matches_astar_50_starts()
    print("All Dijkstra reference tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
