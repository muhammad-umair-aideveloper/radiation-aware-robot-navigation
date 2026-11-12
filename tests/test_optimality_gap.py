"""Unit tests for optimality gap helpers (plain asserts, no pytest dependency)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, START_CELL, GOAL_CELL
from src.dp_reference import dijkstra_search
from src.planners import astar_search
from src.optimality_gap import compute_path_cost, compute_relative_gap


def test_gap_zero_for_optimal() -> None:
    """A path equal to the optimum gives gap exactly 0.0."""
    opt_cost = 150.0
    actual_cost = 150.0
    gap = compute_relative_gap(actual_cost, opt_cost)
    assert gap == 0.0, f"Expected gap 0.0, got {gap}"

    # Slightly sub-optimal due to float jitter should still be close
    gap_jitter = compute_relative_gap(150.0 + 1e-12, 150.0)
    assert math.isclose(gap_jitter, 0.0, abs_tol=1e-6)
    print("PASS: test_gap_zero_for_optimal")


def test_gap_positive_for_suboptimal() -> None:
    """A longer or higher-dose path strictly yields a positive relative gap."""
    opt_cost = 100.0
    higher_cost = 125.0
    gap = compute_relative_gap(higher_cost, opt_cost)
    assert gap == 25.0, f"Expected gap 25.0%, got {gap}%"
    assert gap > 0.0
    print("PASS: test_gap_positive_for_suboptimal")


def test_path_cost_calculation() -> None:
    """Path cost correctly aggregates step length and radiation intensity."""
    env = RadiationEnv()
    # Simple 2-step path: (2,2) -> (2,3) -> (2,4) [both orthogonal, L=1.0]
    test_path = [(2, 2), (2, 3), (2, 4)]
    cost_lam0 = compute_path_cost(test_path, env, lambda_rad=0.0)
    assert cost_lam0 == 2.0, f"Expected cost 2.0 at λ=0, got {cost_lam0}"

    cost_lam25 = compute_path_cost(test_path, env, lambda_rad=2.5)
    expected_extra = 2.5 * (env.radiation[2, 3] + env.radiation[2, 4])
    assert math.isclose(cost_lam25, 2.0 + expected_extra, rel_tol=1e-6)
    print("PASS: test_path_cost_calculation")


def test_astar_aware_zero_gap_vs_dijkstra() -> None:
    """On canonical route, A* aware path matches Dijkstra optimal cost, giving gap 0.0."""
    env = RadiationEnv()
    path_a, _, _, _ = astar_search(START_CELL, GOAL_CELL, mode="aware", env=env, lambda_rad=2.5)
    path_d, cost_d, _, _, _ = dijkstra_search(START_CELL, GOAL_CELL, lambda_rad=2.5, env=env)

    cost_a = compute_path_cost(path_a, env, lambda_rad=2.5)
    gap = compute_relative_gap(cost_a, cost_d)

    assert math.isclose(gap, 0.0, abs_tol=1e-7), f"Expected gap 0.0%, got {gap}%"
    print("PASS: test_astar_aware_zero_gap_vs_dijkstra")


def run_all_tests() -> None:
    print("Running optimality gap tests...")
    test_gap_zero_for_optimal()
    test_gap_positive_for_suboptimal()
    test_path_cost_calculation()
    test_astar_aware_zero_gap_vs_dijkstra()
    print("All optimality gap tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
