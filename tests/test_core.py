"""Core unit tests using plain asserts (no pytest required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import (
    ACTIONS,
    ACTION_LENGTHS,
    GOAL_CELL,
    GRID_SHAPE,
    START_CELL,
    RadiationEnv,
    compute_radiation_field,
    create_wall_mask,
)
from src.planners import astar_search, octile_distance


def test_field_at_source() -> None:
    """Field value at an isolated source must equal I0 / 2.0."""
    sr, sc = 10, 10
    i0 = 100.0
    field = compute_radiation_field(shape=(20, 20), sources=[((sr, sc), i0)])
    expected = i0 / 2.0
    actual = field[sr, sc]
    assert math.isclose(actual, expected, rel_tol=1e-6), (
        f"Expected source intensity {expected}, got {actual}"
    )
    print("PASS: test_field_at_source")


def test_astar_empty_grid() -> None:
    """A* on an empty small grid returns the exact octile path length."""
    # Create an empty environment without walls
    env = RadiationEnv(start=(0, 0), goal=(3, 4), max_steps=100)
    env.shape = (10, 10)
    env.walls = np.zeros((10, 10), dtype=bool)
    env.radiation = np.zeros((10, 10), dtype=np.float64)

    path, dose, steps, expanded = astar_search((0, 0), (3, 4), mode="shortest", env=env)
    assert path is not None, "A* failed to find path on empty grid"

    expected_length = 3 * math.sqrt(2.0) + 1 * 1.0
    # Measure path length from steps
    actual_length = 0.0
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        actual_length += math.sqrt(dr * dr + dc * dc)

    assert math.isclose(actual_length, expected_length, rel_tol=1e-6), (
        f"Expected path length {expected_length}, got {actual_length}"
    )
    assert steps == 4, f"Expected 4 steps, got {steps}"
    print("PASS: test_astar_empty_grid")


def test_astar_never_crosses_walls() -> None:
    """A* path must never visit a wall cell or cut wall corners."""
    env = RadiationEnv()
    for mode in ["shortest", "aware"]:
        path, dose, steps, expanded = astar_search(START_CELL, GOAL_CELL, mode=mode, env=env)
        assert path is not None, f"A* {mode} failed to find path"

        for idx, (r, c) in enumerate(path):
            assert not env.walls[r, c], f"Path step {idx} is a wall cell: ({r}, {c})"
            if idx > 0:
                pr, pc = path[idx - 1]
                dr = r - pr
                dc = c - pc
                if abs(dr) == 1 and abs(dc) == 1:
                    assert not (env.walls[pr + dr, pc] or env.walls[pr, pc + dc]), (
                        f"Corner cutting detected between ({pr},{pc}) and ({r},{c})"
                    )
    print("PASS: test_astar_never_crosses_walls")


def test_collision_rewards_terminate() -> None:
    """Colliding with boundary or wall must terminate with -1000 reward."""
    env = RadiationEnv(start=(0, 0), goal=GOAL_CELL)
    # Action 0 is N: moving north from row 0 is out of bounds
    next_state, reward, done, info = env.step(0)
    assert done is True, "Collision must terminate the episode"
    assert reward == -1000.0, f"Expected reward -1000.0 on collision, got {reward}"
    assert info["outcome"] == "collision", f"Expected collision outcome, got {info['outcome']}"
    assert next_state == (0, 0), "Robot must stay at original position on collision"
    print("PASS: test_collision_rewards_terminate")


def run_all_tests() -> None:
    print("Running core unit tests...")
    test_field_at_source()
    test_astar_empty_grid()
    test_astar_never_crosses_walls()
    test_collision_rewards_terminate()
    print("All core unit tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
