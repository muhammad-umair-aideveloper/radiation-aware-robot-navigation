"""Unit tests for procedural layout generator properties (seeds 100 to 109).

Verifies determinism, wall fraction limits, free-space connectivity,
source placement on free cells, and start/goal validity.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import START_CELL, GOAL_CELL, RadiationEnv
from src.layouts import generate_random_layout, check_connectivity, Layout


def test_layout_determinism() -> None:
    """Same seed must produce bitwise-identical layouts; different seeds must differ."""
    l1 = generate_random_layout(seed=100)
    l2 = generate_random_layout(seed=100)
    l3 = generate_random_layout(seed=101)

    assert np.array_equal(l1.walls, l2.walls), "Determinism failed: walls differ for same seed"
    assert l1.wall_fraction == l2.wall_fraction, "Wall fractions differ for same seed"
    assert l1.sources == l2.sources, "Sources differ for same seed"

    # Different seeds should yield different wall geometries
    assert not np.array_equal(l1.walls, l3.walls), "Different seeds produced identical walls"
    print("PASS: test_layout_determinism")


def test_free_space_connected_all_seeds() -> None:
    """All generated layouts (seeds 100-109) must have fully connected free space."""
    for s in range(100, 110):
        layout = generate_random_layout(seed=s)
        is_conn = check_connectivity(layout.walls, start=layout.start, goal=layout.goal)
        assert is_conn, f"Layout seed {s} failed free space connectivity check"
    print("PASS: test_free_space_connected_all_seeds (seeds 100 to 109 verified)")


def test_sources_on_free_cells() -> None:
    """Sources must reside exclusively on free cells, within valid intensity range."""
    for s in range(100, 110):
        layout = generate_random_layout(seed=s)
        assert len(layout.sources) == 3, f"Expected 3 sources, got {len(layout.sources)}"
        source_positions = set()
        for pos, i0 in layout.sources:
            r, c = pos
            assert not layout.walls[r, c], f"Seed {s}: source at ({r}, {c}) is placed on a wall"
            assert (r, c) != layout.start, f"Seed {s}: source placed on start"
            assert (r, c) != layout.goal, f"Seed {s}: source placed on goal"
            assert 120.0 <= i0 <= 200.0, f"Seed {s}: source intensity {i0} out of range [120, 200]"
            assert pos not in source_positions, f"Seed {s}: duplicate source position {pos}"
            source_positions.add(pos)
    print("PASS: test_sources_on_free_cells")


def test_start_and_goal_free() -> None:
    """Start (2,2) and goal (47,47) must never be obstacle walls."""
    for s in range(100, 110):
        layout = generate_random_layout(seed=s)
        assert not layout.walls[START_CELL], f"Seed {s}: start cell is a wall"
        assert not layout.walls[GOAL_CELL], f"Seed {s}: goal cell is a wall"
    print("PASS: test_start_and_goal_free")


def test_wall_fraction_range() -> None:
    """Wall area fraction must be strictly between 10% and 25%."""
    for s in range(100, 110):
        layout = generate_random_layout(seed=s)
        assert 0.10 <= layout.wall_fraction <= 0.25, (
            f"Seed {s}: wall fraction {layout.wall_fraction:.4f} outside [0.10, 0.25]"
        )
    print("PASS: test_wall_fraction_range")


def test_layout_to_env() -> None:
    """layout.to_env() properly initializes RadiationEnv with layout attributes."""
    layout = generate_random_layout(seed=105)
    env = layout.to_env(max_steps=250)
    assert isinstance(env, RadiationEnv)
    assert np.array_equal(env.walls, layout.walls)
    assert env.max_steps == 250
    assert not env.walls[env.start]
    assert not env.walls[env.goal]
    print("PASS: test_layout_to_env")


def run_all_tests() -> None:
    print("Running procedural layout tests...")
    test_layout_determinism()
    test_free_space_connected_all_seeds()
    test_sources_on_free_cells()
    test_start_and_goal_free()
    test_wall_fraction_range()
    test_layout_to_env()
    print("All procedural layout tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
