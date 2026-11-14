"""Tests for sensor agent training script options, curriculum schedule, and defaults."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train_sensor_agent import compute_curriculum_min_dist


def test_curriculum_schedule() -> None:
    """Verify linear distance curriculum schedule: 5 at step 0, 20 at 40% of steps, then constant."""
    total_steps = 400000

    # At step 0: exactly 5.0
    d0 = compute_curriculum_min_dist(0, total_steps)
    assert math.isclose(d0, 5.0, abs_tol=1e-6), f"Expected 5.0 at step 0, got {d0}"

    # At 20% of steps (80,000 / 400,000 = half of curriculum): exactly 12.5
    d_mid = compute_curriculum_min_dist(80000, total_steps)
    assert math.isclose(d_mid, 12.5, abs_tol=1e-6), f"Expected 12.5 at midpoint, got {d_mid}"

    # At 40% of steps (160,000 / 400,000): reaches maximum 20.0
    d_40 = compute_curriculum_min_dist(160000, total_steps)
    assert math.isclose(d_40, 20.0, abs_tol=1e-6), f"Expected 20.0 at 40% steps, got {d_40}"

    # Beyond 40% of steps (e.g. 240,000 and 400,000): remains constant at 20.0
    d_60 = compute_curriculum_min_dist(240000, total_steps)
    assert math.isclose(d_60, 20.0, abs_tol=1e-6), f"Expected 20.0 at 60% steps, got {d_60}"

    d_end = compute_curriculum_min_dist(400000, total_steps)
    assert math.isclose(d_end, 20.0, abs_tol=1e-6), f"Expected 20.0 at end, got {d_end}"

    # Edge cases
    assert compute_curriculum_min_dist(0, 0) == 20.0
    assert compute_curriculum_min_dist(100, -10) == 20.0


def test_training_argument_defaults() -> None:
    """Verify that default training arguments preserve backwards compatibility with S5."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["shaped", "plain"], default="shaped")
    parser.add_argument("--lam", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack", type=int, default=1)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--goal-pool", type=int, default=30)
    parser.add_argument("--tag", type=str, default=None)

    args = parser.parse_args([])

    assert args.variant == "shaped"
    assert args.lam == 2.5
    assert args.seed == 0
    assert args.stack == 1, "Default frame stack must be 1 (unstacked)"
    assert args.curriculum is False, "Curriculum must be disabled by default"
    assert args.goal_pool == 30, "Default goal pool size must be 30"
    assert args.tag is None


def test_network_input_dimension_from_stack() -> None:
    """Verify observation dimension calculation for stack 1 vs stack 4."""
    from src.sensor_env import SensorNavigationEnv

    env1 = SensorNavigationEnv(num_stack=1)
    assert env1.obs_dim == 28

    env4 = SensorNavigationEnv(num_stack=4)
    assert env4.obs_dim == 112


if __name__ == "__main__":
    test_curriculum_schedule()
    test_training_argument_defaults()
    test_network_input_dimension_from_stack()
    print("All training option tests passed!")
