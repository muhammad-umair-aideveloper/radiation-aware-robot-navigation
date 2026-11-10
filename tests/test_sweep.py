"""Unit tests for radiation weight sweep helpers and Pareto visualization data.

Uses plain asserts following project standards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, START_CELL, GOAL_CELL
from src.sweep_lambda import run_astar_sweep, format_astar_markdown_table, SWEEP_LAMBDAS
from src.plot_pareto import load_astar_sweep_data, load_qlearning_sweep_data


def test_astar_sweep_monotonic_dose_tradeoff() -> None:
    """As lambda increases, mean dose should not increase, and steps should not decrease."""
    env = RadiationEnv()
    # Test on a small 3-start subset for fast verification
    mini_starts = [START_CELL, (10, 10), (5, 40)]
    lambdas = [0.0, 1.0, 5.0]

    res = run_astar_sweep(env, mini_starts, lambdas=lambdas)
    assert "astar_shortest" in res
    assert "astar_aware" in res

    doses = [res["astar_aware"][str(l)]["mean_dose"] for l in lambdas]
    steps = [res["astar_aware"][str(l)]["mean_steps"] for l in lambdas]

    # Dose should decrease or remain equal as radiation penalty increases
    assert doses[0] >= doses[1] >= doses[2], f"Expected non-increasing dose: {doses}"
    # Steps should increase or remain equal to detour around radiation
    assert steps[0] <= steps[1] <= steps[2], f"Expected non-decreasing steps: {steps}"

    print(f"PASS: test_astar_sweep_monotonic_dose_tradeoff (steps={steps}, doses={doses})")


def test_sweep_markdown_table_formatting() -> None:
    """Markdown table formatter produces valid markdown table lines."""
    dummy_res = {
        "astar_shortest": {
            "success_count": 50,
            "total_starts": 50,
            "success_rate": 1.0,
            "mean_steps": 36.18,
            "mean_dose": 119.15,
            "canonical": {"steps": 56.0, "dose": 281.88},
        },
        "astar_aware": {
            "1.0": {
                "lambda": 1.0,
                "success_count": 50,
                "total_starts": 50,
                "success_rate": 1.0,
                "mean_steps": 42.90,
                "mean_dose": 55.68,
                "canonical": {"steps": 76.0, "dose": 59.73},
                "paired_with_shortest": {"mean_dose_diff": -63.46, "mean_step_diff": 6.72},
            }
        },
    }
    table = format_astar_markdown_table(dummy_res)
    assert "| Lambda | Method |" in table
    assert "| 0.0 (ref) | A* Shortest |" in table
    assert "| 1.0 | A* Aware |" in table
    print("PASS: test_sweep_markdown_table_formatting")


def test_pareto_loader_and_file_integrity() -> None:
    """Verify results/sweep_astar.json loads cleanly and contains all required sweep lambdas."""
    json_path = REPO_ROOT / "results" / "sweep_astar.json"
    assert json_path.exists(), f"Missing {json_path}"

    data = load_astar_sweep_data(json_path)
    assert "astar_shortest" in data
    assert "astar_aware" in data

    for lam in [0.0, 0.5, 1.0, 2.5, 5.0, 10.0]:
        assert str(lam) in data["astar_aware"], f"Missing lambda {lam} in A* sweep data"
        entry = data["astar_aware"][str(lam)]
        assert entry["success_count"] == 50
        assert entry["mean_dose"] > 0
        assert entry["mean_steps"] > 0

    print("PASS: test_pareto_loader_and_file_integrity")


def run_all_tests() -> None:
    print("Running sweep helper tests...")
    test_astar_sweep_monotonic_dose_tradeoff()
    test_sweep_markdown_table_formatting()
    test_pareto_loader_and_file_integrity()
    print("All sweep helper tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
