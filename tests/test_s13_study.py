"""Focused tests for S13 runner helpers and controls."""

from __future__ import annotations

import numpy as np

from src.layouts import generate_random_layout
from src.s13_study import (
    bootstrap_layout,
    dose_regret,
    objective_cost,
    offline_estimate,
    random_measurement_cells,
    run_controls,
)


def test_bootstrap_layout_fixed_deterministic() -> None:
    mean, low, high = bootstrap_layout([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert low <= mean <= high
    mean2, low2, high2 = bootstrap_layout([1.0, 2.0, 3.0])
    assert (mean, low, high) == (mean2, low2, high2)


def test_random_measurement_cells_are_free_and_deterministic() -> None:
    env = generate_random_layout(200).to_env()
    first = random_measurement_cells(env, 50, 123)
    second = random_measurement_cells(env, 50, 123)
    assert first == second
    assert len(first) == len(set(first)) == 50
    assert all(not env.walls[cell] for cell in first)


def test_offline_estimate_kernel_and_omp3_helpers() -> None:
    env = generate_random_layout(200).to_env()
    starts = [(2, 2), (47, 47)]
    for estimator in ("kernel", "idw", "omp3"):
        result = offline_estimate(env, estimator, starts, 0)
        assert 0.0 <= result["success_rate"] <= 1.0
        assert result["mean_true_dose"] >= 0.0


def test_objective_cost_and_dose_regret_helpers() -> None:
    field = np.ones((50, 50), dtype=float)
    path = [(0, 0), (0, 1), (1, 1)]
    assert objective_cost(path, field) == 7.0
    assert dose_regret(110.0, 100.0) == 10.0


def test_s13_prestudy_controls() -> None:
    result = run_controls((200, 201, 202), 3)
    assert result["passed"]
