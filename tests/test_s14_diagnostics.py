"""Unit tests for S14 diagnostics runner helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.s13_study import estimate_field, run_online_episode
from src.s14_diagnostics import (
    aggregate_layout_diagnostics,
    compute_matched_source_error,
    compute_relative_l2_error,
    replay_and_fit_diagnostics,
)


def test_matched_source_error_known_cases() -> None:
    # 1. Exact match with same order
    true_s = [(10.0, 15.0), (20.0, 25.0), (30.0, 35.0)]
    fitted_s = [(10.0, 15.0), (20.0, 25.0), (30.0, 35.0)]
    err, all_3 = compute_matched_source_error(true_s, fitted_s)
    assert err == 0.0
    assert all_3 is True

    # 2. Exact match with permuted order (fitted has sources swapped)
    permuted_f = [(30.0, 35.0), (10.0, 15.0), (20.0, 25.0)]
    err_p, all_3_p = compute_matched_source_error(true_s, permuted_f)
    assert err_p == 0.0
    assert all_3_p is True

    # 3. Known translation by 2.0 cells in row for all sources
    shifted_2 = [(12.0, 15.0), (22.0, 25.0), (32.0, 35.0)]
    err_s2, all_3_s2 = compute_matched_source_error(true_s, shifted_2)
    assert pytest.approx(err_s2) == 2.0
    assert all_3_s2 is True

    # 4. One source shifted by 5.0 cells (greater than 3.0 cell threshold)
    # Distances: s1 -> 0, s2 -> 0, s3 -> 5.0
    shifted_one_far = [(10.0, 15.0), (20.0, 25.0), (30.0, 40.0)]
    err_far, all_3_far = compute_matched_source_error(true_s, shifted_one_far)
    assert pytest.approx(err_far) == 5.0 / 3.0
    assert all_3_far is False  # Because 5.0 > 3.0


def test_relative_l2_error() -> None:
    mask = np.ones((5, 5), dtype=bool)
    true_field = np.full((5, 5), 2.0)
    fit_field = np.full((5, 5), 2.0)
    assert compute_relative_l2_error(fit_field, true_field, mask) == 0.0

    fit_field_double = np.full((5, 5), 4.0)
    # ||4 - 2||_2 / ||2||_2 = 2 / 2 = 1.0
    assert pytest.approx(compute_relative_l2_error(fit_field_double, true_field, mask)) == 1.0


def test_readings_replay_equals_simulation_on_small_grid() -> None:
    # 15x15 test grid with known point sources
    grid_size = 15
    free_mask = np.ones((grid_size, grid_size), dtype=bool)
    free_mask[0, :] = False
    free_mask[-1, :] = False
    free_mask[:, 0] = False
    free_mask[:, -1] = False

    true_sources = [((4, 4), 150.0), ((4, 10), 160.0), ((10, 7), 170.0)]
    radiation = np.zeros((grid_size, grid_size), dtype=np.float64)
    for (r, c), i0 in true_sources:
        for gr in range(grid_size):
            for gc in range(grid_size):
                d2 = (gr - r) ** 2 + (gc - c) ** 2
                radiation[gr, gc] += i0 / (d2 + 2.0)

    start = (2, 2)
    goal = (12, 12)

    # Run online episode
    ep = run_online_episode(
        true_field=radiation,
        free_mask=free_mask,
        start=start,
        goal=goal,
        estimator="omp3",
        pattern="3x3",
        dwell=1.0,
        seed=42,
        replan_every=3,
        lam=2.5,
        max_steps=50,
    )

    readings = ep["readings"]
    diag = replay_and_fit_diagnostics(
        readings=readings,
        free_mask=free_mask,
        true_field=radiation,
        true_sources=true_sources,
        k=3,
    )

    # Check that distinct measured cells match
    assert diag["distinct_measured_cells"] == len(readings)
    assert diag["matched_source_position_error"] >= 0.0
    assert 0.0 <= diag["relative_l2_field_error"]

    # Verify that estimate_field produces the same field on the same readings
    direct_est_field = estimate_field(readings, free_mask, "omp3")
    from src.estimators import parametric_sources

    cells = np.array(list(readings.keys()), dtype=int)
    values = np.array([np.mean(v) for v in readings.values()], dtype=float)
    replayed_field, _ = parametric_sources(cells, values, free_mask, k=3)
    np.testing.assert_allclose(direct_est_field, np.maximum(replayed_field, 0.0))


def test_aggregate_layout_diagnostics() -> None:
    episodes = [
        {
            "matched_source_position_error": 1.0,
            "all_sources_within_3": True,
            "relative_l2_field_error": 0.2,
            "distinct_measured_cells": 100,
            "dose": 50.0,
            "steps": 40,
            "success": True,
        },
        {
            "matched_source_position_error": 3.0,
            "all_sources_within_3": False,
            "relative_l2_field_error": 0.4,
            "distinct_measured_cells": 120,
            "dose": 70.0,
            "steps": 44,
            "success": True,
        },
    ]

    agg = aggregate_layout_diagnostics(episodes)
    assert agg["mean_matched_source_position_error"] == 2.0
    assert agg["all_sources_within_3_rate"] == 0.5
    assert pytest.approx(agg["mean_relative_l2_field_error"]) == 0.3
    assert agg["mean_distinct_measured_cells"] == 110.0
    assert agg["mean_true_dose"] == 60.0
    assert agg["mean_steps"] == 42.0
    assert agg["success_rate"] == 1.0

    # Empty case
    assert aggregate_layout_diagnostics([]) == {}
