"""Seeded reproducibility checks for sensing and navigation workflows."""

import numpy as np

from src.estimators import idw, kernel_regression, parametric_sources, parametric_sources_shielded
from src.geiger_model import sample_measurement_cells, simulate_counts
from src.online_sim import run_online_episode
from src.s13_demo import print_compact, run_demo
from src.survey import survey_waypoints


def test_measurement_model_is_seeded_and_noise_changes_with_seed() -> None:
    field = np.full((5, 5), 100.0)
    cells = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert np.array_equal(
        sample_measurement_cells(np.ones((5, 5), dtype=bool), 8, 41),
        sample_measurement_cells(np.ones((5, 5), dtype=bool), 8, 41),
    )
    first = simulate_counts(field, cells, dwell=1.0, seed=41)
    second = simulate_counts(field, cells, dwell=1.0, seed=41)
    changed = simulate_counts(field, cells, dwell=1.0, seed=42)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)


def test_all_field_estimators_are_deterministic() -> None:
    mask = np.ones((5, 5), dtype=bool)
    cells = np.array([[0, 0], [1, 2], [2, 4], [4, 1], [4, 4]])
    values = np.array([4.0, 2.0, 1.0, 3.0, 5.0])
    assert np.array_equal(idw(cells, values, mask), idw(cells, values, mask))
    assert np.array_equal(
        kernel_regression(cells, values, mask)[0],
        kernel_regression(cells, values, mask)[0],
    )
    first, first_sources = parametric_sources(cells, values, mask, k=2, refine_passes=1)
    second, second_sources = parametric_sources(cells, values, mask, k=2, refine_passes=1)
    assert np.array_equal(first, second)
    assert first_sources == second_sources

    attenuation = np.eye(mask.size, dtype=np.float32)
    first, first_sources = parametric_sources_shielded(
        cells, values, mask, attenuation, k=2, refine_passes=1
    )
    second, second_sources = parametric_sources_shielded(
        cells, values, mask, attenuation, k=2, refine_passes=1
    )
    assert np.array_equal(first, second)
    assert first_sources == second_sources


def test_online_episode_is_deterministic_for_each_estimator() -> None:
    field = np.full((8, 8), 20.0)
    mask = np.ones((8, 8), dtype=bool)
    for estimator in ("kernel", "idw", "omp3"):
        first = run_online_episode(
            field, mask, (0, 0), (7, 7), estimator, "3x3", 1.0, 13,
            max_steps=12, replan_every=1,
        )
        second = run_online_episode(
            field, mask, (0, 0), (7, 7), estimator, "3x3", 1.0, 13,
            max_steps=12, replan_every=1,
        )
        assert first == second


def test_online_measurement_noise_changes_with_seed() -> None:
    field = np.full((5, 5), 100.0)
    mask = np.ones((5, 5), dtype=bool)
    first = run_online_episode(field, mask, (0, 0), (1, 1), "idw", "3x3", 1.0, 3)
    second = run_online_episode(field, mask, (0, 0), (1, 1), "idw", "3x3", 1.0, 4)
    assert first["readings"] != second["readings"]


def test_survey_and_online_survey_phase_are_deterministic() -> None:
    mask = np.ones((8, 8), dtype=bool)
    assert survey_waypoints((0, 0), mask) == survey_waypoints((0, 0), mask)
    first = run_online_episode(
        np.ones((8, 8)),
        mask,
        (0, 0),
        (7, 7),
        "idw",
        "cell",
        1.0,
        21,
        max_steps=4,
        survey_budget=4,
    )
    second = run_online_episode(
        np.ones((8, 8)),
        mask,
        (0, 0),
        (7, 7),
        "idw",
        "cell",
        1.0,
        21,
        max_steps=4,
        survey_budget=4,
    )
    assert first == second


def test_fast_demo_printed_result_numbers_are_repeatable(capsys) -> None:
    first = run_demo(fast=True)
    print_compact(first)
    first_output = "\n".join(
        line for line in capsys.readouterr().out.splitlines()
        if not line.startswith("Elapsed:")
    )
    second = run_demo(fast=True)
    print_compact(second)
    second_output = "\n".join(
        line for line in capsys.readouterr().out.splitlines()
        if not line.startswith("Elapsed:")
    )
    assert first_output == second_output
