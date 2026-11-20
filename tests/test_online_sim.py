"""Tests for online radiation measurement and replanning."""

import sys

import numpy as np

sys.path.insert(0, "src")

from map_error_study import evaluate_path_on_true_field
from estimators import idw, kernel_regression
from online_sim import estimate_field, inflate_unknown, run_online_episode


def make_case() -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices((12, 12))
    field = 50 / ((rows - 6) ** 2 + (columns - 6) ** 2 + 2)
    free_mask = np.ones((12, 12), dtype=bool)
    free_mask[3, 3] = False
    free_mask[4, 3] = False
    free_mask[3, 4] = False
    return field, free_mask


def assert_valid_path(result: dict, free_mask: np.ndarray) -> None:
    path = result["path"]
    assert path[0] == (0, 0)
    if result["success"]:
        assert path[-1] == (11, 11)
    for current, nxt in zip(path, path[1:]):
        assert max(abs(nxt[0] - current[0]), abs(nxt[1] - current[1])) == 1
        assert free_mask[nxt]


def test_episode_invariants() -> None:
    field, free_mask = make_case()
    result = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "idw", "3x3", 1.0, 0
    )
    assert_valid_path(result, free_mask)
    expected_dose, _ = evaluate_path_on_true_field(result["path"], field)
    assert abs(result["dose"] - expected_dose) <= 1e-9
    assert result["n_readings"] > 0
    assert result["n_distinct_cells"] > 0


def test_seed_reproducibility_and_low_dwell_validity() -> None:
    field, free_mask = make_case()
    first = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "idw", "cell", 1.0, 7
    )
    second = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "idw", "cell", 1.0, 7
    )
    low_dwell = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "idw", "cell", 0.1, 8
    )
    assert first["path"] == second["path"]
    assert_valid_path(low_dwell, free_mask)


def test_few_readings_are_constant_and_walls_are_zero() -> None:
    _, free_mask = make_case()
    estimated = estimate_field({(0, 0): [2.0], (0, 1): [4.0]}, free_mask, "idw")
    assert np.all(estimated[free_mask] == 3.0)
    assert np.all(estimated[~free_mask] == 0.0)
    assert np.all(estimated >= 0.0)


def test_zero_control_uses_shortest_path() -> None:
    field = np.ones((12, 12), dtype=float)
    free_mask = np.ones((12, 12), dtype=bool)
    result = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "idw", "cell", 1.0, 0, control="zero"
    )
    assert result["steps"] == 11


def test_noise_free_readings_match_true_field() -> None:
    field, free_mask = make_case()
    result = run_online_episode(
        field,
        free_mask,
        (0, 0),
        (11, 11),
        "idw",
        "cell",
        1.0,
        0,
        noise_free=True,
    )
    for (row, column), values in result["readings"].items():
        assert all(abs(value - field[row, column]) <= 1e-9 for value in values)


def test_unknown_cell_inflation_and_zero_beta() -> None:
    field, free_mask = make_case()
    readings = {(0, 0): [2.0, 4.0]}
    estimated = np.zeros_like(field)
    inflated = inflate_unknown(estimated, readings, free_mask, 2.0)
    assert inflated[11, 11] >= 6.0
    assert inflated[0, 0] == estimated[0, 0]
    default = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "idw", "cell", 1.0, 4
    )
    no_inflation = run_online_episode(
        field,
        free_mask,
        (0, 0),
        (11, 11),
        "idw",
        "cell",
        1.0,
        4,
        inflate_beta=0.0,
    )
    assert no_inflation["path"] == default["path"]


def test_true_control_reaches_goal() -> None:
    field, free_mask = make_case()
    result = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "idw", "cell", 1.0, 0, control="true"
    )
    assert result["path"][0] == (0, 0)
    assert result["path"][-1] == (11, 11)


def test_omp_estimate_has_valid_field() -> None:
    """OMP estimation returns a valid masked field."""
    field, free_mask = make_case()
    readings = {
        (row, column): [float(field[row, column])]
        for row, column in zip(range(12), range(12))
    }
    estimated = estimate_field(readings, free_mask, "omp3")
    assert estimated.shape == field.shape
    assert np.all(np.isfinite(estimated))
    assert np.all(estimated >= 0.0)
    assert np.all(estimated[~free_mask] == 0.0)


def test_unknown_estimator_raises() -> None:
    """Unknown estimator names are rejected."""
    _, free_mask = make_case()
    try:
        estimate_field({(0, 0): [1.0], (0, 1): [1.0], (0, 2): [1.0]}, free_mask, "foo")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown estimator did not raise ValueError")


def test_existing_estimators_are_unchanged() -> None:
    """Kernel and IDW dispatch match their direct implementations."""
    field, free_mask = make_case()
    readings = {
        (0, 0): [float(field[0, 0])],
        (1, 2): [float(field[1, 2])],
        (6, 6): [float(field[6, 6])],
        (10, 9): [float(field[10, 9])],
    }
    cells = np.array(list(readings), dtype=int)
    values = np.array([np.mean(values) for values in readings.values()])
    expected_kernel, _ = kernel_regression(cells, values, free_mask)
    expected_idw = idw(cells, values, free_mask)
    assert np.allclose(
        estimate_field(readings, free_mask, "kernel"), expected_kernel, atol=1e-12
    )
    assert np.allclose(
        estimate_field(readings, free_mask, "idw"), expected_idw, atol=1e-12
    )


def test_survey_budget_zero_matches_old_episode() -> None:
    field, free_mask = make_case()
    default = run_online_episode(
        field, free_mask, (0, 0), (11, 11), "kernel", "3x3", 1.0, 11
    )
    explicit = run_online_episode(
        field,
        free_mask,
        (0, 0),
        (11, 11),
        "kernel",
        "3x3",
        1.0,
        11,
        survey_budget=0,
    )
    assert explicit["path"] == default["path"]
    assert abs(explicit["dose"] - default["dose"]) <= 1e-12
    assert explicit["steps"] == default["steps"]
    assert explicit["survey_steps"] == 0


def test_survey_budget_and_step_limit() -> None:
    field, free_mask = make_case()
    result = run_online_episode(
        field,
        free_mask,
        (0, 0),
        (11, 11),
        "kernel",
        "3x3",
        1.0,
        3,
        max_steps=360,
        survey_budget=30,
    )
    assert result["survey_steps"] <= 30
    assert result["steps"] <= 360
    expected_dose, _ = evaluate_path_on_true_field(result["path"], field)
    assert abs(result["dose"] - expected_dose) <= 1e-9
    assert result["survey_dose_share"] >= 0.0


if __name__ == "__main__":
    test_episode_invariants()
    test_seed_reproducibility_and_low_dwell_validity()
    test_few_readings_are_constant_and_walls_are_zero()
    test_zero_control_uses_shortest_path()
    test_noise_free_readings_match_true_field()
    test_unknown_cell_inflation_and_zero_beta()
    test_true_control_reaches_goal()
    test_omp_estimate_has_valid_field()
    test_unknown_estimator_raises()
    test_existing_estimators_are_unchanged()
    test_survey_budget_zero_matches_old_episode()
    test_survey_budget_and_step_limit()
    print("online simulation tests passed")
