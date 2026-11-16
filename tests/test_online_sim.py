"""Tests for online radiation measurement and replanning."""

import sys

import numpy as np

sys.path.insert(0, "src")

from map_error_study import evaluate_path_on_true_field
from online_sim import estimate_field, run_online_episode


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


if __name__ == "__main__":
    test_episode_invariants()
    test_seed_reproducibility_and_low_dwell_validity()
    test_few_readings_are_constant_and_walls_are_zero()
    print("online simulation tests passed")
