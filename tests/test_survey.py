"""Plain-assert tests for S12 survey waypoint planning."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.survey import reachable_free_cells, survey_waypoints


def test_spacing_and_deterministic_selection() -> None:
    mask = np.ones((50, 50), dtype=bool)
    first = survey_waypoints((2, 2), mask, spacing=12)
    second = survey_waypoints((2, 2), mask, spacing=12)
    assert first == second
    assert set(first) == {(6, 6), (6, 18), (6, 30), (6, 42),
                          (18, 6), (18, 18), (18, 30), (18, 42),
                          (30, 6), (30, 18), (30, 30), (30, 42),
                          (42, 6), (42, 18), (42, 30), (42, 42)}


def test_skip_walls_and_reachable_nearest_neighbour_order() -> None:
    mask = np.ones((20, 20), dtype=bool)
    mask[6, 6] = False
    mask[10, 6] = True
    first = survey_waypoints((6, 5), mask, spacing=12)
    assert (6, 6) not in first
    assert first[0] == (18, 6)

    separated = np.ones((20, 20), dtype=bool)
    separated[10, :] = False
    assert (18, 18) not in survey_waypoints((2, 2), separated, spacing=12)


def test_empty_result_handling() -> None:
    mask = np.ones((10, 10), dtype=bool)
    assert survey_waypoints((2, 2), mask, spacing=12) == [(6, 6)]
    mask[2, 2] = False
    assert survey_waypoints((2, 2), mask) == []
    assert reachable_free_cells((2, 2), mask) == set()


def test_spacing_must_be_positive() -> None:
    try:
        survey_waypoints((0, 0), np.ones((10, 10), dtype=bool), spacing=0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive spacing did not raise")


if __name__ == "__main__":
    test_spacing_and_deterministic_selection()
    test_skip_walls_and_reachable_nearest_neighbour_order()
    test_empty_result_handling()
    test_spacing_must_be_positive()
    print("survey tests passed")
