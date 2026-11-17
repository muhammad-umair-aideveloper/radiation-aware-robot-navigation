"""Tests for online measurement patterns."""

import sys

import numpy as np

sys.path.insert(0, "src")

from measurement_patterns import pattern_cells


def test_pattern_sizes() -> None:
    free_mask = np.ones((5, 5), dtype=bool)
    free_mask[1, 1] = False
    assert len(pattern_cells((2, 2), free_mask, "cell")) == 1
    assert len(pattern_cells((2, 2), free_mask, "3x3")) == 8
    assert len(pattern_cells((0, 0), free_mask, "3x3")) == 3


def test_cells_are_free_and_in_bounds() -> None:
    free_mask = np.ones((5, 5), dtype=bool)
    free_mask[1, 1] = False
    cells = pattern_cells((2, 2), free_mask, "3x3")
    assert np.all(cells[:, 0] >= 0)
    assert np.all(cells[:, 0] < free_mask.shape[0])
    assert np.all(cells[:, 1] >= 0)
    assert np.all(cells[:, 1] < free_mask.shape[1])
    assert all(free_mask[tuple(cell)] for cell in cells)


def test_disc_pattern_sizes() -> None:
    free_mask = np.ones((30, 30), dtype=bool)
    assert len(pattern_cells((15, 15), free_mask, "disc3")) == 29
    assert len(pattern_cells((15, 15), free_mask, "disc6")) == 113


def test_disc_cells_are_free_and_in_bounds() -> None:
    free_mask = np.ones((15, 15), dtype=bool)
    free_mask[7, 8] = False
    cells = pattern_cells((7, 7), free_mask, "disc6")
    assert np.all(cells[:, 0] >= 0)
    assert np.all(cells[:, 0] < free_mask.shape[0])
    assert np.all(cells[:, 1] >= 0)
    assert np.all(cells[:, 1] < free_mask.shape[1])
    assert all(free_mask[tuple(cell)] for cell in cells)


def test_disc_zero_is_robot_cell() -> None:
    free_mask = np.ones((5, 5), dtype=bool)
    cells = pattern_cells((2, 3), free_mask, "disc0")
    assert cells.tolist() == [[2, 3]]


def test_unknown_pattern_raises() -> None:
    try:
        pattern_cells((2, 2), np.ones((5, 5), dtype=bool), "unknown")
    except ValueError:
        return
    raise AssertionError("unknown pattern did not raise ValueError")


if __name__ == "__main__":
    test_pattern_sizes()
    test_cells_are_free_and_in_bounds()
    test_disc_pattern_sizes()
    test_disc_cells_are_free_and_in_bounds()
    test_disc_zero_is_robot_cell()
    test_unknown_pattern_raises()
    print("measurement pattern tests passed")
