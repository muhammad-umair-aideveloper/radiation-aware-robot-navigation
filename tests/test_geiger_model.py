"""Plain-assert tests for the noisy Geiger measurement model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import geiger_model


def test_seed_reproducibility() -> None:
    """The same seed reproduces sampled cells and counts."""
    mask = np.ones((8, 8), dtype=bool)
    field = np.full((8, 8), 3.0)
    cells_a = geiger_model.sample_measurement_cells(mask, 6, seed=7)
    cells_b = geiger_model.sample_measurement_cells(mask, 6, seed=7)
    counts_a = geiger_model.simulate_counts(field, cells_a, dwell=2.0, seed=9)
    counts_b = geiger_model.simulate_counts(field, cells_b, dwell=2.0, seed=9)
    assert np.array_equal(cells_a, cells_b)
    assert np.array_equal(counts_a, counts_b)


def test_cells_are_distinct_and_free() -> None:
    """Sampled cells are distinct and never walls."""
    mask = np.zeros((5, 6), dtype=bool)
    mask[0, 1] = True
    mask[2, 3] = True
    mask[4, 5] = True
    cells = geiger_model.sample_measurement_cells(mask, 3, seed=11)
    assert len({tuple(cell) for cell in cells}) == 3
    assert all(mask[tuple(cell)] for cell in cells)


def test_poisson_mean() -> None:
    """The empirical mean is within 5 percent of a Poisson mean of 50."""
    field = np.array([[50.0]])
    cells = np.zeros((1, 2), dtype=int)
    draws = [
        geiger_model.simulate_counts(field, cells, dwell=1.0, seed=seed)[0]
        for seed in range(2000)
    ]
    assert abs(float(np.mean(draws)) - 50.0) / 50.0 < 0.05


def test_relative_noise_decreases_with_counts() -> None:
    """Relative Poisson noise is lower when the expected count is 400."""
    cells = np.zeros((1, 2), dtype=int)
    low = np.array([[4.0]])
    high = np.array([[400.0]])
    low_draws = [
        geiger_model.simulate_counts(low, cells, dwell=1.0, seed=seed)[0]
        for seed in range(2000)
    ]
    high_draws = [
        geiger_model.simulate_counts(high, cells, dwell=1.0, seed=seed)[0]
        for seed in range(2000)
    ]
    low_relative = float(np.std(low_draws)) / 4.0
    high_relative = float(np.std(high_draws)) / 400.0
    assert high_relative < low_relative


if __name__ == "__main__":
    test_seed_reproducibility()
    test_cells_are_distinct_and_free()
    test_poisson_mean()
    test_relative_noise_decreases_with_counts()
    print("PASS: geiger model tests")
