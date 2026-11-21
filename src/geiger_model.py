"""Simple noisy Geiger-count measurement model."""

from __future__ import annotations


import numpy as np


def sample_measurement_cells(
    free_mask: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Sample distinct free cells as ``(row, col)`` pairs."""
    mask = np.asarray(free_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("free_mask must be two-dimensional")
    if n < 0:
        raise ValueError("n must be non-negative")

    free_indices = np.flatnonzero(mask)
    if n > len(free_indices):
        raise ValueError("n cannot exceed the number of free cells")

    rng = np.random.default_rng(seed)
    selected = rng.choice(free_indices, size=n, replace=False)
    return np.column_stack(np.unravel_index(selected, mask.shape)).astype(int)


def simulate_counts(
    field: np.ndarray, cells: np.ndarray, dwell: float, seed: int
) -> np.ndarray:
    """Draw Poisson counts with means equal to dwell times intensity."""
    intensity = np.asarray(field)[tuple(np.asarray(cells, dtype=int).T)]
    rng = np.random.default_rng(seed)
    return rng.poisson(float(dwell) * intensity).astype(float)


def counts_to_intensity(counts: np.ndarray, dwell: float) -> np.ndarray:
    """Convert Geiger counts into intensity estimates."""
    return np.asarray(counts, dtype=float) / float(dwell)
