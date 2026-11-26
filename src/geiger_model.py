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
    field = np.asarray(field, dtype=float)
    cells = np.asarray(cells)
    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cells must have shape (n, 2)")
    if not np.isfinite(dwell) or dwell <= 0.0:
        raise ValueError("dwell must be positive and finite")
    if not np.all(np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError("field must contain finite non-negative intensities")
    if not np.all(np.isfinite(cells)) or np.any(cells != np.floor(cells)):
        raise ValueError("cells must contain integer coordinates")
    if np.any(cells < 0) or np.any(cells[:, 0] >= field.shape[0]) or np.any(
        cells[:, 1] >= field.shape[1]
    ):
        raise ValueError("cells must be within the field shape")
    intensity = field[tuple(cells.astype(int).T)]
    rng = np.random.default_rng(seed)
    return rng.poisson(float(dwell) * intensity).astype(float)


def counts_to_intensity(counts: np.ndarray, dwell: float) -> np.ndarray:
    """Convert Geiger counts into intensity estimates."""
    if not np.isfinite(dwell) or dwell <= 0.0:
        raise ValueError("dwell must be positive and finite")
    return np.asarray(counts, dtype=float) / float(dwell)
