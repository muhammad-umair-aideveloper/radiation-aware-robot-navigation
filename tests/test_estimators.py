"""Plain-assert tests for radiation-field estimators."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import estimators
from estimators import parametric_sources


def synthetic_field() -> np.ndarray:
    """Build the declared smooth synthetic radiation field."""
    rows, columns = np.indices((50, 50))
    return 100.0 / ((rows - 20) ** 2 + (columns - 25) ** 2 + 2.0)


def estimate(name: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the synthetic field from deterministic free-cell samples."""
    field = synthetic_field()
    mask = np.ones(field.shape, dtype=bool)
    mask[10:15, 10:15] = False
    rng = np.random.default_rng(23)
    free = np.flatnonzero(mask)
    cells = np.column_stack(np.unravel_index(rng.choice(free, n, replace=False), mask.shape))
    values = field[tuple(cells.T)]
    if name == "kernel":
        estimate_field, _ = estimators.kernel_regression(cells, values, mask)
    else:
        estimate_field = estimators.idw(cells, values, mask)
    return field, mask, estimate_field


def relative_error(actual: np.ndarray, estimated: np.ndarray, mask: np.ndarray) -> float:
    """Return relative L2 error over free cells."""
    return float(np.linalg.norm((actual - estimated)[mask]) / np.linalg.norm(actual[mask]))


def test_output_shape_and_mask() -> None:
    """Both estimators preserve shape and clear wall cells."""
    for name in ("kernel", "idw"):
        _, mask, result = estimate(name, 10)
        assert result.shape == mask.shape
        assert np.all(result >= 0.0)
        assert np.all(result[~mask] == 0.0)


def test_kernel_measured_values() -> None:
    """Kernel estimates are close at 200 noise-free measurement cells."""
    actual = synthetic_field()
    mask = np.ones(actual.shape, dtype=bool)
    rng = np.random.default_rng(0)
    free = np.flatnonzero(mask)
    cells = np.column_stack(np.unravel_index(rng.choice(free, 200, replace=False), mask.shape))
    measured = actual[tuple(cells.T)]
    result, _ = estimators.kernel_regression(cells, measured, mask)
    predicted = result[tuple(cells.T)]
    error = np.linalg.norm(predicted - measured) / np.linalg.norm(measured)
    assert error < 0.20


def test_more_cells_reduce_error() -> None:
    """Two hundred measurements outperform ten for both estimators."""
    actual = synthetic_field()
    mask = np.ones(actual.shape, dtype=bool)
    for name in ("kernel", "idw"):
        _, _, sparse = estimate(name, 10)
        _, _, dense = estimate(name, 200)
        assert relative_error(actual, dense, mask) < relative_error(actual, sparse, mask)


def test_signatures_do_not_take_field() -> None:
    """Estimator signatures accept measurements and a free mask only."""
    for function in (estimators.kernel_regression, estimators.idw):
        assert "field" not in inspect.signature(function).parameters


def test_parametric_source_shape_and_mask() -> None:
    """Parametric estimates are finite, nonnegative, masked, and k-sized."""
    rows, columns = np.indices((50, 50))
    actual = 150.0 / ((rows - 18) ** 2 + (columns - 20) ** 2 + 2.0)
    mask = np.ones(actual.shape, dtype=bool)
    mask[22:28, 18:35] = False
    mask[34:44, 36:39] = False
    mask[8:15, 26:29] = False
    free = np.argwhere(mask)
    cells = free[np.random.default_rng(0).choice(len(free), 60, replace=False)]
    values = actual[tuple(cells.T)]
    for k in (2, 3, 4):
        field, sources = parametric_sources(cells, values, mask, k=k)
        assert field.shape == (50, 50)
        assert np.all(np.isfinite(field))
        assert np.all(field >= 0.0)
        assert np.all(field[~mask] == 0.0)
        assert len(sources) == k


def test_parametric_source_noise_free_recovery() -> None:
    """Parametric sources recover the declared three-source field."""
    rows, columns = np.indices((50, 50))
    actual = (
        150.0 / ((rows - 18) ** 2 + (columns - 20) ** 2 + 2.0)
        + 200.0 / ((rows - 32) ** 2 + (columns - 30) ** 2 + 2.0)
        + 120.0 / ((rows - 25) ** 2 + (columns - 12) ** 2 + 2.0)
    )
    mask = np.ones(actual.shape, dtype=bool)
    mask[22:28, 18:35] = False
    mask[34:44, 36:39] = False
    mask[8:15, 26:29] = False
    free = np.argwhere(mask)
    recovered = 0
    for seed in range(20):
        cells = free[np.random.default_rng(seed).choice(len(free), 200, replace=False)]
        values = actual[tuple(cells.T)]
        estimated, _ = parametric_sources(cells, values, mask, k=3)
        error = np.linalg.norm((actual - estimated)[mask]) / np.linalg.norm(actual[mask])
        recovered += error < 1e-3
    assert recovered >= 17


def test_parametric_source_two_readings() -> None:
    """Parametric estimation accepts two readings."""
    mask = np.ones((5, 5), dtype=bool)
    field, _ = parametric_sources(
        np.array([[0, 0], [4, 4]]), np.array([1.0, 2.0]), mask
    )
    assert field.shape == mask.shape


def test_parametric_source_is_deterministic() -> None:
    """Repeated parametric estimation returns identical fields."""
    mask = np.ones((5, 5), dtype=bool)
    cells = np.array([[0, 0], [2, 2], [4, 4]])
    values = np.array([1.0, 2.0, 1.0])
    first, _ = parametric_sources(cells, values, mask)
    second, _ = parametric_sources(cells, values, mask)
    assert np.array_equal(first, second)


def test_parametric_source_signature() -> None:
    """Parametric estimation cannot receive a true field."""
    assert list(inspect.signature(parametric_sources).parameters) == [
        "cells", "values", "free_mask", "k", "softening", "refine_passes"
    ]


if __name__ == "__main__":
    test_output_shape_and_mask()
    test_kernel_measured_values()
    test_more_cells_reduce_error()
    test_signatures_do_not_take_field()
    test_parametric_source_shape_and_mask()
    test_parametric_source_noise_free_recovery()
    test_parametric_source_two_readings()
    test_parametric_source_is_deterministic()
    test_parametric_source_signature()
    print("PASS: estimator tests")
