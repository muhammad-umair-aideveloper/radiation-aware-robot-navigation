"""NumPy-only radiation-field estimators from noisy point measurements."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _inputs(
    cells: np.ndarray, values: np.ndarray, free_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize estimator inputs."""
    points = np.asarray(cells, dtype=float)
    measurements = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(free_mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("cells must have shape (n, 2)")
    if len(points) != len(measurements) or len(points) == 0:
        raise ValueError("cells and values must have the same nonzero length")
    if mask.ndim != 2:
        raise ValueError("free_mask must be two-dimensional")
    return points, measurements, mask


def _predict(
    points: np.ndarray,
    measurements: np.ndarray,
    coordinates: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Predict values at coordinates with a Gaussian kernel."""
    distances = coordinates[:, None, :] - points[None, :, :]
    weights = np.exp(-np.sum(distances * distances, axis=2) / (2.0 * bandwidth**2))
    denominator = np.sum(weights, axis=1)
    prediction = weights @ measurements
    fallback = float(np.mean(measurements))
    return np.divide(
        prediction,
        denominator,
        out=np.full(len(coordinates), fallback, dtype=float),
        where=denominator > 0.0,
    )


def kernel_regression(
    cells: np.ndarray,
    values: np.ndarray,
    free_mask: np.ndarray,
    bandwidths: Sequence[float] = (1, 2, 3, 4, 6, 8, 12),
) -> Tuple[np.ndarray, float]:
    """Estimate a field and select bandwidth by leave-one-out error."""
    points, measurements, mask = _inputs(cells, values, free_mask)
    candidates = tuple(float(value) for value in bandwidths)
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("bandwidths must contain positive values")

    errors = []
    for bandwidth in candidates:
        if len(points) == 1:
            errors.append(0.0)
            continue
        distances = points[:, None, :] - points[None, :, :]
        weights = np.exp(
            -np.sum(distances * distances, axis=2) / (2.0 * bandwidth**2)
        )
        np.fill_diagonal(weights, 0.0)
        denominator = np.sum(weights, axis=1)
        prediction = np.divide(
            weights @ measurements,
            denominator,
            out=np.full(len(points), float(np.mean(measurements))),
            where=denominator > 0.0,
        )
        errors.append(float(np.mean((prediction - measurements) ** 2)))

    best_bandwidth = candidates[int(np.argmin(errors))]
    rows, columns = np.indices(mask.shape)
    coordinates = np.column_stack((rows.ravel(), columns.ravel())).astype(float)
    field = _predict(points, measurements, coordinates, best_bandwidth).reshape(mask.shape)
    return np.where(mask, np.maximum(field, 0.0), 0.0), best_bandwidth


def idw(
    cells: np.ndarray,
    values: np.ndarray,
    free_mask: np.ndarray,
    power: float = 2.0,
    eps: float = 0.5,
) -> np.ndarray:
    """Estimate a field with inverse-distance weighting."""
    points, measurements, mask = _inputs(cells, values, free_mask)
    if power <= 0.0 or eps <= 0.0:
        raise ValueError("power and eps must be positive")
    rows, columns = np.indices(mask.shape)
    coordinates = np.column_stack((rows.ravel(), columns.ravel())).astype(float)
    distances = coordinates[:, None, :] - points[None, :, :]
    squared = np.sum(distances * distances, axis=2)
    weights = 1.0 / (squared + eps) ** (power / 2.0)
    field = ((weights @ measurements) / np.sum(weights, axis=1)).reshape(mask.shape)
    return np.where(mask, np.maximum(field, 0.0), 0.0)
