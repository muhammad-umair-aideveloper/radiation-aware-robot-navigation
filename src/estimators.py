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


def parametric_sources(
    cells: np.ndarray,
    values: np.ndarray,
    free_mask: np.ndarray,
    k: int = 3,
    softening: float = 2.0,
    refine_passes: int = 5,
) -> Tuple[np.ndarray, list[Tuple[int, int, float]]]:
    """Fit k inverse-square point sources to measured intensities.

    Model: I(c) = sum_s w_s / (d(c, p_s)**2 + softening), w_s >= 0, p_s any grid cell.
    The functional form, the softening and the number of sources k are ASSUMED KNOWN.
    Only the measurements and the free mask are used (never the true field).
    Returns (field, sources) with sources = [(row, col, weight), ...].
    """
    cells = np.asarray(cells, dtype=int).reshape(-1, 2)
    y = np.asarray(values, dtype=float)
    h, w = free_mask.shape
    cand = np.argwhere(np.ones((h, w), dtype=bool))  # candidate index = row * w + col
    d2 = ((cells[:, None, :] - cand[None, :, :]) ** 2).sum(axis=2)
    phi = 1.0 / (d2 + softening)
    norms = np.linalg.norm(phi, axis=0)

    def solve(idx):
        a = phi[:, idx]
        wt, *_ = np.linalg.lstsq(a, y, rcond=None)
        wt = np.clip(wt, 0.0, None)
        return wt, float(((y - a @ wt) ** 2).sum())

    chosen = []
    resid = y.copy()
    for _ in range(k):
        corr = np.abs(phi.T @ resid) / np.maximum(norms, 1e-12)
        if chosen:
            corr[chosen] = -1.0
        chosen.append(int(np.argmax(corr)))
        wt, _ = solve(chosen)
        resid = y - phi[:, chosen] @ wt
    wt, best = solve(chosen)
    for _ in range(refine_passes):  # exchange refinement: re-pick each source over ALL cells
        improved = False
        for j in range(len(chosen)):
            others = [c for i, c in enumerate(chosen) if i != j]
            if others:
                q, _ = np.linalg.qr(phi[:, others])
                y_perp = y - q @ (q.T @ y)
                phi_perp = phi - q @ (q.T @ phi)
            else:
                y_perp, phi_perp = y, phi
            score = (phi_perp.T @ y_perp) ** 2 / np.maximum((phi_perp ** 2).sum(axis=0), 1e-12)
            score[others] = -1.0
            new_id = int(np.argmax(score))
            if new_id == chosen[j]:
                continue
            trial = chosen.copy()
            trial[j] = new_id
            wt2, err2 = solve(trial)
            if err2 < best - 1e-12:
                chosen, wt, best, improved = trial, wt2, err2, True
        if not improved:
            break
    rows, cols = np.indices((h, w))
    field = np.zeros((h, w), dtype=float)
    sources = []
    for pid, weight in zip(chosen, wt):
        sr, sc = divmod(pid, w)
        field += weight / ((rows - sr) ** 2 + (cols - sc) ** 2 + softening)
        sources.append((int(sr), int(sc), float(weight)))
    field = np.clip(field, 0.0, None)
    field[~free_mask] = 0.0
    return field, sources


def parametric_sources_shielded(
    cells: np.ndarray,
    values: np.ndarray,
    free_mask: np.ndarray,
    atten: np.ndarray,
    k: int = 3,
    softening: float = 2.0,
    refine_passes: int = 5,
) -> Tuple[np.ndarray, list[Tuple[int, int, float]]]:
    """Like parametric_sources, but every source is attenuated by the known walls (atten matrix).

    Model: I(c) = sum_s w_s * atten[p_s, c] / (d(c, p_s)**2 + softening), w_s >= 0.
    The form, the softening, k and the wall attenuation are ASSUMED KNOWN.
    Returns (field, sources) with sources = [(row, col, weight), ...].
    """
    cells = np.asarray(cells, dtype=int).reshape(-1, 2)
    y = np.asarray(values, dtype=float)
    h, w = free_mask.shape
    cand = np.argwhere(np.ones((h, w), dtype=bool))
    d2 = ((cells[:, None, :] - cand[None, :, :]) ** 2).sum(axis=2)
    flat = cells[:, 0] * w + cells[:, 1]
    phi = atten[:, flat].T / (d2 + softening)
    norms = np.linalg.norm(phi, axis=0)

    def solve(idx):
        a = phi[:, idx]
        wt, *_ = np.linalg.lstsq(a, y, rcond=None)
        wt = np.clip(wt, 0.0, None)
        return wt, float(((y - a @ wt) ** 2).sum())

    chosen = []
    resid = y.copy()
    for _ in range(k):
        corr = np.abs(phi.T @ resid) / np.maximum(norms, 1e-12)
        if chosen:
            corr[chosen] = -1.0
        chosen.append(int(np.argmax(corr)))
        wt, _ = solve(chosen)
        resid = y - phi[:, chosen] @ wt
    wt, best = solve(chosen)
    for _ in range(refine_passes):
        improved = False
        for j in range(len(chosen)):
            others = [c for i, c in enumerate(chosen) if i != j]
            if others:
                q, _ = np.linalg.qr(phi[:, others])
                y_perp = y - q @ (q.T @ y)
                phi_perp = phi - q @ (q.T @ phi)
            else:
                y_perp, phi_perp = y, phi
            score = (phi_perp.T @ y_perp) ** 2 / np.maximum((phi_perp ** 2).sum(axis=0), 1e-12)
            score[others] = -1.0
            new_id = int(np.argmax(score))
            if new_id == chosen[j]:
                continue
            trial = chosen.copy()
            trial[j] = new_id
            wt2, err2 = solve(trial)
            if err2 < best - 1e-12:
                chosen, wt, best, improved = trial, wt2, err2, True
        if not improved:
            break
    rows, cols = np.indices((h, w))
    field = np.zeros((h, w), dtype=float)
    sources = []
    for pid, weight in zip(chosen, wt):
        sr, sc = divmod(pid, w)
        field += weight * atten[pid].reshape(h, w) / ((rows - sr) ** 2 + (cols - sc) ** 2 + softening)
        sources.append((int(sr), int(sc), float(weight)))
    field = np.clip(field, 0.0, None)
    field[~free_mask] = 0.0
    return field, sources
