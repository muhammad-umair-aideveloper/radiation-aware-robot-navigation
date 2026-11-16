"""Online measure-estimate-replan simulation utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from src.environment import RadiationEnv
from src.estimators import idw, kernel_regression
from src.geiger_model import counts_to_intensity, simulate_counts
from src.measurement_patterns import pattern_cells
from src.planners import astar_search


def estimate_field(
    readings: Dict[Tuple[int, int], List[float]],
    free_mask: np.ndarray,
    estimator: str,
) -> np.ndarray:
    """Estimate a nonnegative radiation field from pooled readings."""
    cells = np.array(list(readings.keys()), dtype=int)
    values = np.array([np.mean(v) for v in readings.values()], dtype=float)
    mask = np.asarray(free_mask, dtype=bool)
    if len(cells) < 3:
        field = np.zeros(mask.shape, dtype=float)
        field[mask] = float(values.mean())
        return field
    if estimator == "kernel":
        field, _ = kernel_regression(cells, values, mask)
    elif estimator == "idw":
        field = idw(cells, values, mask)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return np.maximum(field, 0.0)


def run_online_episode(
    true_field: np.ndarray,
    free_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    estimator: str,
    pattern: str,
    dwell: float,
    seed: int,
    replan_every: int = 3,
    lam: float = 2.5,
    max_steps: int = 300,
) -> Dict[str, object]:
    """Run one noisy online measurement and replanning episode."""
    readings: Dict[Tuple[int, int], List[float]] = {}
    measurement_index = 0

    def take(cell: Tuple[int, int]) -> None:
        nonlocal measurement_index
        cells = pattern_cells(cell, free_mask, pattern)
        counts = simulate_counts(
            true_field, cells, dwell, seed * 100003 + measurement_index
        )
        measurement_index += 1
        values = counts_to_intensity(counts, dwell)
        for measured_cell, value in zip(cells, values):
            key = (int(measured_cell[0]), int(measured_cell[1]))
            readings.setdefault(key, []).append(float(value))

    cur = (int(start[0]), int(start[1]))
    target = (int(goal[0]), int(goal[1]))
    path: List[Tuple[int, int]] = [cur]
    dose = 0.0
    steps = 0
    n_replans = 0
    take(cur)

    while cur != target and steps < max_steps:
        estimated = estimate_field(readings, free_mask, estimator)
        planning_env = RadiationEnv(
            start=cur,
            goal=target,
            walls=~np.asarray(free_mask, dtype=bool),
            radiation=estimated,
        )
        planning_env.shape = estimated.shape
        plan, _, _, _ = astar_search(
            cur, target, mode="aware", env=planning_env, lambda_rad=lam
        )
        n_replans += 1
        if plan is None or len(plan) < 2:
            break
        for nxt_value in plan[1 : 1 + replan_every]:
            nxt = (int(nxt_value[0]), int(nxt_value[1]))
            dr = nxt[0] - cur[0]
            dc = nxt[1] - cur[1]
            dose += float(true_field[nxt]) * math.sqrt(dr * dr + dc * dc)
            path.append(nxt)
            cur = nxt
            steps += 1
            take(cur)
            if cur == target or steps >= max_steps:
                break

    return {
        "path": path,
        "dose": dose,
        "steps": steps,
        "success": cur == target,
        "n_replans": n_replans,
        "n_readings": sum(len(values) for values in readings.values()),
        "n_distinct_cells": len(readings),
        "readings": readings,
    }
