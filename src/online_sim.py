"""Online measure-estimate-replan simulation utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from src.environment import RadiationEnv
from src.estimators import idw, kernel_regression, parametric_sources
from src.geiger_model import counts_to_intensity, simulate_counts
from src.measurement_patterns import pattern_cells
from src.planners import astar_search
from src.survey import survey_waypoints


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
    elif estimator.startswith("omp"):
        k = int(estimator[3:])
        field, _ = parametric_sources(cells, values, mask, k=k)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return np.maximum(field, 0.0)


def inflate_unknown(
    est: np.ndarray,
    readings: Dict[Tuple[int, int], List[float]],
    free_mask: np.ndarray,
    beta: float,
    radius: int = 4,
) -> np.ndarray:
    """Inflate estimates in free cells far from measured cells."""
    measured = np.array(list(readings.keys()), dtype=int)
    m = float(np.mean([np.mean(v) for v in readings.values()]))
    rows, cols = np.indices(free_mask.shape)
    d2 = np.full(free_mask.shape, np.inf)
    for r, c in measured:
        d2 = np.minimum(d2, (rows - r) ** 2 + (cols - c) ** 2)
    unknown = (d2 > radius**2) & free_mask
    out = est.copy()
    out[unknown] = np.maximum(out[unknown], beta * m)
    return out


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
    control: str | None = None,
    noise_free: bool = False,
    inflate_beta: float = 0.0,
    inflate_radius: int = 4,
    survey_budget: int = 0,
) -> Dict[str, object]:
    """Run one noisy online measurement and replanning episode.

    When ``survey_budget`` is positive, the robot first visits coarse-grid
    waypoints for at most that many steps, then uses the usual online loop to
    reach the goal. A budget of zero preserves the original B=0 behaviour.
    """
    if survey_budget < 0:
        raise ValueError("survey_budget must be nonnegative")
    readings: Dict[Tuple[int, int], List[float]] = {}
    measurement_index = 0

    def take(cell: Tuple[int, int]) -> None:
        nonlocal measurement_index
        cells = pattern_cells(cell, free_mask, pattern)
        if noise_free:
            values = true_field[cells[:, 0], cells[:, 1]]
        else:
            counts = simulate_counts(
                true_field, cells, dwell, seed * 100003 + measurement_index
            )
            values = counts_to_intensity(counts, dwell)
        measurement_index += 1
        for measured_cell, value in zip(cells, values):
            key = (int(measured_cell[0]), int(measured_cell[1]))
            readings.setdefault(key, []).append(float(value))

    cur = (int(start[0]), int(start[1]))
    target = (int(goal[0]), int(goal[1]))
    path: List[Tuple[int, int]] = [cur]
    dose = 0.0
    steps = 0
    n_replans = 0
    survey_steps = 0
    survey_dose = 0.0
    take(cur)

    def move_to(segment_target: Tuple[int, int], survey_phase: bool) -> None:
        nonlocal cur, dose, steps, n_replans, survey_steps, survey_dose
        while cur != segment_target and steps < max_steps:
            if survey_phase and survey_steps >= survey_budget:
                break
            if control == "true":
                planning_field = true_field
            elif control == "zero":
                planning_field = np.zeros_like(true_field)
            else:
                planning_field = estimate_field(readings, free_mask, estimator)
                if inflate_beta > 0:
                    planning_field = inflate_unknown(
                        planning_field,
                        readings,
                        free_mask,
                        inflate_beta,
                        inflate_radius,
                    )
            planning_env = RadiationEnv(
                start=cur,
                goal=segment_target,
                walls=~np.asarray(free_mask, dtype=bool),
                radiation=planning_field,
            )
            planning_env.shape = planning_field.shape
            plan, _, _, _ = astar_search(
                cur, segment_target, mode="aware", env=planning_env, lambda_rad=lam
            )
            n_replans += 1
            if plan is None or len(plan) < 2:
                break
            for nxt_value in plan[1 : 1 + replan_every]:
                if survey_phase and survey_steps >= survey_budget:
                    break
                nxt = (int(nxt_value[0]), int(nxt_value[1]))
                dr = nxt[0] - cur[0]
                dc = nxt[1] - cur[1]
                step_dose = float(true_field[nxt]) * math.sqrt(dr * dr + dc * dc)
                dose += step_dose
                if survey_phase:
                    survey_dose += step_dose
                    survey_steps += 1
                path.append(nxt)
                cur = nxt
                steps += 1
                take(cur)
                if cur == segment_target or steps >= max_steps:
                    break

    if survey_budget > 0:
        for waypoint in survey_waypoints(start, free_mask, spacing=12):
            move_to(waypoint, survey_phase=True)
            if survey_steps >= survey_budget:
                break

    move_to(target, survey_phase=False)

    return {
        "path": path,
        "dose": dose,
        "steps": steps,
        "success": cur == target,
        "n_replans": n_replans,
        "n_readings": sum(len(values) for values in readings.values()),
        "n_distinct_cells": len(readings),
        "readings": readings,
        "survey_steps": survey_steps,
        "survey_dose": survey_dose,
        "survey_dose_share": float(survey_dose / dose) if dose > 0 else 0.0,
    }
