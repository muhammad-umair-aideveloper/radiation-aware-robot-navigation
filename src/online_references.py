"""Reference and matched-offline calculations for study S8."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.environment import RadiationEnv
from src.estimation_study import run_configuration
from src.planners import astar_search


def reference_doses(starts: Iterable[Tuple[int, int]]) -> Dict[str, float]:
    """Compute oracle and shortest-path dose and step means."""
    cells = list(starts)
    env = RadiationEnv()
    oracle = [
        astar_search(cell, env.goal, mode="aware", env=env, lambda_rad=2.5)
        for cell in cells
    ]
    shortest = [
        astar_search(cell, env.goal, mode="shortest", env=env)
        for cell in cells
    ]
    return {
        "oracle_dose_mean": float(np.mean([item[1] for item in oracle])),
        "shortest_dose_mean": float(np.mean([item[1] for item in shortest])),
        "oracle_steps_mean": float(np.mean([item[2] for item in oracle])),
        "shortest_steps_mean": float(np.mean([item[2] for item in shortest])),
        "n_starts": float(len(cells)),
    }


def offline_matched(
    estimator: str,
    n_cells: int,
    dwell: float,
    seeds: Iterable[int],
    starts: List[Tuple[int, int]],
) -> Dict[str, object]:
    """Run S7's offline procedure for a requested seed set."""
    import src.estimation_study as study

    requested_seeds = tuple(int(seed) for seed in seeds)
    previous_seeds = study.SEEDS
    study.SEEDS = requested_seeds
    try:
        result = run_configuration(estimator, n_cells, dwell, RadiationEnv(), starts)
    finally:
        study.SEEDS = previous_seeds

    reference = reference_doses(starts)
    doses = [item["true_dose"] for item in result["seed_metrics"]]
    regrets = [
        (dose - reference["oracle_dose_mean"])
        / reference["oracle_dose_mean"]
        * 100.0
        for dose in doses
    ]
    reductions = [
        (reference["shortest_dose_mean"] - dose)
        / reference["shortest_dose_mean"]
        * 100.0
        for dose in doses
    ]
    return {
        "n_cells": int(n_cells),
        "per_seed_mean_dose": [float(dose) for dose in doses],
        "dose_regret_percent": [float(value) for value in regrets],
        "dose_reduction_percent": [float(value) for value in reductions],
    }
