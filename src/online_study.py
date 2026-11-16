"""Resumable S8 online measure-estimate-replan study runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import generate_evaluation_starts
from src.environment import RadiationEnv
from src.estimation_study import bootstrap
from src.online_references import offline_matched, reference_doses
from src.online_sim import run_online_episode

ESTIMATORS = ("kernel", "idw")
PATTERNS = ("cell", "3x3")
DWELLS = (10, 1, 0.1)
SEEDS = range(5)
REPLAN_EVERY = 3
BASE_ENV = RadiationEnv()
STARTS = tuple(generate_evaluation_starts(BASE_ENV, 50, 123)[::2])
TRUE_FIELD = BASE_ENV.radiation
FREE_MASK = ~BASE_ENV.walls
GOAL = BASE_ENV.goal
LAM = 2.5
MAX_STEPS = 300
OUTPUT = ROOT / "results" / "online"


def _per_seed(
    estimator: str, pattern: str, dwell: float, seed: int
) -> Dict[str, float]:
    """Run one seed over the fixed S8 starts."""
    episodes = [
        run_online_episode(
            TRUE_FIELD,
            FREE_MASK,
            start,
            GOAL,
            estimator,
            pattern,
            dwell,
            seed * 1000 + index,
            REPLAN_EVERY,
            LAM,
            MAX_STEPS,
        )
        for index, start in enumerate(STARTS)
    ]
    successful = [item for item in episodes if item["success"]]
    dose = float(np.mean([item["dose"] for item in successful]))
    return {
        "seed": float(seed),
        "mean_readings": float(np.mean([item["n_readings"] for item in episodes])),
        "mean_distinct_cells": float(
            np.mean([item["n_distinct_cells"] for item in episodes])
        ),
        "mean_replans": float(np.mean([item["n_replans"] for item in episodes])),
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "mean_steps": float(np.mean([item["steps"] for item in successful])),
        "mean_true_dose": dose,
    }


def _metric_summary(
    per_seed: List[Dict[str, float]], reference: Dict[str, float]
) -> None:
    """Add per-seed dose comparisons against the fixed references."""
    for item in per_seed:
        dose = item["mean_true_dose"]
        item["dose_regret_percent"] = (
            (dose - reference["oracle_dose_mean"])
            / reference["oracle_dose_mean"]
            * 100.0
        )
        item["dose_reduction_percent"] = (
            (reference["shortest_dose_mean"] - dose)
            / reference["shortest_dose_mean"]
            * 100.0
        )


def _aggregate(
    per_seed: List[Dict[str, float]], regret_key: str, reduction_key: str
) -> Dict[str, Any]:
    """Bootstrap dose metrics and average the remaining online metrics."""
    return {
        "dose_regret_percent": bootstrap(item[regret_key] for item in per_seed),
        "dose_reduction_percent": bootstrap(
            item[reduction_key] for item in per_seed
        ),
        "success_rate_mean": float(np.mean([item["success_rate"] for item in per_seed])),
        "mean_distinct_cells": float(
            np.mean([item["mean_distinct_cells"] for item in per_seed])
        ),
        "mean_readings": float(np.mean([item["mean_readings"] for item in per_seed])),
        "mean_replans": float(np.mean([item["mean_replans"] for item in per_seed])),
        "mean_steps": float(np.mean([item["mean_steps"] for item in per_seed])),
    }


def run_configuration(
    estimator: str,
    pattern: str,
    dwell: float,
    seeds: Iterable[int],
    reference: Dict[str, float],
) -> Dict[str, Any]:
    """Run and summarize one online configuration."""
    started = time.perf_counter()
    seed_values = [int(seed) for seed in seeds]
    per_seed = [_per_seed(estimator, pattern, dwell, seed) for seed in seed_values]
    _metric_summary(per_seed, reference)
    n_cells = int(round(np.mean([item["mean_distinct_cells"] for item in per_seed])))
    offline = offline_matched(estimator, n_cells, dwell, seed_values, list(STARTS))
    online = _aggregate(per_seed, "dose_regret_percent", "dose_reduction_percent")
    offline_summary = {
        "n_cells": n_cells,
        "dose_regret_percent": bootstrap(offline["dose_regret_percent"]),
        "dose_reduction_percent": bootstrap(offline["dose_reduction_percent"]),
    }
    return {
        "estimator": estimator,
        "pattern": pattern,
        "dwell": dwell,
        "seeds": seed_values,
        "n_starts": len(STARTS),
        "per_seed": per_seed,
        "online": online,
        "offline_matched": offline_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _name(estimator: str, pattern: str, dwell: float) -> str:
    """Return the stable configuration filename."""
    return f"{estimator}_{pattern}_t{dwell:g}.json"


def main() -> None:
    """Run either the one-configuration check or all resumable configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    args = parser.parse_args()
    reference = reference_doses(STARTS)
    if args.one:
        result = run_configuration("kernel", "3x3", 1, [0], reference)
        projected = result["elapsed_seconds"] * 5 * 12 / 60.0
        print(json.dumps({"metrics": result, "projected_total_minutes": projected}))
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "references.json").write_text(json.dumps(reference, indent=2))
    for estimator in ESTIMATORS:
        for pattern in PATTERNS:
            for dwell in DWELLS:
                path = OUTPUT / _name(estimator, pattern, dwell)
                if path.exists():
                    continue
                result = run_configuration(estimator, pattern, dwell, SEEDS, reference)
                path.write_text(json.dumps(result, indent=2))
                print(
                    f"finished {estimator}/{pattern}/t{dwell:g} "
                    f"{result['elapsed_seconds']:.2f}s"
                )


if __name__ == "__main__":
    main()
