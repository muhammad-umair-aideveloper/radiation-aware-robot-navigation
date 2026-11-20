"""Resumable Study S12: survey tour versus direct online replanning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import generate_evaluation_starts
from src.environment import RadiationEnv
from src.estimation_study import bootstrap
from src.online_controls_s12 import run_control
from src.online_references import reference_doses
from src.online_sim import run_online_episode

ESTIMATORS = ("kernel", "omp3")
DWELLS = (10, 1)
BUDGETS = (30, 60)
SEEDS = tuple(range(5))
FALLBACK_SEEDS = tuple(range(3))
OUTPUT = ROOT / "results" / "s12"


def run_configuration(
    estimator: str,
    dwell: float,
    survey_budget: int,
    seeds: Iterable[int],
) -> Dict[str, Any]:
    """Run and summarize one S12 configuration."""
    started = time.perf_counter()
    env = RadiationEnv()
    starts = tuple(generate_evaluation_starts(env, 50, 123)[::2])
    references = reference_doses(list(starts))
    seed_values = [int(seed) for seed in seeds]
    per_seed: List[Dict[str, float]] = []

    for seed in seed_values:
        episodes = [
            run_online_episode(
                env.radiation,
                ~env.walls,
                start,
                env.goal,
                estimator,
                "3x3",
                dwell,
                seed * 1000 + index,
                replan_every=3,
                lam=2.5,
                max_steps=300 + survey_budget,
                survey_budget=survey_budget,
            )
            for index, start in enumerate(starts)
        ]
        successful = [episode for episode in episodes if episode["success"]]
        mean_dose = float(
            np.mean([episode["dose"] for episode in successful])
        ) if successful else float("inf")
        mean_survey_dose = float(
            np.mean([episode["survey_dose"] for episode in successful])
        ) if successful else float("inf")
        mean_dose_share = float(
            np.mean([episode["survey_dose_share"] for episode in successful])
        ) if successful else float("inf")
        per_seed.append({
            "seed": float(seed),
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
            "mean_steps": float(
                np.mean([episode["steps"] for episode in successful])
            ) if successful else float("inf"),
            "mean_true_dose": mean_dose,
            "survey_steps": float(
                np.mean([episode["survey_steps"] for episode in episodes])
            ),
            "survey_dose_share": mean_dose_share,
            "mean_distinct_cells": float(
                np.mean([episode["n_distinct_cells"] for episode in episodes])
            ),
            "dose_regret_percent": (
                (mean_dose - references["oracle_dose_mean"])
                / references["oracle_dose_mean"] * 100.0
            ) if successful else float("inf"),
            "dose_reduction_percent": (
                (references["shortest_dose_mean"] - mean_dose)
                / references["shortest_dose_mean"] * 100.0
            ) if successful else float("inf"),
        })

    metrics = {
        "success_rate_mean": float(np.mean([item["success_rate"] for item in per_seed])),
        "mean_steps": float(np.mean([item["mean_steps"] for item in per_seed])),
        "mean_true_dose": float(np.mean([item["mean_true_dose"] for item in per_seed]))
        if all(np.isfinite(item["mean_true_dose"]) for item in per_seed) else float("inf"),
        "survey_steps": float(np.mean([item["survey_steps"] for item in per_seed])),
        "survey_dose_share": float(np.mean([item["survey_dose_share"] for item in per_seed]))
        if all(np.isfinite(item["survey_dose_share"]) for item in per_seed) else float("inf"),
        "mean_distinct_cells": float(
            np.mean([item["mean_distinct_cells"] for item in per_seed])
        ),
        "dose_regret_percent": bootstrap(
            item["dose_regret_percent"] for item in per_seed
        ),
        "dose_reduction_percent": bootstrap(
            item["dose_reduction_percent"] for item in per_seed
        ),
    }
    return {
        "estimator": estimator,
        "dwell": dwell,
        "survey_budget": survey_budget,
        "pattern": "3x3",
        "seeds": seed_values,
        "n_starts": len(starts),
        "per_seed": per_seed,
        "metrics": metrics,
        "references": references,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def configuration_path(
    estimator: str,
    dwell: float,
    survey_budget: int,
) -> Path:
    """Return the resumable result path for one configuration."""
    return OUTPUT / estimator / f"3x3_t{dwell:g}_b{survey_budget}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    control = run_control()
    if not control["passed"]:
        print(json.dumps({"passed": False, "error": "S12 control failed"}))
        raise SystemExit(1)

    if args.one:
        result = run_configuration("kernel", 1, 30, args.seeds or (0,))
        projected = result["elapsed_seconds"] * 8 * 5 / 60.0
        print(json.dumps({
            "control_passed": True,
            "metrics": result,
            "projected_total_minutes": projected,
            "seed_rule": "seeds 0 to 4 unless projected runtime exceeds 90 minutes",
        }, indent=2))
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    projected_seeds = tuple(args.seeds) if args.seeds else SEEDS
    for estimator in ESTIMATORS:
        for dwell in DWELLS:
            for survey_budget in BUDGETS:
                path = configuration_path(estimator, dwell, survey_budget)
                if path.exists():
                    print(f"resuming: skipping {path.relative_to(ROOT)}")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                result = run_configuration(
                    estimator,
                    dwell,
                    survey_budget,
                    projected_seeds,
                )
                path.write_text(json.dumps(result, indent=2))
                print(
                    f"finished {estimator}/t{dwell:g}/b{survey_budget} "
                    f"{result['elapsed_seconds']:.2f}s"
                )


if __name__ == "__main__":
    main()
