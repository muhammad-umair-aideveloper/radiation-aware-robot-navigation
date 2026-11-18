"""Resumable S10 offline and online parametric-estimator study."""

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

from src.environment import RadiationEnv
from src.estimation_study import bootstrap
from src.geiger_model import counts_to_intensity, sample_measurement_cells, simulate_counts
from src.map_error_study import evaluate_path_on_true_field
from src.online_references import reference_doses
from src.online_sim import estimate_field, run_online_episode
from src.online_study2 import FREE_MASK, GOAL, STARTS, TRUE_FIELD
from src.planners import astar_search

SEEDS = range(5)
SEEDS_FALLBACK = range(3)
LAM = 2.5
MAX_STEPS = 300
REPLAN_EVERY = 3
OUTPUT = ROOT / "results" / "param"


def _references() -> Dict[str, float]:
    """Return the shared oracle and shortest-path reference metrics."""
    return reference_doses(list(STARTS))


def _offline_seed(
    estimator: str,
    n: int,
    dwell: float,
    seed: int,
    references: Dict[str, float],
) -> Dict[str, float]:
    """Run one offline estimator configuration for one seed."""
    cells = sample_measurement_cells(FREE_MASK, n, seed)
    counts = simulate_counts(TRUE_FIELD, cells, dwell, seed + 1000)
    values = counts_to_intensity(counts, dwell)
    readings = {
        (int(row), int(column)): [float(value)]
        for (row, column), value in zip(cells, values)
    }
    estimated = estimate_field(readings, FREE_MASK, estimator)
    planned_doses: List[float] = []
    planned_steps: List[float] = []
    successes = []
    planning_env = RadiationEnv(
        goal=GOAL, walls=~np.asarray(FREE_MASK, dtype=bool), radiation=estimated
    )
    planning_env.shape = estimated.shape
    for start in STARTS:
        path, _, _, _ = astar_search(
            start, GOAL, mode="aware", env=planning_env, lambda_rad=LAM
        )
        success = path is not None and path[-1] == GOAL
        successes.append(success)
        if success:
            dose, steps = evaluate_path_on_true_field(path, TRUE_FIELD)
            planned_doses.append(dose)
            planned_steps.append(steps)
    mean_dose = float(np.mean(planned_doses))
    return {
        "seed": float(seed),
        "mean_true_dose": mean_dose,
        "mean_steps": float(np.mean(planned_steps)),
        "success_rate": float(np.mean(successes)),
        "dose_regret_percent": (
            (mean_dose - references["oracle_dose_mean"])
            / references["oracle_dose_mean"]
            * 100.0
        ),
        "dose_reduction_percent": (
            (references["shortest_dose_mean"] - mean_dose)
            / references["shortest_dose_mean"]
            * 100.0
        ),
        "l2_error": float(
            np.linalg.norm((estimated - TRUE_FIELD)[FREE_MASK])
            / np.linalg.norm(TRUE_FIELD[FREE_MASK])
        ),
    }


def _online_seed(
    estimator: str,
    pattern: str,
    dwell: float,
    seed: int,
    references: Dict[str, float],
) -> Dict[str, float]:
    """Run one online estimator configuration for one seed."""
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
            replan_every=REPLAN_EVERY,
            lam=LAM,
            max_steps=MAX_STEPS,
        )
        for index, start in enumerate(STARTS)
    ]
    successful = [episode for episode in episodes if episode["success"]]
    mean_dose = float(np.mean([episode["dose"] for episode in successful]))
    return {
        "seed": float(seed),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "mean_steps": float(np.mean([episode["steps"] for episode in successful])),
        "mean_true_dose": mean_dose,
        "dose_regret_percent": (
            (mean_dose - references["oracle_dose_mean"])
            / references["oracle_dose_mean"]
            * 100.0
        ),
        "dose_reduction_percent": (
            (references["shortest_dose_mean"] - mean_dose)
            / references["shortest_dose_mean"]
            * 100.0
        ),
        "mean_readings": float(np.mean([episode["n_readings"] for episode in episodes])),
        "mean_distinct_cells": float(
            np.mean([episode["n_distinct_cells"] for episode in episodes])
        ),
        "mean_replans": float(np.mean([episode["n_replans"] for episode in episodes])),
    }


def _summary(per_seed: List[Dict[str, float]], keys: Iterable[str]) -> Dict[str, Any]:
    """Bootstrap each requested seed-mean metric."""
    return {key: bootstrap(item[key] for item in per_seed) for key in keys}


def _offline_configuration(
    estimator: str, n: int, dwell: float, seeds: Iterable[int], references: Dict[str, float]
) -> Dict[str, Any]:
    """Build one offline configuration result."""
    started = time.perf_counter()
    seed_values = [int(seed) for seed in seeds]
    per_seed = [_offline_seed(estimator, n, dwell, seed, references) for seed in seed_values]
    result: Dict[str, Any] = {
        "kind": "offline",
        "estimator": estimator,
        "n": n,
        "dwell": dwell,
        "seeds": seed_values,
        "n_starts": len(STARTS),
        "per_seed": per_seed,
    }
    result.update(
        _summary(
            per_seed,
            (
                "mean_true_dose",
                "mean_steps",
                "success_rate",
                "dose_regret_percent",
                "dose_reduction_percent",
                "l2_error",
            ),
        )
    )
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def _online_configuration(
    estimator: str,
    pattern: str,
    dwell: float,
    seeds: Iterable[int],
    references: Dict[str, float],
) -> Dict[str, Any]:
    """Build one online configuration result."""
    started = time.perf_counter()
    seed_values = [int(seed) for seed in seeds]
    per_seed = [
        _online_seed(estimator, pattern, dwell, seed, references)
        for seed in seed_values
    ]
    result: Dict[str, Any] = {
        "kind": "online",
        "estimator": estimator,
        "pattern": pattern,
        "dwell": dwell,
        "seeds": seed_values,
        "n_starts": len(STARTS),
        "per_seed": per_seed,
    }
    result.update(
        _summary(
            per_seed,
            (
                "success_rate",
                "mean_steps",
                "mean_true_dose",
                "dose_regret_percent",
                "dose_reduction_percent",
                "mean_readings",
                "mean_distinct_cells",
                "mean_replans",
            ),
        )
    )
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def _offline_configurations() -> Iterable[Tuple[str, int, float]]:
    """Yield the 29 pre-declared offline configurations."""
    for estimator in ("kernel", "idw", "omp3"):
        for n in (25, 50, 100):
            for dwell in (10.0, 1.0, 0.1):
                yield estimator, n, dwell
    yield "omp2", 50, 1.0
    yield "omp4", 50, 1.0


def _online_configurations() -> Iterable[Tuple[str, str, float]]:
    """Yield the 8 pre-declared online configurations."""
    for pattern in ("cell", "3x3"):
        for dwell in (10.0, 1.0, 0.1):
            yield "omp3", pattern, dwell
    yield "omp2", "3x3", 1.0
    yield "omp4", "3x3", 1.0


def _controls_pass() -> bool:
    """Require both recorded simulator controls to have passed."""
    path = ROOT / "results" / "controls.json"
    if not path.exists():
        print("controls.json is missing; run online_controls.py first")
        return False
    controls = json.loads(path.read_text())
    if not controls.get("true", {}).get("passed", False):
        print("true simulator control failed; aborting study")
        return False
    if not controls.get("zero", {}).get("passed", False):
        print("zero simulator control failed; aborting study")
        return False
    return True


def _run_one(references: Dict[str, float]) -> None:
    """Run the declared one-seed timing check without writing results."""
    offline_started = time.perf_counter()
    offline = _offline_configuration("omp3", 50, 1.0, (0,), references)
    offline_seconds = time.perf_counter() - offline_started
    online_started = time.perf_counter()
    online = _online_configuration("omp3", "3x3", 1.0, (0,), references)
    online_seconds = time.perf_counter() - online_started
    projected = (offline_seconds * 5 * 29 + online_seconds * 5 * 8) / 60.0
    print(json.dumps({"offline": offline, "online": online}, indent=2))
    print(f"offline_elapsed_seconds: {offline_seconds}")
    print(f"online_elapsed_seconds: {online_seconds}")
    print(
        f"projected_total_minutes: {projected} = "
        "(offline_seconds * 5 * 29 + online_seconds * 5 * 8) / 60"
    )
    if projected > 90.0:
        print("projected total is above 90 minutes; use seeds 0 to 2.")


def main() -> None:
    """Run the resumable study or its one-seed timing check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    args = parser.parse_args()
    if not _controls_pass():
        raise SystemExit(1)
    references = _references()
    if args.one:
        _run_one(references)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    references_path = OUTPUT / "references.json"
    if not references_path.exists():
        references_path.write_text(json.dumps(references, indent=2))
    seeds = SEEDS
    for estimator, n, dwell in _offline_configurations():
        path = OUTPUT / f"offline_{estimator}_N{n}_t{dwell:g}.json"
        if path.exists():
            continue
        result = _offline_configuration(estimator, n, dwell, seeds, references)
        path.write_text(json.dumps(result, indent=2))
        print(f"finished {path.name}")
    for estimator, pattern, dwell in _online_configurations():
        path = OUTPUT / f"online_{estimator}_{pattern}_t{dwell:g}.json"
        if path.exists():
            continue
        result = _online_configuration(estimator, pattern, dwell, seeds, references)
        path.write_text(json.dumps(result, indent=2))
        print(f"finished {path.name}")


if __name__ == "__main__":
    main()
