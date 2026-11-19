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

from src.environment import RadiationEnv
from src.estimation_study import bootstrap
from src.estimators import (
    idw,
    kernel_regression,
    parametric_sources,
    parametric_sources_shielded,
)
from src.geiger_model import counts_to_intensity, sample_measurement_cells, simulate_counts
from src.map_error_study import evaluate_path_on_true_field
from src.online_study2 import FREE_MASK, GOAL, STARTS, TRUE_FIELD
from src.planners import astar_search
from src.shielding import attenuation_matrix, shielded_field

SOURCES = [(18, 20, 150.0), (32, 30, 200.0), (25, 12, 120.0)]
MUS = (0.5, 2.0)
ESTIMATORS = ("kernel", "idw", "omp3", "omp3s")
NS = (50, 100)
DWELLS = (10, 1)
SEEDS = range(5)
SEEDS_FALLBACK = range(3)
LAM = 2.5
OUTPUT = ROOT / "results" / "shield"


def _controls_pass() -> None:
    path = ROOT / "results" / "controls.json"
    if not path.exists():
        raise RuntimeError("controls.json is missing; run online_controls.py first")
    controls = json.loads(path.read_text())
    if not controls.get("true", {}).get("passed", False):
        raise RuntimeError("true simulator control failed")
    if not controls.get("zero", {}).get("passed", False):
        raise RuntimeError("zero simulator control failed")

    wall = ~np.asarray(FREE_MASK, dtype=bool)
    free_space = shielded_field(
        SOURCES, wall, np.ones((2500, 2500), dtype=np.float32)
    )
    if not np.allclose(free_space, TRUE_FIELD, atol=1e-9, rtol=0.0):
        raise RuntimeError("SOURCES do not match the project's true field")


def _attenuation(mu: float) -> tuple[np.ndarray, float]:
    cache = ROOT / "tmp" / f"atten_mu{mu:g}.npy"
    started = time.perf_counter()
    if cache.exists():
        return np.load(cache), float(time.perf_counter() - started)
    wall = ~np.asarray(FREE_MASK, dtype=bool)
    atten = attenuation_matrix(wall, mu)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, atten)
    return atten, float(time.perf_counter() - started)


def _reference_paths(field: np.ndarray) -> tuple[list, list]:
    oracle_env = RadiationEnv(
        goal=GOAL, walls=~FREE_MASK, radiation=field
    )
    shortest_env = RadiationEnv(
        goal=GOAL, walls=~FREE_MASK, radiation=np.zeros_like(field)
    )
    blind_env = RadiationEnv(
        goal=GOAL, walls=~FREE_MASK, radiation=TRUE_FIELD
    )
    oracle_paths = [
        astar_search(start, GOAL, mode="aware", env=oracle_env, lambda_rad=LAM)[0]
        for start in STARTS
    ]
    shortest_paths = [
        astar_search(start, GOAL, mode="shortest", env=shortest_env)[0]
        for start in STARTS
    ]
    blind_paths = [
        astar_search(start, GOAL, mode="aware", env=blind_env, lambda_rad=LAM)[0]
        for start in STARTS
    ]
    return oracle_paths, shortest_paths + blind_paths


def _references(mu: float, atten: np.ndarray) -> Dict[str, float]:
    wall = ~np.asarray(FREE_MASK, dtype=bool)
    shielded = shielded_field(SOURCES, wall, atten)
    oracle_env = RadiationEnv(goal=GOAL, walls=~FREE_MASK, radiation=shielded)
    shortest_env = RadiationEnv(
        goal=GOAL, walls=~FREE_MASK, radiation=np.zeros_like(shielded)
    )
    blind_env = RadiationEnv(goal=GOAL, walls=~FREE_MASK, radiation=TRUE_FIELD)

    oracle_doses = []
    oracle_steps = []
    shortest_doses = []
    shortest_steps = []
    blind_doses = []
    blind_steps = []
    for start in STARTS:
        oracle_path = astar_search(
            start, GOAL, mode="aware", env=oracle_env, lambda_rad=LAM
        )[0]
        shortest_path = astar_search(
            start, GOAL, mode="shortest", env=shortest_env
        )[0]
        blind_path = astar_search(
            start, GOAL, mode="aware", env=blind_env, lambda_rad=LAM
        )[0]
        if oracle_path is not None:
            dose, steps = evaluate_path_on_true_field(oracle_path, shielded)
            oracle_doses.append(dose)
            oracle_steps.append(steps)
        if shortest_path is not None:
            dose, steps = evaluate_path_on_true_field(shortest_path, shielded)
            shortest_doses.append(dose)
            shortest_steps.append(steps)
        if blind_path is not None:
            dose, steps = evaluate_path_on_true_field(blind_path, shielded)
            blind_doses.append(dose)
            blind_steps.append(steps)

    oracle_dose = float(np.mean(oracle_doses))
    blind_dose = float(np.mean(blind_doses))
    result = {
        "oracle_dose_mean": oracle_dose,
        "oracle_steps_mean": float(np.mean(oracle_steps)),
        "shortest_dose_mean": float(np.mean(shortest_doses)),
        "shortest_steps_mean": float(np.mean(shortest_steps)),
        "wall_blind_dose_mean": blind_dose,
        "wall_blind_steps_mean": float(np.mean(blind_steps)),
        "wall_blind_regret_percent": (
            (blind_dose - oracle_dose) / oracle_dose * 100.0
        ),
        "n_starts": len(STARTS),
    }
    return result


def _estimate(
    estimator: str,
    cells: np.ndarray,
    values: np.ndarray,
    atten: np.ndarray,
) -> np.ndarray:
    if estimator == "kernel":
        return kernel_regression(cells, values, FREE_MASK)[0]
    if estimator == "idw":
        return idw(cells, values, FREE_MASK)
    if estimator == "omp3":
        return parametric_sources(cells, values, FREE_MASK, k=3)[0]
    return parametric_sources_shielded(
        cells, values, FREE_MASK, atten, k=3
    )[0]


def _offline_seed(
    estimator: str,
    n: int,
    dwell: float,
    mu: float,
    seed: int,
    atten: np.ndarray,
    shielded: np.ndarray,
    references: Dict[str, float],
) -> Dict[str, float]:
    cells = sample_measurement_cells(FREE_MASK, n, seed)
    counts = simulate_counts(shielded, cells, dwell, seed + 1000)
    values = counts_to_intensity(counts, dwell)
    estimated = _estimate(estimator, cells, values, atten)

    planning_env = RadiationEnv(
        goal=GOAL, walls=~FREE_MASK, radiation=estimated
    )
    planned_doses: List[float] = []
    planned_steps: List[float] = []
    successes = []
    for start in STARTS:
        path = astar_search(
            start, GOAL, mode="aware", env=planning_env, lambda_rad=LAM
        )[0]
        success = path is not None and path[-1] == GOAL
        successes.append(success)
        if success:
            dose, steps = evaluate_path_on_true_field(path, shielded)
            planned_doses.append(dose)
            planned_steps.append(steps)

    mean_dose = (
        float(np.mean(planned_doses)) if planned_doses else float("nan")
    )
    mean_steps = (
        float(np.mean(planned_steps)) if planned_steps else float("nan")
    )
    return {
        "seed": float(seed),
        "success_rate": float(np.mean(successes)),
        "mean_steps": mean_steps,
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
        "l2_error": float(
            np.linalg.norm((estimated - shielded)[FREE_MASK])
            / np.linalg.norm(shielded[FREE_MASK])
        ),
    }


def _configuration(
    estimator: str,
    n: int,
    dwell: float,
    mu: float,
    seeds: Iterable[int],
    atten: np.ndarray,
    shielded: np.ndarray,
    references: Dict[str, float],
) -> Dict[str, Any]:
    started = time.perf_counter()
    seed_values = [int(seed) for seed in seeds]
    per_seed = [
        _offline_seed(
            estimator, n, dwell, mu, seed, atten, shielded, references
        )
        for seed in seed_values
    ]
    summary = {
        key: bootstrap(item[key] for item in per_seed)
        for key in (
            "success_rate",
            "mean_steps",
            "mean_true_dose",
            "dose_regret_percent",
            "dose_reduction_percent",
            "l2_error",
        )
    }
    result = {
        "estimator": estimator,
        "n": n,
        "dwell": dwell,
        "mu": mu,
        "seeds": seed_values,
        "n_starts": len(STARTS),
        "per_seed": per_seed,
        "summary": summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    args = parser.parse_args()

    _controls_pass()
    if args.one:
        mu = 2.0
        atten, cache_seconds = _attenuation(mu)
        wall = ~np.asarray(FREE_MASK, dtype=bool)
        shielded = shielded_field(SOURCES, wall, atten)
        references = _references(mu, atten)
        started = time.perf_counter()
        result = _configuration(
            "omp3s", 50, 1, mu, (0,), atten, shielded, references
        )
        run_seconds = float(time.perf_counter() - started)
        metrics = result["summary"]
        print(json.dumps({
            "metrics": metrics,
            "per_seed": result["per_seed"],
            "wall_blind_regret_percent": references["wall_blind_regret_percent"],
        }, indent=2))
        print(f"attenuation_build_seconds: {cache_seconds:.3f}")
        print(f"run_seconds: {run_seconds:.3f}")
        projected = (2 * cache_seconds + run_seconds * 5 * 32) / 60.0
        print(
            f"projected_total_minutes: {projected:.3f} = "
            "(2 * attenuation_seconds + run_seconds * 5 * 32) / 60"
        )
        if projected > 90.0:
            print("projected total is above 90 minutes; use SEEDS_FALLBACK = range(3).")
        return

    for mu in MUS:
        atten, _ = _attenuation(mu)
        wall = ~np.asarray(FREE_MASK, dtype=bool)
        shielded = shielded_field(SOURCES, wall, atten)
        references = _references(mu, atten)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        reference_path = OUTPUT / f"references_mu{mu:g}.json"
        reference_path.write_text(json.dumps(references, indent=2))

        for estimator in ESTIMATORS:
            for n in NS:
                for dwell in DWELLS:
                    name = f"{estimator}_N{n}_t{dwell:g}_mu{mu:g}.json"
                    path = OUTPUT / name
                    if path.exists():
                        continue
                    result = _configuration(
                        estimator, n, dwell, mu, SEEDS,
                        atten, shielded, references,
                    )
                    path.write_text(json.dumps(result, indent=2))
                    print(f"finished {name}")


if __name__ == "__main__":
    main()
