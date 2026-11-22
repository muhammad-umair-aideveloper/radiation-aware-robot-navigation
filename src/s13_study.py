"""S13 fresh-layout measure-estimate-replan study runner."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import ACTION_LENGTHS, ACTIONS, RadiationEnv
from src.geiger_model import counts_to_intensity, simulate_counts
from src.layouts import generate_random_layout
from src.online_sim import estimate_field, run_online_episode
from src.planners import astar_search
from src.s13_layouts import LAYOUT_SEEDS


LAM = 2.5
NOISE_SEEDS = (0, 1, 2)
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


def bootstrap_layout(values: Iterable[float], seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    """Compute mean and 95% bootstrap interval over layout or seed values."""
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.RandomState(seed)
    boot = rng.randint(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = np.mean(array[boot], axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fixed_layout_starts(layout_seed: int, count: int = 10) -> List[Tuple[int, int]]:
    """Load or regenerate the fixed 10 S13 starts for a layout."""
    from src.s13_layouts import generate_s13_layout_set

    record = generate_s13_layout_set()["layouts"][str(layout_seed)]
    return [tuple(start) for start in record["starts"][:count]]


def random_measurement_cells(
    env: RadiationEnv, count: int, seed: int
) -> List[Tuple[int, int]]:
    free = np.argwhere(~env.walls)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(free), size=count, replace=False)
    return [(int(free[index, 0]), int(free[index, 1])) for index in indices]


def offline_estimate(
    env: RadiationEnv,
    estimator: str,
    starts: List[Tuple[int, int]],
    seed: int,
) -> Dict[str, float]:
    cells = random_measurement_cells(env, 50, seed)
    counts = simulate_counts(env.radiation, np.asarray(cells), 1.0, seed)
    values = counts_to_intensity(counts, 1.0)
    readings = {
        cell: [float(value)]
        for cell, value in zip(cells, values)
    }
    field = estimate_field(readings, ~env.walls, estimator)
    estimated_env = RadiationEnv(
        start=env.start,
        goal=env.goal,
        walls=env.walls,
        radiation=field,
    )

    doses: List[float] = []
    successes = 0
    for start in starts:
        path, _, _, _ = astar_search(
            start, env.goal, mode="aware", env=estimated_env, lambda_rad=LAM
        )
        if path is not None:
            dose = sum(
                env.radiation[path[index]]
                * math.sqrt(
                    (path[index][0] - path[index - 1][0]) ** 2
                    + (path[index][1] - path[index - 1][1]) ** 2
                )
                for index in range(1, len(path))
            )
            doses.append(dose)
            successes += 1
    return {
        "mean_true_dose": float(np.mean(doses)) if doses else float("inf"),
        "success_rate": float(successes / len(starts)),
    }


def reference_metrics(
    env: RadiationEnv, starts: List[Tuple[int, int]]
) -> Dict[str, float]:
    oracle_doses: List[float] = []
    shortest_doses: List[float] = []
    for start in starts:
        oracle_path, oracle_dose, _, _ = astar_search(
            start, env.goal, mode="aware", env=env, lambda_rad=LAM
        )
        shortest_path, shortest_dose, _, _ = astar_search(
            start, env.goal, mode="shortest", env=env
        )
        if oracle_path is not None:
            oracle_doses.append(float(oracle_dose))
        if shortest_path is not None:
            shortest_doses.append(float(shortest_dose))
    return {
        "oracle_mean_dose": float(np.mean(oracle_doses)),
        "shortest_mean_dose": float(np.mean(shortest_doses)),
        "oracle_saving_percent": float(
            100.0 * (np.mean(shortest_doses) - np.mean(oracle_doses))
            / np.mean(shortest_doses)
        ),
    }


def online_estimate(
    env: RadiationEnv,
    estimator: str,
    starts: List[Tuple[int, int]],
    seed: int,
) -> Dict[str, float]:
    episodes = [
        run_online_episode(
            env.radiation,
            ~env.walls,
            start,
            env.goal,
            estimator,
            "3x3",
            1.0,
            seed * 1000 + index,
            replan_every=3,
            lam=LAM,
            max_steps=300,
        )
        for index, start in enumerate(starts)
    ]
    successful = [episode for episode in episodes if episode["success"]]
    return {
        "mean_true_dose": float(np.mean([episode["dose"] for episode in successful]))
        if successful else float("inf"),
        "success_rate": float(np.mean([episode["success"] for episode in episodes]))
    }


def dose_regret(dose: float, oracle_dose: float) -> float:
    if oracle_dose == 0:
        return 0.0 if dose == 0 else float("inf")
    return float((dose - oracle_dose) / oracle_dose * 100.0)


def run_layout(layout_seed: int, starts_count: int = 10, seeds: Iterable[int] = NOISE_SEEDS) -> Dict[str, object]:
    layout = generate_random_layout(layout_seed)
    env = layout.to_env()
    starts = fixed_layout_starts(layout_seed, starts_count)
    references = reference_metrics(env, starts)

    result: Dict[str, object] = {
        "metadata": {
            "layout_seed": layout_seed,
            "lambda": LAM,
            "starts": [list(start) for start in starts],
            "noise_seeds": list(seeds),
            "offline": {"n_cells": 50, "dwell": 1.0, "estimators": ["kernel", "idw", "omp3"]},
            "online": {
                "pattern": "3x3",
                "dwell": 1.0,
                "replan_every": 3,
                "max_steps": 300,
                "estimators": ["kernel", "omp3"],
            },
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "references": references,
        "offline": {},
        "online": {},
    }

    oracle_dose = references["oracle_mean_dose"]
    shortest_dose = references["shortest_mean_dose"]
    for estimator in ("kernel", "idw", "omp3"):
        seed_records = [
            offline_estimate(env, estimator, starts, seed)
            for seed in seeds
        ]
        mean_dose = float(np.mean([record["mean_true_dose"] for record in seed_records]))
        result["offline"][estimator] = {
            "per_seed": seed_records,
            "mean_true_dose": mean_dose,
            "mean_dose_regret_percent": float(
                (mean_dose - oracle_dose) / oracle_dose * 100.0
            ),
            "mean_dose_saving_percent": float(
                (shortest_dose - mean_dose) / shortest_dose * 100.0
            ),
            "success_rate": float(np.mean([record["success_rate"] for record in seed_records])),
        }

    for estimator in ("kernel", "omp3"):
        seed_records = [
            online_estimate(env, estimator, starts, seed)
            for seed in seeds
        ]
        mean_dose = float(np.mean([record["mean_true_dose"] for record in seed_records]))
        result["online"][estimator] = {
            "per_seed": seed_records,
            "mean_true_dose": mean_dose,
            "mean_dose_regret_percent": float(
                (mean_dose - oracle_dose) / oracle_dose * 100.0
            ),
            "mean_dose_saving_percent": float(
                (shortest_dose - mean_dose) / shortest_dose * 100.0
            ),
            "success_rate": float(np.mean([record["success_rate"] for record in seed_records])),
        }
    return result


def objective_cost(path: List[Tuple[int, int]], field: np.ndarray, lam: float = LAM) -> float:
    cost = 0.0
    for previous, current in zip(path, path[1:]):
        dr = current[0] - previous[0]
        dc = current[1] - previous[1]
        length = math.sqrt(dr * dr + dc * dc)
        cost += length * (1.0 + lam * field[current])
    return float(cost)


def path_length(path: List[Tuple[int, int]]) -> float:
    return float(sum(
        math.sqrt(
            (current[0] - previous[0]) ** 2
            + (current[1] - previous[1]) ** 2
        )
        for previous, current in zip(path, path[1:])
    ))


def run_controls(layout_seeds: Iterable[int] = (200, 201, 202), starts_per_layout: int = 3) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    for layout_seed in layout_seeds:
        layout = generate_random_layout(layout_seed)
        env = layout.to_env()
        starts = fixed_layout_starts(layout_seed, starts_per_layout)
        for start in starts:
            oracle_path, _, _, _ = astar_search(
                start, env.goal, mode="aware", env=env, lambda_rad=LAM
            )
            shortest_path, _, shortest_steps, _ = astar_search(
                start, env.goal, mode="shortest", env=env
            )
            if oracle_path is None or shortest_path is None:
                raise RuntimeError(f"Reference path missing for layout {layout_seed} start {start}")
            oracle_cost = objective_cost(oracle_path, env.radiation)

            true_episode = run_online_episode(
                env.radiation, ~env.walls, start, env.goal,
                "kernel", "3x3", 1.0, 0,
                replan_every=3, lam=LAM, max_steps=300, control="true",
            )
            true_cost = objective_cost(true_episode["path"], env.radiation)

            zero_episode = run_online_episode(
                env.radiation, ~env.walls, start, env.goal,
                "kernel", "3x3", 1.0, 0,
                replan_every=3, lam=LAM, max_steps=300, control="zero",
            )
            zero_length = path_length(zero_episode["path"])
            shortest_length = path_length(shortest_path)
            checks.append({
                "layout_seed": layout_seed,
                "start": list(start),
                "true_cost": true_cost,
                "oracle_cost": oracle_cost,
                "true_cost_difference": abs(true_cost - oracle_cost),
                "true_passed": bool(abs(true_cost - oracle_cost) <= 1e-9),
                "zero_length": zero_length,
                "shortest_length": shortest_length,
                "zero_length_difference": abs(zero_length - shortest_length),
                "zero_passed": bool(abs(zero_length - shortest_length) <= 1e-9),
            })
    return {
        "passed": bool(all(check["true_passed"] and check["zero_passed"] for check in checks)),
        "checks": checks,
    }


def projected_runtime(layout_seed: int = 200) -> Tuple[float, int]:
    output_dir = ROOT / "results" / "s13"
    start = time.perf_counter()
    run_layout(layout_seed, 1, (0,))
    elapsed = time.perf_counter() - start
    projected_seconds = elapsed * 20 * 10
    result_files = len(LAYOUT_SEEDS)
    return projected_seconds, result_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S13 fresh-layout replication study.")
    parser.add_argument("--quick", action="store_true", help="Run one layout, one start, seed 0.")
    parser.add_argument("--controls", action="store_true", help="Run the pre-study controls.")
    parser.add_argument("--layout-seed", type=int, default=200)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=list(NOISE_SEEDS))
    args = parser.parse_args()

    if args.controls:
        result = run_controls()
        output = ROOT / "results" / "s13_controls.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return

    if args.quick:
        started = time.perf_counter()
        result = run_layout(args.layout_seed, 1, (0,))
        elapsed = time.perf_counter() - started
        projected = elapsed * 20 * 10 * len(args.noise_seeds)
        print(json.dumps({
            "layout_seed": args.layout_seed,
            "quick_elapsed_seconds": round(elapsed, 3),
            "projected_total_seconds": round(projected, 3),
            "projected_total_minutes": round(projected / 60.0, 3),
            "expected_result_files": len(LAYOUT_SEEDS),
            "result_file_template": "results/s13/layout_<seed>.json",
        }, indent=2))
        return

    for layout_seed in LAYOUT_SEEDS:
        output = ROOT / "results" / "s13" / f"layout_{layout_seed}.json"
        if output.exists():
            print(f"SKIP {layout_seed}: {output}")
            continue
        result = run_layout(layout_seed, args.starts, tuple(args.noise_seeds))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"WROTE {output}")


if __name__ == "__main__":
    main()
