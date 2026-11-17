from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.environment import RadiationEnv
from src.estimation_study import bootstrap
from src.online_references import reference_doses
from src.online_sim import run_online_episode
from src.online_study import STARTS
ESTIMATORS = ("kernel", "idw")
STARTS = tuple(STARTS)
LAM = 2.5
MAX_STEPS = 300
REPLAN_EVERY = 3
RADIUS = 4
SEEDS = range(5)
SEEDS_FALLBACK = range(3)

BASE_ENV = RadiationEnv()
TRUE_FIELD = BASE_ENV.radiation
FREE_MASK = ~BASE_ENV.walls
GOAL = BASE_ENV.goal
OUTPUT = ROOT / "results" / "online2"
def _per_seed(
    estimator: str,
    pattern: str,
    dwell: float,
    beta: float,
    seed: int,
    noise_free: bool,
    reference: Dict[str, float],
) -> Dict[str, float]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        episodes = list(pool.map(lambda pair: run_online_episode(
            TRUE_FIELD, FREE_MASK, pair[1], GOAL, estimator, pattern, dwell,
            seed * 1000 + pair[0], replan_every=REPLAN_EVERY, lam=LAM,
            max_steps=MAX_STEPS, noise_free=noise_free, inflate_beta=beta,
            inflate_radius=RADIUS), enumerate(STARTS)))
    successful = [episode for episode in episodes if episode["success"]]
    mean_dose = float(np.mean([episode["dose"] for episode in successful]))
    return {"seed": float(seed),
        "mean_readings": float(np.mean([x["n_readings"] for x in episodes])),
        "mean_distinct_cells": float(np.mean([x["n_distinct_cells"] for x in episodes])),
        "mean_replans": float(np.mean([x["n_replans"] for x in episodes])),
        "success_rate": float(np.mean([x["success"] for x in episodes])),
        "mean_steps": float(np.mean([x["steps"] for x in successful])),
        "mean_true_dose": mean_dose,
        "dose_regret_percent": (mean_dose - reference["oracle_dose_mean"])
        / reference["oracle_dose_mean"] * 100.0,
        "dose_reduction_percent": (reference["shortest_dose_mean"] - mean_dose)
        / reference["shortest_dose_mean"] * 100.0}
def _online_summary(
    per_seed: List[Dict[str, float]], noisy: bool
) -> Dict[str, Any]:
    if noisy:
        regret: Any = bootstrap(item["dose_regret_percent"] for item in per_seed)
        reduction: Any = bootstrap(
            item["dose_reduction_percent"] for item in per_seed
        )
    else:
        regret = per_seed[0]["dose_regret_percent"]
        reduction = per_seed[0]["dose_reduction_percent"]
    return {"dose_regret_percent": regret, "dose_reduction_percent": reduction,
        "success_rate_mean": float(np.mean([x["success_rate"] for x in per_seed])),
        "mean_distinct_cells": float(np.mean([x["mean_distinct_cells"] for x in per_seed])),
        "mean_readings": float(np.mean([x["mean_readings"] for x in per_seed])),
        "mean_replans": float(np.mean([x["mean_replans"] for x in per_seed])),
        "mean_steps": float(np.mean([x["mean_steps"] for x in per_seed]))}
def run_configuration(
    estimator: str,
    pattern: str,
    dwell: float,
    beta: float,
    seeds: Iterable[int],
    noise_free: bool,
    reference: Dict[str, float],
) -> Dict[str, Any]:
    started = time.perf_counter()
    seed_values = [int(seed) for seed in seeds]
    per_seed = [
        _per_seed(estimator, pattern, dwell, beta, seed, noise_free, reference)
        for seed in seed_values
    ]
    return {"estimator": estimator, "pattern": pattern, "dwell": dwell,
        "seeds": seed_values, "n_starts": len(STARTS), "per_seed": per_seed,
        "online": _online_summary(per_seed, not noise_free), "beta": beta,
        "noise_free": noise_free,
        "elapsed_seconds": float(time.perf_counter() - started)}
def _filename(estimator: str, pattern: str, dwell: float, beta: float, noise_free: bool) -> str:
    dwell_name = "inf" if noise_free else f"{dwell:g}"
    return f"{estimator}_{pattern}_t{dwell_name}_b{beta:g}.json"
def _configurations() -> Iterable[Tuple[str, str, float, float, bool]]:
    for estimator in ESTIMATORS:
        for pattern in ("cell", "3x3", "disc3", "disc6"):
            yield estimator, pattern, 1.0, 0.0, True
        for pattern in ("3x3", "disc3", "disc6"):
            for dwell in (10.0, 1.0):
                for beta in (0.0, 2.0):
                    if pattern != "3x3" or beta != 0.0:
                        yield estimator, pattern, dwell, beta, False
def _controls_pass() -> bool:
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
def _run_one(reference: Dict[str, float]) -> None:
    result = run_configuration("kernel", "disc6", 1.0, 2.0, (0,), False, reference)
    projected = result["elapsed_seconds"] * (20 * 5 + 8) / 60.0
    print(json.dumps(result, indent=2))
    print(f"elapsed_seconds: {result['elapsed_seconds']}")
    print(f"projected_total_minutes: {projected} = "
          "elapsed_seconds * (20 * 5 + 8) / 60")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    args = parser.parse_args()
    if not _controls_pass():
        raise SystemExit(1)
    reference = reference_doses(list(STARTS))
    if args.one:
        _run_one(reference)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    references_path = OUTPUT / "references.json"
    if not references_path.exists():
        references_path.write_text(json.dumps(reference, indent=2))
    for estimator, pattern, dwell, beta, noise_free in _configurations():
        path = OUTPUT / _filename(estimator, pattern, dwell, beta, noise_free)
        if path.exists():
            continue
        seeds = (0,) if noise_free else SEEDS
        result = run_configuration(estimator, pattern, dwell, beta, seeds,
                                   noise_free, reference)
        path.write_text(json.dumps(result, indent=2))
        print(f"finished {estimator}/{pattern}/t"
              f"{'inf' if noise_free else f'{dwell:g}'}/b{beta:g} "
              f"{result['elapsed_seconds']:.2f}s")
if __name__ == "__main__": main()
