"""S7 measure-estimate-plan study runner."""
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
from src.estimators import idw, kernel_regression
from src.geiger_model import counts_to_intensity, sample_measurement_cells, simulate_counts
from src.map_error_study import evaluate_path_on_true_field
from src.planners import astar_search
DWELLS = (100, 10, 1, 0.1)
NS = (10, 25, 50, 100, 200)
ESTIMATORS = ("kernel", "idw")
SEEDS = range(10)
LAMBDA = 2.5
OUTPUT = ROOT / "results" / "estimation"
def bootstrap(values: Iterable[float]) -> Dict[str, float]:
    """Return a mean and fixed-seed percentile bootstrap interval."""
    data = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(42)
    samples = rng.choice(data, size=(10000, len(data)), replace=True)
    means = np.mean(samples, axis=1)
    return {
        "mean": float(np.mean(data)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }
def references(env: RadiationEnv, starts: List[Tuple[int, int]]) -> Dict[str, float]:
    """Compute reusable oracle and shortest reference means."""
    oracle = [astar_search(s, env.goal, mode="aware", env=env, lambda_rad=LAMBDA)[1] for s in starts]
    shortest = [astar_search(s, env.goal, mode="shortest", env=env)[1] for s in starts]
    return {"oracle_dose": float(np.mean(oracle)), "shortest_dose": float(np.mean(shortest))}
def run_configuration(
    estimator: str, n: int, dwell: float, env: RadiationEnv, starts: List[Tuple[int, int]]
) -> Dict[str, Any]:
    """Run one estimator, sample count, and dwell-time configuration."""
    true_field = env.radiation
    free_mask = ~env.walls
    ref = references(env, starts)
    seed_metrics: List[Dict[str, float]] = []
    for seed in SEEDS:
        cells = sample_measurement_cells(free_mask, n, seed)
        counts = simulate_counts(true_field, cells, dwell, seed)
        values = counts_to_intensity(counts, dwell)
        if estimator == "kernel":
            estimated, bandwidth = kernel_regression(cells, values, free_mask)
        else:
            estimated, bandwidth = idw(cells, values, free_mask), None
        planning_env = RadiationEnv(walls=env.walls, radiation=estimated, goal=env.goal)
        doses: List[float] = []
        steps: List[float] = []
        for start in starts:
            path, _, planned_steps, _ = astar_search(
                start, env.goal, mode="aware", env=planning_env, lambda_rad=LAMBDA
            )
            if path is not None:
                dose, _ = evaluate_path_on_true_field(path, true_field)
                doses.append(dose)
                steps.append(planned_steps)
        mean_dose = float(np.mean(doses))
        seed_metrics.append({
            "mean_steps": float(np.mean(steps)),
            "true_dose": mean_dose,
            "dose_regret_percent": (mean_dose - ref["oracle_dose"]) / ref["oracle_dose"] * 100.0,
            "dose_reduction_percent": (ref["shortest_dose"] - mean_dose) / ref["shortest_dose"] * 100.0,
            "relative_l2_error": float(np.linalg.norm((estimated - true_field)[free_mask]) /
                                       np.linalg.norm(true_field[free_mask])),
        })
        if bandwidth is not None:
            seed_metrics[-1]["bandwidth"] = bandwidth
    metrics = {key: bootstrap(item[key] for item in seed_metrics)
               for key in ("mean_steps", "true_dose", "dose_regret_percent",
                           "dose_reduction_percent", "relative_l2_error")}
    return {"estimator": estimator, "n": n, "dwell": dwell, "references": ref,
            "seed_metrics": seed_metrics, "metrics": metrics}
def configuration_path(estimator: str, n: int, dwell: float) -> Path:
    """Return the stable JSON path for a configuration."""
    return OUTPUT / f"{estimator}_N{n}_t{dwell:g}.json"
def run_one(write: bool = True) -> Dict[str, Any]:
    """Run the requested demonstration configuration."""
    env = RadiationEnv()
    starts = generate_evaluation_starts(env, num_starts=50, seed=123)
    result = run_configuration("kernel", 50, 1, env, starts)
    if write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        configuration_path("kernel", 50, 1).write_text(json.dumps(result, indent=2))
    return result
def run_all() -> None:
    """Run every missing study configuration."""
    env = RadiationEnv()
    starts = generate_evaluation_starts(env, num_starts=50, seed=123)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for estimator in ESTIMATORS:
        for n in NS:
            for dwell in DWELLS:
                path = configuration_path(estimator, n, dwell)
                if not path.exists():
                    path.write_text(json.dumps(run_configuration(estimator, n, dwell, env, starts), indent=2))
def write_summary() -> None:
    """Write summary tables and missing-configuration list."""
    files = {path.name: json.loads(path.read_text()) for path in OUTPUT.glob("*.json")
             if path.name != "summary.json"}
    lines = ["# S7 estimation study summary", "", "## Dose regret vs oracle (%)"]
    for estimator in ESTIMATORS:
        lines += [f"### {estimator}", "", "| N | " + " | ".join(map(str, DWELLS)) + " |", "|---|" +
                  "|".join(["---"] * len(DWELLS)) + "|"]
        for n in NS:
            row = [str(n)]
            for dwell in DWELLS:
                item = files.get(configuration_path(estimator, n, dwell).name)
                row.append("missing" if item is None else f"{item['metrics']['dose_regret_percent']['mean']:.3f}")
            lines.append("| " + " | ".join(row) + " |")
    missing = [configuration_path(e, n, d).name for e in ESTIMATORS for n in NS for d in DWELLS
               if configuration_path(e, n, d).name not in files]
    missing_lines = [f"- {name}" for name in missing] or ["- none"]
    lines += ["", "## References", json.dumps(next(iter(files.values()))["references"] if files else {}),
              "", "## Missing configurations"] + missing_lines
    (ROOT / "results" / "estimation_summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results" / "estimation_summary.json").write_text(json.dumps({"missing": missing, "files": sorted(files)}, indent=2))
def main() -> None:
    """Parse flags and run the requested study action."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        write_summary()
    elif args.one:
        started = time.perf_counter()
        result = run_one(write=False)
        elapsed = time.perf_counter() - started
        print(json.dumps(result["metrics"], indent=2))
        print(f"elapsed_seconds: {elapsed:.3f}")
        print(f"projected_total_seconds: {elapsed * 40:.3f}")
    else:
        run_all()
if __name__ == "__main__":
    main()
