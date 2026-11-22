"""Three-minute reproducible demo for the canonical and S13 online studies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import generate_evaluation_starts
from src.environment import RadiationEnv
from src.online_sim import run_online_episode
from src.planners import astar_search

OUTPUT_DIR = ROOT / "demo_output"


def reference_doses(
    env: RadiationEnv, starts: List[Tuple[int, int]]
) -> Dict[str, float]:
    oracle: List[float] = []
    shortest: List[float] = []
    oracle_success = shortest_success = 0
    for start in starts:
        oracle_path, oracle_dose, _, _ = astar_search(
            start, env.goal, mode="aware", env=env, lambda_rad=2.5
        )
        shortest_path, shortest_dose, _, _ = astar_search(
            start, env.goal, mode="shortest", env=env
        )
        if oracle_path is not None:
            oracle.append(float(oracle_dose))
            oracle_success += 1
        if shortest_path is not None:
            shortest.append(float(shortest_dose))
            shortest_success += 1
    return {
        "oracle_mean_dose": float(np.mean(oracle)),
        "shortest_mean_dose": float(np.mean(shortest)),
        "oracle_success": float(oracle_success / len(starts)),
        "shortest_success": float(shortest_success / len(starts)),
    }


def online_replanning(
    env: RadiationEnv,
    starts: List[Tuple[int, int]],
    estimator: str,
    seed: int = 0,
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
            lam=2.5,
            max_steps=300,
        )
        for index, start in enumerate(starts)
    ]
    successful = [episode for episode in episodes if episode["success"]]
    return {
        "mean_dose": float(np.mean([episode["dose"] for episode in successful]))
        if successful else float("inf"),
        "success_rate": float(np.mean([episode["success"] for episode in episodes]))
    }


def audit_canonical_numbers(references: Dict[str, float]) -> Dict[str, object]:
    summary = json.loads(
        (ROOT / "results" / "online_summary.json").read_text(encoding="utf-8")
    )
    committed_references = summary["references"]
    checks = [
        {
            "name": "oracle_mean_dose",
            "computed": references["oracle_mean_dose"],
            "committed": float(committed_references["oracle_dose_mean"]),
        },
        {
            "name": "shortest_mean_dose",
            "computed": references["shortest_mean_dose"],
            "committed": float(committed_references["shortest_dose_mean"]),
        },
    ]
    for check in checks:
        check["difference"] = abs(check["computed"] - check["committed"])
        check["passed"] = bool(check["difference"] <= 1e-9)
    return {
        "passed": bool(all(check["passed"] for check in checks)),
        "checks": checks,
    }


def write_outputs(rows: List[Dict[str, object]], elapsed: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT_DIR / "s13_demo_results.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "starts", "mean_dose_a_u", "success_rate"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    figure_path = OUTPUT_DIR / "s13_demo_comparison.png"
    names = [str(row["method"]) for row in rows]
    doses = [float(row["mean_dose_a_u"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, doses, color=["tab:blue", "tab:gray", "tab:orange", "tab:green"])
    axis.set_ylabel("Mean true dose (a.u.)")
    axis.set_title("Canonical 50x50 radiation navigation demo")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    json_path = OUTPUT_DIR / "s13_demo_summary.json"
    json_path.write_text(
        json.dumps({"elapsed_seconds": elapsed, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def run_demo(fast: bool = False) -> Dict[str, object]:
    started = time.perf_counter()
    env = RadiationEnv()
    fifty_starts = generate_evaluation_starts(env, 50, 123)
    fixed_25 = fifty_starts[::2]
    references = reference_doses(env, fixed_25)
    online_starts = fixed_25[:2] if fast else fixed_25[:5]

    kernel = online_replanning(env, online_starts, "kernel")
    omp3 = online_replanning(env, online_starts, "omp3")
    elapsed = time.perf_counter() - started

    rows = [
        {
            "method": "A* shortest",
            "starts": len(fixed_25),
            "mean_dose_a_u": references["shortest_mean_dose"],
            "success_rate": references["shortest_success"],
        },
        {
            "method": "Oracle A*",
            "starts": len(fixed_25),
            "mean_dose_a_u": references["oracle_mean_dose"],
            "success_rate": references["oracle_success"],
        },
        {
            "method": "Kernel online",
            "starts": len(online_starts),
            "mean_dose_a_u": kernel["mean_dose"],
            "success_rate": kernel["success_rate"],
        },
        {
            "method": "OMP3 online",
            "starts": len(online_starts),
            "mean_dose_a_u": omp3["mean_dose"],
            "success_rate": omp3["success_rate"],
        },
    ]

    audit = audit_canonical_numbers(references)
    if not audit["passed"]:
        raise RuntimeError(f"Committed result audit failed: {audit}")
    write_outputs(rows, elapsed)
    return {
        "elapsed_seconds": elapsed,
        "rows": rows,
        "audit": audit,
        "table_path": str(OUTPUT_DIR / "s13_demo_results.csv"),
        "figure_path": str(OUTPUT_DIR / "s13_demo_comparison.png"),
    }


def print_compact(result: Dict[str, object]) -> None:
    print("Method              Starts  Mean dose (a.u.)  Success")
    for row in result["rows"]:
        print(
            f"{row['method']:<18} {row['starts']:>6}  "
            f"{row['mean_dose_a_u']:>16.6f}  "
            f"{row['success_rate']:>7.3f}"
        )
    print(f"Elapsed: {result['elapsed_seconds']:.3f} s")
    print("Full results: RESULTS_INDEX.md, TECHNICAL_REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible S13 canonical demo.")
    parser.add_argument("--fast", action="store_true", help="Use 2 online starts and finish quickly.")
    args = parser.parse_args()
    result = run_demo(fast=args.fast)
    print_compact(result)


if __name__ == "__main__":
    main()
