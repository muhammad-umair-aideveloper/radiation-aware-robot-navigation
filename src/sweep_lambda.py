"""Sweep radiation weight lambda over classical planners and tabular Q-learning.

Evaluates lambda in {0, 0.5, 1, 2.5, 5, 10} on the fixed 50 evaluation starts
and canonical start (2, 2). Saves results/sweep_astar.json and markdown table.
Includes --qlearning option for offline/resumable RL runs saved to results/sweep_q/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, START_CELL, GOAL_CELL
from src.planners import astar_search, QLearningAgent
from main import generate_evaluation_starts

SWEEP_LAMBDAS: List[float] = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0]
SWEEP_SEEDS: List[int] = [0, 1, 2]


def run_astar_sweep(
    env: RadiationEnv,
    eval_starts: List[Tuple[int, int]],
    lambdas: List[float] = SWEEP_LAMBDAS,
) -> Dict[str, Any]:
    """Run A* Shortest (reference) and A* Aware across all specified lambdas."""
    results: Dict[str, Any] = {
        "metadata": {
            "num_starts": len(eval_starts),
            "lambdas": lambdas,
            "goal": list(env.goal),
            "canonical_start": list(START_CELL),
        },
        "astar_shortest": {},
        "astar_aware": {},
    }

    print("\n--- Evaluating A* Shortest (lambda = 0 reference) ---")
    t0 = time.time()
    s_runs = []
    for s in eval_starts:
        path, dose, steps, _ = astar_search(s, env.goal, mode="shortest", env=env)
        succ = path is not None
        s_runs.append({
            "start": list(s),
            "success": succ,
            "dose": float(dose) if succ else None,
            "steps": float(steps) if succ else None,
        })
    s_time = time.time() - t0

    # Canonical route
    c_path, c_dose, c_steps, _ = astar_search(START_CELL, env.goal, mode="shortest", env=env)
    c_succ = c_path is not None

    succ_runs = [r for r in s_runs if r["success"]]
    results["astar_shortest"] = {
        "lambda": 0.0,
        "success_count": len(succ_runs),
        "total_starts": len(eval_starts),
        "success_rate": len(succ_runs) / len(eval_starts),
        "mean_steps": float(np.mean([r["steps"] for r in succ_runs])) if succ_runs else None,
        "mean_dose": float(np.mean([r["dose"] for r in succ_runs])) if succ_runs else None,
        "planning_time_s": float(s_time),
        "canonical": {
            "success": c_succ,
            "steps": float(c_steps) if c_succ else None,
            "dose": float(c_dose) if c_succ else None,
        },
        "per_start": s_runs,
    }

    # A* Aware for each lambda
    for lam in lambdas:
        print(f"--- Evaluating A* Aware (lambda = {lam}) ---")
        t0 = time.time()
        a_runs = []
        for s in eval_starts:
            path, dose, steps, _ = astar_search(
                s, env.goal, mode="aware", env=env, lambda_rad=lam
            )
            succ = path is not None
            a_runs.append({
                "start": list(s),
                "success": succ,
                "dose": float(dose) if succ else None,
                "steps": float(steps) if succ else None,
            })
        a_time = time.time() - t0

        c_path, c_dose, c_steps, _ = astar_search(
            START_CELL, env.goal, mode="aware", env=env, lambda_rad=lam
        )
        c_succ = c_path is not None

        succ_a = [r for r in a_runs if r["success"]]

        # Paired comparison with shortest on mutual successes
        paired = []
        for s_res, a_res in zip(s_runs, a_runs):
            if s_res["success"] and a_res["success"]:
                paired.append({
                    "start": s_res["start"],
                    "dose_diff": a_res["dose"] - s_res["dose"],
                    "step_diff": a_res["steps"] - s_res["steps"],
                })

        mean_dose_diff = float(np.mean([p["dose_diff"] for p in paired])) if paired else None
        mean_step_diff = float(np.mean([p["step_diff"] for p in paired])) if paired else None

        results["astar_aware"][str(lam)] = {
            "lambda": float(lam),
            "success_count": len(succ_a),
            "total_starts": len(eval_starts),
            "success_rate": len(succ_a) / len(eval_starts),
            "mean_steps": float(np.mean([r["steps"] for r in succ_a])) if succ_a else None,
            "mean_dose": float(np.mean([r["dose"] for r in succ_a])) if succ_a else None,
            "planning_time_s": float(a_time),
            "canonical": {
                "success": c_succ,
                "steps": float(c_steps) if c_succ else None,
                "dose": float(c_dose) if c_succ else None,
            },
            "paired_with_shortest": {
                "n_pairs": len(paired),
                "mean_dose_diff": mean_dose_diff,
                "mean_step_diff": mean_step_diff,
            },
            "per_start": a_runs,
        }

    return results


def format_astar_markdown_table(results: Dict[str, Any]) -> str:
    """Format A* sweep results as markdown table."""
    lines = [
        "| Lambda | Method | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Canonical Steps | Canonical Dose (a.u.) | Dose Diff vs Shortest | Step Diff vs Shortest |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    # Shortest reference
    sh = results["astar_shortest"]
    lines.append(
        f"| 0.0 (ref) | A* Shortest | {sh['success_count']}/{sh['total_starts']} ({sh['success_rate']*100:.0f}%) | "
        f"{sh['mean_steps']:.2f} | {sh['mean_dose']:.2f} | "
        f"{int(sh['canonical']['steps'])} | {sh['canonical']['dose']:.2f} | "
        f"0.00 (ref) | 0.00 (ref) |"
    )

    for lam_str, aw in results["astar_aware"].items():
        p = aw["paired_with_shortest"]
        d_diff_str = f"{p['mean_dose_diff']:+.2f}" if p["mean_dose_diff"] is not None else "N/A"
        s_diff_str = f"{p['mean_step_diff']:+.2f}" if p["mean_step_diff"] is not None else "N/A"
        lines.append(
            f"| {aw['lambda']} | A* Aware | {aw['success_count']}/{aw['total_starts']} ({aw['success_rate']*100:.0f}%) | "
            f"{aw['mean_steps']:.2f} | {aw['mean_dose']:.2f} | "
            f"{int(aw['canonical']['steps'])} | {aw['canonical']['dose']:.2f} | "
            f"{d_diff_str} | {s_diff_str} |"
        )

    return "\n".join(lines)


def run_qlearning_sweep(
    env: RadiationEnv,
    eval_starts: List[Tuple[int, int]],
    out_dir: Path,
    lambdas: List[float] = SWEEP_LAMBDAS,
    seeds: List[int] = SWEEP_SEEDS,
    episodes: int = 50000,
) -> None:
    """Run Q-learning across specified lambdas and seeds with resumable file saving."""
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(lambdas) * len(seeds)
    run_idx = 0

    for lam in lambdas:
        for seed in seeds:
            run_idx += 1
            filename = f"q_sweep_lam{lam}_seed{seed}.json"
            filepath = out_dir / filename

            if filepath.exists():
                print(f"[{run_idx}/{total_runs}] Skipping existing run: {filename}")
                continue

            print(f"[{run_idx}/{total_runs}] Training Q-learning lambda={lam}, seed={seed} ({episodes} eps)...")
            q_agent = QLearningAgent(
                env=env,
                episodes=episodes,
                alpha=0.15,
                gamma=0.98,
                lambda_rad=lam,
            )
            Q, ep_rewards, ep_doses, elapsed = q_agent.train(seed=seed)

            # Evaluate on 50 starts
            eval_runs = []
            for s in eval_starts:
                path, outcome, dose, steps = q_agent.rollout(Q, start=s)
                succ = outcome == "goal"
                eval_runs.append({
                    "start": list(s),
                    "success": succ,
                    "outcome": outcome,
                    "dose": float(dose) if succ else None,
                    "steps": float(steps) if succ else None,
                })

            # Canonical route
            c_path, c_outcome, c_dose, c_steps = q_agent.rollout(Q, start=START_CELL)
            c_succ = c_outcome == "goal"

            succ_runs = [r for r in eval_runs if r["success"]]
            payload = {
                "lambda": float(lam),
                "seed": int(seed),
                "episodes": int(episodes),
                "training_time_s": float(elapsed),
                "success_count": len(succ_runs),
                "total_starts": len(eval_starts),
                "success_rate": len(succ_runs) / len(eval_starts),
                "mean_steps": float(np.mean([r["steps"] for r in succ_runs])) if succ_runs else None,
                "mean_dose": float(np.mean([r["dose"] for r in succ_runs])) if succ_runs else None,
                "canonical": {
                    "success": c_succ,
                    "outcome": c_outcome,
                    "steps": float(c_steps) if c_succ else None,
                    "dose": float(c_dose) if c_succ else None,
                },
                "per_start": eval_runs,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            print(f"     Saved {filename} (elapsed: {elapsed:.2f} s, succ: {len(succ_runs)}/50)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep radiation weight lambda over planners and RL.")
    parser.add_argument("--qlearning", action="store_true", help="Run Q-learning sweep across all lambdas and seeds.")
    parser.add_argument("--episodes", type=int, default=50000, help="Episodes per Q-learning run (default: 50000).")
    args = parser.parse_args()

    env = RadiationEnv()
    eval_starts = generate_evaluation_starts(env, num_starts=50, seed=123)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run A* sweep
    print("=" * 70)
    print("Running A* Radiation Weight Sweep: lambda in {0, 0.5, 1, 2.5, 5, 10}")
    print("=" * 70)
    astar_results = run_astar_sweep(env, eval_starts, SWEEP_LAMBDAS)

    out_json = results_dir / "sweep_astar.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(astar_results, f, indent=2)
    print(f"\nSaved A* sweep results to: {out_json}")

    table_md = format_astar_markdown_table(astar_results)
    out_table = results_dir / "sweep_astar_table.md"
    with open(out_table, "w", encoding="utf-8") as f:
        f.write(table_md + "\n")
    print(f"Saved Markdown table to: {out_table}\n")

    print(table_md)

    # 2. Run Q-learning sweep if flag is passed
    if args.qlearning:
        print("\n" + "=" * 70)
        print("Running Tabular Q-Learning Sweep (Resumable)")
        print("=" * 70)
        q_dir = results_dir / "sweep_q"
        run_qlearning_sweep(
            env=env,
            eval_starts=eval_starts,
            out_dir=q_dir,
            lambdas=SWEEP_LAMBDAS,
            seeds=SWEEP_SEEDS,
            episodes=args.episodes,
        )


if __name__ == "__main__":
    main()
