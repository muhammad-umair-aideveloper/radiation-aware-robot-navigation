"""Evaluation script for sensor-based DQN agents, A* baselines, and Tabular Q-learning.

Supports:
  --set {heldout,val,test,diag} : Evaluation pair set selection
  --runs RUN1 RUN2 ...          : Select specific runs to evaluate (or 'baseline', 'all')
  --out-md PATH                 : Custom output Markdown path
  --out-json PATH               : Custom output JSON path
  --force                       : Override test set guard
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, START_CELL, GOAL_CELL
from src.sensor_env import SensorNavigationEnv
from src.sensors import get_sensor_observation, OBS_DIM
from src.dqn_agent import DQNAgent
from src.sensor_pairs import load_pair_set, load_sensor_pairs
from src.planners import astar_search, QLearningAgent

RESULTS_DIR = REPO_ROOT / "results"
SENSOR_RUNS_DIR = RESULTS_DIR / "sensor_runs"
TEST_LOG_FILE = RESULTS_DIR / "test_usage_log.md"

BASELINE_RUNS: List[str] = [
    "dqn_shaped_lam2.5_seed0",
    "dqn_shaped_lam2.5_seed1",
    "dqn_shaped_lam2.5_seed2",
    "dqn_shaped_lam0.0_seed0",
    "dqn_shaped_lam0.0_seed1",
    "dqn_shaped_lam0.0_seed2",
    "dqn_plain_lam2.5_seed0",
]


def compute_bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> Tuple[float, float, float]:
    """Calculate mean and two-sided 95% bootstrap confidence interval."""
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    mean_val = float(np.mean(diffs))
    if len(diffs) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.RandomState(seed)
    n = len(diffs)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = np.mean(diffs[idx], axis=1)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_lower, ci_upper


def evaluate_astar(
    mode: str,
    pairs: List[Dict[str, Any]],
    lambda_rad: float = 2.5,
    env: Optional[RadiationEnv] = None,
) -> Dict[str, Any]:
    """Evaluate A* search baseline on evaluation pairs."""
    if env is None:
        env = RadiationEnv()

    results = []
    successes = 0

    for p in pairs:
        s = tuple(p["start"])
        g = tuple(p["goal"])
        path, dose, steps, _ = astar_search(s, g, mode=mode, env=env, lambda_rad=lambda_rad)

        succ = path is not None
        if succ:
            successes += 1

        results.append({
            "pair_id": p.get("pair_id"),
            "start": list(s),
            "goal": list(g),
            "success": succ,
            "outcome": "goal" if succ else "no_path",
            "steps": float(steps) if succ else None,
            "dose": float(dose) if succ else None,
        })

    succ_steps = [r["steps"] for r in results if r["success"]]
    succ_doses = [r["dose"] for r in results if r["success"]]

    return {
        "method": f"astar_{mode}",
        "mode": mode,
        "lambda_rad": lambda_rad,
        "total_pairs": len(pairs),
        "success_count": successes,
        "collision_count": 0,
        "timeout_count": len(pairs) - successes,
        "success_rate": successes / max(1, len(pairs)),
        "mean_steps": float(np.mean(succ_steps)) if succ_steps else None,
        "mean_dose": float(np.mean(succ_doses)) if succ_doses else None,
        "per_pair": results,
    }


def evaluate_sensor_agent(
    agent: DQNAgent,
    env: SensorNavigationEnv,
    pairs: List[Dict[str, Any]],
    tag: str,
) -> Dict[str, Any]:
    """Evaluate sensor DQN agent on evaluation pairs under greedy policy."""
    results = []
    succ_count = 0
    coll_count = 0
    timeo_count = 0

    for p in pairs:
        s = tuple(p["start"])
        g = tuple(p["goal"])
        obs = env.reset(start=s, goal=g)

        while not env.done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, done, info = env.step(action)

        outcome = info["outcome"]
        succ = (outcome == "goal")
        if succ:
            succ_count += 1
        elif outcome == "collision":
            coll_count += 1
        else:
            timeo_count += 1

        results.append({
            "pair_id": p.get("pair_id"),
            "start": list(s),
            "goal": list(g),
            "success": succ,
            "outcome": outcome,
            "steps": float(info["steps"]),
            "dose": float(info["cumulative_dose"]),
        })

    succ_steps = [r["steps"] for r in results if r["success"]]
    succ_doses = [r["dose"] for r in results if r["success"]]

    return {
        "run_tag": tag,
        "total_pairs": len(pairs),
        "success_count": succ_count,
        "collision_count": coll_count,
        "timeout_count": timeo_count,
        "success_rate": succ_count / max(1, len(pairs)),
        "mean_steps": float(np.mean(succ_steps)) if succ_steps else None,
        "mean_dose": float(np.mean(succ_doses)) if succ_doses else None,
        "per_pair": results,
    }


def evaluate_canonical_route(
    agents: Dict[str, DQNAgent],
    agent_stacks: Dict[str, int],
    base_env: RadiationEnv,
) -> Dict[str, Any]:
    """Evaluate all methods on canonical route: start (2, 2) -> goal (47, 47)."""
    start = START_CELL
    goal = GOAL_CELL
    canonical_results = {}

    # A* Shortest
    p_s, d_s, st_s, _ = astar_search(start, goal, mode="shortest", env=base_env)
    canonical_results["astar_shortest"] = {
        "success": p_s is not None,
        "outcome": "goal" if p_s else "no_path",
        "steps": float(st_s) if p_s else None,
        "dose": float(d_s) if p_s else None,
    }

    # A* Aware
    p_a, d_a, st_a, _ = astar_search(start, goal, mode="aware", env=base_env, lambda_rad=2.5)
    canonical_results["astar_aware"] = {
        "success": p_a is not None,
        "outcome": "goal" if p_a else "no_path",
        "steps": float(st_a) if p_a else None,
        "dose": float(d_a) if p_a else None,
    }

    # Tabular Q-Learning (from saved models if available)
    q_table_path = REPO_ROOT / "models" / "q_table_best.npy"
    if q_table_path.exists():
        q_tab = np.load(q_table_path)
        q_agent = QLearningAgent(base_env, gamma=0.98, epsilon_start=0.0)
        q_agent.q_table = q_tab
        p_q, d_q, out_q = q_agent.evaluate(start=start, goal=goal, max_steps=300)
        canonical_results["tabular_q_learning"] = {
            "success": (out_q == "goal"),
            "outcome": out_q,
            "steps": float(len(p_q) - 1) if p_q else None,
            "dose": float(d_q) if p_q else None,
        }

    # Sensor Agents
    for tag, agent in agents.items():
        k = agent_stacks.get(tag, 1)
        lam = 0.0 if "lam0.0" in tag else 2.5
        c_env = SensorNavigationEnv(base_env=base_env, lambda_rad=lam, shaping=False, num_stack=k)
        obs = c_env.reset(start=start, goal=goal)
        while not c_env.done:
            act = agent.select_action(obs, epsilon=0.0)
            obs, r, done, info = c_env.step(act)

        canonical_results[tag] = {
            "success": (info["outcome"] == "goal"),
            "outcome": info["outcome"],
            "steps": float(info["steps"]),
            "dose": float(info["cumulative_dose"]),
        }

    return canonical_results


def check_and_record_test_guard(runs: List[str], command_str: str, force: bool = False) -> None:
    """Guard against accidental repeated test set evaluations."""
    has_prior_runs = False
    if TEST_LOG_FILE.exists():
        content = TEST_LOG_FILE.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#") and not line.startswith("| Timestamp")]
        if lines:
            has_prior_runs = True

    if has_prior_runs and not force:
        raise RuntimeError(
            "ERROR: The final test set has already been evaluated! "
            "To prevent data leakage, test evaluation is strictly guarded. "
            "Pass --force to explicitly override if intended."
        )

    # Append audit entry
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    runs_str = ", ".join(runs) if len(runs) <= 4 else f"{len(runs)} runs: {runs[0]}..{runs[-1]}"

    if not TEST_LOG_FILE.exists():
        TEST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("# Test Set Evaluation Usage Log\n\n")
            f.write("Guarded audit log recording every evaluation on the final held-out test set (`results/sensor_test_pairs.json`).\n\n")
            f.write("| Timestamp (UTC) | Models Evaluated | Command | Forced |\n")
            f.write("|---|---|---|---|\n")

    with open(TEST_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"| {now_str} | {runs_str} | `{command_str}` | {force} |\n")
    print(f"Recorded test evaluation audit entry in {TEST_LOG_FILE}")


def run_full_sensor_evaluation(
    pair_set: str = "heldout",
    runs: Optional[List[str]] = None,
    out_json: Optional[Path] = None,
    out_md: Optional[Path] = None,
    force: bool = False,
    include_canonical: bool = True,
) -> Dict[str, Any]:
    """Execute complete sensor benchmark on the specified pair set."""
    if pair_set == "test":
        cmd_str = " ".join(sys.argv)
        check_and_record_test_guard(runs or BASELINE_RUNS, cmd_str, force=force)

    # Determine default output paths
    if out_json is None:
        out_json = RESULTS_DIR / f"sensor_{pair_set}_eval.json" if pair_set != "heldout" else RESULTS_DIR / "sensor_eval.json"
    if out_md is None:
        out_md = RESULTS_DIR / f"sensor_{pair_set}_eval.md" if pair_set != "heldout" else RESULTS_DIR / "sensor_eval.md"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Sensor Agent Evaluation Benchmark")
    print(f"Target Pair Set: {pair_set} | Output: {out_md.name}")
    print(f"=======================================================")

    eval_pairs = load_pair_set(pair_set)
    base_env = RadiationEnv()

    # 1. Classical Baselines on the selected pair set
    print(f"Evaluating A* Shortest baseline on {len(eval_pairs)} {pair_set} pairs...")
    astar_sh_res = evaluate_astar(mode="shortest", pairs=eval_pairs, env=base_env)

    print(f"Evaluating A* Aware baseline (lambda=2.5) on {len(eval_pairs)} {pair_set} pairs...")
    astar_aw_res = evaluate_astar(mode="aware", pairs=eval_pairs, lambda_rad=2.5, env=base_env)

    # 2. Determine runs to evaluate
    if runs is None or len(runs) == 0 or runs == ["baseline"]:
        target_runs = BASELINE_RUNS
    elif runs == ["all"]:
        # Find all weight files
        target_runs = [f.stem.replace("_weights", "") for f in SENSOR_RUNS_DIR.glob("*_weights.npz")]
    else:
        target_runs = runs

    sensor_results: Dict[str, Any] = {}
    loaded_agents: Dict[str, DQNAgent] = {}
    agent_stacks: Dict[str, int] = {}

    for tag in target_runs:
        weights_path = SENSOR_RUNS_DIR / f"{tag}_weights.npz"
        if not weights_path.exists():
            print(f"Warning: Weights file missing for {tag}: {weights_path}")
            continue

        # Inspect weight matrix to determine obs_dim and stack count
        data = np.load(weights_path)
        obs_dim = int(data["w0"].shape[0])
        num_stack = max(1, obs_dim // OBS_DIM)
        agent_stacks[tag] = num_stack

        agent = DQNAgent(obs_dim=obs_dim, num_actions=8, hidden_sizes=(64, 64))
        agent.load(weights_path)
        loaded_agents[tag] = agent

        lam = 0.0 if "lam0.0" in tag else 2.5
        eval_env = SensorNavigationEnv(base_env=base_env, lambda_rad=lam, shaping=False, num_stack=num_stack)

        print(f"Evaluating {tag} (obs_dim={obs_dim}, stack={num_stack}, lam={lam}) on {len(eval_pairs)} {pair_set} pairs...")
        res = evaluate_sensor_agent(agent, eval_env, eval_pairs, tag=tag)
        sensor_results[tag] = res
        print(f"  -> Succ: {res['success_count']}/{len(eval_pairs)}, Coll: {res['collision_count']}, Timeo: {res['timeout_count']}")

    # 3. Canonical Route Evaluation (optional, evaluated on canonical pair)
    canonical_res = {}
    if include_canonical:
        print("Evaluating canonical route (2,2) -> (47,47)...")
        canonical_res = evaluate_canonical_route(loaded_agents, agent_stacks, base_env)

    # 4. Paired Comparisons on Mutual Successes
    paired_comparisons: Dict[str, Any] = {}

    def compare_pairs(m1_data: Dict[str, Any], m2_data: Dict[str, Any], label: str) -> Dict[str, Any]:
        p1 = m1_data["per_pair"]
        p2 = m2_data["per_pair"]

        diff_dose = []
        diff_steps = []

        for r1, r2 in zip(p1, p2):
            if r1["success"] and r2["success"]:
                diff_dose.append(r1["dose"] - r2["dose"])
                diff_steps.append(r1["steps"] - r2["steps"])

        n_mutual = len(diff_dose)
        if n_mutual > 0:
            m_dose, ci_d_l, ci_d_u = compute_bootstrap_ci(np.array(diff_dose, dtype=np.float64))
            m_step, ci_s_l, ci_s_u = compute_bootstrap_ci(np.array(diff_steps, dtype=np.float64))
        else:
            m_dose, ci_d_l, ci_d_u = float("nan"), float("nan"), float("nan")
            m_step, ci_s_l, ci_s_u = float("nan"), float("nan"), float("nan")

        return {
            "comparison": label,
            "mutual_successes": n_mutual,
            "mean_dose_diff": m_dose,
            "ci_95_dose_diff": [ci_d_l, ci_d_u],
            "mean_step_diff": m_step,
            "ci_95_step_diff": [ci_s_l, ci_s_u],
        }

    # Compare key pairs
    for s in [0, 1, 2]:
        tag_lam25 = f"dqn_shaped_lam2.5_seed{s}"
        tag_lam0 = f"dqn_shaped_lam0.0_seed{s}"
        if tag_lam25 in sensor_results and tag_lam0 in sensor_results:
            paired_comparisons[f"shaped_lam2.5_vs_lam0_seed{s}"] = compare_pairs(
                sensor_results[tag_lam25], sensor_results[tag_lam0],
                f"Sensor DQN Shaped λ=2.5 vs λ=0.0 (Seed {s})"
            )

    for tag in sensor_results:
        paired_comparisons[f"{tag}_vs_astar_shortest"] = compare_pairs(
            sensor_results[tag], astar_sh_res,
            f"{tag} vs. A* Shortest"
        )
        paired_comparisons[f"{tag}_vs_astar_aware"] = compare_pairs(
            sensor_results[tag], astar_aw_res,
            f"{tag} vs. A* Aware"
        )

    # 5. Assemble and save full report
    report: Dict[str, Any] = {
        "metadata": {
            "pair_set": pair_set,
            "total_pairs": len(eval_pairs),
            "timestamp": time.time(),
        },
        "baselines": {
            "astar_shortest": astar_sh_res,
            "astar_aware": astar_aw_res,
        },
        "sensor_agents": sensor_results,
        "canonical_route": canonical_res,
        "paired_comparisons": paired_comparisons,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation JSON to: {out_json}")

    # Generate Markdown Report
    write_evaluation_markdown(report, out_md)
    return report


def write_evaluation_markdown(report: Dict[str, Any], filepath: Path) -> None:
    """Format evaluation results into a Markdown report."""
    pair_set = report["metadata"]["pair_set"]
    total_pairs = report["metadata"]["total_pairs"]

    md_lines: List[str] = [
        f"# Sensor-Based Agent Evaluation Report ({pair_set.upper()} Set)",
        "",
        f"Evaluation across {total_pairs} pairs from the **{pair_set}** set against classical A* baselines.",
        "",
        f"## 1. Benchmark on {total_pairs} Evaluation Pairs ({pair_set})",
        "",
        "| Method / Model | Variant | Objective λ | Seed | Success Rate | Collisions | Timeouts | Mean Steps | Mean Dose (a.u.) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    # Baselines
    sh_res = report["baselines"]["astar_shortest"]
    aw_res = report["baselines"]["astar_aware"]

    md_lines.append(
        f"| **A* Shortest** | Classical | 0.0 | - | "
        f"{sh_res['success_count']}/{total_pairs} ({sh_res['success_rate']*100:.0f}%) | "
        f"{sh_res['collision_count']} | {sh_res['timeout_count']} | "
        f"{sh_res['mean_steps']:.2f} | {sh_res['mean_dose']:.2f} |"
    )
    md_lines.append(
        f"| **A* Aware** | Classical | 2.5 | - | "
        f"{aw_res['success_count']}/{total_pairs} ({aw_res['success_rate']*100:.0f}%) | "
        f"{aw_res['collision_count']} | {aw_res['timeout_count']} | "
        f"{aw_res['mean_steps']:.2f} | {aw_res['mean_dose']:.2f} |"
    )

    # Sensor models
    for tag, res in report["sensor_agents"].items():
        steps_str = f"{res['mean_steps']:.2f}" if res['mean_steps'] is not None else "N/A"
        dose_str = f"{res['mean_dose']:.2f}" if res['mean_dose'] is not None else "N/A"

        # Parse tag elements
        var = "shaped" if "shaped" in tag else ("plain" if "plain" in tag else "variant")
        lam = "0.0" if "lam0.0" in tag else "2.5"
        seed_val = tag.split("seed")[-1] if "seed" in tag else "-"

        md_lines.append(
            f"| **{tag}** | {var} | {lam} | {seed_val} | "
            f"{res['success_count']}/{total_pairs} ({res['success_rate']*100:.0f}%) | "
            f"{res['collision_count']} | {res['timeout_count']} | {steps_str} | {dose_str} |"
        )

    # Canonical Route (if present)
    canonical_res = report.get("canonical_route", {})
    if canonical_res:
        md_lines.extend([
            "",
            "## 2. Canonical Route Evaluation: (2, 2) → (47, 47)",
            "",
            "| Method / Model | Success | Outcome | Path Steps | Cumulative Dose (a.u.) |",
            "|---|---|---|---|---|",
        ])

        for m_name, c_res in canonical_res.items():
            if c_res is None:
                continue
            succ_mark = "✓" if c_res["success"] else "✗"
            st_str = f"{c_res['steps']:.0f}" if c_res['steps'] is not None else "N/A"
            ds_str = f"{c_res['dose']:.2f}" if c_res['dose'] is not None else "N/A"
            md_lines.append(f"| **{m_name}** | {succ_mark} | {c_res['outcome']} | {st_str} | {ds_str} |")

    # Paired comparisons
    paired_comp = report.get("paired_comparisons", {})
    if paired_comp:
        md_lines.extend([
            "",
            "## 3. Paired Statistical Comparisons",
            "",
            "| Comparison | Mutual Successes | Mean Dose Diff (a.u.) | 95% Bootstrap CI | Mean Step Diff | 95% Bootstrap CI |",
            "|---|---|---|---|---|---|",
        ])

        for comp_key, comp in paired_comp.items():
            n = comp["mutual_successes"]
            if n > 0:
                ci_d = f"[{comp['ci_95_dose_diff'][0]:.2f}, {comp['ci_95_dose_diff'][1]:.2f}]"
                ci_s = f"[{comp['ci_95_step_diff'][0]:.2f}, {comp['ci_95_step_diff'][1]:.2f}]"
                md_lines.append(
                    f"| {comp['comparison']} | N={n} | {comp['mean_dose_diff']:+.2f} | {ci_d} | {comp['mean_step_diff']:+.2f} | {ci_s} |"
                )
            else:
                md_lines.append(f"| {comp['comparison']} | N=0 (no mutual success) | N/A | N/A | N/A | N/A |")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown report to: {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sensor-based DQN agents and baselines.")
    parser.add_argument(
        "--set",
        choices=["heldout", "val", "test", "diag"],
        default="heldout",
        help="Evaluation pair set to test on",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run tags to evaluate (or 'baseline', 'all')",
    )
    parser.add_argument("--out-md", type=str, default=None, help="Custom path for output Markdown report")
    parser.add_argument("--out-json", type=str, default=None, help="Custom path for output JSON report")
    parser.add_argument("--force", action="store_true", help="Force test set evaluation override")
    parser.add_argument("--no-canonical", action="store_true", help="Skip canonical route evaluation")

    args = parser.parse_args()

    out_md = Path(args.out_md) if args.out_md else None
    out_json = Path(args.out_json) if args.out_json else None

    run_full_sensor_evaluation(
        pair_set=args.set,
        runs=args.runs,
        out_json=out_json,
        out_md=out_md,
        force=args.force,
        include_canonical=(not args.no_canonical),
    )


if __name__ == "__main__":
    main()
