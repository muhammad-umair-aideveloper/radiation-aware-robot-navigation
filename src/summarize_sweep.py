"""Aggregate A*, Tabular Q-learning, and SARSA sweep results across radiation weights.

Produces:
  results/sweep_summary.json
  results/sweep_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
ASTAR_SWEEP_PATH = RESULTS_DIR / "sweep_astar.json"
Q_SWEEP_DIR = RESULTS_DIR / "sweep_q"
SARSA_SWEEP_DIR = RESULTS_DIR / "sweep_sarsa"

SWEEP_LAMBDAS: List[float] = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0]
EXPECTED_SEEDS: List[int] = [0, 1, 2]


def load_sweep_data() -> Dict[str, Any]:
    """Load and aggregate A*, Q-learning, and SARSA sweep files."""
    # 1. A* Sweep Data
    astar_data = {}
    if ASTAR_SWEEP_PATH.exists():
        with open(ASTAR_SWEEP_PATH, "r", encoding="utf-8") as f:
            astar_data = json.load(f)

    # 2. Q-learning Sweep Files
    q_runs: Dict[float, List[Dict[str, Any]]] = {lam: [] for lam in SWEEP_LAMBDAS}
    unfinished_or_failed: List[str] = []

    if Q_SWEEP_DIR.exists():
        for lam in SWEEP_LAMBDAS:
            for seed in EXPECTED_SEEDS:
                fpath = Q_SWEEP_DIR / f"q_sweep_lam{lam}_seed{seed}.json"
                if fpath.exists():
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            d = json.load(f)
                            q_runs[lam].append(d)
                            if d.get("success_count", 0) < d.get("total_starts", 50):
                                unfinished_or_failed.append(
                                    f"Q-learning λ={lam} seed={seed}: success={d.get('success_count')}/50"
                                )
                    except Exception as e:
                        unfinished_or_failed.append(f"Q-learning λ={lam} seed={seed}: read error {e}")
                else:
                    unfinished_or_failed.append(f"Q-learning λ={lam} seed={seed}: missing file")
    else:
        unfinished_or_failed.append("Q-learning directory results/sweep_q/ missing")

    # 3. SARSA Sweep Files
    sarsa_runs: Dict[float, List[Dict[str, Any]]] = {lam: [] for lam in SWEEP_LAMBDAS}
    if SARSA_SWEEP_DIR.exists():
        for fpath in SARSA_SWEEP_DIR.glob("sarsa_*.json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    lam = float(d.get("lambda", 2.5))
                    sarsa_runs.setdefault(lam, []).append(d)
            except Exception as e:
                unfinished_or_failed.append(f"SARSA read error {fpath.name}: {e}")
    else:
        unfinished_or_failed.append("SARSA runs not executed (--sarsa option was not present in sweep_lambda.py)")

    # 4. Aggregations per lambda
    summary: Dict[str, Any] = {
        "metadata": {
            "lambdas": SWEEP_LAMBDAS,
            "expected_seeds": EXPECTED_SEEDS,
        },
        "per_lambda": {},
        "unfinished_or_failed_runs": unfinished_or_failed,
    }

    for lam in SWEEP_LAMBDAS:
        lam_str = str(lam)
        lam_summary: Dict[str, Any] = {
            "lambda": lam,
            "astar_aware": None,
            "q_learning": None,
            "sarsa": None,
        }

        # A* values
        if "astar_aware" in astar_data and lam_str in astar_data["astar_aware"]:
            aw = astar_data["astar_aware"][lam_str]
            lam_summary["astar_aware"] = {
                "success_count": aw["success_count"],
                "total_starts": aw["total_starts"],
                "success_rate": aw["success_rate"],
                "mean_steps": aw["mean_steps"],
                "mean_dose": aw["mean_dose"],
                "canonical_steps": aw["canonical"]["steps"],
                "canonical_dose": aw["canonical"]["dose"],
                "paired_dose_diff_vs_shortest": aw["paired_with_shortest"]["mean_dose_diff"],
                "paired_step_diff_vs_shortest": aw["paired_with_shortest"]["mean_step_diff"],
            }

        # Q-learning values
        q_list = q_runs.get(lam, [])
        if q_list:
            succ_counts = [r["success_count"] for r in q_list]
            succ_rates = [r["success_rate"] for r in q_list]
            m_steps = [r["mean_steps"] for r in q_list if r["mean_steps"] is not None]
            m_doses = [r["mean_dose"] for r in q_list if r["mean_dose"] is not None]
            t_times = [r["training_time_s"] for r in q_list if "training_time_s" in r]

            c_doses = [r["canonical"]["dose"] for r in q_list if r.get("canonical", {}).get("dose") is not None]
            c_steps = [r["canonical"]["steps"] for r in q_list if r.get("canonical", {}).get("steps") is not None]

            lam_summary["q_learning"] = {
                "n_seeds": len(q_list),
                "mean_success_count": float(np.mean(succ_counts)),
                "std_success_count": float(np.std(succ_counts)),
                "mean_success_rate": float(np.mean(succ_rates)),
                "mean_steps": float(np.mean(m_steps)) if m_steps else None,
                "std_steps": float(np.std(m_steps)) if m_steps else None,
                "mean_dose": float(np.mean(m_doses)) if m_doses else None,
                "std_dose": float(np.std(m_doses)) if m_doses else None,
                "mean_training_time_s": float(np.mean(t_times)) if t_times else None,
                "std_training_time_s": float(np.std(t_times)) if t_times else None,
                "canonical_mean_dose": float(np.mean(c_doses)) if c_doses else None,
                "canonical_mean_steps": float(np.mean(c_steps)) if c_steps else None,
            }

        # SARSA values
        s_list = sarsa_runs.get(lam, [])
        if s_list:
            succ_counts = [r["success_count"] for r in s_list]
            m_steps = [r["mean_steps"] for r in s_list if r["mean_steps"] is not None]
            m_doses = [r["mean_dose"] for r in s_list if r["mean_dose"] is not None]
            lam_summary["sarsa"] = {
                "n_seeds": len(s_list),
                "mean_success_count": float(np.mean(succ_counts)),
                "mean_steps": float(np.mean(m_steps)) if m_steps else None,
                "mean_dose": float(np.mean(m_doses)) if m_doses else None,
            }

        summary["per_lambda"][lam_str] = lam_summary

    return summary


def format_markdown_summary(summary: Dict[str, Any]) -> str:
    """Format aggregated sweep results as a comprehensive markdown table and report."""
    lines = [
        "# Radiation Weight Sweep Summary (A* vs. Tabular Q-Learning)",
        "",
        "Aggregated benchmark results across radiation weights $\\lambda \\in \\{0.0, 0.5, 1.0, 2.5, 5.0, 10.0\\}$.",
        "Evaluated on the fixed 50 evaluation starts and canonical route $(2, 2) \\to (47, 47)$.",
        "Q-learning metrics are reported as mean $\\pm$ std over 3 independent training seeds ($50,000$ episodes each).",
        "",
        "| Lambda (λ) | Method | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Canonical Steps | Canonical Dose (a.u.) | Training / Planning Time |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for lam_str, d in summary["per_lambda"].items():
        lam = d["lambda"]
        # A* row
        aw = d.get("astar_aware")
        if aw:
            lines.append(
                f"| {lam} | A* Aware | {aw['success_count']}/{aw['total_starts']} ({aw['success_rate']*100:.0f}%) | "
                f"{aw['mean_steps']:.2f} | {aw['mean_dose']:.2f} | "
                f"{int(aw['canonical_steps'])} | {aw['canonical_dose']:.2f} | "
                f"deterministic planner |"
            )
        # Q-learning row
        ql = d.get("q_learning")
        if ql:
            lines.append(
                f"| {lam} | Q-Learning (3 seeds) | {ql['mean_success_count']:.1f} ± {ql['std_success_count']:.1f}/50 ({ql['mean_success_rate']*100:.1f}%) | "
                f"{ql['mean_steps']:.2f} ± {ql['std_steps']:.2f} | {ql['mean_dose']:.2f} ± {ql['std_dose']:.2f} | "
                f"{ql['canonical_mean_steps']:.1f} | {ql['canonical_mean_dose']:.2f} | "
                f"{ql['mean_training_time_s']:.1f} ± {ql['std_training_time_s']:.1f} s |"
            )
        # SARSA row if available
        sa = d.get("sarsa")
        if sa:
            lines.append(
                f"| {lam} | SARSA | {sa['mean_success_count']:.1f}/50 | "
                f"{sa['mean_steps']:.2f} | {sa['mean_dose']:.2f} | - | - | - |"
            )

    lines.extend([
        "",
        "### Unfinished or Failed Runs",
    ])
    for item in summary["unfinished_or_failed_runs"]:
        lines.append(f"- {item}")

    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_sweep_data()

    json_path = RESULTS_DIR / "sweep_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to: {json_path}")

    md_text = format_markdown_summary(summary)
    md_path = RESULTS_DIR / "sweep_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text + "\n")
    print(f"Saved Markdown summary to: {md_path}")


if __name__ == "__main__":
    main()
