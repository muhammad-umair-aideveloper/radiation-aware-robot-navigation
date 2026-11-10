"""Plot Pareto frontier: dose versus path length across radiation weights.

Visualizes the A* curve over lambda in {0, 0.5, 1, 2.5, 5, 10}.
If Q-learning run files exist in results/sweep_q/, also plots Q-learning points
with mean and standard deviation across seeds.

Saves results/pareto_dose_vs_length.png.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
ASTAR_SWEEP_PATH = RESULTS_DIR / "sweep_astar.json"
Q_SWEEP_DIR = RESULTS_DIR / "sweep_q"
OUT_PLOT_PATH = RESULTS_DIR / "pareto_dose_vs_length.png"


def load_astar_sweep_data(json_path: Path) -> Dict[str, Any]:
    """Load A* sweep results from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qlearning_sweep_data(q_dir: Path) -> Dict[float, Dict[str, Any]]:
    """Load all available Q-learning sweep runs grouped by lambda."""
    if not q_dir.exists():
        return {}

    q_data: Dict[float, List[Dict[str, Any]]] = {}
    for json_file in q_dir.glob("q_sweep_lam*_seed*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                lam = float(d["lambda"])
                q_data.setdefault(lam, []).append(d)
        except Exception as e:
            print(f"Warning reading {json_file}: {e}")

    summary: Dict[float, Dict[str, Any]] = {}
    for lam, runs in q_data.items():
        succ_rates = [r["success_rate"] for r in runs]
        mean_steps = [r["mean_steps"] for r in runs if r["mean_steps"] is not None]
        mean_doses = [r["mean_dose"] for r in runs if r["mean_dose"] is not None]

        summary[lam] = {
            "n_runs": len(runs),
            "mean_success_rate": float(np.mean(succ_rates)) if succ_rates else 0.0,
            "mean_steps": float(np.mean(mean_steps)) if mean_steps else None,
            "std_steps": float(np.std(mean_steps)) if mean_steps else None,
            "mean_dose": float(np.mean(mean_doses)) if mean_doses else None,
            "std_dose": float(np.std(mean_doses)) if mean_doses else None,
        }

    return summary


def plot_pareto_front(
    astar_data: Dict[str, Any],
    q_data: Dict[float, Dict[str, Any]],
    out_path: Path,
    dpi: int = 200,
) -> None:
    """Generate publication-ready dark-theme Pareto plot: Dose vs Path Length."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#111111")
    ax.set_facecolor("#181818")

    # 1. A* Shortest reference point
    sh = astar_data["astar_shortest"]
    ax.scatter(
        [sh["mean_steps"]],
        [sh["mean_dose"]],
        color="cyan",
        marker="s",
        s=120,
        zorder=5,
        label=f"A* Shortest (λ=0, steps={sh['mean_steps']:.1f}, dose={sh['mean_dose']:.1f})",
    )

    # 2. A* Aware curve over lambda
    lambdas = sorted([float(k) for k in astar_data["astar_aware"].keys()])
    a_steps = []
    a_doses = []
    for lam in lambdas:
        entry = astar_data["astar_aware"][str(lam)]
        a_steps.append(entry["mean_steps"])
        a_doses.append(entry["mean_dose"])

    ax.plot(
        a_steps,
        a_doses,
        color="#39ff14",
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=7,
        zorder=4,
        label="A* Aware Pareto Frontier",
    )

    # Annotate lambda values on A* curve
    for lam, x, y in zip(lambdas, a_steps, a_doses):
        offset_x = 0.4
        offset_y = 1.5 if lam != 0.0 else -3.0
        ax.annotate(
            f"λ={lam}",
            xy=(x, y),
            xytext=(x + offset_x, y + offset_y),
            color="#aaffaa",
            fontsize=9,
            weight="bold",
            arrowprops=dict(arrowstyle="->", color="#55aa55", lw=0.8),
        )

    # 3. Q-learning points (if available)
    if q_data:
        q_lams = sorted(q_data.keys())
        q_steps_vals = []
        q_steps_errs = []
        q_dose_vals = []
        q_dose_errs = []
        valid_lams = []

        for lam in q_lams:
            qd = q_data[lam]
            if qd["mean_steps"] is not None and qd["mean_dose"] is not None:
                valid_lams.append(lam)
                q_steps_vals.append(qd["mean_steps"])
                q_steps_errs.append(qd["std_steps"] if qd["std_steps"] is not None else 0.0)
                q_dose_vals.append(qd["mean_dose"])
                q_dose_errs.append(qd["std_dose"] if qd["std_dose"] is not None else 0.0)

        if valid_lams:
            ax.errorbar(
                q_steps_vals,
                q_dose_vals,
                xerr=q_steps_errs,
                yerr=q_dose_errs,
                fmt="^",
                color="yellow",
                ecolor="orange",
                elinewidth=1.5,
                capsize=4,
                markersize=8,
                zorder=6,
                label=f"Q-Learning (3 seeds mean ± std)",
            )
            for lam, x, y in zip(valid_lams, q_steps_vals, q_dose_vals):
                ax.annotate(
                    f"Q(λ={lam})",
                    xy=(x, y),
                    xytext=(x + 0.5, y + 1.5),
                    color="gold",
                    fontsize=8,
                )
    else:
        # Note that Q-learning sweep is pending
        ax.text(
            0.98,
            0.95,
            "Q-Learning sweep: pending user execution\n(--qlearning)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#888888",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222", edgecolor="#444444", alpha=0.8),
        )

    ax.set_title("Pareto Trade-Off: Absorbed Radiation Dose vs Path Length", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Mean Path Length on 50 Evaluation Starts (steps / cells)", color="white", fontsize=11)
    ax.set_ylabel("Mean Absorbed Radiation Dose (a.u.)", color="white", fontsize=11)
    ax.grid(True, linestyle=":", color="#444444", alpha=0.7)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Pareto plot saved to: {out_path}")


def main() -> None:
    if not ASTAR_SWEEP_PATH.exists():
        print(f"Error: {ASTAR_SWEEP_PATH} not found. Run src/sweep_lambda.py first.")
        sys.exit(1)

    astar_data = load_astar_sweep_data(ASTAR_SWEEP_PATH)
    q_data = load_qlearning_sweep_data(Q_SWEEP_DIR)

    plot_pareto_front(astar_data, q_data, OUT_PLOT_PATH)


if __name__ == "__main__":
    main()
