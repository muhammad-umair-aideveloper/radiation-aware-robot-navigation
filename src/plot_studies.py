"""Generate visualization plots for procedural layouts overview and map-error robustness.

Outputs:
  results/layouts_overview.png
  results/robustness.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import START_CELL, GOAL_CELL, RadiationEnv
from src.layouts import generate_random_layout
from src.planners import astar_search

RESULTS_DIR = REPO_ROOT / "results"
MAP_ERROR_JSON = RESULTS_DIR / "map_error_study.json"


def plot_layouts_overview(out_path: Path, dpi: int = 200) -> None:
    """Plot the first 4 procedural layouts (seeds 100-103) with both A* paths from (2,2)."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor="#111111")
    axes = axes.flatten()

    seeds = [100, 101, 102, 103]

    for idx, (ax, seed) in enumerate(zip(axes, seeds)):
        ax.set_facecolor("#181818")
        layout = generate_random_layout(seed=seed)
        env = layout.to_env()

        # Heatmap
        im = ax.imshow(
            env.radiation,
            cmap="inferno",
            origin="upper",
            extent=(0, env.shape[1], env.shape[0], 0),
        )

        # Grey walls
        wall_overlay = np.ma.masked_where(~env.walls, np.ones_like(env.walls, dtype=float))
        ax.imshow(
            wall_overlay,
            cmap="gray",
            vmin=0,
            vmax=2,
            origin="upper",
            extent=(0, env.shape[1], env.shape[0], 0),
            alpha=0.65,
        )

        # Plan paths from canonical start (2,2)
        path_s, dose_s, steps_s, _ = astar_search(START_CELL, GOAL_CELL, mode="shortest", env=env)
        path_a, dose_a, steps_a, _ = astar_search(
            START_CELL, GOAL_CELL, mode="aware", env=env, lambda_rad=2.5
        )

        def to_xy(p):
            return [c + 0.5 for _, c in p], [r + 0.5 for r, _ in p]

        if path_s:
            xs_s, ys_s = to_xy(path_s)
            ax.plot(xs_s, ys_s, linestyle="--", color="cyan", linewidth=2.0, label="A* Shortest")
        if path_a:
            xs_a, ys_a = to_xy(path_a)
            ax.plot(xs_a, ys_a, linestyle="-", color="#39ff14", linewidth=2.2, label="A* Aware (λ=2.5)")

        # Sources
        for s_pos, i0 in layout.sources:
            ax.scatter(s_pos[1] + 0.5, s_pos[0] + 0.5, color="orange", marker="x", s=70, linewidths=2)

        # Start & Goal
        ax.scatter([START_CELL[1] + 0.5], [START_CELL[0] + 0.5], color="deepskyblue", s=70, label="Start (2,2)")
        ax.scatter([GOAL_CELL[1] + 0.5], [GOAL_CELL[0] + 0.5], color="red", marker="*", s=90, label="Goal (47,47)")

        d_red = (dose_s - dose_a) / dose_s * 100.0 if dose_s > 0 else 0.0
        ax.set_title(
            f"Layout Seed {seed} (Wall: {layout.wall_fraction*100:.1f}%)\n"
            f"Shortest: {dose_s:.1f} a.u. ({int(steps_s)} st) | Aware: {dose_a:.1f} a.u. ({int(steps_a)} st) [-{d_red:.1f}%]",
            color="white",
            fontsize=10,
            pad=8,
        )
        ax.set_xlim(0, env.shape[1])
        ax.set_ylim(env.shape[0], 0)
        ax.axis("off")
        if idx == 0:
            ax.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved layouts overview to: {out_path}")


def plot_robustness_curve(json_path: Path, out_path: Path, dpi: int = 200) -> None:
    """Plot true absorbed dose versus map perturbation level with oracle and shortest baselines."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_s = data["baselines"]["shortest"]["mean_dose"]
    base_o = data["baselines"]["oracle_aware"]["mean_dose"]

    levels = [1, 2, 3]
    means = [data["levels"][str(l)]["mean_true_dose"] for l in levels]
    stds = [data["levels"][str(l)]["std_true_dose"] for l in levels]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#111111")
    ax.set_facecolor("#181818")

    # Baselines
    ax.axhline(
        base_s,
        color="cyan",
        linestyle="--",
        linewidth=2.0,
        label=f"A* Shortest (Radiation-blind: {base_s:.2f} a.u.)",
    )
    ax.axhline(
        base_o,
        color="#39ff14",
        linestyle=":",
        linewidth=2.2,
        label=f"Oracle A* Aware (True field: {base_o:.2f} a.u.)",
    )

    # Plot each individual run as jittered scatter
    rng = np.random.RandomState(42)
    for lvl in levels:
        run_doses = [r["mean_true_dose"] for r in data["levels"][str(lvl)]["runs"]]
        jitter = rng.uniform(-0.06, 0.06, size=len(run_doses))
        ax.scatter(
            [lvl + j for j in jitter],
            run_doses,
            color="orange",
            alpha=0.55,
            s=45,
            zorder=4,
            label="Individual perturbation runs (10/level)" if lvl == 1 else "",
        )

    # Plot mean with error bars (including level 0 as Oracle)
    x_all = [0] + levels
    y_all = [base_o] + means
    y_err = [0.0] + stds

    ax.errorbar(
        x_all,
        y_all,
        yerr=y_err,
        fmt="o-",
        color="yellow",
        ecolor="orange",
        elinewidth=2.0,
        capsize=5,
        capthick=1.5,
        linewidth=2.2,
        markersize=8,
        zorder=5,
        label="A* Aware (Mean ± 1 std on True Field)",
    )

    # Annotate points
    for x, y, std in zip(x_all, y_all, y_err):
        label_text = f"{y:.2f}" if std == 0 else f"{y:.2f} ± {std:.2f}"
        ax.annotate(
            label_text,
            xy=(x, y),
            xytext=(x, y + 2.5),
            color="white",
            fontsize=9,
            weight="bold",
            ha="center",
        )

    ax.set_title("Robustness to Radiation Map Uncertainty (L2)", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Perturbation Level (Max Source Shift in Cells, Intensity Error ±20%)", color="white", fontsize=11)
    ax.set_ylabel("True Absorbed Radiation Dose on 50 Starts (a.u.)", color="white", fontsize=11)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Level 0\n(Oracle)", "Level 1\n(±1 cell)", "Level 2\n(±2 cells)", "Level 3\n(±3 cells)"])
    ax.grid(True, linestyle=":", color="#444444", alpha=0.7)
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", fontsize=9)
    ax.set_ylim(40, 130)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved robustness curve to: {out_path}")


def main() -> None:
    plot_layouts_overview(RESULTS_DIR / "layouts_overview.png")
    plot_robustness_curve(MAP_ERROR_JSON, RESULTS_DIR / "robustness.png")


if __name__ == "__main__":
    main()
