"""Plots for the S7 measure-estimate-plan study."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

RESULTS = ROOT / "results"
ESTIMATION = RESULTS / "estimation"
DWELLS = (100, 10, 1, 0.1)
NS = (10, 25, 50, 100, 200)


def load(name: str, n: int, dwell: float) -> Dict:
    """Load one configuration result."""
    return json.loads((ESTIMATION / f"{name}_N{n}_t{dwell:g}.json").read_text())


def setup() -> None:
    """Use a dark plotting theme."""
    plt.style.use("dark_background")
    plt.rcParams.update({"axes.facecolor": "#202020", "figure.facecolor": "#121212"})


def plot_regret() -> None:
    """Plot dose regret against measurement count."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DWELLS)))
    for axis, estimator in zip(axes, ("kernel", "idw")):
        for color, dwell in zip(colors, DWELLS):
            means = [load(estimator, n, dwell)["metrics"]["dose_regret_percent"]["mean"] for n in NS]
            lows = [load(estimator, n, dwell)["metrics"]["dose_regret_percent"]["ci_low"] for n in NS]
            highs = [load(estimator, n, dwell)["metrics"]["dose_regret_percent"]["ci_high"] for n in NS]
            axis.plot(NS, means, marker="o", color=color, label=f"t={dwell:g}")
            axis.fill_between(NS, lows, highs, color=color, alpha=0.16)
        axis.set_title(estimator)
        axis.set_xlabel("Measurement cells (N)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Dose regret vs oracle (%)")
    axes[1].legend(title="Dwell time")
    fig.tight_layout()
    fig.savefig(RESULTS / "estimation_regret.png", dpi=200)
    plt.close(fig)


def estimate_fields(env: RadiationEnv) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the two N=50, dwell=1 estimates for seed zero."""
    cells = sample_measurement_cells(~env.walls, 50, 0)
    values = counts_to_intensity(simulate_counts(env.radiation, cells, 1, 0), 1)
    kernel, _ = kernel_regression(cells, values, ~env.walls)
    inverse = idw(cells, values, ~env.walls)
    return kernel, inverse, cells


def plot_fields() -> None:
    """Plot the true field and both seed-zero estimates."""
    env = RadiationEnv()
    kernel, inverse, cells = estimate_fields(env)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    images = [(env.radiation, "True field"), (kernel, "Kernel estimate"), (inverse, "IDW estimate")]
    for axis, (field, title) in zip(axes, images):
        image = axis.imshow(field, cmap="inferno")
        axis.scatter(cells[:, 1], cells[:, 0], s=8, facecolors="none", edgecolors="cyan")
        axis.set_title(title)
        axis.set_xlabel("Column")
        fig.colorbar(image, ax=axis, fraction=0.046)
    axes[0].set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(RESULTS / "estimation_fields.png", dpi=200)
    plt.close(fig)


def best_estimator() -> str:
    """Select the lower-regret estimator for N=50 and dwell one."""
    scores = {name: load(name, 50, 1)["metrics"]["dose_regret_percent"]["mean"]
              for name in ("kernel", "idw")}
    return min(scores, key=scores.get)


def plot_paths() -> None:
    """Plot shortest, oracle, and best-estimator paths on the true field."""
    env = RadiationEnv()
    starts = [(2, 2)]
    best = best_estimator()
    cells = sample_measurement_cells(~env.walls, 50, 0)
    values = counts_to_intensity(simulate_counts(env.radiation, cells, 1, 0), 1)
    estimated = (kernel_regression(cells, values, ~env.walls)[0] if best == "kernel"
                 else idw(cells, values, ~env.walls))
    planning_env = RadiationEnv(walls=env.walls, radiation=estimated, goal=env.goal)
    paths: List[Tuple[str, List[Tuple[int, int]]]] = []
    for label, mode, planner_env in (
        ("A* shortest", "shortest", env),
        ("Oracle A* aware", "aware", env),
        (f"Best estimate ({best})", "aware", planning_env),
    ):
        path, _, _, _ = astar_search(starts[0], env.goal, mode=mode, env=planner_env, lambda_rad=2.5)
        if path is not None:
            paths.append((label, path))
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(env.radiation, cmap="inferno")
    styles = ["white", "cyan", "lime"]
    for (label, path), color in zip(paths, styles):
        points = np.asarray(path)
        axis.plot(points[:, 1], points[:, 0], color=color, linewidth=1.6, label=label)
    axis.scatter([2, env.goal[1]], [2, env.goal[0]], c=["white", "red"], s=30)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "estimation_paths.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Create all S7 figures."""
    setup()
    plot_regret()
    plot_fields()
    plot_paths()


if __name__ == "__main__":
    main()
