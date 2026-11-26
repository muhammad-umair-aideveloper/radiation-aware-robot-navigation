"""Plots for the S8 online measure-estimate-replan study."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RadiationEnv
from src.online_sim import estimate_field, run_online_episode
from src.planners import astar_search

RESULTS = ROOT / "results"
ONLINE = RESULTS / "online"
DWELLS = (10, 1, 0.1)
ESTIMATORS = ("kernel", "idw")
PATTERNS = ("cell", "3x3")


def _load(estimator: str, pattern: str, dwell: float) -> Dict[str, Any]:
    """Load one saved result."""
    name = f"{estimator}_{pattern}_t{dwell:g}.json"
    return json.loads((ONLINE / name).read_text(encoding="utf-8"))


def _setup() -> None:
    """Use the repository's dark plotting theme."""
    plt.style.use("dark_background")
    plt.rcParams.update({"axes.facecolor": "#202020", "figure.facecolor": "#121212"})


def plot_regret() -> None:
    """Plot online and offline dose regret with bootstrap intervals."""
    references = json.loads((ONLINE / "references.json").read_text(encoding="utf-8"))
    shortest_regret = (
        (references["shortest_dose_mean"] - references["oracle_dose_mean"])
        / references["oracle_dose_mean"]
        * 100.0
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    positions = np.arange(len(DWELLS))
    width = 0.18
    for axis, estimator in zip(axes.flat, ESTIMATORS * 2):
        pattern = PATTERNS[0] if axis in axes[0] else PATTERNS[1]
        for offset, label, key in (
            (-width / 2, "Online", "online"),
            (width / 2, "Offline matched", "offline_matched"),
        ):
            means = []
            lows = []
            highs = []
            for dwell in DWELLS:
                item = _load(estimator, pattern, dwell)[key]["dose_regret_percent"]
                means.append(item["mean"])
                lows.append(item["mean"] - item["ci_low"])
                highs.append(item["ci_high"] - item["mean"])
            axis.errorbar(
                positions + offset,
                means,
                yerr=[lows, highs],
                fmt="o-",
                capsize=3,
                label=label,
            )
        axis.set_title(f"{estimator}, {pattern}")
        axis.set_ylabel("Dose regret vs oracle (%)")
        axis.set_xticks(positions, [f"{d:g}" for d in DWELLS])
        axis.set_xlabel("Dwell")
        axis.grid(alpha=0.2)
    axes[0, 0].axhline(0.0, color="cyan", linestyle="--", label="Oracle")
    axes[0, 0].axhline(shortest_regret, color="orange", linestyle=":", label="A* shortest")
    axes[0, 0].legend()
    axes[0, 1].axhline(0.0, color="cyan", linestyle="--")
    axes[0, 1].axhline(shortest_regret, color="orange", linestyle=":")
    axes[1, 0].axhline(0.0, color="cyan", linestyle="--")
    axes[1, 0].axhline(shortest_regret, color="orange", linestyle=":")
    axes[1, 1].axhline(0.0, color="cyan", linestyle="--")
    axes[1, 1].axhline(shortest_regret, color="orange", linestyle=":")
    fig.tight_layout()
    fig.savefig(RESULTS / "online_vs_offline.png", dpi=200)
    plt.close(fig)


def plot_episode() -> None:
    """Plot the canonical seed-zero online episode and its final estimate."""
    env = RadiationEnv()
    start = (2, 2)
    oracle_path, oracle_dose, _, _ = astar_search(
        start, env.goal, mode="aware", env=env, lambda_rad=2.5
    )
    shortest_path, shortest_dose, _, _ = astar_search(
        start, env.goal, mode="shortest", env=env
    )
    episode = run_online_episode(
        env.radiation, ~env.walls, start, env.goal, "kernel", "3x3", 1, 0
    )
    estimated = estimate_field(episode["readings"], ~env.walls, "kernel")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#121212")
    axes[0].imshow(env.radiation, cmap="inferno")
    for path, color, label in (
        (oracle_path, "cyan", f"Oracle ({oracle_dose:.1f} a.u.)"),
        (shortest_path, "white", f"A* shortest ({shortest_dose:.1f} a.u.)"),
        (episode["path"], "lime", f"Online ({episode['dose']:.1f} a.u.)"),
    ):
        points = np.asarray(path)
        axes[0].plot(points[:, 1], points[:, 0], color=color, linewidth=1.5, label=label)
    axes[0].set_title("True field and paths")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    axes[0].legend(fontsize=8)
    axes[1].imshow(estimated, cmap="inferno")
    cells = np.asarray(list(episode["readings"]), dtype=int)
    axes[1].scatter(cells[:, 1], cells[:, 0], s=5, c="cyan", label="Measured cells")
    axes[1].set_title("Measured cells and final kernel estimate")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "online_episode.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Create both S8 figures."""
    _setup()
    plot_regret()
    plot_episode()


if __name__ == "__main__":
    main()
