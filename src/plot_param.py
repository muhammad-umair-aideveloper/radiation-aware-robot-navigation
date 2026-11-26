"""Plot S10 offline and online estimator comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUMMARY = ROOT / "results" / "param_summary.json"


def _metric(summary: dict, estimator: str) -> dict:
    """Return one estimator metric from the summary."""
    return summary[estimator]["dose_regret_percent"]


def plot_offline(summary: dict) -> None:
    """Plot offline regret against the number of measurements."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    colors = {"kernel": "#56b4e9", "idw": "#e69f00", "omp3": "#009e73"}
    for axis, dwell in zip(axes, ("10", "1", "0.1")):
        for estimator in ("kernel", "idw", "omp3"):
            means = []
            lows = []
            highs = []
            for n in (25, 50, 100):
                metric = _metric(summary["offline"][f"N{n}_t{dwell}"], estimator)
                means.append(metric["mean"])
                lows.append(metric["mean"] - metric["ci_low"])
                highs.append(metric["ci_high"] - metric["mean"])
            axis.errorbar(
                (25, 50, 100),
                means,
                yerr=np.array([lows, highs]),
                marker="o",
                capsize=3,
                label=estimator,
                color=colors[estimator],
            )
        axis.set_title(f"dwell = {dwell}")
        axis.set_xlabel("measurement cells, N")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("dose regret vs oracle (%)")
    axes[-1].legend()
    fig.suptitle("S10 offline estimator comparison")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "param_offline.png", dpi=200)
    plt.close(fig)


def plot_online(summary: dict) -> None:
    """Plot online regret with S8 baselines and reference lines."""
    groups = [
        ("cell", "10"),
        ("cell", "1"),
        ("cell", "0.1"),
        ("3x3", "10"),
        ("3x3", "1"),
        ("3x3", "0.1"),
    ]
    labels = [f"{pattern}, t={dwell}" for pattern, dwell in groups]
    estimators = ("omp3", "kernel", "idw")
    names = ("omp3", "S8 kernel", "S8 idw")
    colors = ("#009e73", "#56b4e9", "#e69f00")
    values = np.zeros((len(estimators), len(groups)))
    lower = np.zeros_like(values)
    upper = np.zeros_like(values)
    for index, (pattern, dwell) in enumerate(groups):
        data = summary["online"][f"{pattern}_t{dwell}"]
        for estimator_index, estimator in enumerate(estimators):
            source = estimator if estimator == "omp3" else estimator
            metric = data[source]["dose_regret_percent"]
            values[estimator_index, index] = metric["mean"]
            lower[estimator_index, index] = metric["mean"] - metric["ci_low"]
            upper[estimator_index, index] = metric["ci_high"] - metric["mean"]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(groups))
    width = 0.25
    for index, (name, color) in enumerate(zip(names, colors)):
        axis.bar(
            positions + (index - 1) * width,
            values[index],
            width,
            yerr=np.array([lower[index], upper[index]]),
            capsize=3,
            label=name,
            color=color,
        )
    references = json.loads((ROOT / "results" / "param" / "references.json").read_text(encoding="utf-8"))
    shortest_regret = (
        (references["shortest_dose_mean"] - references["oracle_dose_mean"])
        / references["oracle_dose_mean"]
        * 100.0
    )
    axis.axhline(shortest_regret, color="#f0e442", linestyle="--", label="A* shortest")
    axis.axhline(0.0, color="#ffffff", linewidth=1, label="oracle")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("dose regret vs oracle (%)")
    axis.set_title("S10 online estimator comparison")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(ROOT / "results" / "param_online.png", dpi=200)
    plt.close(figure)


def main() -> None:
    """Create both S10 figures."""
    with SUMMARY.open() as handle:
        summary = json.load(handle)
    plt.style.use("dark_background")
    plot_offline(summary)
    plot_online(summary)


if __name__ == "__main__":
    main()
