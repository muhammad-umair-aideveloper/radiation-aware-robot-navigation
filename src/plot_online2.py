"""Plot S9 sensor-range and unexplored-cell results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_summary() -> Dict[str, Any]:
    """Load the generated S9 summary."""
    return json.loads((RESULTS / "online2_summary.json").read_text())


def noisy_row(
    summary: Dict[str, Any], estimator: str, pattern: str, dwell: int, beta: int
) -> Dict[str, Any]:
    """Find one noisy summary row."""
    return next(
        row
        for row in summary["noisy"][estimator]
        if row["pattern"] == pattern
        and row["dwell"] == dwell
        and row["beta"] == beta
    )


def plot_regret(summary: Dict[str, Any]) -> None:
    """Plot noisy dose regret with bootstrap intervals."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, facecolor="#111111")
    patterns = ("3x3", "disc3", "disc6")
    positions = np.arange(len(patterns))
    for axis, estimator in zip(axes, ("kernel", "idw")):
        for offset, beta, color in ((-0.18, 0, "#39ff14"), (0.18, 2, "#f39c12")):
            for dwell, marker in ((10, "10"), (1, "1")):
                means: List[float] = []
                lows: List[float] = []
                highs: List[float] = []
                for pattern in patterns:
                    value = noisy_row(summary, estimator, pattern, dwell, beta)
                    metric = value["dose_regret_percent"]
                    means.append(metric["mean"])
                    lows.append(metric["mean"] - metric["ci_low"])
                    highs.append(metric["ci_high"] - metric["mean"])
                shift = offset + (-0.07 if dwell == 10 else 0.07)
                label = f"beta={beta}, dwell={marker}"
                axis.bar(
                    positions + shift,
                    means,
                    width=0.13,
                    yerr=[lows, highs],
                    capsize=3,
                    color=color,
                    alpha=0.85 if dwell == 10 else 0.55,
                    label=label,
                )
        refs = summary["references"]
        shortest_regret = (
            (refs["shortest_dose_mean"] - refs["oracle_dose_mean"])
            / refs["oracle_dose_mean"]
            * 100.0
        )
        axis.axhline(0.0, color="white", linewidth=1.0, label="oracle")
        axis.axhline(
            shortest_regret,
            color="#00ffff",
            linestyle="--",
            linewidth=1.2,
            label="A* shortest",
        )
        axis.set_title(estimator)
        axis.set_xticks(positions)
        axis.set_xticklabels(patterns)
        axis.set_xlabel("Measurement pattern")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Dose regret (%)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "online2_regret.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_noise_free(summary: Dict[str, Any]) -> None:
    """Plot noise-free regret beside noisy dwell-10 beta-zero regret."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, facecolor="#111111")
    patterns = ("cell", "3x3", "disc3", "disc6")
    positions = np.arange(len(patterns))
    for axis, estimator in zip(axes, ("kernel", "idw")):
        free = {
            row["pattern"]: row["dose_regret_percent"]
            for row in summary["noise_free"]
            if row["estimator"] == estimator
        }
        noisy = {}
        for pattern in ("3x3", "disc3", "disc6"):
            noisy[pattern] = noisy_row(summary, estimator, pattern, 10, 0)
        axis.bar(
            positions - 0.2,
            [free[pattern] for pattern in patterns],
            width=0.36,
            color="#39ff14",
            label="noise-free",
        )
        noisy_values = [
            noisy[pattern]["dose_regret_percent"]["mean"]
            if pattern in noisy
            else np.nan
            for pattern in patterns
        ]
        axis.bar(
            positions + 0.2,
            noisy_values,
            width=0.36,
            color="#00bfff",
            label="dwell 10, beta 0",
        )
        axis.set_title(estimator)
        axis.set_xticks(positions)
        axis.set_xticklabels(patterns)
        axis.set_xlabel("Measurement pattern")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Dose regret (%)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(
        RESULTS / "online2_noise_free.png", dpi=200, facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def main() -> None:
    """Generate both S9 figures."""
    summary = load_summary()
    plot_regret(summary)
    plot_noise_free(summary)


if __name__ == "__main__":
    main()
