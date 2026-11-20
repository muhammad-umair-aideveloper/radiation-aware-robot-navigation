"""Plot the S12 survey dose and step trade-off."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "s12_summary.json"


def bootstrap_ci(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    rng = np.random.RandomState(42)
    means = [
        float(np.mean(rng.choice(array, size=len(array), replace=True)))
        for _ in range(10000)
    ]
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def plot_tradeoff(summary: Dict[str, object]) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"kernel": "#4cc9f0", "omp3": "#f72585"}
    markers = {10: "o", 1: "s"}
    configs = summary["configurations"]
    for estimator in ("kernel", "omp3"):
        for dwell in (10, 1):
            item = configs[f"{estimator}_t{dwell:g}"]["rows"]
            steps = []
            doses = []
            for budget in ("0", "30", "60"):
                row = item[budget]
                steps.append(row["mean_steps"])
                doses.append(row["mean_true_dose"])
            step_errors_low = []
            step_errors_high = []
            dose_errors_low = []
            dose_errors_high = []
            for budget in ("0", "30", "60"):
                if budget == "0":
                    if estimator == "kernel":
                        source = ROOT / "results" / "online" / f"kernel_3x3_t{dwell:g}.json"
                    else:
                        source = ROOT / "results" / "param" / f"online_omp3_3x3_t{dwell:g}.json"
                    per_seed = json.loads(source.read_text())["per_seed"]
                    step_values = [seed["mean_steps"] for seed in per_seed]
                    dose_values = [seed["mean_true_dose"] for seed in per_seed]
                else:
                    source = ROOT / "results" / "s12" / estimator / f"3x3_t{dwell:g}_b{budget}.json"
                    per_seed = json.loads(source.read_text())["per_seed"]
                    step_values = [seed["mean_steps"] for seed in per_seed]
                    dose_values = [seed["mean_true_dose"] for seed in per_seed]
                step_ci = bootstrap_ci(step_values)
                dose_ci = bootstrap_ci(dose_values)
                step_errors_low.append(step_ci["mean"] - step_ci["ci_low"])
                step_errors_high.append(step_ci["ci_high"] - step_ci["mean"])
                dose_errors_low.append(dose_ci["mean"] - dose_ci["ci_low"])
                dose_errors_high.append(dose_ci["ci_high"] - dose_ci["mean"])
            ax.plot(
                steps,
                doses,
                color=colors[estimator],
                alpha=0.75,
                linewidth=1.5,
                marker=markers[dwell],
                markersize=7,
                label=f"{estimator}, dwell {dwell:g}",
            )
            ax.errorbar(
                steps,
                doses,
                xerr=(step_errors_low, step_errors_high),
                yerr=(dose_errors_low, dose_errors_high),
                fmt="none",
                ecolor=colors[estimator],
                alpha=0.45,
                capsize=3,
            )
            for budget, x, y in zip(("0", "30", "60"), steps, doses):
                ax.annotate(budget, (x, y), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Total steps")
    ax.set_ylabel("Total true dose (a.u.)")
    ax.set_title("S12 Survey: dose-step trade-off")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "s12_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_regret(summary: Dict[str, object]) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    configs = summary["configurations"]
    budgets = (0, 30, 60)
    colors = {10: "#4cc9f0", 1: "#f72585"}
    for estimator_index, estimator in enumerate(("kernel", "omp3")):
        ax_panel = ax[estimator_index]
        ax_panel.set_title(estimator)
        for dwell in (10, 1):
            means = []
            lows = []
            highs = []
            rows = configs[f"{estimator}_t{dwell:g}"]["rows"]
            for budget in ("0", "30", "60"):
                ci = rows[budget]["dose_regret_percent"]
                means.append(ci["mean"])
                lows.append(ci["ci_low"])
                highs.append(ci["ci_high"])
            x = np.asarray(budgets, dtype=float)
            y = np.asarray(means)
            yerr = np.vstack((y - np.asarray(lows), np.asarray(highs) - y))
            ax_panel.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=4,
                color=colors[dwell],
                label=f"dwell {dwell:g}",
            )
        ax_panel.set_xticks(budgets)
        ax_panel.set_xlabel("Survey budget B")
        ax_panel.grid(True, alpha=0.2)
        ax_panel.legend()
    ax[0].set_ylabel("Dose regret versus oracle (%)")
    fig.suptitle("S12 Dose Regret by Survey Budget")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "s12_regret.png", dpi=180)
    plt.close(fig)


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    plot_tradeoff(summary)
    plot_regret(summary)


if __name__ == "__main__":
    main()
