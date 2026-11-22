"""Plot the S13 fresh-layout replication results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "s13_summary.json"


def set_dark_theme() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 180,
    })


def plot_paired_online_regret(summary: dict) -> None:
    rows = sorted(
        summary["per_layout"],
        key=lambda row: (row["online_kernel_regret_percent"], row["layout_seed"]),
    )
    seeds = [str(row["layout_seed"]) for row in rows]
    kernel = np.asarray([row["online_kernel_regret_percent"] for row in rows])
    omp3 = np.asarray([row["online_omp3_regret_percent"] for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, kernel, color="#4cc9f0", marker="o", linewidth=1.5, label="kernel")
    ax.plot(x, omp3, color="#f72585", marker="o", linewidth=1.5, label="omp3")
    ax.fill_between(x, kernel, omp3, color="#bdbdbd", alpha=0.08)
    ax.axhline(
        summary["pooled"]["online_kernel_regret_percent"]["mean"],
        color="#4cc9f0",
        linestyle="--",
        linewidth=1.5,
        label="kernel pooled mean",
    )
    ax.axhline(
        summary["pooled"]["online_omp3_regret_percent"]["mean"],
        color="#f72585",
        linestyle="--",
        linewidth=1.5,
        label="omp3 pooled mean",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=45)
    ax.set_xlabel("Layout seed (sorted by kernel online regret)")
    ax.set_ylabel("Online dose regret versus oracle (%)")
    ax.set_title("S13 Kernel versus OMP3 Online Dose Regret")
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "s13_online_paired.png")
    plt.close(fig)


def plot_oracle_saving(summary: dict) -> None:
    values = np.asarray([
        row["oracle_saving_percent"] for row in summary["per_layout"]
    ])
    pooled = summary["pooled"]["oracle_saving_percent"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=np.arange(20, 71, 5), color="#4cc9f0", edgecolor="#101820", alpha=0.9)
    ax.axvline(pooled["mean"], color="#ffd166", linewidth=2, label="Pooled mean")
    ax.axvspan(
        pooled["ci_low"],
        pooled["ci_high"],
        color="#ffd166",
        alpha=0.18,
        label="95% bootstrap interval",
    )
    ax.set_xlabel("Oracle dose saving versus A* shortest (%)")
    ax.set_ylabel("Layout count")
    ax.set_title("S13 Distribution of Oracle Dose Saving")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "s13_oracle_saving.png")
    plt.close(fig)


def plot_offline_online(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    estimators = ("kernel", "omp3")
    y = np.arange(len(estimators))
    offline_means = []
    offline_lows = []
    offline_highs = []
    online_means = []
    online_lows = []
    online_highs = []

    for estimator in estimators:
        offline = summary["pooled"][f"offline_{estimator}_regret_percent"]
        online = summary["pooled"][f"online_{estimator}_regret_percent"]
        offline_means.append(offline["mean"])
        offline_lows.append(offline["mean"] - offline["ci_low"])
        offline_highs.append(offline["ci_high"] - offline["mean"])
        online_means.append(online["mean"])
        online_lows.append(online["mean"] - online["ci_low"])
        online_highs.append(online["ci_high"] - online["mean"])

    height = 0.32
    ax.barh(
        y - height / 2,
        offline_means,
        height,
        xerr=(offline_lows, offline_highs),
        capsize=5,
        color="#4cc9f0",
        label="offline",
    )
    ax.barh(
        y + height / 2,
        online_means,
        height,
        xerr=(online_lows, online_highs),
        capsize=5,
        color="#f72585",
        label="online",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(estimators)
    ax.set_xlabel("Pooled dose regret versus oracle (%)")
    ax.set_title("S13 Offline versus Online Pooled Dose Regret")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "s13_offline_online_regret.png")
    plt.close(fig)


def main() -> None:
    set_dark_theme()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    plot_paired_online_regret(summary)
    plot_oracle_saving(summary)
    plot_offline_online(summary)
    print(
        "Wrote s13_online_paired.png, s13_oracle_saving.png, "
        "and s13_offline_online_regret.png"
    )


if __name__ == "__main__":
    main()
