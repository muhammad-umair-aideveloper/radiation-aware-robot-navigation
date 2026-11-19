"""Plot S11 wall-shielding regret and seed-zero fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RadiationEnv
from src.estimators import (
    kernel_regression,
    idw,
    parametric_sources,
    parametric_sources_shielded,
)
from src.geiger_model import counts_to_intensity, sample_measurement_cells, simulate_counts
from src.online_study2 import FREE_MASK, TRUE_FIELD
from src.shielding import attenuation_matrix, shielded_field

SOURCES = [(18, 20, 150.0), (32, 30, 200.0), (25, 12, 120.0)]
MUS = (0.5, 2.0)
ESTIMATORS = ("kernel", "idw", "omp3", "omp3s")
NS = (50, 100)
DWELLS = (10, 1)
SUMMARY = ROOT / "results" / "shield_summary.json"


def setup() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#202020",
        "figure.facecolor": "#121212",
    })


def plot_regret() -> None:
    summary = json.loads(SUMMARY.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {
        "kernel": "#56b4e9",
        "idw": "#e69f00",
        "omp3": "#009e73",
        "omp3s": "#cc79a7",
    }
    groups = [(n, dwell) for n in NS for dwell in DWELLS]
    positions = np.arange(len(groups))
    width = 0.18

    for axis, mu in zip(axes, MUS):
        mu_key = f"{mu:.1f}"
        for index, estimator in enumerate(ESTIMATORS):
            means = []
            lows = []
            highs = []
            for n, dwell in groups:
                item = summary["configurations"][mu_key][f"N{n}_t{dwell}"][estimator]
                metric = item["dose_regret_percent"]
                means.append(metric["mean"])
                lows.append(metric["mean"] - metric["ci_low"])
                highs.append(metric["ci_high"] - metric["mean"])
            axis.bar(
                positions + (index - 1.5) * width,
                means,
                width,
                yerr=np.array([lows, highs]),
                capsize=3,
                label=estimator,
                color=colors[estimator],
            )
        blind = summary["references"][mu_key]["wall_blind_regret_percent"]
        axis.axhline(blind, color="#f0e442", linestyle="--",
                     label="wall-blind oracle")
        axis.axhline(0.0, color="#ffffff", linewidth=1, label="oracle")
        axis.set_title(f"mu = {mu:.1f}")
        axis.set_xticks(positions, [f"N={n}, t={dwell}" for n, dwell in groups])
        axis.set_xlabel("configuration")
        axis.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("dose regret vs oracle (%)")
    axes[1].legend(ncol=2)
    fig.suptitle("S11 offline dose regret")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "shield_regret.png", dpi=200)
    plt.close(fig)


def plot_fields() -> None:
    mu = 2.0
    wall = ~np.asarray(FREE_MASK, dtype=bool)
    cache = ROOT / "tmp" / f"atten_mu{mu:g}.npy"
    if cache.exists():
        atten = np.load(cache)
    else:
        atten = attenuation_matrix(wall, mu)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, atten)

    shielded = shielded_field(SOURCES, wall, atten)
    cells = sample_measurement_cells(FREE_MASK, 50, 0)
    counts = simulate_counts(shielded, cells, 1, 1000)
    values = counts_to_intensity(counts, 1)
    omp3 = parametric_sources(cells, values, FREE_MASK, k=3)[0]
    omp3s = parametric_sources_shielded(
        cells, values, FREE_MASK, atten, k=3
    )[0]

    fields = (
        (TRUE_FIELD, "Free-space field"),
        (shielded, "Shielded field"),
        (omp3, "omp3 estimate"),
        (omp3s, "omp3s estimate"),
    )
    finite = [field[FREE_MASK] for field, _ in fields]
    maximum = max(float(np.max(field)) for field in finite)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for axis, (field, title) in zip(axes.ravel(), fields):
        image = axis.imshow(field, cmap="inferno", vmin=0.0, vmax=maximum)
        wall_overlay = np.ma.masked_where(~wall, wall)
        axis.imshow(wall_overlay, cmap="Greys", alpha=0.65, vmin=0, vmax=1)
        axis.scatter(
            cells[:, 1], cells[:, 0],
            s=7, facecolors="none", edgecolors="cyan", linewidths=0.5,
        )
        axis.set_title(title)
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    fig.suptitle("S11 fields at mu = 2.0, N = 50, dwell = 1, seed = 0")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "shield_fields.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    plot_regret()
    plot_fields()


if __name__ == "__main__":
    main()
