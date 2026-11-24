"""Plot figures for S14 exploratory failure analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
ANALYSIS_JSON = RESULTS / "s14_analysis.json"
S14_DIR = RESULTS / "s14"
S13_LAYOUTS_JSON = RESULTS / "s13_layouts.json"

from src.layouts import generate_random_layout


def set_dark_theme() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 180,
    })


def plot_paired_difference(data: dict) -> None:
    """Figure (a): Paired difference per layout, sorted, with losing layouts highlighted."""
    rows = sorted(data["per_layout"], key=lambda r: r["paired_online_difference"])
    seeds = [str(r["layout_seed"]) for r in rows]
    diffs = [r["paired_online_difference"] for r in rows]
    colors = ["#ff4d6d" if r["group"] == "omp3 loses" else "#4cc9f0" for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    bars = ax.bar(x, diffs, color=colors, edgecolor="#ffffff", linewidth=0.5, width=0.7)
    ax.axhline(0.0, color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.6)

    # Highlight legend proxy
    ax.bar(0, 0, color="#ff4d6d", label="omp3 loses (n=6, paired diff < 0)")
    ax.bar(0, 0, color="#4cc9f0", label="omp3 wins (n=14, paired diff > 0)")

    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Layout seed (sorted by paired difference)")
    ax.set_ylabel("Paired online difference (%)\n(kernel regret - omp3 regret)")
    ax.set_title("S14: Paired Online Difference per Layout (Sorted)")
    ax.legend(loc="upper left")

    # Add values on top of bars
    for bar, val in zip(bars, diffs):
        y_pos = val + (2.5 if val >= 0 else -6.0)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color="#e0e0e0")

    fig.tight_layout()
    out_path = RESULTS / "s14_paired_difference.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def plot_feature_scatters(data: dict) -> None:
    """Figure (b): Six small scatter plots of feature versus difference with Spearman annotated."""
    features = [
        ("oracle_saving_percent", "Oracle dose saving (%)"),
        ("wall_fraction", "Wall fraction"),
        ("min_source_distance", "Min source distance (cells)"),
        ("mean_source_to_shortest_dist", "Mean source-path dist (cells)"),
        ("corridor_intensity_share", "Corridor intensity share"),
        ("mean_shortest_path_length", "Mean shortest path length (cells)"),
    ]

    per_layout = data["per_layout"]
    diffs = np.array([r["paired_online_difference"] for r in per_layout])
    groups = [r["group"] for r in per_layout]
    colors = ["#ff4d6d" if g == "omp3 loses" else "#4cc9f0" for g in groups]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (fname, label) in enumerate(features):
        ax = axes[i]
        vals = np.array([r[fname] for r in per_layout])
        corr = data["correlations"][fname]

        ax.scatter(vals, diffs, c=colors, edgecolors="#ffffff", s=45, alpha=0.85, linewidth=0.5)
        ax.axhline(0.0, color="#ffffff", linestyle=":", linewidth=0.8, alpha=0.4)

        # Regression line for visual trend
        z = np.polyfit(vals, diffs, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(vals), max(vals), 50)
        ax.plot(x_trend, p(x_trend), color="#ffd166", linestyle="--", linewidth=1.2, alpha=0.7)

        rho_str = f"Spearman ρ = {corr['spearman_rho']:.3f}\n95% CI [{corr['ci_low']:.3f}, {corr['ci_high']:.3f}]\np = {corr['permutation_p']:.4f}"
        ax.text(
            0.05, 0.95, rho_str,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#101820", edgecolor="#555555", alpha=0.8),
        )

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Paired difference (%)", fontsize=9)
        ax.set_title(fname, fontsize=10, pad=6)

    fig.suptitle("S14: Pre-declared Layout Features vs Paired Online Difference", fontsize=13, y=1.00)
    fig.tight_layout()
    out_path = RESULTS / "s14_feature_scatters.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def plot_source_error_by_group(data: dict) -> None:
    """Figure (c): Matched source position error by group."""
    per_layout = data["per_layout"]
    gc = data["group_comparison"]

    loses = [r["mean_matched_source_error"] for r in per_layout if r["group"] == "omp3 loses"]
    wins = [r["mean_matched_source_error"] for r in per_layout if r["group"] == "omp3 wins"]

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [1, 2]
    box = ax.boxplot(
        [loses, wins],
        positions=positions,
        widths=0.45,
        patch_artist=True,
        boxprops=dict(facecolor="#2b2d42", edgecolor="#ffffff", linewidth=1.0),
        medianprops=dict(color="#ffd166", linewidth=2.0),
        whiskerprops=dict(color="#ffffff", linewidth=1.0),
        capprops=dict(color="#ffffff", linewidth=1.0),
    )

    box["boxes"][0].set_facecolor("#590d22")
    box["boxes"][1].set_facecolor("#003049")

    # Overlay jittered points
    rng = np.random.RandomState(42)
    jitter_loses = rng.uniform(-0.08, 0.08, len(loses)) + 1
    jitter_wins = rng.uniform(-0.08, 0.08, len(wins)) + 2
    ax.scatter(jitter_loses, loses, color="#ff4d6d", edgecolors="#ffffff", s=50, zorder=3, alpha=0.9, label="omp3 loses (n=6)")
    ax.scatter(jitter_wins, wins, color="#4cc9f0", edgecolors="#ffffff", s=50, zorder=3, alpha=0.9, label="omp3 wins (n=14)")

    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"omp3 loses\nmedian = {gc['loses_median_matched_error']:.2f}",
        f"omp3 wins\nmedian = {gc['wins_median_matched_error']:.2f}",
    ], fontsize=10)
    ax.set_ylabel("Mean matched source position error (cells)")
    ax.set_title("S14: Matched Source Position Error by Outcome Group")

    annot = (
        f"Diff of medians: {gc['diff_of_medians']:.2f} cells\n"
        f"95% CI: [{gc['diff_ci_low']:.2f}, {gc['diff_ci_high']:.2f}]\n"
        f"H4: not supported (CI covers zero)"
    )
    ax.text(
        0.05, 0.95, annot,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#101820", edgecolor="#666666", alpha=0.85),
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = RESULTS / "s14_source_error_by_group.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def reconstruct_omp3_field(fitted_sources: list[list[float]], shape: tuple[int, int], softening: float = 2.0) -> np.ndarray:
    """Reconstruct 2D parametric field from fitted sources."""
    field = np.zeros(shape, dtype=float)
    grid_r, grid_c = np.indices(shape)
    for s in fitted_sources:
        r, c, w = s[0], s[1], s[2]
        if w > 0:
            d2 = (grid_r - r) ** 2 + (grid_c - c) ** 2
            field += w / (d2 + softening)
    return field


def plot_extreme_field_comparison(data: dict) -> None:
    """Figure (d): Side-by-side picture of true field and final omp3 estimate for extreme layouts."""
    # Choice rule: layout with minimum and maximum paired online difference
    sorted_layouts = sorted(data["per_layout"], key=lambda r: r["paired_online_difference"])
    min_row = sorted_layouts[0]   # most negative (layout 204)
    max_row = sorted_layouts[-1]  # most positive (layout 201)

    min_seed = min_row["layout_seed"]
    max_seed = max_row["layout_seed"]

    min_diag = json.loads((S14_DIR / f"layout_{min_seed}.json").read_text(encoding="utf-8"))
    max_diag = json.loads((S14_DIR / f"layout_{max_seed}.json").read_text(encoding="utf-8"))

    # Episode selection rule: first episode (noise seed 0, start index 0)
    ep_min = min_diag["episodes"][0]
    ep_max = max_diag["episodes"][0]

    env_min = generate_random_layout(min_seed).to_env()
    env_max = generate_random_layout(max_seed).to_env()

    field_min_est = reconstruct_omp3_field(ep_min["fitted_sources"], env_min.shape)
    field_max_est = reconstruct_omp3_field(ep_max["fitted_sources"], env_max.shape)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    configs = [
        (
            0,
            env_min,
            field_min_est,
            min_seed,
            min_row["paired_online_difference"],
            ep_min,
            "Most Negative Paired Diff (omp3 loses)",
        ),
        (
            1,
            env_max,
            field_max_est,
            max_seed,
            max_row["paired_online_difference"],
            ep_max,
            "Most Positive Paired Diff (omp3 wins)",
        ),
    ]

    for row_idx, env, est_field, seed, diff, ep, group_title in configs:
        ax_true = axes[row_idx, 0]
        ax_est = axes[row_idx, 1]

        # Mask walls
        true_masked = env.radiation.copy()
        true_masked[env.walls] = np.nan
        est_masked = est_field.copy()
        est_masked[env.walls] = np.nan

        vmax = max(np.nanmax(true_masked), np.nanmax(est_masked))

        im_true = ax_true.imshow(true_masked, cmap="inferno", vmin=0, vmax=vmax, origin="upper")
        im_est = ax_est.imshow(est_masked, cmap="inferno", vmin=0, vmax=vmax, origin="upper")

        # Overlay walls in dark gray
        ax_true.imshow(np.where(env.walls, 1.0, np.nan), cmap="gray", vmin=0, vmax=1, origin="upper", alpha=0.6)
        ax_est.imshow(np.where(env.walls, 1.0, np.nan), cmap="gray", vmin=0, vmax=1, origin="upper", alpha=0.6)

        # Plot true sources
        for s in env.sources:
            pos = s[0] if isinstance(s[0], (tuple, list)) else s[:2]
            ax_true.plot(pos[1], pos[0], marker="*", color="#00ffcc", markersize=12, markeredgecolor="#000000", label="True source")
            ax_est.plot(pos[1], pos[0], marker="*", color="#00ffcc", markersize=10, markeredgecolor="#000000", alpha=0.6, label="True source")

        # Plot fitted sources on estimated field
        for fs in ep["fitted_sources"]:
            if fs[2] > 0:  # non-zero weight
                ax_est.plot(fs[1], fs[0], marker="X", color="#ff0055", markersize=10, markeredgecolor="#ffffff", label="Fitted source")

        # Plot executed path
        path = np.array(ep["path"])
        ax_est.plot(path[:, 1], path[:, 0], color="#ffd166", linewidth=1.5, alpha=0.8, label="Robot path")

        # Titles and labels
        ax_true.set_title(f"Layout {seed} True Field\n({group_title}, diff={diff:.1f}%)", fontsize=10)
        ax_est.set_title(
            f"Layout {seed} Final OMP3 Estimate\n(Matched pos err={ep['matched_source_position_error']:.1f} cells, L2={ep['relative_l2_field_error']:.3f})",
            fontsize=10,
        )

        ax_true.set_ylabel("Grid Row")
        ax_true.set_xlabel("Grid Col")
        ax_est.set_xlabel("Grid Col")

        fig.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04, label="Intensity (a.u.)")
        fig.colorbar(im_est, ax=ax_est, fraction=0.046, pad=0.04, label="Intensity (a.u.)")

        # Deduplicate legend handles
        handles, labels = ax_est.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_est.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=7.5)

    caption = (
        "Figure: True radiation fields versus final online OMP3 estimates for the extreme layouts.\n"
        "Selection rule: The layouts with the most negative paired difference (Layout 204, -19.44%) and the most\n"
        "positive paired difference (Layout 201, +160.89%) were chosen. Evaluated on noise seed 0, start 0."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color="#cccccc")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = RESULTS / "s14_extreme_field_comparisons.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def main() -> None:
    set_dark_theme()
    data = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    plot_paired_difference(data)
    plot_feature_scatters(data)
    plot_source_error_by_group(data)
    plot_extreme_field_comparison(data)
    print("S14 figure generation complete.")


if __name__ == "__main__":
    main()
