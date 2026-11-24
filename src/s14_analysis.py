"""S14 exploratory failure analysis: statistics, group comparison and verdicts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES_JSON = ROOT / "results" / "s14_features.json"
S13_SUMMARY_JSON = ROOT / "results" / "s13_summary.json"
S14_DIR = ROOT / "results" / "s14"
OUTPUT_JSON = ROOT / "results" / "s14_analysis.json"
OUTPUT_MD = ROOT / "results" / "s14_analysis.md"

FEATURE_NAMES = [
    "oracle_saving_percent",
    "wall_fraction",
    "min_source_distance",
    "mean_source_to_shortest_dist",
    "corridor_intensity_share",
    "mean_shortest_path_length",
]

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42
PERMUTATION_N = 10000
PERMUTATION_SEED = 42


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two arrays."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d = rx - ry
    return float(1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def _rank(arr: np.ndarray) -> np.ndarray:
    """Average ranks for array values, handling ties."""
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[order[j + 1]] == arr[order[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman correlation resampling paired observations."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rhos[b] = spearman_rank_correlation(x[idx], y[idx])
    lo = float(np.percentile(rhos, 100 * alpha / 2))
    hi = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return lo, hi


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = PERMUTATION_N,
    seed: int = PERMUTATION_SEED,
) -> float:
    """Two-sided permutation test p-value for Spearman correlation."""
    rng = np.random.RandomState(seed)
    obs = abs(spearman_rank_correlation(x, y))
    count = 0
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        if abs(spearman_rank_correlation(x, perm_y)) >= obs:
            count += 1
    return float(count / n_perm)


def bootstrap_median_diff_ci(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for difference of medians (A minus B)."""
    rng = np.random.RandomState(seed)
    na, nb = len(group_a), len(group_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sa = group_a[rng.randint(0, na, size=na)]
        sb = group_b[rng.randint(0, nb, size=nb)]
        diffs[b] = float(np.median(sa)) - float(np.median(sb))
    median_diff = float(np.median(group_a)) - float(np.median(group_b))
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return median_diff, lo, hi


def run_analysis() -> dict:
    """Run the full S14 analysis and return the results dict."""
    features_data = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    records = features_data["records"]

    # Load diagnostics summaries
    diag_summaries: dict[int, dict] = {}
    for rec in records:
        seed = rec["layout_seed"]
        diag_path = S14_DIR / f"layout_{seed}.json"
        if diag_path.exists():
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            diag_summaries[seed] = diag["summary"]

    # Build arrays
    seeds = [r["layout_seed"] for r in records]
    diffs = np.array([r["paired_online_difference"] for r in records])
    groups = [r["group"] for r in records]

    # Feature arrays
    feature_arrays: dict[str, np.ndarray] = {}
    for fname in FEATURE_NAMES:
        feature_arrays[fname] = np.array(
            [r["features"][fname] for r in records]
        )

    # Spearman correlations with bootstrap CIs and permutation p-values
    correlations: dict[str, dict] = {}
    for fname in FEATURE_NAMES:
        x = feature_arrays[fname]
        rho = spearman_rank_correlation(x, diffs)
        ci_lo, ci_hi = bootstrap_spearman_ci(x, diffs)
        p_val = permutation_p_value(x, diffs)
        correlations[fname] = {
            "spearman_rho": rho,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "permutation_p": p_val,
        }

    # Group comparison: matched source position error
    loses_mask = np.array([g == "omp3 loses" for g in groups])
    wins_mask = np.array([g == "omp3 wins" for g in groups])

    loses_errors = np.array([
        diag_summaries[s]["mean_matched_source_position_error"]
        for s, m in zip(seeds, loses_mask) if m and s in diag_summaries
    ])
    wins_errors = np.array([
        diag_summaries[s]["mean_matched_source_position_error"]
        for s, m in zip(seeds, wins_mask) if m and s in diag_summaries
    ])

    loses_median = float(np.median(loses_errors)) if len(loses_errors) > 0 else 0.0
    wins_median = float(np.median(wins_errors)) if len(wins_errors) > 0 else 0.0

    diff_median, diff_ci_lo, diff_ci_hi = bootstrap_median_diff_ci(
        loses_errors, wins_errors,
    )

    group_comparison = {
        "loses_count": int(np.sum(loses_mask)),
        "wins_count": int(np.sum(wins_mask)),
        "loses_median_matched_error": loses_median,
        "wins_median_matched_error": wins_median,
        "diff_of_medians": diff_median,
        "diff_ci_low": diff_ci_lo,
        "diff_ci_high": diff_ci_hi,
    }

    # H4 verdict
    h4_larger = loses_median > wins_median
    h4_excludes_zero = diff_ci_lo > 0.0
    if h4_larger and h4_excludes_zero:
        h4_verdict = "supported"
    elif h4_larger and not h4_excludes_zero:
        h4_verdict = "not supported"
    else:
        h4_verdict = "not supported"

    # H5 verdict: oracle saving positively correlated with paired diff
    oracle_corr = correlations["oracle_saving_percent"]
    h5_positive = oracle_corr["spearman_rho"] > 0.0
    h5_excludes_zero = oracle_corr["ci_low"] > 0.0
    if h5_positive and h5_excludes_zero:
        h5_verdict = "supported"
    elif h5_positive and not h5_excludes_zero:
        h5_verdict = "not supported"
    else:
        h5_verdict = "not supported"

    # Per-layout table with diagnostics
    per_layout_table: list[dict] = []
    for rec in records:
        seed = rec["layout_seed"]
        entry = {
            "layout_seed": seed,
            "paired_online_difference": rec["paired_online_difference"],
            "group": rec["group"],
        }
        entry.update(rec["features"])
        if seed in diag_summaries:
            ds = diag_summaries[seed]
            entry["mean_matched_source_error"] = ds["mean_matched_source_position_error"]
            entry["all_sources_within_3_rate"] = ds["all_sources_within_3_rate"]
            entry["mean_relative_l2_error"] = ds["mean_relative_l2_field_error"]
            entry["mean_distinct_cells"] = ds["mean_distinct_measured_cells"]
        per_layout_table.append(entry)

    result = {
        "metadata": {
            "study": "S14",
            "status": "post hoc, exploratory",
            "layout_count": len(records),
            "bootstrap_resamples": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "permutation_count": PERMUTATION_N,
            "permutation_seed": PERMUTATION_SEED,
            "features_compared": len(FEATURE_NAMES),
            "note": "Six features compared without multiplicity adjustment; all p-values descriptive.",
        },
        "per_layout": per_layout_table,
        "correlations": correlations,
        "group_comparison": group_comparison,
        "verdicts": {
            "H4": h4_verdict,
            "H4_detail": (
                f"Loses median {loses_median:.2f} vs wins median {wins_median:.2f}; "
                f"diff of medians {diff_median:.2f} [{diff_ci_lo:.2f}, {diff_ci_hi:.2f}]"
            ),
            "H5": h5_verdict,
            "H5_detail": (
                f"Oracle saving Spearman rho = {oracle_corr['spearman_rho']:.3f} "
                f"[{oracle_corr['ci_low']:.3f}, {oracle_corr['ci_high']:.3f}]"
            ),
            "H1_unchanged": "not supported",
        },
    }
    return result


def write_markdown(result: dict) -> None:
    """Write the analysis markdown summary."""
    lines = [
        "# S14 Exploratory Failure Analysis Results",
        "",
        "Status: post hoc, exploratory. All p-values are descriptive. "
        "Six features compared without multiplicity adjustment.",
        "",
        "## Per-Layout Table",
        "",
        "| Layout | Group | Paired diff (%) | Oracle saving (%) | "
        "Matched source error | Relative L2 error | Distinct cells |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_layout"]:
        mse = row.get("mean_matched_source_error", float("nan"))
        l2 = row.get("mean_relative_l2_error", float("nan"))
        dc = row.get("mean_distinct_cells", float("nan"))
        lines.append(
            f"| {row['layout_seed']} | {row['group']} | "
            f"{row['paired_online_difference']:.2f} | "
            f"{row['oracle_saving_percent']:.1f} | "
            f"{mse:.2f} | {l2:.4f} | {dc:.1f} |"
        )

    lines += [
        "",
        "## Feature-Difference Spearman Correlations",
        "",
        "| Feature | Spearman rho | 95% CI | Permutation p |",
        "|---|---:|---:|---:|",
    ]
    for fname, corr in result["correlations"].items():
        lines.append(
            f"| {fname} | {corr['spearman_rho']:.3f} | "
            f"[{corr['ci_low']:.3f}, {corr['ci_high']:.3f}] | "
            f"{corr['permutation_p']:.4f} |"
        )

    gc = result["group_comparison"]
    lines += [
        "",
        "## Group Comparison: Matched Source Position Error",
        "",
        f"- Losing group (n={gc['loses_count']}): median = {gc['loses_median_matched_error']:.2f} cells",
        f"- Winning group (n={gc['wins_count']}): median = {gc['wins_median_matched_error']:.2f} cells",
        f"- Difference of medians (loses minus wins): {gc['diff_of_medians']:.2f} "
        f"[{gc['diff_ci_low']:.2f}, {gc['diff_ci_high']:.2f}]",
        "",
        "## Verdicts",
        "",
        f"- H4: **{result['verdicts']['H4']}**. {result['verdicts']['H4_detail']}.",
        f"- H5: **{result['verdicts']['H5']}**. {result['verdicts']['H5_detail']}.",
        f"- H1: **{result['verdicts']['H1_unchanged']}** (unchanged from S13).",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = run_analysis()
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)
    print(f"Wrote {OUTPUT_JSON.name} and {OUTPUT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
