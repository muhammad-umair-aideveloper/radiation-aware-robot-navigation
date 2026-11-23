"""Summarize the S13 fresh-layout replication study."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
S13_DIR = ROOT / "results" / "s13"
OUTPUT_JSON = ROOT / "results" / "s13_summary.json"
OUTPUT_MD = ROOT / "results" / "s13_summary.md"

LAYOUT_SEEDS = list(range(200, 220))
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


def bootstrap(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(array))
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = np.mean(array[indices], axis=1)
    return {
        "mean": mean,
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def ci_text(value: dict[str, float]) -> str:
    return f"{value['mean']:.1f} [{value['ci_low']:.1f}, {value['ci_high']:.1f}]"


def exact_binomial_sign_test(wins: int, losses: int) -> dict[str, float | int]:
    """Exact two-sided binomial sign test against H0: p = 0.5 with ties dropped."""
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "n": 0, "p_value": 1.0}
    p_value = sum(
        math.comb(n, i)
        for i in range(n + 1)
        if abs(i - n / 2) >= abs(wins - n / 2)
    ) / (2**n)
    return {
        "wins": wins,
        "losses": losses,
        "n": n,
        "p_value": float(p_value),
    }


def compute_sign_test(diffs: Iterable[float]) -> dict[str, float | int]:
    """Compute exact two-sided binomial sign test from differences, dropping ties."""
    diff_list = [float(d) for d in diffs]
    wins = int(sum(d > 0.0 for d in diff_list))
    losses = int(sum(d < 0.0 for d in diff_list))
    ties = int(sum(d == 0.0 for d in diff_list))
    test_res = exact_binomial_sign_test(wins, losses)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n": test_res["n"],
        "p_value": test_res["p_value"],
    }


def paired_regret_difference(
    kernel_regrets: Iterable[float],
    omp3_regrets: Iterable[float],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    """Paired difference of dose regret per layout: kernel minus omp3 (%).
    
    Positive values mean kernel regret is higher than omp3 (omp3 performs better).
    """
    k_arr = np.asarray(list(kernel_regrets), dtype=np.float64)
    o_arr = np.asarray(list(omp3_regrets), dtype=np.float64)
    diff = k_arr - o_arr
    mean_val = float(np.mean(diff))
    median_val = float(np.median(diff))
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(diff), size=(resamples, len(diff)))
    means = np.mean(diff[indices], axis=1)
    return {
        "mean": mean_val,
        "median": median_val,
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def verdict(count: int, threshold: int = 15) -> str:
    return "supported" if count >= threshold else "not supported"


def main() -> None:
    records = []
    for seed in LAYOUT_SEEDS:
        path = S13_DIR / f"layout_{seed}.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))

    per_layout = []
    oracle_savings = []
    offline_kernel_regrets = []
    offline_omp3_regrets = []
    online_kernel_regrets = []
    online_omp3_regrets = []
    offline_kernel_savings = []
    offline_omp3_savings = []
    online_kernel_savings = []
    online_omp3_savings = []

    offline_omp3_beats_kernel = 0
    online_omp3_beats_kernel = 0
    oracle_saving_at_least_10 = 0
    kernel_online_over_50 = 0

    for record in records:
        seed = record["metadata"]["layout_seed"]
        oracle_saving = record["references"]["oracle_saving_percent"]
        offline_kernel = record["offline"]["kernel"]["mean_dose_regret_percent"]
        offline_omp3 = record["offline"]["omp3"]["mean_dose_regret_percent"]
        online_kernel = record["online"]["kernel"]["mean_dose_regret_percent"]
        online_omp3 = record["online"]["omp3"]["mean_dose_regret_percent"]

        offline_omp3_beats_kernel += int(offline_omp3 < offline_kernel)
        online_omp3_beats_kernel += int(online_omp3 < online_kernel)
        oracle_saving_at_least_10 += int(oracle_saving >= 10.0)
        kernel_online_over_50 += int(online_kernel > 50.0)

        per_layout.append({
            "layout_seed": seed,
            "oracle_saving_percent": oracle_saving,
            "offline_kernel_regret_percent": offline_kernel,
            "offline_idw_regret_percent": record["offline"]["idw"]["mean_dose_regret_percent"],
            "offline_omp3_regret_percent": offline_omp3,
            "online_kernel_regret_percent": online_kernel,
            "online_omp3_regret_percent": online_omp3,
            "offline_kernel_success_rate": record["offline"]["kernel"]["success_rate"],
            "offline_idw_success_rate": record["offline"]["idw"]["success_rate"],
            "offline_omp3_success_rate": record["offline"]["omp3"]["success_rate"],
            "online_kernel_success_rate": record["online"]["kernel"]["success_rate"],
            "online_omp3_success_rate": record["online"]["omp3"]["success_rate"],
        })
        oracle_savings.append(oracle_saving)
        offline_kernel_regrets.append(offline_kernel)
        offline_omp3_regrets.append(offline_omp3)
        online_kernel_regrets.append(online_kernel)
        online_omp3_regrets.append(online_omp3)
        offline_kernel_savings.append(record["offline"]["kernel"]["mean_dose_saving_percent"])
        offline_omp3_savings.append(record["offline"]["omp3"]["mean_dose_saving_percent"])
        online_kernel_savings.append(record["online"]["kernel"]["mean_dose_saving_percent"])
        online_omp3_savings.append(record["online"]["omp3"]["mean_dose_saving_percent"])

    pooled = {
        "oracle_saving_percent": bootstrap(oracle_savings),
        "offline_kernel_regret_percent": bootstrap(offline_kernel_regrets),
        "offline_idw_regret_percent": bootstrap(
            [row["offline_idw_regret_percent"] for row in per_layout]
        ),
        "offline_omp3_regret_percent": bootstrap(offline_omp3_regrets),
        "online_kernel_regret_percent": bootstrap(online_kernel_regrets),
        "online_omp3_regret_percent": bootstrap(online_omp3_regrets),
        "offline_kernel_saving_percent": bootstrap(offline_kernel_savings),
        "offline_omp3_saving_percent": bootstrap(offline_omp3_savings),
        "online_kernel_saving_percent": bootstrap(online_kernel_savings),
        "online_omp3_saving_percent": bootstrap(online_omp3_savings),
    }
    paired_counts = {
        "offline_omp3_beats_kernel_layouts": offline_omp3_beats_kernel,
        "online_omp3_beats_kernel_layouts": online_omp3_beats_kernel,
        "oracle_saving_at_least_10_layouts": oracle_saving_at_least_10,
        "kernel_online_regret_over_50_layouts": kernel_online_over_50,
        "layout_count": len(records),
    }
    hypotheses = {
        "H1": verdict(online_omp3_beats_kernel),
        "H2": verdict(oracle_saving_at_least_10),
        "H3": verdict(kernel_online_over_50),
    }

    offline_diffs = [
        row["offline_kernel_regret_percent"] - row["offline_omp3_regret_percent"]
        for row in per_layout
    ]
    online_diffs = [
        row["online_kernel_regret_percent"] - row["online_omp3_regret_percent"]
        for row in per_layout
    ]
    offline_sign = compute_sign_test(offline_diffs)
    online_sign = compute_sign_test(online_diffs)
    offline_paired_diff = paired_regret_difference(offline_kernel_regrets, offline_omp3_regrets)
    online_paired_diff = paired_regret_difference(online_kernel_regrets, online_omp3_regrets)

    supplementary = {
        "status": "post hoc (decided after the S13 results were seen)",
        "comparisons_count": 2,
        "adjusted_p_values": False,
        "nature": "descriptive",
        "offline": {
            "sign_test": offline_sign,
            "paired_difference": offline_paired_diff,
        },
        "online": {
            "sign_test": online_sign,
            "paired_difference": online_paired_diff,
        },
    }

    summary = {
        "metadata": {
            "layout_count": len(records),
            "layout_seeds": LAYOUT_SEEDS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "per_layout": per_layout,
        "pooled": pooled,
        "paired_counts": paired_counts,
        "hypothesis_verdicts": hypotheses,
        "supplementary_post_hoc": supplementary,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# S13 Fresh-Layout Replication Summary",
        "",
        "## Per-layout results",
        "",
        "| Layout | Oracle saving (%) | Offline kernel regret (%) | Offline IDW regret (%) | Offline omp3 regret (%) | Online kernel regret (%) | Online omp3 regret (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_layout:
        lines.append(
            f"| {row['layout_seed']} | {row['oracle_saving_percent']:.1f} | "
            f"{row['offline_kernel_regret_percent']:.1f} | "
            f"{row['offline_idw_regret_percent']:.1f} | "
            f"{row['offline_omp3_regret_percent']:.1f} | "
            f"{row['online_kernel_regret_percent']:.1f} | "
            f"{row['online_omp3_regret_percent']:.1f} |"
        )

    lines += [
        "",
        "## Pooled layout means",
        "",
        "| Metric | Mean [95% bootstrap interval] |",
        "|---|---:|",
    ]
    for name in (
        "oracle_saving_percent",
        "offline_kernel_regret_percent",
        "offline_idw_regret_percent",
        "offline_omp3_regret_percent",
        "online_kernel_regret_percent",
        "online_omp3_regret_percent",
        "offline_kernel_saving_percent",
        "offline_omp3_saving_percent",
        "online_kernel_saving_percent",
        "online_omp3_saving_percent",
    ):
        lines.append(f"| {name} | {ci_text(pooled[name])} |")

    lines += [
        "",
        "## Paired layout counts",
        "",
        "| Comparison or threshold | Layout count |",
        "|---|---:|",
        f"| omp3 offline beats kernel offline | {offline_omp3_beats_kernel}/20 |",
        f"| omp3 online beats kernel online | {online_omp3_beats_kernel}/20 |",
        f"| Oracle saving at least 10% | {oracle_saving_at_least_10}/20 |",
        f"| Kernel online regret over 50% | {kernel_online_over_50}/20 |",
        "",
        "## Hypothesis verdicts",
        "",
        "| Hypothesis | Verdict |",
        "|---|---|",
        f"| H1: omp3 online beats kernel in at least 15 layouts | {hypotheses['H1']} |",
        f"| H2: oracle saving is at least 10% in at least 15 layouts | {hypotheses['H2']} |",
        f"| H3: kernel online regret exceeds 50% in at least 15 layouts | {hypotheses['H3']} |",
        "",
        "## Supplementary post-hoc analysis",
        "",
        "Status: post hoc (decided after S13 results were seen). Note: 2 planned comparisons; p-values are unadjusted and descriptive. Pre-declared verdicts (H1: not supported, H2: supported, H3: supported) remain unchanged under their pre-declared thresholds.",
        "",
        "### Exact two-sided binomial sign test (omp3 vs kernel)",
        "",
        "| Mode | Wins (omp3) | Losses (kernel) | Ties | n | Two-sided p-value |",
        "|---|---:|---:|---:|---:|---:|",
        f"| offline | {offline_sign['wins']} | {offline_sign['losses']} | {offline_sign['ties']} | {offline_sign['n']} | {offline_sign['p_value']:.4f} |",
        f"| online | {online_sign['wins']} | {online_sign['losses']} | {online_sign['ties']} | {online_sign['n']} | {online_sign['p_value']:.4f} |",
        "",
        "### Paired dose regret difference (kernel minus omp3, %)",
        "",
        "| Mode | Mean diff [%] | Median diff [%] | 95% bootstrap interval |",
        "|---|---:|---:|---:|",
        f"| offline | {offline_paired_diff['mean']:.1f} | {offline_paired_diff['median']:.1f} | [{offline_paired_diff['ci_low']:.1f}, {offline_paired_diff['ci_high']:.1f}] |",
        f"| online | {online_paired_diff['mean']:.1f} | {online_paired_diff['median']:.1f} | [{online_paired_diff['ci_low']:.1f}, {online_paired_diff['ci_high']:.1f}] |",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": pooled, "paired_counts": paired_counts, "hypothesis_verdicts": hypotheses}, indent=2))


if __name__ == "__main__":
    main()
