"""Summarize the S13 fresh-layout replication study."""

from __future__ import annotations

import json
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
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": pooled, "paired_counts": paired_counts, "hypothesis_verdicts": hypotheses}, indent=2))


if __name__ == "__main__":
    main()
