"""Write markdown and JSON summaries for the S12 survey study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "results" / "s12"
OUTPUT_JSON = ROOT / "results" / "s12_summary.json"
OUTPUT_MD = ROOT / "results" / "s12_summary.md"


def ci(value: Dict[str, float]) -> str:
    return f"{value['mean']:.1f} [{value['ci_low']:.1f}, {value['ci_high']:.1f}]"


def b0_result(estimator: str, dwell: float) -> Dict[str, Any]:
    if estimator == "kernel":
        path = ROOT / "results" / "online" / f"kernel_3x3_t{dwell:g}.json"
        item = json.loads(path.read_text())
        online = item["online"]
        return {
            "success_rate": online["success_rate_mean"],
            "mean_steps": online["mean_steps"],
            "mean_true_dose": float(
                np.mean([seed["mean_true_dose"] for seed in item["per_seed"]])
            ),
            "survey_steps": 0.0,
            "survey_dose_share": 0.0,
            "mean_distinct_cells": online["mean_distinct_cells"],
            "dose_regret_percent": online["dose_regret_percent"],
            "dose_reduction_percent": online["dose_reduction_percent"],
        }
    path = ROOT / "results" / "param" / f"online_omp3_3x3_t{dwell:g}.json"
    item = json.loads(path.read_text())
    return {
        "success_rate": item["success_rate"]["mean"],
        "mean_steps": item["mean_steps"]["mean"],
        "mean_true_dose": item["mean_true_dose"]["mean"],
        "survey_steps": 0.0,
        "survey_dose_share": 0.0,
        "mean_distinct_cells": item["mean_distinct_cells"]["mean"],
        "dose_regret_percent": item["dose_regret_percent"],
        "dose_reduction_percent": item["dose_reduction_percent"],
    }


def bnew_result(estimator: str, dwell: float, budget: int) -> Dict[str, Any]:
    path = S12 / estimator / f"3x3_t{dwell:g}_b{budget}.json"
    item = json.loads(path.read_text())
    metrics = item["metrics"]
    return {
        "success_rate": metrics["success_rate_mean"],
        "mean_steps": metrics["mean_steps"],
        "mean_true_dose": metrics["mean_true_dose"],
        "survey_steps": metrics["survey_steps"],
        "survey_dose_share": metrics["survey_dose_share"],
        "mean_distinct_cells": metrics["mean_distinct_cells"],
        "dose_regret_percent": metrics["dose_regret_percent"],
        "dose_reduction_percent": metrics["dose_reduction_percent"],
    }


def main() -> None:
    configurations: Dict[str, Any] = {}
    missing = []
    for estimator in ("kernel", "omp3"):
        for dwell in (10, 1):
            key = f"{estimator}_t{dwell:g}"
            configurations[key] = {
                "estimator": estimator,
                "dwell": dwell,
                "rows": {
                    "0": b0_result(estimator, dwell),
                    "30": bnew_result(estimator, dwell, 30),
                    "60": bnew_result(estimator, dwell, 60),
                },
            }

    summary = {"configurations": configurations, "missing": missing}
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))

    lines = [
        "# S12 Survey Dose Summary\n",
        "\n",
        "| Estimator | Dwell | B | Success | Total steps | Total true dose (a.u.) | Survey steps | Survey dose share | Distinct cells | Dose regret % [95% CI] | Dose reduction % [95% CI] |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for estimator in ("kernel", "omp3"):
        for dwell in (10, 1):
            row = configurations[f"{estimator}_t{dwell:g}"]["rows"]
            for budget in ("0", "30", "60"):
                item = row[budget]
                lines.append(
                    f"| {estimator} | {dwell:g} | {budget} | "
                    f"{item['success_rate']:.3f} | {item['mean_steps']:.1f} | "
                    f"{item['mean_true_dose']:.1f} | {item['survey_steps']:.1f} | "
                    f"{item['survey_dose_share']:.3f} | {item['mean_distinct_cells']:.1f} | "
                    f"{ci(item['dose_regret_percent'])} | "
                    f"{ci(item['dose_reduction_percent'])} |\n"
                )
    OUTPUT_MD.write_text("".join(lines))


if __name__ == "__main__":
    main()
