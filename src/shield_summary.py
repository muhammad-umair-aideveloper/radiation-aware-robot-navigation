"""Summarize the completed S11 wall-shielding study."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHIELD_DIR = ROOT / "results" / "shield"
SUMMARY_JSON = ROOT / "results" / "shield_summary.json"
SUMMARY_MD = ROOT / "results" / "shield_summary.md"
MUS = (0.5, 2.0)
ESTIMATORS = ("kernel", "idw", "omp3", "omp3s")
NS = (50, 100)
DWELLS = (10, 1)


def _format_metric(metric: Dict[str, float]) -> str:
    return f"{metric['mean']:.1f} [{metric['ci_low']:.1f}, {metric['ci_high']:.1f}]"


def _build_summary() -> Dict[str, Any]:
    references = {}
    configurations = {}
    missing = []

    for mu in MUS:
        mu_key = f"{mu:.1f}"
        reference_path = SHIELD_DIR / f"references_mu{mu:g}.json"
        if not reference_path.exists():
            missing.append(reference_path.name)
            continue
        reference = json.loads(reference_path.read_text())
        reference["oracle_saving_percent"] = (
            (reference["shortest_dose_mean"] - reference["oracle_dose_mean"])
            / reference["shortest_dose_mean"]
            * 100.0
        )
        references[mu_key] = reference

        configurations[mu_key] = {}
        for n in NS:
            for dwell in DWELLS:
                key = f"N{n}_t{dwell}"
                configurations[mu_key][key] = {}
                for estimator in ESTIMATORS:
                    name = f"{estimator}_N{n}_t{dwell}_mu{mu:g}.json"
                    path = SHIELD_DIR / name
                    if not path.exists():
                        missing.append(name)
                        continue
                    result = json.loads(path.read_text())
                    configurations[mu_key][key][estimator] = {
                        "dose_regret_percent": result["summary"]["dose_regret_percent"],
                        "l2_error": result["summary"]["l2_error"],
                        "dose_reduction_percent": result["summary"]["dose_reduction_percent"],
                        "success_rate": result["summary"]["success_rate"],
                    }

    return {
        "references": references,
        "configurations": configurations,
        "missing_configurations": sorted(set(missing)),
    }


def _markdown(summary: Dict[str, Any]) -> str:
    lines = ["# S11 Wall Shielding Study Results", ""]

    lines.extend([
        "## References",
        "",
        "| mu | oracle mean dose | oracle mean steps | A* shortest mean dose | A* shortest mean steps | oracle saving (%) | wall-blind mean dose | wall-blind mean steps | wall-blind regret (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mu in ("0.5", "2.0"):
        reference = summary["references"][mu]
        lines.append(
            f"| {mu} | {reference['oracle_dose_mean']:.1f} | "
            f"{reference['oracle_steps_mean']:.1f} | "
            f"{reference['shortest_dose_mean']:.1f} | "
            f"{reference['shortest_steps_mean']:.1f} | "
            f"{reference['oracle_saving_percent']:.1f} | "
            f"{reference['wall_blind_dose_mean']:.1f} | "
            f"{reference['wall_blind_steps_mean']:.1f} | "
            f"{reference['wall_blind_regret_percent']:.1f} |"
        )

    for mu in ("0.5", "2.0"):
        lines.extend(["", f"## mu = {mu}", "", "| N | dwell | kernel regret [95% CI] / L2 | idw regret [95% CI] / L2 | omp3 regret [95% CI] / L2 | omp3s regret [95% CI] / L2 |", "|---:|---:|---:|---:|---:|---:|"])
        for n in NS:
            for dwell in DWELLS:
                values = summary["configurations"][mu][f"N{n}_t{dwell}"]
                cells = []
                for estimator in ESTIMATORS:
                    item = values[estimator]
                    cells.append(
                        f"{_format_metric(item['dose_regret_percent'])} / "
                        f"{item['l2_error']['mean']:.2f}"
                    )
                lines.append(
                    f"| {n} | {dwell} | " + " | ".join(cells) + " |"
                )

    lines.extend(["", "## Missing configurations", ""])
    missing = summary["missing_configurations"]
    lines.append("- " + (", ".join(missing) if missing else "none"))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summary = _build_summary()
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    SUMMARY_MD.write_text(_markdown(summary))


if __name__ == "__main__":
    main()
