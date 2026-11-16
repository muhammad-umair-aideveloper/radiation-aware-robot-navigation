"""Summarize saved S8 online study results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "online"
SUMMARY_MD = ROOT / "results" / "online_summary.md"
SUMMARY_JSON = ROOT / "results" / "online_summary.json"
ESTIMATORS = ("kernel", "idw")
PATTERNS = ("cell", "3x3")
DWELLS = (10, 1, 0.1)


def _config_name(estimator: str, pattern: str, dwell: float) -> str:
    """Return a saved configuration filename."""
    return f"{estimator}_{pattern}_t{dwell:g}.json"


def _one_decimal(value: float) -> str:
    """Format a markdown value to one decimal place."""
    return f"{value:.1f}"


def _ci(item: Dict[str, Any]) -> str:
    """Format a bootstrap result as a one-decimal confidence interval."""
    return (
        f"{_one_decimal(item['mean'])} "
        f"[{_one_decimal(item['ci_low'])}, {_one_decimal(item['ci_high'])}]"
    )


def _table(estimator: str, configs: Dict[str, Any]) -> List[str]:
    """Build the table for one estimator."""
    lines = [
        f"### {estimator}\n",
        "\n",
        "| Pattern | Dwell | Success rate | Distinct cells | Replans | Online regret % [95% CI] | Offline regret % [95% CI] | Online reduction % [95% CI] |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for pattern in PATTERNS:
        for dwell in DWELLS:
            item = configs[_config_name(estimator, pattern, dwell)]
            online = item["online"]
            offline = item["offline_matched"]
            lines.append(
                f"| {pattern} | {dwell:g} | "
                f"{_one_decimal(online['success_rate_mean'] * 100)}% | "
                f"{_one_decimal(online['mean_distinct_cells'])} | "
                f"{_one_decimal(online['mean_replans'])} | "
                f"{_ci(online['dose_regret_percent'])} | "
                f"{_ci(offline['dose_regret_percent'])} | "
                f"{_ci(online['dose_reduction_percent'])} |\n"
            )
    return lines


def main() -> None:
    """Read configurations and write markdown and JSON summaries."""
    references = json.loads((RESULTS / "references.json").read_text())
    configs: Dict[str, Any] = {}
    missing: List[str] = []
    for estimator in ESTIMATORS:
        for pattern in PATTERNS:
            for dwell in DWELLS:
                name = _config_name(estimator, pattern, dwell)
                path = RESULTS / name
                if path.exists():
                    configs[name] = json.loads(path.read_text())
                else:
                    missing.append(name)

    oracle_saving = (
        (references["shortest_dose_mean"] - references["oracle_dose_mean"])
        / references["shortest_dose_mean"]
        * 100.0
    )
    summary = {
        "references": references,
        "oracle_saving_percent": oracle_saving,
        "configurations": configs,
        "missing_configurations": missing,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    lines = [
        "# S8 online measure-estimate-replan summary\n",
        "\n",
        "## References\n",
        "\n",
        "| Reference | Mean dose (a.u.) | Mean steps |\n",
        "|---|---:|---:|\n",
        f"| Oracle | {_one_decimal(references['oracle_dose_mean'])} | {_one_decimal(references['oracle_steps_mean'])} |\n",
        f"| A* shortest | {_one_decimal(references['shortest_dose_mean'])} | {_one_decimal(references['shortest_steps_mean'])} |\n",
        "\n",
        f"Oracle saving versus A* shortest: **{_one_decimal(oracle_saving)}%**.\n",
    ]
    for estimator in ESTIMATORS:
        lines.append("\n")
        lines.extend(_table(estimator, configs))
    lines.append("\n")
    lines.extend(["## Missing configurations\n", "\n"])
    if missing:
        lines.extend(f"- {name}\n" for name in missing)
    else:
        lines.append("- none\n")
    SUMMARY_MD.write_text("".join(lines))


if __name__ == "__main__":
    main()
