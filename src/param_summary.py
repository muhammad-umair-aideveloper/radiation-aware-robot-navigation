"""Summarize the completed S10 parametric-estimator study."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARAM_DIR = ROOT / "results" / "param"
SUMMARY_JSON = ROOT / "results" / "param_summary.json"
SUMMARY_MD = ROOT / "results" / "param_summary.md"
S8_DIR = ROOT / "results" / "online"


def _read(path: Path) -> Dict[str, Any]:
    """Read one study result."""
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(result: Dict[str, Any], name: str) -> Dict[str, float]:
    """Return a bootstrap metric from a result."""
    if name in result:
        return result[name]
    return result["online"][name]


def _format_metric(metric: Dict[str, float]) -> str:
    """Format a mean and confidence interval to one decimal place."""
    return f"{metric['mean']:.1f} [{metric['ci_low']:.1f}, {metric['ci_high']:.1f}]"


def _format_l2(metric: Dict[str, float]) -> str:
    """Format an L2 metric to two decimal places."""
    return f"{metric['mean']:.2f}"


def _offline_results() -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Load offline results keyed by estimator, N, and dwell."""
    results = {}
    for path in PARAM_DIR.glob("offline_*.json"):
        result = _read(path)
        results[(int(result["n"]), f"{result['dwell']:g}", result["estimator"])] = result
    return results


def _online_results() -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Load online S10 and S8 results keyed by estimator, pattern, and dwell."""
    results = {}
    for path in PARAM_DIR.glob("online_*.json"):
        result = _read(path)
        results[("s10", result["pattern"], f"{result['dwell']:g}", result["estimator"])] = result
    for estimator in ("kernel", "idw"):
        for pattern in ("cell", "3x3"):
            for dwell in ("10", "1", "0.1"):
                path = S8_DIR / f"{estimator}_{pattern}_t{dwell}.json"
                results[("s8", pattern, dwell, estimator)] = _read(path)
    return results


def _build_summary() -> Dict[str, Any]:
    """Build the machine-readable S10 summary."""
    references = _read(PARAM_DIR / "references.json")
    offline = _offline_results()
    online = _online_results()
    missing: List[str] = []
    expected = []
    for estimator in ("kernel", "idw", "omp3"):
        for n in (25, 50, 100):
            for dwell in ("10", "1", "0.1"):
                expected.append(f"offline_{estimator}_N{n}_t{dwell}.json")
    expected += ["offline_omp2_N50_t1.json", "offline_omp4_N50_t1.json"]
    for pattern in ("cell", "3x3"):
        for dwell in ("10", "1", "0.1"):
            expected.append(f"online_omp3_{pattern}_t{dwell}.json")
    expected += ["online_omp2_3x3_t1.json", "online_omp4_3x3_t1.json"]
    missing = [name for name in expected if not (PARAM_DIR / name).exists()]
    oracle_saving = (
        (references["shortest_dose_mean"] - references["oracle_dose_mean"])
        / references["shortest_dose_mean"]
        * 100.0
    )
    return {
        "references": {
            **references,
            "oracle_saving_percent": oracle_saving,
        },
        "offline": {
            f"N{n}_t{dwell}": {
                estimator: {
                    "dose_regret_percent": _metric(
                        offline[(n, dwell, estimator)], "dose_regret_percent"
                    ),
                    "l2_error": _metric(offline[(n, dwell, estimator)], "l2_error"),
                }
                for estimator in ("kernel", "idw", "omp3")
                if (n, dwell, estimator) in offline
            }
            for n in (25, 50, 100)
            for dwell in ("10", "1", "0.1")
        },
        "online": {
            f"{pattern}_t{dwell}": {
                estimator: {
                    "dose_regret_percent": _metric(
                        online[(kind, pattern, dwell, estimator)],
                        "dose_regret_percent",
                    ),
                    **(
                        {
                            "success_rate": _metric(
                                online[(kind, pattern, dwell, estimator)],
                                "success_rate",
                            ),
                            "mean_distinct_cells": _metric(
                                online[(kind, pattern, dwell, estimator)],
                                "mean_distinct_cells",
                            ),
                        }
                        if kind == "s10"
                        else {}
                    ),
                }
                for kind, estimator in (
                    ("s10", "omp3"),
                    ("s8", "kernel"),
                    ("s8", "idw"),
                )
            }
            for pattern in ("cell", "3x3")
            for dwell in ("10", "1", "0.1")
        },
        "misspecification": {
            estimator: {
                "offline": {
                    "dose_regret_percent": _metric(
                        offline[(50, "1", estimator)], "dose_regret_percent"
                    ),
                    "l2_error": _metric(offline[(50, "1", estimator)], "l2_error"),
                },
                "online": {
                    "dose_regret_percent": _metric(
                        online[("s10", "3x3", "1", estimator)],
                        "dose_regret_percent",
                    )
                },
            }
            for estimator in ("omp2", "omp3", "omp4")
        },
        "missing_configurations": missing,
    }


def _offline_table(summary: Dict[str, Any]) -> List[str]:
    """Render the offline comparison table."""
    rows = [
        "## Offline dose regret and field error",
        "",
        "| N | dwell | kernel regret [95% CI] / L2 | idw regret [95% CI] / L2 | omp3 regret [95% CI] / L2 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["offline"].items():
        n, dwell = key[1:].split("_t")
        cells = []
        for estimator in ("kernel", "idw", "omp3"):
            item = values[estimator]
            cells.append(
                f"{_format_metric(item['dose_regret_percent'])} / "
                f"{_format_l2(item['l2_error'])}"
            )
        rows.append(f"| {n} | {dwell} | " + " | ".join(cells) + " |")
    return rows


def _online_table(summary: Dict[str, Any]) -> List[str]:
    """Render the online comparison table."""
    rows = [
        "## Online dose regret",
        "",
        "| pattern | dwell | omp3 regret [95% CI] | omp3 success | omp3 distinct cells | S8 kernel regret [95% CI] | S8 idw regret [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["online"].items():
        pattern, dwell = key.split("_t")
        omp3 = values["omp3"]
        rows.append(
            f"| {pattern} | {dwell} | {_format_metric(omp3['dose_regret_percent'])} | "
            f"{omp3['success_rate']['mean']:.1f} | {omp3['mean_distinct_cells']['mean']:.1f} | "
            f"{_format_metric(values['kernel']['dose_regret_percent'])} | "
            f"{_format_metric(values['idw']['dose_regret_percent'])} |"
        )
    return rows


def _misspecification_table(summary: Dict[str, Any]) -> List[str]:
    """Render the misspecification comparison table."""
    rows = [
        "## Misspecification",
        "",
        "| estimator | offline regret [95% CI] | offline L2 error | online regret [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for estimator in ("omp2", "omp3", "omp4"):
        item = summary["misspecification"][estimator]
        rows.append(
            f"| {estimator} | {_format_metric(item['offline']['dose_regret_percent'])} | "
            f"{_format_l2(item['offline']['l2_error'])} | "
            f"{_format_metric(item['online']['dose_regret_percent'])} |"
        )
    return rows


def _markdown(summary: Dict[str, Any]) -> str:
    """Render the complete summary markdown."""
    references = summary["references"]
    lines = [
        "# S10 Physics-Informed Estimator Results",
        "",
        "## Reference paths",
        "",
        "| metric | mean |",
        "|---|---:|",
        f"| oracle dose | {references['oracle_dose_mean']:.1f} |",
        f"| A* shortest dose | {references['shortest_dose_mean']:.1f} |",
        f"| oracle steps | {references['oracle_steps_mean']:.1f} |",
        f"| A* shortest steps | {references['shortest_steps_mean']:.1f} |",
        f"| oracle saving percent | {references['oracle_saving_percent']:.1f} |",
        "",
    ]
    lines.extend(_offline_table(summary))
    lines.extend([""])
    lines.extend(_online_table(summary))
    lines.extend([""])
    lines.extend(_misspecification_table(summary))
    lines.extend(["", "## Missing configurations", ""])
    lines.append(
        "- " + (", ".join(summary["missing_configurations"]) or "none")
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Write the machine-readable and markdown S10 summaries."""
    summary = _build_summary()
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
