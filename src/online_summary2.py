"""Generate the S9 online study summary from result JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ONLINE2 = RESULTS / "online2"


def load_results() -> Dict[str, Dict[str, Any]]:
    """Load all S9 configuration results by filename."""
    return {
        path.name: json.loads(path.read_text())
        for path in ONLINE2.glob("*.json")
        if path.name != "references.json"
    }


def one(value: float) -> str:
    """Format a markdown value to one decimal place."""
    return f"{value:.1f}"


def metric(item: Dict[str, Any], key: str) -> float:
    """Read a scalar or bootstrap mean from a result block."""
    value = item["online"][key]
    return float(value["mean"] if isinstance(value, dict) else value)


def ci(item: Dict[str, Any], key: str) -> str:
    """Format a bootstrap metric and its confidence interval."""
    value = item.get("online", item)[key]
    return f"{one(value['mean'])} [{one(value['ci_low'])}, {one(value['ci_high'])}]"


def table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a markdown table with one newline per row."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the S9 markdown and JSON summaries."""
    data = load_results()
    references = json.loads((ONLINE2 / "references.json").read_text())
    controls = json.loads((RESULTS / "controls.json").read_text())
    s8: Dict[str, Dict[str, Any]] = {}
    for path in (RESULTS / "online").glob("*_3x3_t*.json"):
        item = json.loads(path.read_text())
        if item["dwell"] in (10, 1):
            s8[f"{item['estimator']}_{int(item['dwell'])}"] = item

    expected = [
        f"{estimator}_{pattern}_tinf_b0.json"
        for estimator in ("kernel", "idw")
        for pattern in ("cell", "3x3", "disc3", "disc6")
    ]
    expected += [
        f"{estimator}_{pattern}_t{dwell}_b{beta}.json"
        for estimator in ("kernel", "idw")
        for pattern in ("3x3", "disc3", "disc6")
        for dwell in (10, 1)
        for beta in (0, 2)
        if not (pattern == "3x3" and beta == 0)
    ]
    missing = [name for name in expected if name not in data]
    oracle_saving = (
        (references["shortest_dose_mean"] - references["oracle_dose_mean"])
        / references["shortest_dose_mean"]
        * 100.0
    )
    summary: Dict[str, Any] = {
        "controls": controls,
        "references": {**references, "oracle_saving_percent": oracle_saving},
        "noise_free": [],
        "noisy": {"kernel": [], "idw": []},
        "missing": missing,
    }
    for estimator in ("kernel", "idw"):
        for pattern in ("cell", "3x3", "disc3", "disc6"):
            item = data[f"{estimator}_{pattern}_tinf_b0.json"]
            summary["noise_free"].append({
                "estimator": estimator,
                "pattern": pattern,
                "success_rate": metric(item, "success_rate_mean"),
                "distinct_cells": metric(item, "mean_distinct_cells"),
                "dose_regret_percent": metric(item, "dose_regret_percent"),
                "dose_reduction_percent": metric(item, "dose_reduction_percent"),
            })
        for pattern in ("3x3", "disc3", "disc6"):
            for dwell in (10, 1):
                for beta in (0, 2):
                    source = "S9"
                    if pattern == "3x3" and beta == 0:
                        item = s8[f"{estimator}_{dwell}"]
                        source = "S8"
                    else:
                        item = data[f"{estimator}_{pattern}_t{dwell}_b{beta}.json"]
                    summary["noisy"][estimator].append({
                        "pattern": pattern,
                        "dwell": dwell,
                        "beta": beta,
                        "source": source,
                        "success_rate": metric(item, "success_rate_mean"),
                        "distinct_cells": metric(item, "mean_distinct_cells"),
                        "replans": metric(item, "mean_replans"),
                        "mean_steps": metric(item, "mean_steps"),
                        "dose_regret_percent": item["online"]["dose_regret_percent"],
                        "dose_reduction_percent": item["online"]["dose_reduction_percent"],
                    })

    lines = ["", "# S9 online study summary", "", "## Controls", "",
             (RESULTS / "controls.md").read_text().strip(), "",
             "## References", "",
             table(["reference", "mean dose (a.u.)", "mean steps", "oracle saving %"], [
                 ["oracle", one(references["oracle_dose_mean"]),
                  one(references["oracle_steps_mean"]), one(oracle_saving)],
                 ["A* shortest", one(references["shortest_dose_mean"]),
                  one(references["shortest_steps_mean"]), one(0.0)],
             ]), "## Noise-free", "",
             table(["estimator / pattern", "success rate", "distinct cells",
                    "dose regret %", "dose reduction %"], [
                 [f"{row['estimator']} / {row['pattern']}", one(row["success_rate"]),
                  one(row["distinct_cells"]), one(row["dose_regret_percent"]),
                  one(row["dose_reduction_percent"])]
                 for row in summary["noise_free"]
             ])]
    for estimator in ("kernel", "idw"):
        lines += [f"## Noisy: {estimator}", "",
                  table(["pattern / dwell / beta", "success rate", "distinct cells",
                         "replans", "mean steps", "dose regret % [95% CI]",
                         "dose reduction % [95% CI]"], [
                      [f"{row['pattern']} / {row['dwell']} / {row['beta']} ({row['source']})",
                       one(row["success_rate"]), one(row["distinct_cells"]),
                       one(row["replans"]), one(row["mean_steps"]),
                       ci(row, "dose_regret_percent"),
                       ci(row, "dose_reduction_percent")]
                      for row in summary["noisy"][estimator]
                  ])]
    lines += ["## Missing configurations", "",
              "none" if not missing else "\n".join(missing), ""]
    (RESULTS / "online2_summary.md").write_text("\n".join(lines))
    (RESULTS / "online2_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
