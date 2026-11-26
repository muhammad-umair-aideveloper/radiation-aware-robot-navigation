"""Generate a provenance index for all files under results/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "RESULTS_INDEX.md"

CLAIMS: Dict[str, Tuple[str, str]] = {
    "s13": ("src/s13_summary.py", "python src/s13_summary.py"),
    "s13_layout": ("src/s13_study.py", "python src/s13_study.py"),
    "s13_layouts": ("src/s13_layouts.py", "python src/s13_layouts.py"),
    "s13_controls": ("src/s13_study.py", "python src/s13_study.py --controls"),
    "s13_online_paired": ("src/plot_s13.py", "python src/plot_s13.py"),
    "s13_oracle_saving": ("src/plot_s13.py", "python src/plot_s13.py"),
    "s13_offline_online_regret": ("src/plot_s13.py", "python src/plot_s13.py"),
    "s14_features": ("src/s14_features.py", "python src/s14_features.py"),
    "s14_analysis": ("src/s14_analysis.py", "python src/s14_analysis.py"),
    "s14_paired_difference": ("src/plot_s14.py", "python src/plot_s14.py"),
    "s14_feature_scatters": ("src/plot_s14.py", "python src/plot_s14.py"),
    "s14_source_error_by_group": ("src/plot_s14.py", "python src/plot_s14.py"),
    "s14_extreme_field_comparisons": ("src/plot_s14.py", "python src/plot_s14.py"),
    "s14": ("src/s14_diagnostics.py", "python src/s14_diagnostics.py"),
    "s12": ("src/s12_study.py", "python src/s12_study.py"),
    "s12_summary": ("src/s12_summary.py", "python src/s12_summary.py"),
    "s12_tradeoff": ("src/plot_s12.py", "python src/plot_s12.py"),
    "s12_regret": ("src/plot_s12.py", "python src/plot_s12.py"),
    "s12_control": ("src/online_controls_s12.py", "python src/online_controls_s12.py"),
    "online": ("src/online_study.py", "python src/online_study.py"),
    "online_summary": ("src/online_summary.py", "python src/online_summary.py"),
    "online2": ("src/online_study2.py", "python src/online_study2.py"),
    "online2_summary": ("src/online_summary2.py", "python src/online_summary2.py"),
    "param": ("src/param_study.py", "python src/param_study.py"),
    "param_summary": ("src/param_summary.py", "python src/param_summary.py"),
    "shield": ("src/shield_study.py", "python src/shield_study.py"),
    "shield_summary": ("src/shield_summary.py", "python src/shield_summary.py"),
    "estimation": ("src/estimation_study.py", "python src/estimation_study.py"),
    "estimation_summary": ("src/estimation_study.py", "python src/estimation_study.py"),
}


def claim(path: Path) -> Tuple[str, str]:
    rel = path.relative_to(RESULTS).as_posix()
    stem = path.stem
    for key, value in CLAIMS.items():
        if rel == key or rel.startswith(key + "/") or stem.startswith(key):
            return value
    return ("not claimed", "not recorded")


def elapsed(path: Path) -> str:
    if path.suffix != ".json":
        return ""
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        value = item.get("elapsed_seconds")
        return f"{float(value):.2f}s" if value is not None else ""
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return ""


def main() -> None:
    lines = [
        "# Results index\n",
        "\n",
        "| Path | Producer script | Command | Runtime |\n",
        "|---|---|---|---:|\n",
    ]
    unclaimed: list[str] = []
    entries: list[Tuple[str, str, str, str]] = []
    for directory in sorted((p for p in RESULTS.rglob("*") if p.is_dir()), key=lambda p: p.as_posix()):
        rel = directory.relative_to(ROOT).as_posix()
        script, command = claim(directory)
        entries.append((rel + "/", script, command, ""))
        if script == "not claimed":
            unclaimed.append(rel + "/")
    for path in sorted((p for p in RESULTS.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        rel = path.relative_to(ROOT).as_posix()
        script, command = claim(path)
        runtime = elapsed(path)
        entries.append((rel, script, command, runtime))
        if script == "not claimed":
            unclaimed.append(rel)
    for rel, script, command, runtime in entries:
        lines.append(f"| {rel} | {script} | `{command}` | {runtime} |\n")
    lines.extend(["\n", "## Files or folders with no claimed script\n", "\n"])
    if unclaimed:
        lines.extend(f"- {rel}\n" for rel in unclaimed)
    else:
        lines.append("- none\n")
    OUTPUT.write_text("".join(lines), encoding="utf-8")
    print(f"indexed {len(entries)} paths; unclaimed={len(unclaimed)}")


if __name__ == "__main__":
    main()
