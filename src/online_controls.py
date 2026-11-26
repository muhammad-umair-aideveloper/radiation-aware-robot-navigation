"""Validate online simulator controls against oracle paths."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RadiationEnv
from src.online_sim import run_online_episode
from src.online_study import FREE_MASK, GOAL, STARTS, TRUE_FIELD
from src.planners import astar_search

Cell = Tuple[int, int]


def path_cost(path: Iterable[Cell], field, lam: float = 2.5) -> float:
    """Return aware-planning cost for a path on a radiation field."""
    cells = list(path)
    total = 0.0
    for current, new in zip(cells, cells[1:]):
        dr = new[0] - current[0]
        dc = new[1] - current[1]
        length = math.sqrt(2.0) if dr and dc else 1.0
        total += length * (1.0 + lam * float(field[new]))
    return total


def path_length(path: Iterable[Cell]) -> float:
    """Return geometric length for a path."""
    cells = list(path)
    return sum(
        math.sqrt(2.0) if (new[0] - current[0]) and (new[1] - current[1]) else 1.0
        for current, new in zip(cells, cells[1:])
    )


def run_controls(starts: Iterable[Cell]) -> Dict[str, object]:
    """Run true-field and zero-field control comparisons."""
    env = RadiationEnv()
    oracle_differences: List[float] = []
    shortest_differences: List[float] = []
    true_lower_bound = True
    for start in starts:
        oracle_path, _, _, _ = astar_search(
            start, GOAL, mode="aware", env=env, lambda_rad=2.5
        )
        shortest_path, _, _, _ = astar_search(
            start, GOAL, mode="shortest", env=env
        )
        true_result = run_online_episode(
            TRUE_FIELD,
            FREE_MASK,
            start,
            GOAL,
            "kernel",
            "cell",
            1.0,
            0,
            control="true",
        )
        zero_result = run_online_episode(
            TRUE_FIELD,
            FREE_MASK,
            start,
            GOAL,
            "kernel",
            "cell",
            1.0,
            0,
            control="zero",
        )
        oracle_cost = path_cost(oracle_path, TRUE_FIELD)
        executed_cost = path_cost(true_result["path"], TRUE_FIELD)
        oracle_differences.append((executed_cost - oracle_cost) / oracle_cost)
        true_lower_bound = true_lower_bound and executed_cost >= oracle_cost - 1e-9
        shortest_differences.append(
            path_length(zero_result["path"]) - path_length(shortest_path)
        )

    true_max = max(abs(value) for value in oracle_differences)
    zero_max = max(abs(value) for value in shortest_differences)
    return {
        "true": {
            "differences": oracle_differences,
            "max_abs_difference": true_max,
            "passed": true_max <= 1e-9 and true_lower_bound,
        },
        "zero": {
            "differences": shortest_differences,
            "max_abs_difference": zero_max,
            "passed": zero_max <= 1e-9,
        },
    }


def main() -> None:
    """Run controls and write the JSON and Markdown reports."""
    results = run_controls(STARTS)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "controls.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = [
        "| control | max abs difference | passed |",
        "|---|---:|---|",
    ]
    for name in ("true", "zero"):
        item = results[name]
        markdown.append(
            f"| {name} | {item['max_abs_difference']:.12g} | {item['passed']} |"
        )
    report = "\n".join(markdown) + "\n"
    (output / "controls.md").write_text(report, encoding="utf-8")
    print(report, end="")
    print(json.dumps(results, indent=2))
    if not all(results[name]["passed"] for name in ("true", "zero")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
