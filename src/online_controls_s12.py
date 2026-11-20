"""Executed B=0 control for S12 integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import generate_evaluation_starts
from src.environment import RadiationEnv
from src.online_sim import run_online_episode


def _kernel_seed(start_cells: Iterable[Tuple[int, int]], seed: int) -> Dict[str, float]:
    env = RadiationEnv()
    episodes = [
        run_online_episode(
            env.radiation,
            ~env.walls,
            start,
            env.goal,
            "kernel",
            "3x3",
            1.0,
            seed * 1000 + index,
            replan_every=3,
            lam=2.5,
            max_steps=300,
            survey_budget=0,
        )
        for index, start in enumerate(start_cells)
    ]
    successful = [episode for episode in episodes if episode["success"]]
    return {
        "mean_true_dose": float(np.mean([episode["dose"] for episode in successful])),
        "mean_steps": float(np.mean([episode["steps"] for episode in successful])),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
    }


def _omp_seed(start_cells: Iterable[Tuple[int, int]], seed: int) -> Dict[str, float]:
    env = RadiationEnv()
    episodes = [
        run_online_episode(
            env.radiation,
            ~env.walls,
            start,
            env.goal,
            "omp3",
            "3x3",
            1.0,
            seed * 1000 + index,
            replan_every=3,
            lam=2.5,
            max_steps=300,
            survey_budget=0,
        )
        for index, start in enumerate(start_cells)
    ]
    successful = [episode for episode in episodes if episode["success"]]
    return {
        "mean_true_dose": float(np.mean([episode["dose"] for episode in successful])),
        "mean_steps": float(np.mean([episode["steps"] for episode in successful])),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
    }


def run_control() -> Dict[str, object]:
    """Recompute S8/S10 seed 0 and 1 and compare saved per-seed values."""
    env = RadiationEnv()
    starts = tuple(generate_evaluation_starts(env, 50, 123)[::2])
    kernel_reference = json.loads(
        (ROOT / "results" / "online" / "kernel_3x3_t1.json").read_text()
    )
    omp_reference = json.loads(
        (ROOT / "results" / "param" / "online_omp3_3x3_t1.json").read_text()
    )

    checks: list[Dict[str, object]] = []
    for seed, expected in zip((0, 1), kernel_reference["per_seed"][:2]):
        actual = _kernel_seed(starts, seed)
        differences = {
            key: abs(actual[key] - float(expected[key]))
            for key in ("mean_true_dose", "mean_steps", "success_rate")
        }
        checks.append({
            "estimator": "kernel",
            "seed": seed,
            "actual": actual,
            "reference": {
                key: float(expected[key])
                for key in ("mean_true_dose", "mean_steps", "success_rate")
            },
            "max_abs_difference": float(max(differences.values())),
            "passed": bool(max(differences.values()) <= 1e-9),
        })
    for seed, expected in zip((0, 1), omp_reference["per_seed"][:2]):
        actual = _omp_seed(starts, seed)
        differences = {
            key: abs(actual[key] - float(expected[key]))
            for key in ("mean_true_dose", "mean_steps", "success_rate")
        }
        checks.append({
            "estimator": "omp3",
            "seed": seed,
            "actual": actual,
            "reference": {
                key: float(expected[key])
                for key in ("mean_true_dose", "mean_steps", "success_rate")
            },
            "max_abs_difference": float(max(differences.values())),
            "passed": bool(max(differences.values()) <= 1e-9),
        })

    passed = all(bool(check["passed"]) for check in checks)
    return {"passed": passed, "checks": checks}


if __name__ == "__main__":
    result = run_control()
    output = ROOT / "results" / "s12_control.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
