"""Golden-reference recomputation test.

To update intentionally, rerun the computation command recorded in
tests/golden_reference.json only after a committed code or layout change,
then review every changed number and update the tolerance-approved reference.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import generate_evaluation_starts
from src.environment import RadiationEnv
from src.planners import astar_search


ROOT = Path(__file__).resolve().parents[1]


def test_golden_references() -> None:
    reference = json.loads(
        (ROOT / "tests" / "golden_reference.json").read_text()
    )
    env = RadiationEnv()
    starts = tuple(generate_evaluation_starts(env, 50, 123)[::2])
    oracle = [
        astar_search(start, env.goal, mode="aware", env=env, lambda_rad=2.5)
        for start in starts
    ]
    shortest = [
        astar_search(start, env.goal, mode="shortest", env=env)
        for start in starts
    ]
    canonical_aware = astar_search(
        (2, 2), env.goal, mode="aware", env=env, lambda_rad=2.5
    )
    canonical_shortest = astar_search(
        (2, 2), env.goal, mode="shortest", env=env
    )

    actual = {
        "oracle_dose_mean": float(np.mean([item[1] for item in oracle]))
        if oracle else 0.0,
        "oracle_steps_mean": float(np.mean([item[2] for item in oracle]))
        if oracle else 0.0,
        "shortest_dose_mean": float(np.mean([item[1] for item in shortest]))
        if shortest else 0.0,
        "shortest_steps_mean": float(np.mean([item[2] for item in shortest]))
        if shortest else 0.0,
        "canonical_aware_dose": float(canonical_aware[1]),
        "canonical_aware_steps": float(canonical_aware[2]),
        "canonical_shortest_dose": float(canonical_shortest[1]),
        "canonical_shortest_steps": float(canonical_shortest[2]),
        "canonical_aware_path_len": len(canonical_aware[0]) - 1,
        "canonical_shortest_path_len": len(canonical_shortest[0]) - 1,
    }
    for key, expected in reference.items():
        assert abs(actual[key] - expected) <= 1e-6, (key, actual[key], expected)


if __name__ == "__main__":
    test_golden_references()
    print("golden reference tests passed")
