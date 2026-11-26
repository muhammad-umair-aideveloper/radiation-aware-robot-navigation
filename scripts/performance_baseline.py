"""Time the navigation and estimation workloads on the canonical layout."""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from PIL import __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import GOAL_CELL, START_CELL, RadiationEnv
from src.estimators import (
    idw,
    kernel_regression,
    parametric_sources,
    parametric_sources_shielded,
)
from src.geiger_model import sample_measurement_cells
from src.online_sim import run_online_episode
from src.planners import astar_search
from src.shielding import attenuation_matrix

REPETITIONS = 3


def _median_seconds(operation: Callable[[], object], repetitions: int = REPETITIONS) -> float:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def run_baseline(output: Path = ROOT / "results" / "performance_baseline.json") -> dict:
    """Run each workload repeatedly and save a machine-specific baseline."""
    env = RadiationEnv()
    free_mask = ~env.walls
    cells = sample_measurement_cells(free_mask, 50, seed=0)
    values = env.radiation[tuple(cells.T)]
    attenuation_started = time.perf_counter()
    atten = attenuation_matrix(env.walls, mu=0.5)
    attenuation_seconds = time.perf_counter() - attenuation_started

    timings = {
        "astar_shortest": _median_seconds(
            lambda: astar_search(START_CELL, GOAL_CELL, mode="shortest", env=env)
        ),
        "astar_aware": _median_seconds(
            lambda: astar_search(START_CELL, GOAL_CELL, mode="aware", env=env)
        ),
        "estimator_kernel": _median_seconds(
            lambda: kernel_regression(cells, values, free_mask)
        ),
        "estimator_idw": _median_seconds(lambda: idw(cells, values, free_mask)),
        "estimator_omp3": _median_seconds(
            lambda: parametric_sources(cells, values, free_mask, k=3)
        ),
        "estimator_omp3_shielded": _median_seconds(
            lambda: parametric_sources_shielded(
                cells, values, free_mask, atten, k=3
            )
        ),
        "online_episode_idw": _median_seconds(
            lambda: run_online_episode(
                env.radiation,
                free_mask,
                START_CELL,
                GOAL_CELL,
                "idw",
                "cell",
                1.0,
                seed=42,
                max_steps=300,
            )
        ),
    }

    result = {
        "machine": {
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": pillow_version,
        },
        "layout": "canonical 50x50",
        "repetitions": REPETITIONS,
        "timings_median_seconds": timings,
        "shield_attenuation_matrix_seconds": attenuation_seconds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run_baseline()
    print(json.dumps(result["timings_median_seconds"], indent=2))


if __name__ == "__main__":
    main()
