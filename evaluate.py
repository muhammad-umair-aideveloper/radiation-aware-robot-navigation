"""Standalone inference script: load pre-trained Q-table and run a greedy rollout.

Usage:
    python evaluate.py

Loads models/q_table_best.npy (zero training time), runs a greedy (epsilon=0)
rollout from (2, 2) to (47, 47), and prints step count, dose, and latency.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import (
    ACTIONS,
    ACTION_LENGTHS,
    GOAL_CELL,
    START_CELL,
    RadiationEnv,
)


def greedy_rollout(
    Q: np.ndarray,
    env: RadiationEnv,
    start: tuple[int, int] = START_CELL,
    max_steps: int = 300,
) -> tuple[list[tuple[int, int]], str, float, int]:
    """Pure-greedy rollout (epsilon=0) from start.

    Returns:
        (path, outcome, total_dose, step_count)
        outcome in ("goal", "collision", "timeout")
    """
    r, c = start
    path: list[tuple[int, int]] = [(r, c)]
    total_dose = 0.0

    for step in range(max_steps):
        if (r, c) == env.goal:
            return path, "goal", total_dose, step

        a = int(np.argmax(Q[r, c]))
        dr, dc = ACTIONS[a]
        length = ACTION_LENGTHS[a]

        if env.is_collision(r, c, dr, dc):
            return path, "collision", total_dose, step

        nr, nc = r + dr, c + dc
        total_dose += float(env.radiation[nr, nc] * length)
        path.append((nr, nc))

        if (nr, nc) == env.goal:
            return path, "goal", total_dose, step + 1

        r, c = nr, nc

    return path, "timeout", total_dose, max_steps


def main() -> None:
    q_path = REPO_ROOT / "models" / "q_table_best.npy"
    if not q_path.exists():
        print(f"ERROR: {q_path} not found. Run `python main.py` first.")
        raise SystemExit(1)

    print("=" * 55)
    print("Radiation Navigation — Pre-Trained Inference")
    print("=" * 55)
    print(f"Loading Q-table from: {q_path}")

    t_load = time.perf_counter()
    Q = np.load(q_path)
    t_load_ms = (time.perf_counter() - t_load) * 1000

    env = RadiationEnv()

    t_run = time.perf_counter()
    path, outcome, total_dose, steps = greedy_rollout(Q, env)
    t_run_ms = (time.perf_counter() - t_run) * 1000

    print(f"\nRoute : {START_CELL} -> {GOAL_CELL}")
    print(f"Outcome         : {outcome}")
    print(f"Steps           : {steps}")
    print(f"Total dose      : {total_dose:.4f} a.u.")
    print(f"Load latency    : {t_load_ms:.3f} ms")
    print(f"Rollout latency : {t_run_ms:.3f} ms")
    print(f"Q-table shape   : {Q.shape}")
    print("=" * 55)

    if outcome != "goal":
        raise SystemExit(f"Rollout did not reach the goal (outcome={outcome}). Check the saved model.")


if __name__ == "__main__":
    main()

