"""Trajectory safety and Pareto metrics evaluator.

Computes three research-grade metrics across evaluation episodes:

1. **Success Rate** (%)
       Fraction of episodes where the robot reached the goal.

2. **Path Efficiency Ratio** (dimensionless, in (0, 1])
       PER = L_optimal / L_actual
       where L_optimal is the free-space shortest path length (octile distance
       from start to goal on the obstacle-free grid, computed by A*-shortest),
       and L_actual is the length of the agent's executed path.
       PER = 1.0 means the agent took the geometrically shortest path;
       PER < 1.0 means the path was longer than necessary.

3. **Radiation Exposure**
       - Total cumulative dose (arbitrary units, a.u.) over successful episodes.
       - Peak instantaneous dose: max I(cell) * step_length across all steps
         of all successful episodes.

Failed episodes are NEVER averaged into dose statistics (reported separately).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Result record for one evaluation episode."""

    start: Tuple[int, int]
    goal: Tuple[int, int]
    outcome: str                     # "goal" | "collision" | "timeout"
    path: List[Tuple[int, int]]      # waypoints including start
    total_dose: float                # cumulative absorbed dose (a.u.)
    peak_step_dose: float            # max single-step dose in this episode
    path_length: float               # actual geometric path length (cells)
    optimal_length: Optional[float]  # free-space A* length (None if unreachable)


@dataclass
class ParetoMetrics:
    """Aggregated research metrics over a set of evaluation episodes."""

    n_episodes: int
    n_success: int
    n_collision: int
    n_timeout: int

    # Computed only over SUCCESSFUL episodes
    mean_total_dose: float
    std_total_dose: float
    peak_instantaneous_dose: float    # max over all successful episodes
    mean_path_efficiency: float       # mean PER
    std_path_efficiency: float

    @property
    def success_rate_pct(self) -> float:
        """Success rate in percent."""
        return 100.0 * self.n_success / self.n_episodes if self.n_episodes > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"ParetoMetrics(\n"
            f"  n_episodes           = {self.n_episodes}\n"
            f"  success_rate         = {self.success_rate_pct:.1f}%\n"
            f"  mean_total_dose      = {self.mean_total_dose:.4f} a.u.\n"
            f"  peak_instant_dose    = {self.peak_instantaneous_dose:.4f} a.u.\n"
            f"  mean_path_efficiency = {self.mean_path_efficiency:.4f}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Free-space shortest path length via Dijkstra (obstacle-aware)
# ---------------------------------------------------------------------------

def _free_space_path_length(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: np.ndarray,
    actions: List[Tuple[int, int]],
    action_lengths: List[float],
) -> Optional[float]:
    """Return the obstacle-aware free-space shortest path length using Dijkstra.

    Returns None if the goal is unreachable from start.
    """
    h, w = walls.shape
    dist = np.full((h, w), np.inf, dtype=np.float64)
    dist[start] = 0.0
    pq: List[Tuple[float, int, int]] = [(0.0, start[0], start[1])]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        if (r, c) == goal:
            return float(d)
        for (dr, dc), length in zip(actions, action_lengths):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if walls[nr, nc]:
                continue
            if abs(dr) == 1 and abs(dc) == 1:
                if walls[r + dr, c] or walls[r, c + dc]:
                    continue
            nd = d + length
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(pq, (nd, nr, nc))

    return None  # unreachable


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_episode_result(
    path: List[Tuple[int, int]],
    outcome: str,
    intensity_field: np.ndarray,
    walls: np.ndarray,
    actions: List[Tuple[int, int]],
    action_lengths: List[float],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> EpisodeResult:
    """Compute per-episode metrics from a completed rollout path.

    Args:
        path:            Ordered list of (row, col) cells visited, start first.
        outcome:         "goal", "collision", or "timeout".
        intensity_field: Precomputed radiation intensity array.
        walls:           Boolean wall mask.
        actions:         List of (dr, dc) action deltas.
        action_lengths:  Corresponding step lengths.
        start:           Start cell.
        goal:            Goal cell.

    Returns:
        EpisodeResult with all metrics populated.
    """
    total_dose = 0.0
    peak_step_dose = 0.0
    actual_length = 0.0

    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        length = math.sqrt(dr * dr + dc * dc)
        step_d = float(intensity_field[path[i + 1]]) * length
        total_dose += step_d
        peak_step_dose = max(peak_step_dose, step_d)
        actual_length += length

    optimal_length = _free_space_path_length(
        start, goal, walls, actions, action_lengths
    )

    return EpisodeResult(
        start=start,
        goal=goal,
        outcome=outcome,
        path=path,
        total_dose=total_dose,
        peak_step_dose=peak_step_dose,
        path_length=actual_length,
        optimal_length=optimal_length,
    )


def aggregate_metrics(results: Sequence[EpisodeResult]) -> ParetoMetrics:
    """Aggregate per-episode results into research Pareto metrics.

    Failed episodes are excluded from dose and efficiency statistics.

    Args:
        results: Sequence of EpisodeResult objects.

    Returns:
        ParetoMetrics summary.
    """
    n = len(results)
    successes = [r for r in results if r.outcome == "goal"]
    n_succ = len(successes)
    n_coll = sum(1 for r in results if r.outcome == "collision")
    n_time = sum(1 for r in results if r.outcome == "timeout")

    if successes:
        doses = [r.total_dose for r in successes]
        mean_dose = float(np.mean(doses))
        std_dose = float(np.std(doses))
        peak_dose = max(r.peak_step_dose for r in successes)

        pers = []
        for r in successes:
            if r.optimal_length is not None and r.path_length > 0:
                pers.append(r.optimal_length / r.path_length)
        mean_per = float(np.mean(pers)) if pers else float("nan")
        std_per = float(np.std(pers)) if pers else float("nan")
    else:
        mean_dose = std_dose = peak_dose = float("nan")
        mean_per = std_per = float("nan")

    return ParetoMetrics(
        n_episodes=n,
        n_success=n_succ,
        n_collision=n_coll,
        n_timeout=n_time,
        mean_total_dose=mean_dose,
        std_total_dose=std_dose,
        peak_instantaneous_dose=peak_dose,
        mean_path_efficiency=mean_per,
        std_path_efficiency=std_per,
    )

