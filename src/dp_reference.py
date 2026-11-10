"""Exact Dynamic Programming / Dijkstra reference solver for radiation-aware grid navigation.

Provides an exact baseline reference without heuristics, obeying identical
8-connected grid movement and corner-cutting collision rules as the planners.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.environment import (
    ACTIONS,
    ACTION_LENGTHS,
    GOAL_CELL,
    GRID_SHAPE,
    RadiationEnv,
)


def dijkstra_search(
    start: Tuple[int, int],
    goal: Tuple[int, int] = GOAL_CELL,
    lambda_rad: float = 2.5,
    env: Optional[RadiationEnv] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], float, float, float, int]:
    """Exact Dijkstra shortest path search on the 8-connected grid.

    Args:
        start: Starting (row, col) cell.
        goal: Goal (row, col) cell.
        lambda_rad: Radiation weight multiplier (step cost = L * (1 + lambda_rad * I)).
        env: RadiationEnv instance (created if None).

    Returns:
        (path, total_cost, total_dose, total_steps, nodes_expanded)
        If no path is found, path is None and metrics are inf.
    """
    if env is None:
        env = RadiationEnv(start=start, goal=goal)

    if start == goal:
        return [start], 0.0, 0.0, 0.0, 0

    # Priority queue: (cost, count, r, c)
    open_set: List[Tuple[float, int, int, int]] = []
    count = 0
    heapq.heappush(open_set, (0.0, count, start[0], start[1]))

    cost_to_reach = np.full(env.shape, np.inf, dtype=np.float64)
    cost_to_reach[start] = 0.0

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed_set = np.zeros(env.shape, dtype=bool)
    nodes_expanded = 0

    while open_set:
        c_val, _, r, c = heapq.heappop(open_set)
        curr = (r, c)

        if closed_set[r, c]:
            continue
        closed_set[r, c] = True
        nodes_expanded += 1

        if curr == goal:
            # Reconstruct path
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()

            total_dose = 0.0
            total_cost = cost_to_reach[goal]
            for i in range(len(path) - 1):
                p_curr = path[i]
                p_next = path[i + 1]
                dr = p_next[0] - p_curr[0]
                dc = p_next[1] - p_curr[1]
                step_l = math.sqrt(dr * dr + dc * dc)
                total_dose += env.radiation[p_next] * step_l

            steps = float(len(path) - 1)
            return path, total_cost, total_dose, steps, nodes_expanded

        # Expand neighbours
        for a_idx, (dr, dc) in enumerate(ACTIONS):
            if env.is_collision(r, c, dr, dc):
                continue

            nr, nc = r + dr, c + dc
            if closed_set[nr, nc]:
                continue

            length = ACTION_LENGTHS[a_idx]
            step_cost = length * (1.0 + lambda_rad * env.radiation[nr, nc])

            tentative_cost = cost_to_reach[r, c] + step_cost
            if tentative_cost < cost_to_reach[nr, nc]:
                came_from[(nr, nc)] = (r, c)
                cost_to_reach[nr, nc] = tentative_cost
                count += 1
                heapq.heappush(open_set, (tentative_cost, count, nr, nc))

    return None, float("inf"), float("inf"), float("inf"), nodes_expanded

