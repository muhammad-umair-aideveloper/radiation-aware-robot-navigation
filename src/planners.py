"""A* search planners and Tabular Q-Learning agent for radiation-aware navigation."""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.environment import (
    ACTIONS,
    ACTION_LENGTHS,
    GOAL_CELL,
    GRID_SHAPE,
    START_CELL,
    RadiationEnv,
    compute_radiation_field,
    create_wall_mask,
)


def octile_distance(r1: int, c1: int, r2: int, c2: int) -> float:
    """Admissible octile distance heuristic on an 8-connected grid."""
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    return abs(dr - dc) * 1.0 + min(dr, dc) * math.sqrt(2.0)


def astar_search(
    start: Tuple[int, int],
    goal: Tuple[int, int] = GOAL_CELL,
    mode: str = "shortest",
    env: Optional[RadiationEnv] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], float, float, int]:
    """A* search on the 8-connected grid.

    Args:
        start: Starting (row, col) cell.
        goal: Goal (row, col) cell.
        mode: "shortest" (cost = L) or "aware" (cost = L * (1 + 2.5 * I(new))).
        env: RadiationEnv instance (created if None).

    Returns:
        (path, total_dose, total_steps, nodes_expanded)
        If no path found, path is None and metrics are inf.
    """
    if env is None:
        env = RadiationEnv(start=start, goal=goal)

    if start == goal:
        return [start], 0.0, 0.0, 0

    # Priority queue: (f_score, count, r, c)
    open_set: List[Tuple[float, int, int, int]] = []
    count = 0
    h_start = octile_distance(start[0], start[1], goal[0], goal[1])
    heapq.heappush(open_set, (h_start, count, start[0], start[1]))

    g_score = np.full(env.shape, np.inf, dtype=np.float64)
    g_score[start] = 0.0

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed_set = np.zeros(env.shape, dtype=bool)
    nodes_expanded = 0

    while open_set:
        f, _, r, c = heapq.heappop(open_set)
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

            # Compute total dose and step count
            total_dose = 0.0
            for i in range(len(path) - 1):
                p_curr = path[i]
                p_next = path[i + 1]
                dr = p_next[0] - p_curr[0]
                dc = p_next[1] - p_curr[1]
                step_l = math.sqrt(dr * dr + dc * dc)
                total_dose += env.radiation[p_next] * step_l

            steps = float(len(path) - 1)
            return path, total_dose, steps, nodes_expanded

        # Expand neighbours
        for a_idx, (dr, dc) in enumerate(ACTIONS):
            if env.is_collision(r, c, dr, dc):
                continue

            nr, nc = r + dr, c + dc
            if closed_set[nr, nc]:
                continue

            length = ACTION_LENGTHS[a_idx]
            if mode == "shortest":
                step_cost = length
            elif mode == "aware":
                step_cost = length * (1.0 + 2.5 * env.radiation[nr, nc])
            else:
                raise ValueError(f"Unknown mode: {mode}")

            tentative_g = g_score[r, c] + step_cost
            if tentative_g < g_score[nr, nc]:
                came_from[(nr, nc)] = (r, c)
                g_score[nr, nc] = tentative_g
                h = octile_distance(nr, nc, goal[0], goal[1])
                count += 1
                heapq.heappush(open_set, (tentative_g + h, count, nr, nc))

    return None, float("inf"), float("inf"), nodes_expanded


class QLearningAgent:
    """Tabular Q-Learning agent with precomputed transition tables for fast training."""

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.98,
        eps_start: float = 1.0,
        eps_min: float = 0.02,
        episodes: int = 50000,
        max_steps: int = 300,
        env: Optional[RadiationEnv] = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.episodes = episodes
        self.max_steps = max_steps
        self.env = env if env is not None else RadiationEnv()

        # Precompute transitions to maximize training speed
        self._build_transition_tables()

        # Valid free cells for random starts
        self.free_cells = [
            (int(r), int(c))
            for r, c in zip(*np.where(~self.env.walls))
            if (r, c) != self.env.goal
        ]

    def _build_transition_tables(self) -> None:
        """Precompute deterministic transitions for all (row, col, action) tuples."""
        h, w = self.env.shape
        num_act = len(ACTIONS)

        self.next_r = np.zeros((h, w, num_act), dtype=np.int32)
        self.next_c = np.zeros((h, w, num_act), dtype=np.int32)
        self.rewards = np.zeros((h, w, num_act), dtype=np.float32)
        self.terminals = np.zeros((h, w, num_act), dtype=bool)
        self.doses = np.zeros((h, w, num_act), dtype=np.float32)

        for r in range(h):
            for c in range(w):
                if self.env.walls[r, c]:
                    continue
                for a, (dr, dc) in enumerate(ACTIONS):
                    length = ACTION_LENGTHS[a]
                    if self.env.is_collision(r, c, dr, dc):
                        self.next_r[r, c, a] = r
                        self.next_c[r, c, a] = c
                        self.rewards[r, c, a] = -1000.0
                        self.terminals[r, c, a] = True
                        self.doses[r, c, a] = 0.0
                    else:
                        nr, nc = r + dr, c + dc
                        self.next_r[r, c, a] = nr
                        self.next_c[r, c, a] = nc
                        step_dose = float(self.env.radiation[nr, nc] * length)
                        self.doses[r, c, a] = step_dose
                        if (nr, nc) == self.env.goal:
                            self.rewards[r, c, a] = 1000.0
                            self.terminals[r, c, a] = True
                        else:
                            self.rewards[r, c, a] = -length * (
                                1.0 + 2.5 * self.env.radiation[nr, nc]
                            )
                            self.terminals[r, c, a] = False

    def train(self, seed: int) -> Tuple[np.ndarray, List[float], List[float], float]:
        """Train Q-learning table for self.episodes.

        Returns:
            (Q_table, episode_rewards, episode_doses, elapsed_time_seconds)
        """
        rng = np.random.RandomState(seed)
        Q = np.zeros((*self.env.shape, len(ACTIONS)), dtype=np.float32)

        # Geometric decay factor for epsilon
        eps_decay = (self.eps_min / self.eps_start) ** (1.0 / max(1, self.episodes - 1))
        epsilon = self.eps_start

        episode_rewards: List[float] = []
        episode_doses: List[float] = []

        start_time = time.time()
        num_free = len(self.free_cells)

        for ep in range(self.episodes):
            # 50% start at (2,2), 50% at a random free cell
            if rng.rand() < 0.5:
                r, c = START_CELL
            else:
                idx = rng.randint(0, num_free)
                r, c = self.free_cells[idx]

            ep_reward = 0.0
            ep_dose = 0.0

            for _ in range(self.max_steps):
                # Epsilon-greedy action selection
                if rng.rand() < epsilon:
                    a = rng.randint(0, 8)
                else:
                    # Break ties randomly for unvisited states
                    q_vals = Q[r, c]
                    max_q = np.max(q_vals)
                    best_actions = np.where(q_vals == max_q)[0]
                    a = int(best_actions[rng.randint(len(best_actions))])

                nr = int(self.next_r[r, c, a])
                nc = int(self.next_c[r, c, a])
                rew = float(self.rewards[r, c, a])
                term = bool(self.terminals[r, c, a])
                dose = float(self.doses[r, c, a])

                ep_reward += rew
                ep_dose += dose

                if term:
                    target = rew
                else:
                    target = rew + self.gamma * float(np.max(Q[nr, nc]))

                Q[r, c, a] += self.alpha * (target - Q[r, c, a])

                if term:
                    break

                r, c = nr, nc

            episode_rewards.append(ep_reward)
            episode_doses.append(ep_dose)
            epsilon *= eps_decay

        elapsed = time.time() - start_time
        return Q, episode_rewards, episode_doses, elapsed

    def rollout(
        self, Q: np.ndarray, start: Tuple[int, int] = START_CELL
    ) -> Tuple[List[Tuple[int, int]], str, float, int]:
        """Greedy evaluation rollout from start with a 300-step loop guard.

        Returns:
            (path, outcome, total_dose, step_count)
            outcome in ("goal", "collision", "timeout")
        """
        r, c = start
        path = [(r, c)]
        total_dose = 0.0
        visited = { (r, c) }

        for step in range(self.max_steps):
            if (r, c) == self.env.goal:
                return path, "goal", total_dose, step

            a = int(np.argmax(Q[r, c]))
            dr, dc = ACTIONS[a]
            length = ACTION_LENGTHS[a]

            if self.env.is_collision(r, c, dr, dc):
                return path, "collision", total_dose, step

            nr, nc = r + dr, c + dc
            path.append((nr, nc))
            total_dose += self.env.radiation[nr, nc] * length

            if (nr, nc) == self.env.goal:
                return path, "goal", total_dose, step + 1

            r, nc_curr = nr, nc
            c = nc_curr

        return path, "timeout", total_dose, self.max_steps
