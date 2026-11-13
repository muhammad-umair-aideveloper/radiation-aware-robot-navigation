"""Sensor-based navigation environment with LiDAR, Geiger perception, and potential-based shaping."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import (
    RadiationEnv,
    ACTIONS,
    ACTION_LENGTHS,
    START_CELL,
    GOAL_CELL,
)
from src.sensors import get_sensor_observation, OBS_DIM
from main import compute_free_space_distances


class SensorNavigationEnv:
    """2D navigation environment using egocentric sensor observations and relative goal guidance."""

    def __init__(
        self,
        base_env: Optional[RadiationEnv] = None,
        lambda_rad: float = 2.5,
        shaping: bool = True,
        gamma: float = 0.99,
        max_steps: int = 300,
        scale_reward: bool = True,
    ) -> None:
        self.base_env = RadiationEnv() if base_env is None else base_env
        self.lambda_rad = float(lambda_rad)
        self.shaping = bool(shaping)
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self.scale_reward = bool(scale_reward)

        self.robot_pos: Tuple[int, int] = self.base_env.start
        self.goal_pos: Tuple[int, int] = self.base_env.goal
        self.steps: int = 0
        self.cumulative_dose: float = 0.0
        self.done: bool = False
        self.outcome: str = "init"

        # Global cache for free-space distance maps (keyed by goal cell)
        self._distance_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get_distance_map(self, goal: Tuple[int, int]) -> np.ndarray:
        """Retrieve or compute the 8-connected free-space geodesic distance map to goal."""
        if goal not in self._distance_cache:
            self._distance_cache[goal] = compute_free_space_distances(self.base_env, goal)
        return self._distance_cache[goal]

    def _potential(self, pos: Tuple[int, int], is_terminal: bool = False) -> float:
        """Potential function Phi(s) = -d_free(s) / 100.0, with Phi(terminal) = 0.0."""
        if is_terminal:
            return 0.0
        dist_map = self.get_distance_map(self.goal_pos)
        d_val = float(dist_map[pos[0], pos[1]])
        if math.isinf(d_val):
            d_val = 100.0
        return -d_val / 100.0

    def reset(
        self,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Reset environment to specified or random start and goal."""
        self.robot_pos = self.base_env.start if start is None else (int(start[0]), int(start[1]))
        self.goal_pos = self.base_env.goal if goal is None else (int(goal[0]), int(goal[1]))
        self.steps = 0
        self.cumulative_dose = 0.0
        self.done = False
        self.outcome = "in_progress"

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return 28-dimensional egocentric sensor observation."""
        return get_sensor_observation(
            robot_pos=self.robot_pos,
            goal_pos=self.goal_pos,
            walls=self.base_env.walls,
            radiation=self.base_env.radiation,
            shape=self.base_env.shape,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one 8-connected discrete movement step."""
        if self.done:
            raise RuntimeError("Cannot step in a terminated environment. Call reset() first.")

        self.steps += 1
        dr, dc = ACTIONS[action]
        step_len = ACTION_LENGTHS[action]

        curr_pos = self.robot_pos
        is_coll = self.base_env.is_collision(curr_pos[0], curr_pos[1], dr, dc)

        phi_curr = self._potential(curr_pos, is_terminal=False)

        if is_coll:
            self.done = True
            self.outcome = "collision"
            raw_reward = -1000.0
            step_dose = 0.0
            phi_next = self._potential(curr_pos, is_terminal=True)  # Terminal -> 0.0
        else:
            next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
            self.robot_pos = next_pos
            step_rad = float(self.base_env.radiation[next_pos[0], next_pos[1]])
            step_dose = step_rad * step_len
            self.cumulative_dose += step_dose

            if next_pos == self.goal_pos:
                self.done = True
                self.outcome = "goal"
                raw_reward = 1000.0
                phi_next = self._potential(next_pos, is_terminal=True)  # Terminal -> 0.0
            elif self.steps >= self.max_steps:
                self.done = True
                self.outcome = "timeout"
                raw_reward = -step_len * (1.0 + self.lambda_rad * step_rad)
                phi_next = self._potential(next_pos, is_terminal=True)  # Terminal -> 0.0
            else:
                raw_reward = -step_len * (1.0 + self.lambda_rad * step_rad)
                phi_next = self._potential(next_pos, is_terminal=False)

        # Potential-based reward shaping: F = gamma * Phi(s') - Phi(s)
        f_shaping = (self.gamma * phi_next - phi_curr) if self.shaping else 0.0

        # Reward for network training: divided by 100
        scaled_base = raw_reward / 100.0
        total_training_reward = scaled_base + f_shaping

        returned_reward = total_training_reward if self.scale_reward else (raw_reward + f_shaping * 100.0)

        info: Dict[str, Any] = {
            "outcome": self.outcome,
            "steps": self.steps,
            "step_dose": step_dose,
            "cumulative_dose": self.cumulative_dose,
            "raw_reward": raw_reward,
            "shaping": f_shaping,
            "phi_curr": phi_curr,
            "phi_next": phi_next,
            "training_reward": total_training_reward,
            "robot_pos": list(self.robot_pos),
            "goal_pos": list(self.goal_pos),
        }

        obs = self._get_obs()
        return obs, float(returned_reward), self.done, info
