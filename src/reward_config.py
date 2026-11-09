"""Multi-objective reward shaping configuration.

Formal reward equation (non-terminal step):

    R_t = -alpha * L - lambda_rad * I(r_t) * L - 0.0 * I_collision

Terminal overrides:
    goal      -> R = R_goal       (large positive)
    collision -> R = R_collision  (large negative)

where:
    alpha       -- step-length penalty weight (discourages long paths)
    lambda_rad  -- radiation penalty weight (discourages high-dose cells)
    L           -- step length (1 for cardinal, sqrt(2) for diagonal)
    I(r_t)      -- radiation intensity at the new cell
    R_goal      -- terminal goal bonus
    R_collision -- terminal collision penalty

This module purposely contains NO training logic; it is a typed, immutable
configuration container that can be passed to the environment and planners.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Typed, immutable reward weight configuration.

    All weights are declared here and must NOT be hard-coded elsewhere.
    Change these values with a dedicated commit and document the reason.

    Attributes:
        R_goal:       Reward received when the robot reaches the goal cell.
        R_collision:  Reward (penalty) received on a wall or boundary collision.
        alpha:        Per-unit-length step cost multiplier (path-length penalty).
        lambda_rad:   Per-unit-length radiation dose penalty multiplier.
                      Set to 0.0 for the radiation-blind baseline agent.

    Reward equation (non-terminal step):
        R_t = -alpha * L  -  lambda_rad * I(r_t) * L

    Reward equation (terminal):
        R_t = R_goal          if outcome == "goal"
        R_t = R_collision     if outcome == "collision"
    """

    R_goal: float = 1000.0
    R_collision: float = -1000.0
    alpha: float = 1.0          # step penalty weight
    lambda_rad: float = 2.5     # radiation penalty weight (matches planners.py)

    def step_reward(self, step_length: float, intensity: float) -> float:
        """Non-terminal per-step reward.

        Args:
            step_length: Euclidean length of the action (1 or sqrt(2)).
            intensity:   Radiation field value I(r_new) at the destination cell.

        Returns:
            R_t = -(alpha + lambda_rad * I) * L
        """
        return -(self.alpha + self.lambda_rad * intensity) * step_length

    def goal_reward(self) -> float:
        """Terminal goal reward."""
        return self.R_goal

    def collision_reward(self) -> float:
        """Terminal collision penalty."""
        return self.R_collision


# ---------------------------------------------------------------------------
# Pre-defined configurations used in the project
# ---------------------------------------------------------------------------

#: Radiation-aware agent (lambda_rad = 2.5 as used in planners.py A* aware).
AWARE_CONFIG = RewardConfig(
    R_goal=1000.0,
    R_collision=-1000.0,
    alpha=1.0,
    lambda_rad=2.5,
)

#: Radiation-blind baseline agent (lambda_rad = 0).
BLIND_CONFIG = RewardConfig(
    R_goal=1000.0,
    R_collision=-1000.0,
    alpha=1.0,
    lambda_rad=0.0,
)
