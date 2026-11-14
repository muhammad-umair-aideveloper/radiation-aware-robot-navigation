"""Unit tests for SensorNavigationEnv: observations, collision, goal, timeout, and potential shaping."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, ACTIONS, START_CELL, GOAL_CELL
from src.sensor_env import SensorNavigationEnv, FrameStack
from src.sensors import OBS_DIM


def test_observation_shape_and_ranges() -> None:
    """Verify observation dimension (28), data types, and numerical ranges."""
    env = SensorNavigationEnv()
    obs = env.reset(start=(10, 10), goal=(40, 40))

    assert obs.shape == (OBS_DIM,), f"Expected shape ({OBS_DIM},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    # LiDAR distances in [0, 1]
    lidar = obs[:16]
    assert np.all(lidar >= 0.0) and np.all(lidar <= 1.0), "LiDAR values must be in [0, 1]"

    # Geiger intensities >= 0
    geiger = obs[16:25]
    assert np.all(geiger >= 0.0), "Geiger readings must be non-negative"

    # Goal vector in [-1, 1]
    goal_vec = obs[25:27]
    assert np.all(goal_vec >= -1.0) and np.all(goal_vec <= 1.0), "Goal vector must be in [-1, 1]"

    # Goal distance in [0, 1]
    goal_dist = obs[27]
    assert 0.0 <= goal_dist <= 1.0, "Goal distance must be in [0, 1]"

    print("PASS: test_observation_shape_and_ranges")


def test_collision_terminates_and_penalizes() -> None:
    """Verify that hitting a wall or moving out of grid bounds yields -1000 and terminates."""
    env = SensorNavigationEnv()

    # 1. Collision with obstacle wall
    # Canonical wall rect: (22, 27, 18, 34). Cell (22, 20) is a wall.
    # Place robot at (21, 20). Action 4 (South: dr=1, dc=0) moves into (22, 20).
    env.reset(start=(21, 20), goal=(47, 47))
    obs, reward, done, info = env.step(4)

    assert done is True, "Collision must terminate the episode"
    assert info["outcome"] == "collision", f"Expected collision outcome, got {info['outcome']}"
    assert info["raw_reward"] == -1000.0, f"Expected raw reward -1000, got {info['raw_reward']}"

    # 2. Collision with outer grid boundary
    env.reset(start=(0, 10), goal=(47, 47))
    obs, reward, done, info = env.step(0)  # North out of bounds (r = -1)

    assert done is True, "Border collision must terminate the episode"
    assert info["outcome"] == "collision", f"Expected collision outcome, got {info['outcome']}"
    assert info["raw_reward"] == -1000.0

    print("PASS: test_collision_terminates_and_penalizes")


def test_goal_terminates_and_rewards() -> None:
    """Verify that reaching the target goal yields +1000 and terminates."""
    env = SensorNavigationEnv()
    target_goal = (10, 15)

    # Place robot at adjacent cell (10, 14). Action 2 (East: dr=0, dc=1) moves into goal.
    env.reset(start=(10, 14), goal=target_goal)
    obs, reward, done, info = env.step(2)

    assert done is True, "Reaching goal must terminate the episode"
    assert info["outcome"] == "goal", f"Expected goal outcome, got {info['outcome']}"
    assert info["raw_reward"] == 1000.0, f"Expected raw reward +1000, got {info['raw_reward']}"

    print("PASS: test_goal_terminates_and_rewards")


def test_timeout_after_300_steps() -> None:
    """Verify that an episode strictly terminates on step 300 with timeout outcome."""
    env = SensorNavigationEnv(max_steps=300)
    # Oscillate between adjacent free cells (2, 2) and (2, 3)
    env.reset(start=(2, 2), goal=(47, 47))

    for step_num in range(1, 300):
        # East on odd, West on even
        act = 2 if step_num % 2 == 1 else 6
        obs, reward, done, info = env.step(act)
        assert done is False, f"Episode terminated prematurely at step {step_num}"

    # Step 300: must terminate on timeout
    act = 2
    obs, reward, done, info = env.step(act)
    assert done is True, "Episode must terminate on step 300"
    assert info["outcome"] == "timeout", f"Expected timeout outcome, got {info['outcome']}"
    assert env.steps == 300

    print("PASS: test_timeout_after_300_steps")


def test_shaping_terms_telescope_along_trajectory() -> None:
    """Verify potential-based shaping terms telescope to -Phi(s0) = d_free(s0) / 100.0."""
    gamma = 0.99
    env = SensorNavigationEnv(shaping=True, gamma=gamma)
    start = (10, 10)
    goal = (10, 15)

    env.reset(start=start, goal=goal)
    phi_0 = env._potential(start, is_terminal=False)

    shaping_terms = []
    # Walk 5 steps East to goal: (10, 11), (10, 12), (10, 13), (10, 14), (10, 15)
    for _ in range(5):
        obs, reward, done, info = env.step(2)
        shaping_terms.append(info["shaping"])
        if done:
            break

    assert done is True and info["outcome"] == "goal"

    discounted_sum = sum((gamma ** t) * F for t, F in enumerate(shaping_terms))
    expected_sum = -phi_0  # Because Phi(terminal) = 0, sum_{t=0}^{T-1} gamma^t F_t = -Phi(s0)

    assert math.isclose(discounted_sum, expected_sum, abs_tol=1e-6), (
        f"Shaping sum {discounted_sum} != expected {expected_sum}"
    )

    # Also verify on a collision trajectory
    env.reset(start=(21, 20), goal=(47, 47))
    phi_0_coll = env._potential((21, 20), is_terminal=False)
    obs, reward, done, info = env.step(4)  # Hits wall on step 0
    assert done is True and info["outcome"] == "collision"
    assert math.isclose(info["shaping"], -phi_0_coll, abs_tol=1e-6)

    print("PASS: test_shaping_terms_telescope_along_trajectory")


def test_same_seed_reproducibility() -> None:
    """Verify that identical starting parameters and actions produce identical trajectory states."""
    env1 = SensorNavigationEnv()
    env2 = SensorNavigationEnv()

    start = (5, 5)
    goal = (45, 45)
    actions = [2, 3, 3, 4, 3, 2, 4]

    obs1 = env1.reset(start=start, goal=goal)
    obs2 = env2.reset(start=start, goal=goal)
    assert np.allclose(obs1, obs2)

    for a in actions:
        o1, r1, d1, i1 = env1.step(a)
        o2, r2, d2, i2 = env2.step(a)
        assert np.allclose(o1, o2), "Observations diverged"
        assert math.isclose(r1, r2), "Rewards diverged"
        assert d1 == d2, "Done flag diverged"
        assert i1["outcome"] == i2["outcome"]
        assert math.isclose(i1["step_dose"], i2["step_dose"])

    print("PASS: test_same_seed_reproducibility")


def test_frame_stacking_shape_and_mechanics() -> None:
    """Verify that 4-frame stacking produces 112 features, repeats on reset, and drops oldest frame."""
    env = SensorNavigationEnv(num_stack=4)
    obs = env.reset(start=(10, 10), goal=(40, 40))

    # 1. Shape must be 112
    assert obs.shape == (112,), f"Expected shape (112,), got {obs.shape}"

    # 2. First frame repeated 4 times at reset
    f0 = obs[0:28]
    f1 = obs[28:56]
    f2 = obs[56:84]
    f3 = obs[84:112]
    assert np.allclose(f0, f1) and np.allclose(f1, f2) and np.allclose(f2, f3), "Frames must be identical at reset"

    # 3. Oldest frame dropped after step
    # Action 2 is East (0, 1) -> moves to (10, 11)
    next_obs, r, done, info = env.step(2)
    assert next_obs.shape == (112,)

    # The new next_obs frames 0, 1, 2 should equal the previous obs frames 1, 2, 3
    assert np.allclose(next_obs[0:84], obs[28:112]), "Oldest frame must be dropped and prior frames shifted"
    # The newest frame (84:112) must differ from the initial frame (position changed from (10,10) to (10,11))
    assert not np.allclose(next_obs[84:112], f0), "New frame must reflect updated agent state"

    # 4. Also test standalone FrameStack wrapper
    base_env = SensorNavigationEnv(num_stack=1)
    wrapper = FrameStack(base_env, k=4)
    w_obs = wrapper.reset(start=(10, 10), goal=(40, 40))
    assert w_obs.shape == (112,)
    assert np.allclose(w_obs[:28], w_obs[28:56])

    w_next, _, _, _ = wrapper.step(2)
    assert w_next.shape == (112,)
    assert np.allclose(w_next[0:84], w_obs[28:112])

    print("PASS: test_frame_stacking_shape_and_mechanics")


def test_frame_stacking_reproducibility() -> None:
    """Verify that running with identical actions produces identical stacked observations and rewards."""
    env1 = FrameStack(SensorNavigationEnv(num_stack=1), k=4)
    env2 = FrameStack(SensorNavigationEnv(num_stack=1), k=4)

    start = (15, 15)
    goal = (35, 35)
    actions = [2, 3, 2, 4, 3, 1, 0]

    obs1 = env1.reset(start=start, goal=goal)
    obs2 = env2.reset(start=start, goal=goal)
    assert np.allclose(obs1, obs2)

    for a in actions:
        o1, r1, d1, _ = env1.step(a)
        o2, r2, d2, _ = env2.step(a)
        assert np.allclose(o1, o2)
        assert math.isclose(r1, r2)
        assert d1 == d2

    print("PASS: test_frame_stacking_reproducibility")


def run_all_tests() -> None:
    print("Running SensorNavigationEnv unit tests...")
    test_observation_shape_and_ranges()
    test_collision_terminates_and_penalizes()
    test_goal_terminates_and_rewards()
    test_timeout_after_300_steps()
    test_shaping_terms_telescope_along_trajectory()
    test_same_seed_reproducibility()
    test_frame_stacking_shape_and_mechanics()
    test_frame_stacking_reproducibility()
    print("All sensor environment tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()

