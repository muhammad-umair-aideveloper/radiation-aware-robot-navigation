"""Unit tests for DQNAgent: replay buffer sampling, epsilon decay schedule, and target network synchronization."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dqn_agent import ReplayBuffer, compute_epsilon, DQNAgent


def test_replay_buffer_operations() -> None:
    """Verify replay buffer capacity, circular overwrite, and batch sampling."""
    capacity = 10
    obs_dim = 4
    buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim)

    assert len(buf) == 0

    # Add 5 items
    for i in range(5):
        obs = np.full(obs_dim, float(i), dtype=np.float32)
        buf.add(obs, action=i % 8, reward=float(i * 10), next_obs=obs + 1.0, done=(i == 4))

    assert len(buf) == 5

    # Sample batch
    rng = np.random.RandomState(42)
    b_obs, b_act, b_rew, b_next_obs, b_done = buf.sample(batch_size=4, rng=rng)
    assert b_obs.shape == (4, obs_dim)
    assert b_act.shape == (4,)
    assert b_rew.shape == (4,)
    assert b_next_obs.shape == (4, obs_dim)
    assert b_done.shape == (4,)

    # Add 10 more items to trigger circular overwrite
    for i in range(10):
        obs = np.full(obs_dim, float(100 + i), dtype=np.float32)
        buf.add(obs, action=0, reward=1.0, next_obs=obs, done=False)

    assert len(buf) == capacity, f"Buffer size {len(buf)} must be capped at {capacity}"
    assert buf.ptr == (5 + 10) % capacity  # 15 % 10 = 5

    print("PASS: test_replay_buffer_operations")


def test_epsilon_schedule() -> None:
    """Verify geometric epsilon decay schedule and clamping after decay fraction."""
    total_steps = 10000
    decay_fraction = 0.6
    decay_steps = int(total_steps * decay_fraction)  # 6000

    # Step 0: start_eps = 1.0
    eps_0 = compute_epsilon(0, total_steps, start_eps=1.0, end_eps=0.05, decay_fraction=0.6)
    assert math.isclose(eps_0, 1.0, abs_tol=1e-5), f"Expected 1.0, got {eps_0}"

    # Intermediate steps: strictly decreasing
    prev_eps = eps_0
    for s in [500, 1000, 2000, 4000, 5999]:
        eps_s = compute_epsilon(s, total_steps, start_eps=1.0, end_eps=0.05, decay_fraction=0.6)
        assert eps_s < prev_eps, f"Epsilon not strictly decreasing at step {s}: {eps_s} >= {prev_eps}"
        assert 0.05 <= eps_s <= 1.0
        prev_eps = eps_s

    # At decay_steps: end_eps = 0.05
    eps_end = compute_epsilon(decay_steps, total_steps, start_eps=1.0, end_eps=0.05, decay_fraction=0.6)
    assert math.isclose(eps_end, 0.05, abs_tol=1e-5), f"Expected 0.05, got {eps_end}"

    # Beyond decay_steps: strictly clamped at 0.05
    for s in [decay_steps + 1, decay_steps + 1000, total_steps, total_steps * 2]:
        eps_post = compute_epsilon(s, total_steps, start_eps=1.0, end_eps=0.05, decay_fraction=0.6)
        assert math.isclose(eps_post, 0.05, abs_tol=1e-5), f"Expected 0.05 clamp, got {eps_post}"

    print("PASS: test_epsilon_schedule")


def test_target_network_sync() -> None:
    """Verify target network remains static until explicitly copied."""
    agent = DQNAgent(obs_dim=10, num_actions=4, hidden_sizes=(16, 16), seed=123)

    # Initial check: online and target have identical weights
    for w_on, w_tg in zip(agent.online_net.weights, agent.target_net.weights):
        assert np.array_equal(w_on, w_tg)

    # Modify online network weights by training
    rng = np.random.RandomState(42)
    for _ in range(100):
        o = rng.randn(10).astype(np.float32)
        no = rng.randn(10).astype(np.float32)
        agent.buffer.add(o, 0, 1.0, no, False)

    for _ in range(5):
        agent.update()

    # Online and target should now diverge
    weights_diverged = False
    for w_on, w_tg in zip(agent.online_net.weights, agent.target_net.weights):
        if not np.allclose(w_on, w_tg):
            weights_diverged = True
            break
    assert weights_diverged is True, "Target network should not update automatically before sync"

    # Explicit sync
    agent.target_net.copy_from(agent.online_net)

    # Online and target should now match exactly
    for w_on, w_tg in zip(agent.online_net.weights, agent.target_net.weights):
        assert np.array_equal(w_on, w_tg), "Target network failed to sync with online network"

    print("PASS: test_target_network_sync")


def run_all_tests() -> None:
    print("Running DQNAgent unit tests...")
    test_replay_buffer_operations()
    test_epsilon_schedule()
    test_target_network_sync()
    print("All DQNAgent tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()

