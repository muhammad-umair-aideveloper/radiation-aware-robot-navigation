"""NumPy Deep Q-Network (DQN) agent with experience replay, target network, and epsilon schedule."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.nn import MLP, AdamOptimizer, compute_huber_loss_and_grad


class ReplayBuffer:
    """Preallocated circular FIFO replay buffer for efficient transition sampling."""

    def __init__(self, capacity: int = 50000, obs_dim: int = 28) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.obs_buf = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.action_buf = np.empty(self.capacity, dtype=np.int64)
        self.reward_buf = np.empty(self.capacity, dtype=np.float32)
        self.done_buf = np.empty(self.capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single environment transition."""
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = int(action)
        self.reward_buf[self.ptr] = float(reward)
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a uniform random batch of transitions."""
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer with size {self.size}")
        indices = rng.randint(0, self.size, size=batch_size)
        return (
            self.obs_buf[indices],
            self.action_buf[indices],
            self.reward_buf[indices],
            self.next_obs_buf[indices],
            self.done_buf[indices],
        )

    def __len__(self) -> int:
        return self.size


def compute_epsilon(
    step: int,
    total_steps: int,
    start_eps: float = 1.0,
    end_eps: float = 0.05,
    decay_fraction: float = 0.6,
) -> float:
    """Compute geometric epsilon decay over the first decay_fraction of total steps."""
    decay_steps = int(total_steps * decay_fraction)
    if decay_steps <= 0 or step >= decay_steps:
        return float(end_eps)
    if step <= 0:
        return float(start_eps)

    # Geometric decay: eps(t) = end_eps * (start_eps / end_eps) ** (1 - t / decay_steps)
    ratio = start_eps / end_eps
    progress = 1.0 - (float(step) / float(decay_steps))
    return float(end_eps * (ratio ** progress))


class DQNAgent:
    """DQN Agent using NumPy MLP, target network, and Huber loss."""

    def __init__(
        self,
        obs_dim: int = 28,
        num_actions: int = 8,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        update_frequency: int = 4,
        target_update_frequency: int = 1000,
        huber_delta: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.update_freq = int(update_frequency)
        self.target_update_freq = int(target_update_frequency)
        self.huber_delta = float(huber_delta)
        self.seed = int(seed)

        self.rng = np.random.RandomState(seed)

        # Networks
        layer_sizes = (obs_dim,) + tuple(hidden_sizes) + (num_actions,)
        self.online_net = MLP(layer_sizes=layer_sizes, seed=seed, dtype=np.float32)
        self.target_net = MLP(layer_sizes=layer_sizes, seed=seed + 1, dtype=np.float32)
        self.target_net.copy_from(self.online_net)

        # Optimizer
        self.optimizer = AdamOptimizer(self.online_net.get_params(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim)

        self.total_steps: int = 0
        self.update_steps: int = 0

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action via epsilon-greedy policy."""
        if epsilon > 0.0 and self.rng.rand() < epsilon:
            return int(self.rng.randint(0, self.num_actions))

        q_values = self.online_net.forward(obs)
        return int(np.argmax(q_values))

    def step_and_update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition in replay buffer and perform gradient update if scheduled."""
        self.total_steps += 1
        self.buffer.add(obs, action, reward, next_obs, done)

        # Target network sync
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.copy_from(self.online_net)

        # Gradient update step
        if self.total_steps % self.update_freq == 0 and len(self.buffer) >= self.batch_size:
            return self.update()

        return None

    def update(self) -> float:
        """Perform one batch DQN parameter update."""
        obs_b, act_b, rew_b, next_obs_b, done_b = self.buffer.sample(self.batch_size, self.rng)

        # Target values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        next_q_target = self.target_net.forward(next_obs_b)
        max_next_q = np.max(next_q_target, axis=1)
        targets = rew_b + (1.0 - done_b.astype(np.float32)) * self.gamma * max_next_q

        # Online network predictions
        q_pred = self.online_net.forward(obs_b)

        # Huber loss and analytical gradients
        loss, grad_out = compute_huber_loss_and_grad(q_pred, act_b, targets, delta=self.huber_delta)

        grad_w, grad_b = self.online_net.backward(grad_out)
        all_grads: List[np.ndarray] = []
        for w, b in zip(grad_w, grad_b):
            all_grads.extend([w, b])

        self.optimizer.step(all_grads)
        self.update_steps += 1
        return loss

    def save(self, filepath: Path | str) -> None:
        """Save online network parameters."""
        self.online_net.save(filepath)

    def load(self, filepath: Path | str) -> None:
        """Load online network parameters and synchronize target network."""
        self.online_net.load(filepath)
        self.target_net.copy_from(self.online_net)
