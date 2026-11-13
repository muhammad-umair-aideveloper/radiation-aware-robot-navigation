"""NumPy-only neural network implementation: Multi-Layer Perceptron (MLP) with Adam optimizer and Huber loss."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class AdamOptimizer:
    """Adam optimizer for NumPy parameter arrays."""

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Perform one Adam optimization step given parameter gradients."""
        self.t += 1
        b1_t = 1.0 - (self.beta1 ** self.t)
        b2_t = 1.0 - (self.beta2 ** self.t)

        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[i] / b1_t
            v_hat = self.v[i] / b2_t

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        """Reset momentum and step counters."""
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]


class MLP:
    """Multi-Layer Perceptron with ReLU activations and linear output."""

    def __init__(
        self,
        layer_sizes: Sequence[int] = (28, 64, 64, 8),
        seed: Optional[int] = 42,
        dtype: type = np.float32,
    ) -> None:
        self.layer_sizes = list(layer_sizes)
        self.dtype = dtype
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        rng = np.random.RandomState(seed)

        # He / Kaiming normal initialization
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            scale = np.sqrt(2.0 / n_in)
            W = (rng.randn(n_in, n_out) * scale).astype(self.dtype)
            b = np.zeros(n_out, dtype=self.dtype)
            self.weights.append(W)
            self.biases.append(b)

        # Cache for forward pass activations
        self._inputs: List[np.ndarray] = []
        self._zs: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the MLP.
        
        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,)
        Returns:
            Output array of shape (batch_size, output_dim)
        """
        x_arr = np.asarray(x, dtype=self.dtype)
        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        self._inputs = [x_arr]
        self._zs = []
        out = x_arr

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = out @ self.weights[i] + self.biases[i]
            self._zs.append(z)
            out = np.maximum(0.0, z)
            self._inputs.append(out)

        # Output layer with linear activation
        z_out = out @ self.weights[-1] + self.biases[-1]
        self._zs.append(z_out)
        self._inputs.append(z_out)

        if is_1d:
            return z_out[0]
        return z_out

    def backward(self, grad_output: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Analytical backpropagation through the network.
        
        Args:
            grad_output: dL/d(out), shape (batch_size, output_dim)
        Returns:
            grad_weights: List of dL/dW for each layer
            grad_biases: List of dL/db for each layer
        """
        grad = np.asarray(grad_output, dtype=self.dtype)
        if grad.ndim == 1:
            grad = grad.reshape(1, -1)

        grad_w: List[np.ndarray] = []
        grad_b: List[np.ndarray] = []

        for i in reversed(range(len(self.weights))):
            a_prev = self._inputs[i]
            grad_w.insert(0, a_prev.T @ grad)
            grad_b.insert(0, np.sum(grad, axis=0))

            if i > 0:
                grad_a = grad @ self.weights[i].T
                # Derivative of ReLU: 1 if z > 0 else 0
                grad = grad_a * (self._zs[i - 1] > 0.0)

        return grad_w, grad_b

    def get_params(self) -> List[np.ndarray]:
        """Return list of all mutable parameters [W1, b1, W2, b2, ...]."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w, b])
        return params

    def copy_from(self, source: MLP) -> None:
        """Deep copy parameters from another MLP instance."""
        assert len(self.weights) == len(source.weights), "Network architectures must match"
        for i in range(len(self.weights)):
            self.weights[i] = np.copy(source.weights[i])
            self.biases[i] = np.copy(source.biases[i])

    def save(self, filepath: Path | str) -> None:
        """Save model parameters to a compressed .npz file."""
        data_dict = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data_dict[f"w_{i}"] = w
            data_dict[f"b_{i}"] = b
        np.savez_compressed(filepath, **data_dict)

    def load(self, filepath: Path | str) -> None:
        """Load model parameters from a .npz file."""
        data = np.load(filepath)
        for i in range(len(self.weights)):
            self.weights[i] = data[f"w_{i}"].astype(self.dtype)
            self.biases[i] = data[f"b_{i}"].astype(self.dtype)


def compute_huber_loss_and_grad(
    q_pred: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    delta: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """Compute Huber loss and gradient w.r.t q_pred for selected action indices.
    
    Args:
        q_pred: shape (batch_size, num_actions)
        actions: shape (batch_size,) integer action indices
        targets: shape (batch_size,) TD target values
        delta: Huber loss threshold parameter (default: 1.0)
    Returns:
        loss: mean scalar Huber loss over the batch
        grad: shape (batch_size, num_actions) gradient dL/d(q_pred)
    """
    batch_size = len(actions)
    chosen_q = q_pred[np.arange(batch_size), actions]
    errors = chosen_q - targets

    abs_errors = np.abs(errors)
    quadratic_mask = (abs_errors <= delta)
    linear_mask = ~quadratic_mask

    # Huber loss values
    loss_vals = np.empty_like(errors)
    loss_vals[quadratic_mask] = 0.5 * (errors[quadratic_mask] ** 2)
    loss_vals[linear_mask] = delta * (abs_errors[linear_mask] - 0.5 * delta)
    mean_loss = float(np.mean(loss_vals))

    # Analytical gradient dL / d(chosen_q)
    grad_chosen = np.empty_like(errors)
    grad_chosen[quadratic_mask] = errors[quadratic_mask]
    grad_chosen[linear_mask] = delta * np.sign(errors[linear_mask])

    # Normalize by batch size
    grad_chosen /= batch_size

    # Full gradient matrix
    grad = np.zeros_like(q_pred)
    grad[np.arange(batch_size), actions] = grad_chosen

    return mean_loss, grad

