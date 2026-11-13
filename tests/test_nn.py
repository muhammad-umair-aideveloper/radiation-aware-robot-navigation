"""Unit tests for NumPy MLP and Adam optimizer: finite differences gradient verification and optimization."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nn import MLP, AdamOptimizer, compute_huber_loss_and_grad


def test_analytic_gradients_against_finite_differences() -> None:
    """Verify analytical backpropagation gradients against two-sided finite differences in float64."""
    # Test network with 28 inputs, two hidden layers of 16, and 4 outputs in float64
    mlp = MLP(layer_sizes=(10, 16, 16, 4), seed=2026, dtype=np.float64)

    rng = np.random.RandomState(42)
    batch_size = 5
    x = rng.randn(batch_size, 10).astype(np.float64)
    y_target = rng.randn(batch_size, 4).astype(np.float64)

    # MSE Loss: L = 0.5 * sum((out - y_target)^2) / batch_size
    def compute_loss() -> float:
        out = mlp.forward(x)
        return float(0.5 * np.sum((out - y_target) ** 2) / batch_size)

    out = mlp.forward(x)
    grad_out = (out - y_target) / batch_size
    grad_w, grad_b = mlp.backward(grad_out)

    eps = 1e-7
    tolerance = 1e-5

    # 1. Check weight gradients
    for layer_idx, W in enumerate(mlp.weights):
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                orig = W[r, c]

                W[r, c] = orig + eps
                l_plus = compute_loss()

                W[r, c] = orig - eps
                l_minus = compute_loss()

                W[r, c] = orig

                num_grad = (l_plus - l_minus) / (2.0 * eps)
                ana_grad = grad_w[layer_idx][r, c]

                rel_err = abs(num_grad - ana_grad) / max(1e-8, abs(num_grad) + abs(ana_grad))
                assert rel_err < tolerance, (
                    f"Weight grad mismatch at layer {layer_idx}, ({r}, {c}): "
                    f"num={num_grad:.8e}, ana={ana_grad:.8e}, rel_err={rel_err:.8e} >= {tolerance}"
                )

    # 2. Check bias gradients
    for layer_idx, b in enumerate(mlp.biases):
        for i in range(b.shape[0]):
            orig = b[i]

            b[i] = orig + eps
            l_plus = compute_loss()

            b[i] = orig - eps
            l_minus = compute_loss()

            b[i] = orig

            num_grad = (l_plus - l_minus) / (2.0 * eps)
            ana_grad = grad_b[layer_idx][i]

            rel_err = abs(num_grad - ana_grad) / max(1e-8, abs(num_grad) + abs(ana_grad))
            assert rel_err < tolerance, (
                f"Bias grad mismatch at layer {layer_idx}, ({i}): "
                f"num={num_grad:.8e}, ana={ana_grad:.8e}, rel_err={rel_err:.8e} >= {tolerance}"
            )

    print("PASS: test_analytic_gradients_against_finite_differences (all rel errors < 1e-5 in float64)")


def test_huber_loss_and_gradient() -> None:
    """Verify Huber loss value and gradient with finite differences."""
    rng = np.random.RandomState(99)
    batch_size = 4
    num_actions = 8

    q_pred = rng.randn(batch_size, num_actions).astype(np.float64)
    actions = np.array([0, 2, 5, 7], dtype=np.int64)
    # Include both small errors (< 1.0) and large errors (> 1.0)
    targets = q_pred[np.arange(batch_size), actions] + np.array([-0.3, 0.7, -2.5, 4.0], dtype=np.float64)

    loss, grad = compute_huber_loss_and_grad(q_pred, actions, targets, delta=1.0)

    eps = 1e-7
    for b in range(batch_size):
        for a in range(num_actions):
            orig = q_pred[b, a]

            q_pred[b, a] = orig + eps
            l_plus, _ = compute_huber_loss_and_grad(q_pred, actions, targets, delta=1.0)

            q_pred[b, a] = orig - eps
            l_minus, _ = compute_huber_loss_and_grad(q_pred, actions, targets, delta=1.0)

            q_pred[b, a] = orig

            num_grad = (l_plus - l_minus) / (2.0 * eps)
            ana_grad = grad[b, a]

            rel_err = abs(num_grad - ana_grad) / max(1e-8, abs(num_grad) + abs(ana_grad))
            assert rel_err < 1e-5, f"Huber grad mismatch at ({b}, {a}): rel_err={rel_err}"

    print("PASS: test_huber_loss_and_gradient")


def test_adam_step_decreases_loss() -> None:
    """Verify that multiple Adam optimizer steps strictly decrease training loss."""
    mlp = MLP(layer_sizes=(10, 20, 2), seed=42, dtype=np.float32)
    optimizer = AdamOptimizer(mlp.get_params(), lr=0.01)

    rng = np.random.RandomState(123)
    x = rng.randn(8, 10).astype(np.float32)
    y = rng.randn(8, 2).astype(np.float32)

    initial_loss = float(np.mean((mlp.forward(x) - y) ** 2))

    for _ in range(50):
        out = mlp.forward(x)
        grad_out = 2.0 * (out - y) / 8.0
        gw, gb = mlp.backward(grad_out)
        grads = []
        for w, b in zip(gw, gb):
            grads.extend([w, b])
        optimizer.step(grads)

    final_loss = float(np.mean((mlp.forward(x) - y) ** 2))
    assert final_loss < 0.1 * initial_loss, f"Adam failed to decrease loss: {initial_loss} -> {final_loss}"

    print("PASS: test_adam_step_decreases_loss")


def test_save_load_parameters(tmp_path_fn: Path | None = None) -> None:
    """Verify serialization and deserialization preserves exact network outputs."""
    mlp1 = MLP(layer_sizes=(28, 64, 64, 8), seed=777)
    x = np.random.randn(3, 28).astype(np.float32)
    out1 = mlp1.forward(x)

    tmp_file = REPO_ROOT / "models" / "_tmp_test_weights.npz"
    tmp_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        mlp1.save(tmp_file)

        mlp2 = MLP(layer_sizes=(28, 64, 64, 8), seed=999)
        mlp2.load(tmp_file)
        out2 = mlp2.forward(x)

        assert np.array_equal(out1, out2), "Loaded weights produced divergent outputs"
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

    print("PASS: test_save_load_parameters")


def run_all_tests() -> None:
    print("Running NN unit tests...")
    test_analytic_gradients_against_finite_differences()
    test_huber_loss_and_gradient()
    test_adam_step_decreases_loss()
    test_save_load_parameters()
    print("All NN tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()

