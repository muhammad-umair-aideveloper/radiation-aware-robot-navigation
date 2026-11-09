"""Edge-inference latency and memory benchmark for the Q-table agent.

Measures single-step action selection latency over 1,000 dry-run states on CPU
using time.perf_counter() for sub-millisecond resolution.

Usage:
    python benchmark_inference.py

Output: mean and 99th-percentile latency in milliseconds, peak RSS (MB).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, create_wall_mask

N_SAMPLES: int = 1_000
Q_PATH: Path = REPO_ROOT / "models" / "q_table_best.npy"


def _peak_rss_mb() -> float:
    """Return process peak RSS in megabytes (best-effort, Windows & Linux)."""
    try:
        import resource  # Unix
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    return float("nan")  # not available without optional deps


def run_latency_benchmark(
    Q: np.ndarray,
    n_samples: int = N_SAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Benchmark greedy action selection latency over *n_samples* random states.

    For each sample a random free cell is chosen as the state; the action is
    selected as argmax(Q[r, c]) — the exact operation used at inference time.

    Args:
        Q:         Loaded Q-table, shape (H, W, num_actions).
        n_samples: Number of single-step timing measurements.
        seed:      RNG seed for reproducible state sampling.

    Returns:
        dict with keys: mean_ms, p99_ms, min_ms, max_ms, n_samples.
    """
    env = RadiationEnv()
    free_cells = [
        (int(r), int(c))
        for r, c in zip(*np.where(~env.walls))
    ]
    rng = np.random.RandomState(seed)
    sample_states = [
        free_cells[rng.randint(len(free_cells))]
        for _ in range(n_samples)
    ]

    latencies_ns: list[float] = []
    for r, c in sample_states:
        t0 = time.perf_counter()
        _ = int(np.argmax(Q[r, c]))          # single-step inference
        t1 = time.perf_counter()
        latencies_ns.append((t1 - t0) * 1e6)  # microseconds -> stored as µs

    arr = np.array(latencies_ns)
    return {
        "mean_us": float(np.mean(arr)),
        "p99_us": float(np.percentile(arr, 99)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)) / 1000.0,
        "p99_ms": float(np.percentile(arr, 99)) / 1000.0,
        "n_samples": float(n_samples),
    }


def main() -> None:
    if not Q_PATH.exists():
        print(f"ERROR: {Q_PATH} not found. Run `python main.py` first.")
        raise SystemExit(1)

    print("=" * 55)
    print("Edge-Inference Latency Benchmark")
    print("=" * 55)
    Q = np.load(Q_PATH)
    print(f"Q-table: {Q.shape}, dtype={Q.dtype}, "
          f"size={Q.nbytes / 1024:.1f} KB")

    stats = run_latency_benchmark(Q, n_samples=N_SAMPLES)

    print(f"\nAction-selection latency over {N_SAMPLES:,} dry-run states (CPU):")
    print(f"  Mean    : {stats['mean_ms']:.4f} ms  ({stats['mean_us']:.2f} µs)")
    print(f"  P99     : {stats['p99_ms']:.4f} ms  ({stats['p99_us']:.2f} µs)")
    print(f"  Min     : {stats['min_us']:.2f} µs")
    print(f"  Max     : {stats['max_us']:.2f} µs")

    rss = _peak_rss_mb()
    if not (rss != rss):  # not NaN
        print(f"\nPeak process RSS : {rss:.1f} MB")
    else:
        print("\nPeak RSS: not available (install psutil for Windows support)")
    print("=" * 55)


if __name__ == "__main__":
    main()
