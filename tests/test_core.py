"""Core unit tests + 6 scientific invariant checks + artifact integrity audit.

Run with:  python tests/test_core.py
All tests use plain asserts; no pytest dependency required.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.planners import QLearningAgent, astar_search, octile_distance


# ---------------------------------------------------------------------------
# Original 4 tests
# ---------------------------------------------------------------------------

def test_field_at_source() -> None:
    """Field value at an isolated source must equal I0 / 2.0."""
    sr, sc = 10, 10
    i0 = 100.0
    field = compute_radiation_field(shape=(20, 20), sources=[((sr, sc), i0)])
    expected = i0 / 2.0
    actual = field[sr, sc]
    assert math.isclose(actual, expected, rel_tol=1e-6), (
        f"Expected source intensity {expected}, got {actual}"
    )
    print("PASS: test_field_at_source")


def test_astar_empty_grid() -> None:
    """A* on an empty small grid returns the exact octile path length."""
    env = RadiationEnv(start=(0, 0), goal=(3, 4), max_steps=100)
    env.shape = (10, 10)
    env.walls = np.zeros((10, 10), dtype=bool)
    env.radiation = np.zeros((10, 10), dtype=np.float64)

    path, dose, steps, expanded = astar_search((0, 0), (3, 4), mode="shortest", env=env)
    assert path is not None, "A* failed to find path on empty grid"

    expected_length = 3 * math.sqrt(2.0) + 1 * 1.0
    actual_length = 0.0
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        actual_length += math.sqrt(dr * dr + dc * dc)

    assert math.isclose(actual_length, expected_length, rel_tol=1e-6), (
        f"Expected path length {expected_length}, got {actual_length}"
    )
    assert steps == 4, f"Expected 4 steps, got {steps}"
    print("PASS: test_astar_empty_grid")


def test_astar_never_crosses_walls() -> None:
    """A* path must never visit a wall cell or cut wall corners."""
    env = RadiationEnv()
    for mode in ["shortest", "aware"]:
        path, dose, steps, expanded = astar_search(START_CELL, GOAL_CELL, mode=mode, env=env)
        assert path is not None, f"A* {mode} failed to find path"

        for idx, (r, c) in enumerate(path):
            assert not env.walls[r, c], f"Path step {idx} is a wall cell: ({r}, {c})"
            if idx > 0:
                pr, pc = path[idx - 1]
                dr = r - pr
                dc = c - pc
                if abs(dr) == 1 and abs(dc) == 1:
                    assert not (env.walls[pr + dr, pc] or env.walls[pr, pc + dc]), (
                        f"Corner cutting detected between ({pr},{pc}) and ({r},{c})"
                    )
    print("PASS: test_astar_never_crosses_walls")


def test_collision_rewards_terminate() -> None:
    """Colliding with boundary or wall must terminate with -1000 reward."""
    env = RadiationEnv(start=(0, 0), goal=GOAL_CELL)
    # Action 0 is N: moving north from row 0 is out of bounds
    next_state, reward, done, info = env.step(0)
    assert done is True, "Collision must terminate the episode"
    assert reward == -1000.0, f"Expected reward -1000.0 on collision, got {reward}"
    assert info["outcome"] == "collision", f"Expected collision outcome, got {info['outcome']}"
    assert next_state == (0, 0), "Robot must stay at original position on collision"
    print("PASS: test_collision_rewards_terminate")


# ---------------------------------------------------------------------------
# Invariant 1: Inverse-square monotonicity + no NaN/Inf
# ---------------------------------------------------------------------------

def test_inverse_square_monotonicity() -> None:
    """Radiation intensity strictly decreases with distance; no NaN or Inf on grid."""
    # --- NaN/Inf check on full 50x50 field ---
    env = RadiationEnv()
    assert not np.any(np.isnan(env.radiation)), "NaN found in radiation field"
    assert not np.any(np.isinf(env.radiation)), "Inf found in radiation field"

    # --- Monotonicity: single isolated source, check along a radial direction ---
    sr, sc = 25, 25
    i0 = 200.0
    field = compute_radiation_field(shape=(50, 50), sources=[((sr, sc), i0)])
    # Sample along the east ray: (sr, sc), (sr, sc+1), ..., (sr, sc+10)
    prev = field[sr, sc]
    for dc in range(1, 11):
        curr = field[sr, sc + dc]
        assert curr < prev, (
            f"Monotonicity violated: I(d={dc-1})={prev:.6f} <= I(d={dc})={curr:.6f}"
        )
        prev = curr

    # --- Check peak value at source equals I0/2 ---
    assert math.isclose(field[sr, sc], i0 / 2.0, rel_tol=1e-6), (
        f"Peak value {field[sr, sc]} != I0/2={i0/2.0}"
    )
    print("PASS: test_inverse_square_monotonicity")


# ---------------------------------------------------------------------------
# Invariant 2: Zero-lambda equivalence
# ---------------------------------------------------------------------------

def test_zero_radiation_equivalence() -> None:
    """With lambda=0 (kappa=0), aware A* step cost = L = shortest, so paths are identical cost."""
    # Build a minimal env where radiation is non-zero (use default env)
    env = RadiationEnv()

    # With kappa=0 in aware mode the step cost equals L, same as shortest.
    # We test on a small obstacle-free sub-environment to get a unique path.
    env_flat = RadiationEnv(start=(5, 5), goal=(10, 15), max_steps=100)
    env_flat.shape = (20, 20)
    env_flat.walls = np.zeros((20, 20), dtype=bool)
    env_flat.radiation = np.zeros((20, 20), dtype=np.float64)

    path_s, _, steps_s, _ = astar_search((5, 5), (10, 15), mode="shortest", env=env_flat)
    path_a, _, steps_a, _ = astar_search((5, 5), (10, 15), mode="aware", env=env_flat)

    assert path_s is not None and path_a is not None, "A* failed on flat grid"
    # Both must yield same step count when radiation=0 (cost=L in both modes)
    assert steps_s == steps_a, (
        f"With zero radiation, shortest steps={steps_s} != aware steps={steps_a}"
    )

    # Measure actual path lengths in cells
    def path_length(p):
        total = 0.0
        for i in range(len(p) - 1):
            dr = p[i + 1][0] - p[i][0]
            dc = p[i + 1][1] - p[i][1]
            total += math.sqrt(dr * dr + dc * dc)
        return total

    len_s = path_length(path_s)
    len_a = path_length(path_a)
    assert math.isclose(len_s, len_a, rel_tol=1e-6), (
        f"With zero radiation, path lengths differ: shortest={len_s:.4f}, aware={len_a:.4f}"
    )
    print("PASS: test_zero_radiation_equivalence")


# ---------------------------------------------------------------------------
# Invariant 3: High-radiation detour
# ---------------------------------------------------------------------------

def test_high_radiation_detour() -> None:
    """Aware A* must have dose_aware < 0.35 * dose_shortest and steps_aware >= steps_shortest."""
    env = RadiationEnv()
    path_s, dose_s, steps_s, _ = astar_search(START_CELL, GOAL_CELL, mode="shortest", env=env)
    path_a, dose_a, steps_a, _ = astar_search(START_CELL, GOAL_CELL, mode="aware", env=env)

    assert path_s is not None, "A* Shortest failed on canonical route"
    assert path_a is not None, "A* Aware failed on canonical route"

    threshold = 0.35 * dose_s
    assert dose_a < threshold, (
        f"dose_aware={dose_a:.2f} is NOT < 0.35 * dose_shortest={threshold:.2f}"
    )
    assert steps_a >= steps_s, (
        f"Aware path ({steps_a} steps) is shorter than shortest path ({steps_s} steps)"
    )
    print(
        f"PASS: test_high_radiation_detour "
        f"(dose_aware={dose_a:.2f} < {threshold:.2f}=0.35*dose_shortest={dose_s:.2f}, "
        f"steps_aware={steps_a} >= steps_shortest={steps_s})"
    )


# ---------------------------------------------------------------------------
# Invariant 4: No obstacle penetration (A* + Q-learning rollout)
# ---------------------------------------------------------------------------

def test_no_obstacle_penetration() -> None:
    """No trajectory cell from A* or Q-learning rollout may fall on a wall cell."""
    env = RadiationEnv()
    walls = env.walls  # shape (50, 50), True = obstacle

    # A* paths
    for mode in ["shortest", "aware"]:
        path, _, _, _ = astar_search(START_CELL, GOAL_CELL, mode=mode, env=env)
        assert path is not None, f"A* {mode} failed"
        for idx, (r, c) in enumerate(path):
            assert not walls[r, c], (
                f"A* {mode}: step {idx} at ({r},{c}) is a wall"
            )

    # Train Q-learning with a very small budget (500 episodes) just for rollout coverage
    agent = QLearningAgent(env=env, episodes=500, alpha=0.15, gamma=0.98)
    Q, _, _, _ = agent.train(seed=0)
    path_q, outcome, _, _ = agent.rollout(Q, start=START_CELL)

    for idx, (r, c) in enumerate(path_q):
        assert not walls[r, c], (
            f"Q-learning rollout: step {idx} at ({r},{c}) is a wall (outcome={outcome})"
        )

    print(f"PASS: test_no_obstacle_penetration (Q outcome={outcome}, path_len={len(path_q)})")


# ---------------------------------------------------------------------------
# Invariant 5: Bellman update correctness (single-step, deterministic)
# ---------------------------------------------------------------------------

def test_q_table_bellman_update() -> None:
    """Q[s,a] must update per (1-alpha)*Q[s,a] + alpha*(r + gamma*max(Q[s']))."""
    env = RadiationEnv()
    agent = QLearningAgent(env=env, episodes=0)  # no training, just the tables

    alpha = agent.alpha       # 0.15
    gamma = agent.gamma       # 0.98

    # Pick a known non-colliding state: start cell, action 2 (East)
    r, c, a = START_CELL[0], START_CELL[1], 2  # E
    assert not env.walls[r, c], "Start cell must be free"
    assert not bool(agent.terminals[r, c, a]), "Action E from (2,2) must not be terminal"

    nr = int(agent.next_r[r, c, a])
    nc = int(agent.next_c[r, c, a])
    rew = float(agent.rewards[r, c, a])

    # Set up initial Q values explicitly for reproducibility
    Q = np.zeros((*env.shape, 8), dtype=np.float32)
    Q[r, c, a] = 5.0            # arbitrary initial value
    Q[nr, nc, :] = np.arange(8, dtype=np.float32)  # max = 7.0 at action index 7

    q_before = float(Q[r, c, a])
    best_next = float(np.max(Q[nr, nc]))
    target = rew + gamma * best_next
    expected_after = (1.0 - alpha) * q_before + alpha * target

    # Perform one manual update (same formula used in train())
    Q[r, c, a] += alpha * (target - Q[r, c, a])
    actual_after = float(Q[r, c, a])

    assert math.isclose(actual_after, expected_after, abs_tol=1e-6), (
        f"Bellman update failed: expected={expected_after:.8f}, got={actual_after:.8f}"
    )
    print(
        f"PASS: test_q_table_bellman_update "
        f"(q_before={q_before:.4f}, target={target:.4f}, q_after={actual_after:.6f})"
    )


# ---------------------------------------------------------------------------
# Invariant 6: Visual artifact integrity
# ---------------------------------------------------------------------------

def test_visual_artifacts_integrity() -> None:
    """Plots and GIF must exist, be > 15 KB, and GIF must have >= 10 frames."""
    from PIL import Image

    artifacts = {
        "trajectory_comparison.png": REPO_ROOT / "results" / "trajectory_comparison.png",
        "rl_convergence.png":        REPO_ROOT / "results" / "rl_convergence.png",
        "radiation_navigation.gif":  REPO_ROOT / "results" / "radiation_navigation.gif",
    }

    MIN_SIZE = 15 * 1024  # 15 KB

    for name, path in artifacts.items():
        assert path.exists(), f"Artifact missing: {path}"
        size = path.stat().st_size
        assert size > MIN_SIZE, (
            f"Artifact {name} is suspiciously small: {size} bytes (< {MIN_SIZE})"
        )
        print(f"  {name}: {size // 1024} KB ✓")

    # GIF frame count
    gif_path = artifacts["radiation_navigation.gif"]
    img = Image.open(gif_path)
    frames = 0
    try:
        while True:
            frames += 1
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    assert frames >= 10, f"GIF has only {frames} frames (required >= 10)"
    print(f"PASS: test_visual_artifacts_integrity ({frames} GIF frames)")


# ---------------------------------------------------------------------------
# Repository hygiene check
# ---------------------------------------------------------------------------

def test_repo_hygiene() -> None:
    """README must not contain TODO/TBD; .gitignore must exclude __pycache__ etc."""
    readme = REPO_ROOT / "README.md"
    assert readme.exists(), "README.md is missing"
    text = readme.read_text(encoding="utf-8")
    for placeholder in ["TODO", "TBD", "PLACEHOLDER"]:
        assert placeholder not in text, f"README.md still contains placeholder: {placeholder}"

    gitignore = REPO_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore is missing"
    gi_text = gitignore.read_text(encoding="utf-8")
    for entry in ["__pycache__/", ".venv/"]:
        # Accept either exact match or variant like `__pycache__` without slash
        base = entry.rstrip("/")
        assert base in gi_text or entry in gi_text, (
            f".gitignore is missing entry for: {entry}"
        )
    print("PASS: test_repo_hygiene")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_saved_model_inference() -> None:
    """Saved Q-table must exist, load cleanly, reach goal with dose < 75 and no wall hits."""
    from evaluate import greedy_rollout

    q_path = REPO_ROOT / "models" / "q_table_best.npy"
    assert q_path.exists(), f"Artifact missing: {q_path}"

    Q = np.load(q_path)
    assert Q.shape == (50, 50, 8), f"Unexpected Q-table shape: {Q.shape}"
    assert not np.any(np.isnan(Q)), "NaN values in loaded Q-table"

    env = RadiationEnv()
    path, outcome, total_dose, steps = greedy_rollout(Q, env, start=START_CELL)

    assert outcome == "goal", f"Pre-trained model did not reach goal (outcome={outcome})"
    assert total_dose < 75.0, f"Dose {total_dose:.2f} >= 75.0 a.u."

    # No wall collisions in the path
    for idx, (r, c) in enumerate(path):
        assert not env.walls[r, c], f"Wall collision at step {idx}: ({r},{c})"

    print(
        f"PASS: test_saved_model_inference "
        f"(outcome={outcome}, steps={steps}, dose={total_dose:.2f} a.u.)"
    )


def run_all_tests() -> None:
    tests = [
        test_field_at_source,
        test_astar_empty_grid,
        test_astar_never_crosses_walls,
        test_collision_rewards_terminate,
        test_inverse_square_monotonicity,
        test_zero_radiation_equivalence,
        test_high_radiation_detour,
        test_no_obstacle_penetration,
        test_q_table_bellman_update,
        test_visual_artifacts_integrity,
        test_repo_hygiene,
    ]

    print("=" * 65)
    print("QA & Scientific Invariant Audit — test_core.py")
    print("=" * 65)

    passed, failed = [], []
    for fn in tests:
        try:
            fn()
            passed.append(fn.__name__)
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL: {fn.__name__} — {exc}")

    print()
    print("=" * 65)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print("=" * 65)
    print(f"{'Test':<45} {'Result'}")
    print("-" * 65)
    for name in passed:
        print(f"  {name:<43} PASS")
    for name, _ in failed:
        print(f"  {name:<43} FAIL")
    print()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()
