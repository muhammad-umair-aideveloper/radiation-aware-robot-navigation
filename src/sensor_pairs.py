"""Generation and serialization of seeded training goal pools, held-out goal pools, and evaluation pairs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, GOAL_CELL
from main import compute_free_space_distances

RESULTS_DIR = REPO_ROOT / "results"
SENSOR_PAIRS_FILE = RESULTS_DIR / "sensor_pairs.json"


def get_low_radiation_cells(
    env: RadiationEnv,
    max_intensity: float = 0.05,
    exclude_cell: Tuple[int, int] = GOAL_CELL,
) -> List[Tuple[int, int]]:
    """Return free cells with low radiation intensity, strictly excluding the canonical goal."""
    # Check if raw radiation values < max_intensity exist
    raw_cells = [
        (r, c)
        for r in range(env.shape[0])
        for c in range(env.shape[1])
        if not env.walls[r, c] and env.radiation[r, c] < max_intensity and (r, c) != exclude_cell
    ]
    if len(raw_cells) >= 50:
        return sorted(raw_cells)

    # Use normalized intensity I / max(I) < max_intensity
    max_rad = float(np.max(env.radiation))
    norm_cells = [
        (r, c)
        for r in range(env.shape[0])
        for c in range(env.shape[1])
        if not env.walls[r, c] and (env.radiation[r, c] / max_rad) < max_intensity and (r, c) != exclude_cell
    ]
    return sorted(norm_cells)


def generate_goal_pools_and_eval_pairs(
    env: RadiationEnv,
    seed: int = 42,
    num_train: int = 30,
    num_held_out: int = 20,
    starts_per_goal: int = 5,
    min_dist: float = 20.0,
    max_intensity: float = 0.05,
) -> Dict[str, Any]:
    """Sample disjoint training and held-out goal pools, and generate fixed evaluation pairs."""
    candidates = get_low_radiation_cells(env, max_intensity=max_intensity, exclude_cell=env.goal)

    if len(candidates) < num_train + num_held_out:
        raise RuntimeError(
            f"Not enough low-radiation candidates ({len(candidates)}) to sample "
            f"{num_train} train and {num_held_out} held-out goals."
        )

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(candidates))

    train_goals = [candidates[i] for i in shuffled_indices[:num_train]]
    held_out_goals = [candidates[i] for i in shuffled_indices[num_train : num_train + num_held_out]]

    assert set(train_goals).isdisjoint(set(held_out_goals)), "Train and held-out goals must be disjoint"
    assert env.goal not in train_goals, "Canonical goal must not be in training pool"
    assert env.goal not in held_out_goals, "Canonical goal must not be in held-out pool"

    eval_pairs: List[Dict[str, Any]] = []

    for g_idx, goal in enumerate(held_out_goals):
        dist_map = compute_free_space_distances(env, goal)
        valid_starts: List[Tuple[int, int]] = []
        for r in range(env.shape[0]):
            for c in range(env.shape[1]):
                if not env.walls[r, c] and dist_map[r, c] >= min_dist and dist_map[r, c] < 1e9:
                    valid_starts.append((r, c))

        valid_starts.sort()
        if len(valid_starts) < starts_per_goal:
            raise RuntimeError(
                f"Held-out goal {goal} only has {len(valid_starts)} valid starts with distance >= {min_dist}"
            )

        start_rng = np.random.RandomState(seed * 100 + g_idx)
        chosen_indices = start_rng.choice(len(valid_starts), size=starts_per_goal, replace=False)

        for s_idx, chosen_i in enumerate(chosen_indices):
            start = valid_starts[chosen_i]
            eval_pairs.append({
                "pair_id": g_idx * starts_per_goal + s_idx,
                "goal_index": g_idx,
                "goal": list(goal),
                "start_index": s_idx,
                "start": list(start),
                "free_space_distance": float(dist_map[start]),
            })

    return {
        "metadata": {
            "num_train_goals": num_train,
            "num_held_out_goals": num_held_out,
            "num_eval_pairs": len(eval_pairs),
            "starts_per_held_out_goal": starts_per_goal,
            "min_free_space_distance": min_dist,
            "max_goal_radiation_threshold": max_intensity,
            "seed": seed,
            "canonical_goal_excluded": list(env.goal),
        },
        "train_goals": [list(g) for g in train_goals],
        "held_out_goals": [list(g) for g in held_out_goals],
        "eval_pairs": eval_pairs,
    }


def save_sensor_pairs(
    filepath: Path = SENSOR_PAIRS_FILE,
    env: RadiationEnv | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate and write sensor pairs JSON."""
    if env is None:
        env = RadiationEnv()
    data = generate_goal_pools_and_eval_pairs(env, seed=seed)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['eval_pairs'])} evaluation pairs to {filepath}")
    return data


def load_sensor_pairs(filepath: Path = SENSOR_PAIRS_FILE) -> Dict[str, Any]:
    """Load pre-generated sensor pairs JSON."""
    if not filepath.exists():
        raise FileNotFoundError(f"Sensor pairs file not found: {filepath}. Run src/sensor_pairs.py first.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    save_sensor_pairs()
