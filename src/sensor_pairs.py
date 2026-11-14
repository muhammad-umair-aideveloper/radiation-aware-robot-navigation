"""Generation and serialization of seeded training goal pools, held-out goal pools, and evaluation pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, GOAL_CELL
from main import compute_free_space_distances
from config import VAL_PAIRS_SEED

RESULTS_DIR = REPO_ROOT / "results"
SENSOR_PAIRS_FILE = RESULTS_DIR / "sensor_pairs.json"
SENSOR_PAIRS_S6_FILE = RESULTS_DIR / "sensor_pairs_s6.json"
VAL_PAIRS_FILE = RESULTS_DIR / "sensor_val_pairs.json"
TEST_PAIRS_FILE = RESULTS_DIR / "sensor_test_pairs.json"
DIAG_PAIRS_FILE = RESULTS_DIR / "sensor_diag_pairs.json"
POOL_150_FILE = RESULTS_DIR / "sensor_pool_150.json"


def get_low_radiation_cells(
    env: RadiationEnv,
    max_intensity: float = 0.05,
    exclude_cell: Tuple[int, int] = GOAL_CELL,
) -> List[Tuple[int, int]]:
    """Return free cells with low radiation intensity, strictly excluding the canonical goal."""
    raw_cells = [
        (r, c)
        for r in range(env.shape[0])
        for c in range(env.shape[1])
        if not env.walls[r, c] and env.radiation[r, c] < max_intensity and (r, c) != exclude_cell
    ]
    if len(raw_cells) >= 50:
        return sorted(raw_cells)

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
    """Sample disjoint training and held-out goal pools, and generate fixed evaluation pairs (S5)."""
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


def dist_euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def min_dist_to_set(pt: Tuple[int, int], pt_set: List[Tuple[int, int]]) -> float:
    if not pt_set:
        return 1e9
    return min(dist_euclidean(pt, p) for p in pt_set)


def generate_s6_pairs(
    env: Optional[RadiationEnv] = None,
    seed: int = VAL_PAIRS_SEED,
    min_separation: float = 5.0,
    min_dist: float = 20.0,
    starts_per_goal: int = 5,
    max_intensity: float = 0.05,
) -> Dict[str, Any]:
    """Generate S6 goal pools (150-train), validation, test, and diagnostic pair sets.

    Constraints strictly upheld:
      - Pairwise disjoint goal sets.
      - Minimum 5-cell Euclidean separation between sets.
      - Free cells with normalized I < 0.05 at starts and goals.
      - Free-space geodesic distance >= 20.0 cells.
      - Canonical goal (47, 47) excluded from all sets.
    """
    if env is None:
        env = RadiationEnv()

    # Load existing S5 training and held-out goals for continuity
    s5_data = load_sensor_pairs(filepath=SENSOR_PAIRS_FILE)
    train_goals_30 = [tuple(g) for g in s5_data["train_goals"]]
    held_out_goals_20 = [tuple(g) for g in s5_data["held_out_goals"]]
    held_out_pairs = s5_data["eval_pairs"]

    candidates = get_low_radiation_cells(env, max_intensity=max_intensity, exclude_cell=env.goal)
    cand_set = set(candidates)

    rng = np.random.RandomState(seed)
    shuffled = [candidates[i] for i in rng.permutation(len(candidates))]

    # 1. Validation Goals (20 goals)
    val_goals: List[Tuple[int, int]] = []
    for c in shuffled:
        if len(val_goals) >= 20:
            break
        if min_dist_to_set(c, train_goals_30) >= min_separation and min_dist_to_set(c, held_out_goals_20) >= min_separation:
            val_goals.append(c)

    # 2. Test Goals (20 goals)
    test_goals: List[Tuple[int, int]] = []
    for c in shuffled:
        if len(test_goals) >= 20:
            break
        if c in val_goals:
            continue
        if (
            min_dist_to_set(c, train_goals_30) >= min_separation
            and min_dist_to_set(c, held_out_goals_20) >= min_separation
            and min_dist_to_set(c, val_goals) >= min_separation
        ):
            test_goals.append(c)

    # 3. Extra Training Goals for 150-goal pool (120 goals)
    extra_train_goals: List[Tuple[int, int]] = []
    for c in shuffled:
        if len(extra_train_goals) >= 120:
            break
        if c in train_goals_30 or c in val_goals or c in test_goals:
            continue
        if (
            min_dist_to_set(c, held_out_goals_20) >= min_separation
            and min_dist_to_set(c, val_goals) >= min_separation
            and min_dist_to_set(c, test_goals) >= min_separation
        ):
            extra_train_goals.append(c)

    train_goals_150 = train_goals_30 + extra_train_goals

    # Verify disjointness and separation
    assert len(train_goals_30) == 30
    assert len(train_goals_150) == 150
    assert len(held_out_goals_20) == 20
    assert len(val_goals) == 20
    assert len(test_goals) == 20

    s_train150 = set(train_goals_150)
    s_heldout = set(held_out_goals_20)
    s_val = set(val_goals)
    s_test = set(test_goals)

    assert s_train150.isdisjoint(s_heldout), "Train150 and heldout goals must be disjoint"
    assert s_train150.isdisjoint(s_val), "Train150 and val goals must be disjoint"
    assert s_train150.isdisjoint(s_test), "Train150 and test goals must be disjoint"
    assert s_heldout.isdisjoint(s_val), "Heldout and val goals must be disjoint"
    assert s_heldout.isdisjoint(s_test), "Heldout and test goals must be disjoint"
    assert s_val.isdisjoint(s_test), "Val and test goals must be disjoint"

    for v in val_goals:
        assert min_dist_to_set(v, train_goals_150) >= min_separation
        assert min_dist_to_set(v, held_out_goals_20) >= min_separation
        assert min_dist_to_set(v, test_goals) >= min_separation

    for t in test_goals:
        assert min_dist_to_set(t, train_goals_150) >= min_separation
        assert min_dist_to_set(t, held_out_goals_20) >= min_separation
        assert min_dist_to_set(t, val_goals) >= min_separation

    for e in extra_train_goals:
        assert min_dist_to_set(e, held_out_goals_20) >= min_separation
        assert min_dist_to_set(e, val_goals) >= min_separation
        assert min_dist_to_set(e, test_goals) >= min_separation

    def sample_pairs(goals: List[Tuple[int, int]], seed_offset: int) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        for g_idx, goal in enumerate(goals):
            dist_map = compute_free_space_distances(env, goal)
            valid_starts = [
                c for c in cand_set
                if dist_map[c] >= min_dist and dist_map[c] < 1e9 and c != goal
            ]
            valid_starts.sort()
            start_rng = np.random.RandomState(seed + seed_offset + g_idx * 17)
            chosen_indices = start_rng.choice(len(valid_starts), size=starts_per_goal, replace=False)
            for s_idx, ci in enumerate(chosen_indices):
                start = valid_starts[ci]
                pairs.append({
                    "pair_id": g_idx * starts_per_goal + s_idx,
                    "goal_index": g_idx,
                    "goal": list(goal),
                    "start_index": s_idx,
                    "start": list(start),
                    "free_space_distance": float(dist_map[start]),
                })
        return pairs

    val_pairs = sample_pairs(val_goals, seed_offset=1000)
    test_pairs = sample_pairs(test_goals, seed_offset=2000)
    diag_pairs = sample_pairs(train_goals_30[:20], seed_offset=3000)

    assert len(val_pairs) == 100
    assert len(test_pairs) == 100
    assert len(diag_pairs) == 100

    s6_data = {
        "metadata": {
            "seed": seed,
            "min_separation_between_sets": min_separation,
            "min_free_space_distance": min_dist,
            "max_intensity_threshold": max_intensity,
            "num_train_goals_30": len(train_goals_30),
            "num_train_goals_150": len(train_goals_150),
            "num_held_out_goals": len(held_out_goals_20),
            "num_val_goals": len(val_goals),
            "num_test_goals": len(test_goals),
            "num_heldout_pairs": len(held_out_pairs),
            "num_val_pairs": len(val_pairs),
            "num_test_pairs": len(test_pairs),
            "num_diag_pairs": len(diag_pairs),
            "canonical_goal_excluded": list(env.goal),
        },
        "train_goals_30": [list(g) for g in train_goals_30],
        "train_goals_150": [list(g) for g in train_goals_150],
        "held_out_goals": [list(g) for g in held_out_goals_20],
        "val_goals": [list(g) for g in val_goals],
        "test_goals": [list(g) for g in test_goals],
        "held_out_pairs": held_out_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "diag_pairs": diag_pairs,
    }
    return s6_data


def generate_and_save_s6_pairs(
    filepath: Path = SENSOR_PAIRS_S6_FILE,
    env: Optional[RadiationEnv] = None,
    seed: int = VAL_PAIRS_SEED,
) -> Dict[str, Any]:
    """Generate and serialize S6 pair sets to disk."""
    data = generate_s6_pairs(env=env, seed=seed)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved complete S6 pair bundle to {filepath}")

    # Write separate companion files for direct access
    with open(VAL_PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump({"goals": data["val_goals"], "pairs": data["val_pairs"]}, f, indent=2)
    with open(TEST_PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump({"goals": data["test_goals"], "pairs": data["test_pairs"]}, f, indent=2)
    with open(DIAG_PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump({"goals": data["train_goals_30"][:20], "pairs": data["diag_pairs"]}, f, indent=2)
    with open(POOL_150_FILE, "w", encoding="utf-8") as f:
        json.dump({"train_goals_150": data["train_goals_150"]}, f, indent=2)

    print("Saved companion files: val, test, diag pairs, and 150-goal pool.")
    return data


def save_sensor_pairs(
    filepath: Path = SENSOR_PAIRS_FILE,
    env: RadiationEnv | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate and write sensor pairs JSON (S5)."""
    if env is None:
        env = RadiationEnv()
    data = generate_goal_pools_and_eval_pairs(env, seed=seed)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['eval_pairs'])} evaluation pairs to {filepath}")
    return data


def load_sensor_pairs(
    set_name: str = "heldout",
    filepath: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load pre-generated sensor pairs JSON.

    If set_name is 'heldout' and no filepath given, loads S5 results/sensor_pairs.json.
    Supports set_name in {'heldout', 'val', 'test', 'diag'}.
    """
    if filepath is not None:
        if not filepath.exists():
            raise FileNotFoundError(f"Sensor pairs file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    if set_name in ("heldout", "held_out"):
        target = SENSOR_PAIRS_FILE
        if not target.exists():
            raise FileNotFoundError(f"Held-out sensor pairs file not found: {target}")
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    # For S6 sets, load from SENSOR_PAIRS_S6_FILE
    target_s6 = SENSOR_PAIRS_S6_FILE
    if not target_s6.exists():
        raise FileNotFoundError(
            f"S6 sensor pairs file not found: {target_s6}. Run 'python src/sensor_pairs.py --s6' first."
        )
    with open(target_s6, "r", encoding="utf-8") as f:
        s6 = json.load(f)

    if set_name == "val":
        return {"eval_pairs": s6["val_pairs"], "goals": s6["val_goals"], "metadata": s6["metadata"]}
    elif set_name == "test":
        return {"eval_pairs": s6["test_pairs"], "goals": s6["test_goals"], "metadata": s6["metadata"]}
    elif set_name == "diag":
        return {"eval_pairs": s6["diag_pairs"], "goals": s6["train_goals_30"][:20], "metadata": s6["metadata"]}
    else:
        return s6


def load_pair_set(set_name: str) -> List[Dict[str, Any]]:
    """Convenience helper to return a list of 100 evaluation pairs by set name."""
    if set_name in ("heldout", "held_out"):
        data = load_sensor_pairs("heldout")
        return data["eval_pairs"]
    data = load_sensor_pairs(set_name)
    return data.get("eval_pairs", data.get("pairs", []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor pair generation utility.")
    parser.add_argument("--s6", action="store_true", help="Generate S6 validation, test, and diagnostic pair sets.")
    parser.add_argument("--seed", type=int, default=VAL_PAIRS_SEED, help="Random seed for sampling.")
    args = parser.parse_args()

    if args.s6:
        generate_and_save_s6_pairs(seed=args.seed)
    else:
        save_sensor_pairs(seed=args.seed)
