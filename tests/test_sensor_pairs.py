"""Tests for S6 sensor goal pools, pair sets, separations, and constraints."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, GOAL_CELL
from src.sensor_pairs import (
    load_sensor_pairs,
    load_pair_set,
    SENSOR_PAIRS_S6_FILE,
    dist_euclidean,
    min_dist_to_set,
)


def test_sensor_pair_set_sizes() -> None:
    """Check set sizes of all goal pools and evaluation pair sets."""
    s6 = load_sensor_pairs(filepath=SENSOR_PAIRS_S6_FILE)

    assert len(s6["train_goals_30"]) == 30
    assert len(s6["train_goals_150"]) == 150
    assert len(s6["held_out_goals"]) == 20
    assert len(s6["val_goals"]) == 20
    assert len(s6["test_goals"]) == 20

    assert len(s6["held_out_pairs"]) == 100
    assert len(s6["val_pairs"]) == 100
    assert len(s6["test_pairs"]) == 100
    assert len(s6["diag_pairs"]) == 100

    # Also test convenience loader
    assert len(load_pair_set("heldout")) == 100
    assert len(load_pair_set("val")) == 100
    assert len(load_pair_set("test")) == 100
    assert len(load_pair_set("diag")) == 100


def test_goal_sets_pairwise_disjoint() -> None:
    """Verify all goal sets are strictly pairwise disjoint."""
    s6 = load_sensor_pairs(filepath=SENSOR_PAIRS_S6_FILE)

    s_train150 = set(tuple(g) for g in s6["train_goals_150"])
    s_heldout = set(tuple(g) for g in s6["held_out_goals"])
    s_val = set(tuple(g) for g in s6["val_goals"])
    s_test = set(tuple(g) for g in s6["test_goals"])

    assert s_train150.isdisjoint(s_heldout)
    assert s_train150.isdisjoint(s_val)
    assert s_train150.isdisjoint(s_test)
    assert s_heldout.isdisjoint(s_val)
    assert s_heldout.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)


def test_minimum_5_cell_separation_between_sets() -> None:
    """Verify that every new goal maintains at least 5.0 cells Euclidean distance from other sets."""
    s6 = load_sensor_pairs(filepath=SENSOR_PAIRS_S6_FILE)

    train150 = [tuple(g) for g in s6["train_goals_150"]]
    heldout = [tuple(g) for g in s6["held_out_goals"]]
    val_goals = [tuple(g) for g in s6["val_goals"]]
    test_goals = [tuple(g) for g in s6["test_goals"]]
    extra_train = train150[30:]  # 120 new goals

    min_sep = 5.0

    # Validation goals separated from train150, heldout, and test
    for v in val_goals:
        assert min_dist_to_set(v, train150) >= min_sep
        assert min_dist_to_set(v, heldout) >= min_sep
        assert min_dist_to_set(v, test_goals) >= min_sep

    # Test goals separated from train150, heldout, and val
    for t in test_goals:
        assert min_dist_to_set(t, train150) >= min_sep
        assert min_dist_to_set(t, heldout) >= min_sep
        assert min_dist_to_set(t, val_goals) >= min_sep

    # Extra training goals separated from heldout, val, and test
    for e in extra_train:
        assert min_dist_to_set(e, heldout) >= min_sep
        assert min_dist_to_set(e, val_goals) >= min_sep
        assert min_dist_to_set(e, test_goals) >= min_sep


def test_radiation_threshold_starts_and_goals() -> None:
    """Verify starts and goals in S6 pair sets have normalized radiation intensity < 0.05."""
    env = RadiationEnv()
    max_rad = float(np.max(env.radiation))
    s6 = load_sensor_pairs(filepath=SENSOR_PAIRS_S6_FILE)

    # Check goals in all goal sets
    all_goals = (
        s6["train_goals_150"]
        + s6["held_out_goals"]
        + s6["val_goals"]
        + s6["test_goals"]
    )
    for g in all_goals:
        assert (env.radiation[g[0], g[1]] / max_rad) < 0.05

    # Check starts and goals in S6 pair sets
    for set_key in ["val_pairs", "test_pairs", "diag_pairs"]:
        for pair in s6[set_key]:
            s_r, s_c = pair["start"]
            g_r, g_c = pair["goal"]
            assert (env.radiation[s_r, s_c] / max_rad) < 0.05
            assert (env.radiation[g_r, g_c] / max_rad) < 0.05


def test_free_space_distance_rule() -> None:
    """Verify all pairs in evaluation sets have free-space distance >= 20.0."""
    s6 = load_sensor_pairs(filepath=SENSOR_PAIRS_S6_FILE)

    for set_key in ["held_out_pairs", "val_pairs", "test_pairs", "diag_pairs"]:
        for pair in s6[set_key]:
            assert pair["free_space_distance"] >= 20.0


def test_canonical_goal_excluded() -> None:
    """Verify canonical goal (47, 47) is excluded from all goal sets and pair sets."""
    canonical = list(GOAL_CELL)
    s6 = load_sensor_pairs(filepath=SENSOR_PAIRS_S6_FILE)

    assert canonical not in s6["train_goals_150"]
    assert canonical not in s6["held_out_goals"]
    assert canonical not in s6["val_goals"]
    assert canonical not in s6["test_goals"]

    for set_key in ["held_out_pairs", "val_pairs", "test_pairs", "diag_pairs"]:
        for pair in s6[set_key]:
            assert pair["goal"] != canonical
            assert pair["start"] != canonical


if __name__ == "__main__":
    test_sensor_pair_set_sizes()
    test_goal_sets_pairwise_disjoint()
    test_minimum_5_cell_separation_between_sets()
    test_radiation_threshold_starts_and_goals()
    test_free_space_distance_rule()
    test_canonical_goal_excluded()
    print("All sensor pair tests passed!")

