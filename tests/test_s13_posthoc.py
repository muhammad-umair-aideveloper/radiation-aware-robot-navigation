"""Unit tests for S13 post-hoc supplementary statistical analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.s13_summary import (
    compute_sign_test,
    exact_binomial_sign_test,
    paired_regret_difference,
    verdict,
)


def test_exact_binomial_sign_test_known_cases() -> None:
    # All-wins (20 wins, 0 losses)
    res_all_wins = exact_binomial_sign_test(20, 0)
    assert res_all_wins["n"] == 20
    assert pytest.approx(res_all_wins["p_value"]) == 2.0 / (2**20)

    # All-losses (0 wins, 20 losses)
    res_all_losses = exact_binomial_sign_test(0, 20)
    assert res_all_losses["n"] == 20
    assert pytest.approx(res_all_losses["p_value"]) == 2.0 / (2**20)

    # Balanced cases
    res_balanced_20 = exact_binomial_sign_test(10, 10)
    assert res_balanced_20["n"] == 20
    assert res_balanced_20["p_value"] == 1.0

    res_balanced_10 = exact_binomial_sign_test(5, 5)
    assert res_balanced_10["n"] == 10
    assert res_balanced_10["p_value"] == 1.0

    # Empty case
    res_empty = exact_binomial_sign_test(0, 0)
    assert res_empty["n"] == 0
    assert res_empty["p_value"] == 1.0

    # Known S13 cases: 16 wins out of 20 (offline)
    # sum_{i=0..4} C(20, i) = 1 + 20 + 190 + 1140 + 4845 = 6196
    # 2 * 6196 / 1048576 = 12392 / 1048576 = 0.01181793212890625
    res_offline = exact_binomial_sign_test(16, 4)
    assert res_offline["n"] == 20
    assert pytest.approx(res_offline["p_value"]) == 12392.0 / 1048576.0

    # Known S13 cases: 14 wins out of 20 (online)
    # sum_{i=0..6} C(20, i) = 6196 + 15504 + 38760 = 60460
    # 2 * 60460 / 1048576 = 120920 / 1048576 = 0.11531829833984375
    res_online = exact_binomial_sign_test(14, 6)
    assert res_online["n"] == 20
    assert pytest.approx(res_online["p_value"]) == 120920.0 / 1048576.0


def test_tie_handling() -> None:
    # Mixed with ties: 3 positive, 1 negative, 2 ties
    diffs = [10.0, 5.0, 2.0, -4.0, 0.0, 0.0]
    res = compute_sign_test(diffs)
    assert res["wins"] == 3
    assert res["losses"] == 1
    assert res["ties"] == 2
    assert res["n"] == 4
    assert pytest.approx(res["p_value"]) == 10.0 / 16.0

    # All ties
    diffs_all_ties = [0.0, 0.0, 0.0]
    res_ties = compute_sign_test(diffs_all_ties)
    assert res_ties["wins"] == 0
    assert res_ties["losses"] == 0
    assert res_ties["ties"] == 3
    assert res_ties["n"] == 0
    assert res_ties["p_value"] == 1.0

    # Balanced with ties: 4 wins, 4 losses, 3 ties
    diffs_balanced_ties = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0]
    res_bt = compute_sign_test(diffs_balanced_ties)
    assert res_bt["wins"] == 4
    assert res_bt["losses"] == 4
    assert res_bt["ties"] == 3
    assert res_bt["n"] == 8
    assert res_bt["p_value"] == 1.0


def test_sign_convention_difference() -> None:
    # Convention is kernel minus omp3: positive when kernel regret > omp3 regret (omp3 is better)
    kernel_regret = [50.0, 40.0, 30.0]
    omp3_regret = [20.0, 10.0, 25.0]
    res = paired_regret_difference(kernel_regret, omp3_regret, seed=42, resamples=1000)
    assert pytest.approx(res["mean"]) == float(np.mean([30.0, 30.0, 5.0]))
    assert pytest.approx(res["median"]) == float(np.median([30.0, 30.0, 5.0]))
    assert res["mean"] > 0
    assert res["median"] > 0
    assert res["ci_low"] <= res["mean"] <= res["ci_high"]


def test_bootstrap_determinism() -> None:
    k = [10.0, 25.0, 30.0, 15.0, 50.0]
    o = [5.0, 20.0, 10.0, 18.0, 20.0]
    res1 = paired_regret_difference(k, o, seed=42, resamples=5000)
    res2 = paired_regret_difference(k, o, seed=42, resamples=5000)
    assert res1 == res2
    assert res1["ci_low"] <= res1["mean"] <= res1["ci_high"]


def test_h1_to_h3_verdicts_unchanged() -> None:
    summary_path = ROOT / "results" / "s13_summary.json"
    assert summary_path.exists(), "results/s13_summary.json must exist"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    verdicts = summary["hypothesis_verdicts"]
    assert verdicts["H1"] == "not supported"
    assert verdicts["H2"] == "supported"
    assert verdicts["H3"] == "supported"

    counts = summary["paired_counts"]
    assert counts["online_omp3_beats_kernel_layouts"] == 14
    assert counts["oracle_saving_at_least_10_layouts"] == 19
    assert counts["kernel_online_regret_over_50_layouts"] == 16

    assert verdict(14, 15) == "not supported"
    assert verdict(19, 15) == "supported"
    assert verdict(16, 15) == "supported"
