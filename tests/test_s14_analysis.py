"""Unit tests for S14 analysis statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.s14_analysis import (
    _rank,
    bootstrap_median_diff_ci,
    bootstrap_spearman_ci,
    permutation_p_value,
    spearman_rank_correlation,
)


class TestSpearmanRankCorrelation:
    """Test Spearman rank correlation computation."""

    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        rho = spearman_rank_correlation(x, y)
        assert abs(rho - 1.0) < 1e-12

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        rho = spearman_rank_correlation(x, y)
        assert abs(rho - (-1.0)) < 1e-12

    def test_zero_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
        rho = spearman_rank_correlation(x, y)
        # Known Spearman for this permutation: 1 - 6*10/(5*24) = 1 - 0.5 = 0.5
        # Let me compute by hand: ranks of x = [1,2,3,4,5], ranks of y = [3,1,5,2,4]
        # d = [-2,1,-2,2,1], d^2 = [4,1,4,4,1] = 14
        # rho = 1 - 6*14/(5*24) = 1 - 84/120 = 1 - 0.7 = 0.3
        assert abs(rho - 0.3) < 1e-12

    def test_ties(self):
        x = np.array([1.0, 2.0, 2.0, 4.0])
        y = np.array([1.0, 3.0, 2.0, 4.0])
        rho = spearman_rank_correlation(x, y)
        # With ties, rank of x = [1, 2.5, 2.5, 4], rank of y = [1, 3, 2, 4]
        # d = [0, -0.5, 0.5, 0], d^2 = [0, 0.25, 0.25, 0] = 0.5
        # rho = 1 - 6*0.5/(4*15) = 1 - 3/60 = 1 - 0.05 = 0.95
        assert abs(rho - 0.95) < 1e-12

    def test_determinism(self):
        rng = np.random.RandomState(99)
        x = rng.randn(20)
        y = rng.randn(20)
        r1 = spearman_rank_correlation(x, y)
        r2 = spearman_rank_correlation(x, y)
        assert r1 == r2


class TestRanking:
    """Test the internal ranking function."""

    def test_no_ties(self):
        arr = np.array([3.0, 1.0, 2.0])
        ranks = _rank(arr)
        np.testing.assert_array_equal(ranks, [3.0, 1.0, 2.0])

    def test_all_ties(self):
        arr = np.array([1.0, 1.0, 1.0])
        ranks = _rank(arr)
        np.testing.assert_array_equal(ranks, [2.0, 2.0, 2.0])


class TestBootstrapCI:
    """Test bootstrap CI for Spearman."""

    def test_perfect_correlation_narrow_ci(self):
        x = np.arange(20, dtype=float)
        y = x * 2.0 + 1.0
        ci_lo, ci_hi = bootstrap_spearman_ci(x, y)
        assert ci_lo > 0.9
        assert ci_hi <= 1.0 + 1e-12

    def test_determinism(self):
        rng = np.random.RandomState(42)
        x = rng.randn(20)
        y = rng.randn(20)
        ci1 = bootstrap_spearman_ci(x, y)
        ci2 = bootstrap_spearman_ci(x, y)
        assert ci1 == ci2


class TestPermutationTest:
    """Test permutation p-value."""

    def test_perfect_correlation_low_p(self):
        x = np.arange(20, dtype=float)
        y = x * 3.0
        p = permutation_p_value(x, y, n_perm=1000, seed=42)
        assert p < 0.01

    def test_no_correlation_high_p(self):
        # Random independent data should have high p
        rng = np.random.RandomState(123)
        x = rng.randn(20)
        y = rng.randn(20)
        p = permutation_p_value(x, y, n_perm=1000, seed=42)
        assert p > 0.05

    def test_determinism(self):
        rng = np.random.RandomState(42)
        x = rng.randn(20)
        y = rng.randn(20)
        p1 = permutation_p_value(x, y, n_perm=500, seed=42)
        p2 = permutation_p_value(x, y, n_perm=500, seed=42)
        assert p1 == p2


class TestGroupComparison:
    """Test bootstrap median difference."""

    def test_known_difference(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([1.0, 2.0, 3.0])
        diff, _, _ = bootstrap_median_diff_ci(a, b, n_boot=100, seed=42)
        assert abs(diff - 18.0) < 1e-12  # median(a) - median(b) = 20 - 2

    def test_sign_convention(self):
        """Difference is group_a minus group_b (loses minus wins)."""
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        diff, ci_lo, ci_hi = bootstrap_median_diff_ci(a, b, n_boot=1000, seed=42)
        assert diff > 0  # a > b
        assert ci_lo > 0  # interval should exclude zero for well-separated groups
