import numpy as np
import pytest
from core.stats import (
    lift_ci_bootstrap,
    paired_query_test,
    required_n,
    two_proportion_test,
)


# ---------------------------------------------------------------------------
# two_proportion_test
# ---------------------------------------------------------------------------

class TestTwoProportionTest:
    def test_lift_formula_exact(self):
        # Hand-computed: p_a=0.10, p_b=0.30 → lift = (0.30-0.10)/0.10 = 2.0
        result = two_proportion_test(cited_a=10, n_a=100, cited_b=30, n_b=100)
        assert result["p_a"] == pytest.approx(0.10)
        assert result["p_b"] == pytest.approx(0.30)
        assert result["lift"] == pytest.approx(2.0)

    def test_clearly_significant(self):
        # 10% vs 30% citation rate over 200 trials each — must reject at p<0.001
        result = two_proportion_test(cited_a=20, n_a=200, cited_b=60, n_b=200)
        assert result["p_value"] < 0.001
        assert result["z"] > 0  # positive z means B > A

    def test_clearly_not_significant(self):
        # 15% vs 17% — tiny difference, small sample — must not reject
        result = two_proportion_test(cited_a=15, n_a=100, cited_b=17, n_b=100)
        assert result["p_value"] > 0.05

    def test_zero_baseline_returns_nan_lift(self):
        result = two_proportion_test(cited_a=0, n_a=100, cited_b=10, n_b=100)
        assert np.isnan(result["lift"])

    def test_equal_rates_not_significant(self):
        result = two_proportion_test(cited_a=30, n_a=100, cited_b=30, n_b=100)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["lift"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# required_n (power analysis)
# ---------------------------------------------------------------------------

class TestRequiredN:
    def test_returns_positive_integer(self):
        n = required_n(baseline_rate=0.30, mde_relative=0.20)
        assert isinstance(n, int)
        assert n > 0

    def test_plausible_range_for_our_experiment(self):
        # baseline=0.30, 20% relative lift → Cohen's h ≈ 0.128.
        # NormalIndPower returns n_per_arm using effective-n = n/2, so n ≈ 2*(481) ≈ 963.
        # This is queries per arm needed; our 40-query pilot is intentionally underpowered.
        n = required_n(baseline_rate=0.30, mde_relative=0.20)
        assert 850 < n < 1100

    def test_larger_mde_requires_fewer_queries(self):
        # Larger detectable effect = stronger signal = less data needed
        n_small = required_n(baseline_rate=0.30, mde_relative=0.10)
        n_large = required_n(baseline_rate=0.30, mde_relative=0.40)
        assert n_small > n_large

    def test_higher_power_requires_more_queries(self):
        # Wanting to detect the effect 90% of the time costs more than 80%
        n_80 = required_n(baseline_rate=0.30, mde_relative=0.20, power=0.80)
        n_90 = required_n(baseline_rate=0.30, mde_relative=0.20, power=0.90)
        assert n_90 > n_80

    def test_stricter_alpha_requires_more_queries(self):
        # Tighter false-positive budget means we need more evidence
        n_05 = required_n(baseline_rate=0.30, mde_relative=0.20, alpha=0.05)
        n_01 = required_n(baseline_rate=0.30, mde_relative=0.20, alpha=0.01)
        assert n_01 > n_05


# ---------------------------------------------------------------------------
# paired_query_test
# ---------------------------------------------------------------------------

class TestPairedQueryTest:
    def test_detects_consistent_lift(self):
        # Every query improves by exactly +0.20 — paired t must be highly significant
        rng = np.random.default_rng(42)
        rate_a = rng.uniform(0.1, 0.7, size=40)
        rate_b = np.clip(rate_a + 0.20, 0.0, 1.0)
        result = paired_query_test(rate_a, rate_b)
        assert result["p_value"] < 0.001
        assert result["mean_b"] > result["mean_a"]
        assert result["mean_diff"] == pytest.approx(0.20, abs=1e-6)

    def test_no_lift_not_significant(self):
        # Identical rates → zero variance in differences → degenerate case returns p=1
        rate = np.array([0.2, 0.4, 0.6, 0.3, 0.5] * 8)
        result = paired_query_test(rate, rate.copy())
        assert result["p_value"] == pytest.approx(1.0)
        assert result["mean_diff"] == pytest.approx(0.0)
        assert result["t"] == pytest.approx(0.0)

    def test_ci_contains_true_diff(self):
        # True mean diff ≈ 0.15; add noise so diffs have nonzero variance (CI has width).
        # Without noise all 40 diffs are identical → std=0 → CI collapses to a point.
        rng = np.random.default_rng(7)
        rate_a = rng.uniform(0.1, 0.7, size=40)
        noise = rng.normal(0, 0.05, size=40)
        rate_b = np.clip(rate_a + 0.15 + noise, 0.0, 1.0)
        result = paired_query_test(rate_a, rate_b)
        assert result["diff_ci_low"] < result["diff_ci_high"]
        assert result["mean_diff"] == pytest.approx(0.15, abs=0.05)

    def test_mean_diff_equals_mean_b_minus_mean_a(self):
        rng = np.random.default_rng(0)
        rate_a = rng.uniform(0.2, 0.6, size=40)
        rate_b = rng.uniform(0.2, 0.6, size=40)
        result = paired_query_test(rate_a, rate_b)
        assert result["mean_diff"] == pytest.approx(result["mean_b"] - result["mean_a"])


# ---------------------------------------------------------------------------
# lift_ci_bootstrap
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_ci_is_ordered(self):
        rng = np.random.default_rng(0)
        arm_a = rng.binomial(1, 0.30, size=200).astype(float)
        arm_b = rng.binomial(1, 0.40, size=200).astype(float)
        result = lift_ci_bootstrap(arm_a, arm_b, iters=3000)
        assert result["lift_ci_low"] < result["lift_ci_high"]

    def test_larger_effect_shifts_ci_up(self):
        # Fixed arm_a; larger arm_b lift should produce higher CI bounds
        arm_a = np.array([0.0] * 100 + [1.0] * 100)      # p=0.50
        arm_b_small = np.array([0.0] * 90 + [1.0] * 110)  # p=0.55, +10% lift
        arm_b_large = np.array([0.0] * 50 + [1.0] * 150)  # p=0.75, +50% lift
        r_small = lift_ci_bootstrap(arm_a, arm_b_small, iters=2000)
        r_large = lift_ci_bootstrap(arm_a, arm_b_large, iters=2000)
        assert r_large["lift_ci_low"] > r_small["lift_ci_low"]
        assert r_large["lift_point"] > r_small["lift_point"]

    def test_zero_arm_a_returns_nan(self):
        # All arm_a zeros → every bootstrap sample a=0 → no valid lifts collected.
        # Guard must return NaN rather than crash on np.percentile([]).
        arm_a = np.zeros(100)
        arm_b = np.ones(100)
        result = lift_ci_bootstrap(arm_a, arm_b, iters=100)
        assert np.isnan(result["lift_point"])
        assert np.isnan(result["lift_ci_low"])
        assert np.isnan(result["lift_ci_high"])
