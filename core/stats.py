import numpy as np
from scipy import stats as sp
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower


def two_proportion_test(cited_a: int, n_a: int, cited_b: int, n_b: int) -> dict:
    # NAIVE: treats all trials as IID. Biased because same queries run both arms.
    # Include as a comparison baseline; lead results with paired_query_test instead.
    count = np.array([cited_b, cited_a])
    nobs = np.array([n_b, n_a])
    z, p = proportions_ztest(count, nobs)
    p_a = cited_a / n_a
    p_b = cited_b / n_b
    lift = (p_b - p_a) / p_a if p_a > 0 else float("nan")
    return {"p_a": p_a, "p_b": p_b, "z": float(z), "p_value": float(p), "lift": lift}


def lift_ci_bootstrap(
    arm_a: np.ndarray, arm_b: np.ndarray, iters: int = 10_000
) -> dict:
    # Bootstrap avoids distributional assumptions that lift (a ratio) violates.
    # Seed=0 makes the CI reproducible across runs.
    rng = np.random.default_rng(0)
    lifts = []
    for _ in range(iters):
        a = rng.choice(arm_a, size=len(arm_a), replace=True).mean()
        b = rng.choice(arm_b, size=len(arm_b), replace=True).mean()
        if a > 0:
            lifts.append((b - a) / a)
    lifts = np.array(lifts)
    if len(lifts) == 0:
        return {"lift_point": float("nan"), "lift_ci_low": float("nan"), "lift_ci_high": float("nan")}
    lo, hi = np.percentile(lifts, [2.5, 97.5])
    p_a = arm_a.mean()
    lift_point = (arm_b.mean() - p_a) / p_a if p_a > 0 else float("nan")
    return {
        "lift_point": float(lift_point),
        "lift_ci_low": float(lo),
        "lift_ci_high": float(hi),
    }


def paired_query_test(
    rate_a_per_query: np.ndarray, rate_b_per_query: np.ndarray
) -> dict:
    # PRIMARY TEST. Input: per-query citation rates (cited / n_repeats), one per query.
    # Paired design eliminates between-query variance: a "hard" query is hard for both
    # arms, so its difficulty cancels out in the per-query difference.
    diffs = rate_b_per_query - rate_a_per_query
    n = len(diffs)
    # Degenerate case: all differences identical → no variance to test.
    if diffs.std(ddof=1) == 0:
        mean_d = float(diffs.mean())
        return {
            "t": 0.0,
            "p_value": 1.0,
            "mean_a": float(rate_a_per_query.mean()),
            "mean_b": float(rate_b_per_query.mean()),
            "mean_diff": mean_d,
            "diff_ci_low": mean_d,
            "diff_ci_high": mean_d,
        }
    t, p = sp.ttest_rel(rate_b_per_query, rate_a_per_query)
    # Use t-distribution for CI (not normal) — correct for finite n_queries.
    t_crit = sp.t.ppf(0.975, df=n - 1)
    se = diffs.std(ddof=1) / np.sqrt(n)
    return {
        "t": float(t),
        "p_value": float(p),
        "mean_a": float(rate_a_per_query.mean()),
        "mean_b": float(rate_b_per_query.mean()),
        "mean_diff": float(diffs.mean()),
        "diff_ci_low": float(diffs.mean() - t_crit * se),
        "diff_ci_high": float(diffs.mean() + t_crit * se),
    }


def required_n(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    # baseline_rate is a design-time ASSUMPTION, not measured data.
    # Cohen's h converts two proportions into a single effect-size number that
    # accounts for the variance-stabilising arcsine transformation.
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde_relative)
    effect = proportion_effectsize(p2, p1)
    n = NormalIndPower().solve_power(
        effect_size=effect, alpha=alpha, power=power, alternative="two-sided"
    )
    return int(np.ceil(n))
