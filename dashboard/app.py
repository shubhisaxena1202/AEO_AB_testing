"""
Streamlit dashboard for the AEO A/B experiment.
Run with:  streamlit run dashboard/app.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st
import yaml
from scipy import stats as sp
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

# ── repo root on path so core/ is importable ──────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEO A/B Experiment",
    page_icon="📊",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_results(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_stats(results: list[dict]) -> dict:
    arm_a = [r for r in results if r["arm"] == "A"]
    arm_b = [r for r in results if r["arm"] == "B"]

    outcomes_a = np.array([int(r["recommended"]) for r in arm_a])
    outcomes_b = np.array([int(r["recommended"]) for r in arm_b])

    n_a, n_b = len(outcomes_a), len(outcomes_b)
    cited_a, cited_b = outcomes_a.sum(), outcomes_b.sum()
    rate_a = cited_a / n_a
    rate_b = cited_b / n_b
    lift = (rate_b - rate_a) / rate_a if rate_a > 0 else float("nan")

    # Two-proportion z-test
    _, p_z = proportions_ztest([cited_b, cited_a], [n_b, n_a])

    # Paired t-test
    by_a = defaultdict(list)
    by_b = defaultdict(list)
    for r in results:
        (by_a if r["arm"] == "A" else by_b)[r["query_id"]].append(int(r["recommended"]))
    query_ids = sorted(by_a.keys())
    rate_a_q = np.array([np.mean(by_a[q]) for q in query_ids])
    rate_b_q = np.array([np.mean(by_b[q]) for q in query_ids])
    t_stat, p_t = sp.ttest_rel(rate_b_q, rate_a_q)

    # Bootstrap CI on lift
    rng = np.random.default_rng(42)
    lifts = []
    for _ in range(10_000):
        a = rng.choice(outcomes_a, len(outcomes_a), replace=True).mean()
        b = rng.choice(outcomes_b, len(outcomes_b), replace=True).mean()
        if a > 0:
            lifts.append((b - a) / a)
    lifts = np.array(lifts)
    ci_low, ci_high = np.percentile(lifts, [2.5, 97.5]) if len(lifts) else (float("nan"), float("nan"))

    return {
        "rate_a": rate_a, "rate_b": rate_b,
        "n_a": n_a, "n_b": n_b,
        "cited_a": cited_a, "cited_b": cited_b,
        "lift": lift,
        "p_z": p_z, "p_t": p_t, "t_stat": t_stat,
        "ci_low": ci_low, "ci_high": ci_high,
        "bootstrapped_lifts": lifts,
        "query_ids": query_ids,
        "rate_a_q": rate_a_q,
        "rate_b_q": rate_b_q,
        "diffs": rate_b_q - rate_a_q,
    }


def power_required(baseline, mde_rel, alpha=0.05, power=0.80):
    p2 = baseline * (1 + mde_rel)
    es = proportion_effectsize(p2, baseline)
    return int(np.ceil(NormalIndPower().solve_power(
        effect_size=es, alpha=alpha, power=power, alternative="two-sided"
    )))


def power_at_n(n, baseline, mde_rel, alpha=0.05):
    p2 = baseline * (1 + mde_rel)
    es = proportion_effectsize(p2, baseline)
    return NormalIndPower().solve_power(
        effect_size=es, alpha=alpha, nobs1=n, alternative="two-sided"
    )


# ── load data ─────────────────────────────────────────────────────────────────
config = load_config()
results_path = ROOT / config["paths"]["results"]

if not results_path.exists():
    st.error("No results file found. Run `python -m core.runner` first.")
    st.stop()

results = load_results(str(results_path))
s = compute_stats(results)

baseline = config["baseline_citation_rate"]
mde_rel  = config["min_detectable_effect"]
n_queries = len(s["query_ids"])
req_n     = power_required(baseline, mde_rel)
pilot_power = power_at_n(n_queries, baseline, mde_rel)

# ── layout ────────────────────────────────────────────────────────────────────
st.title("📊 AEO A/B Experiment — Results Dashboard")
st.caption(
    "Does AEO-optimized content get recommended more by an LLM answer engine? "
    f"Domain: **{config['domain']}** · Product: **{config['target_product_name']}**"
)
st.divider()

# ── SECTION 1: Key metrics ────────────────────────────────────────────────────
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

sig = s["p_t"] < 0.05
lift_pct = f"{s['lift']:+.1%}"
p_display = f"{s['p_t']:.3f}"

col1.metric(
    "Arm A — Control",
    f"{s['rate_a']:.1%}",
    help=f"{s['cited_a']} recommended out of {s['n_a']} trials",
)
col2.metric(
    "Arm B — Treatment (AEO)",
    f"{s['rate_b']:.1%}",
    delta=lift_pct,
    delta_color="normal",
    help=f"{s['cited_b']} recommended out of {s['n_b']} trials",
)
col3.metric(
    "Paired t-test p-value",
    p_display,
    delta="significant ✓" if sig else "not significant",
    delta_color="normal" if sig else "off",
    help="p < 0.05 = statistically significant",
)
col4.metric(
    "Bootstrap 95% CI",
    f"[{s['ci_low']:+.1%}, {s['ci_high']:+.1%}]",
    help="Range of plausible true lifts",
)

st.divider()

# ── SECTION 2: Significance + Power ──────────────────────────────────────────
col_sig, col_power = st.columns([1, 1])

with col_sig:
    st.subheader("Significance Tests")

    verdict_color = "green" if sig else "orange"
    verdict_text  = "Significant — unlikely to be noise" if sig else "Not significant — cannot rule out chance"
    st.markdown(f"**Verdict:** :{verdict_color}[{verdict_text}]")

    st.markdown(f"""
| Test | Statistic | p-value | Result |
|---|---|---|---|
| Two-proportion z-test *(naive)* | z = {s['t_stat']:.2f} | {s['p_z']:.4f} | {'✓ sig.' if s['p_z'] < 0.05 else '✗ not sig.'} |
| Paired t-test *(rigorous)* | t = {s['t_stat']:.2f} | {s['p_t']:.4f} | {'✓ sig.' if s['p_t'] < 0.05 else '✗ not sig.'} |
""")
    st.caption(
        "The paired test is the lead result — it accounts for within-query correlation "
        "by comparing per-query rates rather than treating all trials as independent."
    )

with col_power:
    st.subheader("Power Analysis")

    st.markdown(f"""
| Parameter | Value |
|---|---|
| Baseline recommendation rate | {baseline:.0%} |
| Minimum detectable lift | +{mde_rel:.0%} relative |
| Target power | 80% |
| Required queries / arm | **{req_n}** |
| This pilot | **{n_queries}** queries/arm |
| Power of this pilot | **{pilot_power:.0%}** |
""")
    st.caption(
        f"At {n_queries} queries/arm, the pilot has only {pilot_power:.0%} power — "
        f"it would miss a {mde_rel:.0%} relative lift ~{(1-pilot_power):.0%} of the time. "
        "This run demonstrates methodology, not a powered conclusion."
    )

st.divider()

# ── SECTION 3: Bootstrap CI chart ─────────────────────────────────────────────
import matplotlib.pyplot as plt

st.subheader("Bootstrap Distribution of Lift")
st.caption("Each bar is one of 10,000 resampled experiments. The spread shows uncertainty; the red dashes are the 95% CI.")

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(s["bootstrapped_lifts"], bins=80, color="#0d6efd", alpha=0.75, edgecolor="none")
ax.axvline(s["lift"],    color="black",   lw=2,   linestyle="-",  label=f"Observed lift = {s['lift']:+.1%}")
ax.axvline(s["ci_low"],  color="#d9534f", lw=2,   linestyle="--", label=f"95% CI: {s['ci_low']:+.1%} to {s['ci_high']:+.1%}")
ax.axvline(s["ci_high"], color="#d9534f", lw=2,   linestyle="--")
ax.axvline(0,            color="gray",    lw=1.5, linestyle=":",  label="Zero lift")
ax.set_xlabel("Bootstrapped relative lift (B vs A)")
ax.set_ylabel("Count")
ax.set_title("Bootstrap Sampling Distribution of Lift (10,000 resamples)")
ax.legend(fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── SECTION 4: Per-query breakdown ────────────────────────────────────────────
st.subheader("Per-Query Breakdown")
st.caption(
    "Each point / bar is one of the 40 queries. "
    "Blue = Arm B outperformed A on that query; red = A outperformed B."
)

fig2, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Waterfall: B - A per query
colors = ["#0d6efd" if d > 0 else "#d9534f" if d < 0 else "#adb5bd" for d in s["diffs"]]
axes[0].bar(range(len(s["diffs"])), s["diffs"], color=colors, width=0.7)
axes[0].axhline(0, color="black", lw=1)
axes[0].axhline(s["diffs"].mean(), color="orange", lw=2, linestyle="--",
                label=f"Mean diff = {s['diffs'].mean():+.3f}")
axes[0].set_xlabel("Query index")
axes[0].set_ylabel("rate_B − rate_A")
axes[0].set_title("Per-Query Difference (B − A)\nBlue = B won · Red = A won")
axes[0].legend(fontsize=9)

# Scatter: rate_A vs rate_B
axes[1].scatter(s["rate_a_q"], s["rate_b_q"], c=colors, s=65,
                alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
diag = np.linspace(0, 1, 100)
axes[1].plot(diag, diag, "k--", lw=1, label="A = B (no lift)")
axes[1].set_xlabel("rate_A (control)")
axes[1].set_ylabel("rate_B (treatment)")
axes[1].set_title("Rate A vs Rate B per Query\nAbove diagonal = B better")
axes[1].set_xlim(-0.05, 1.05)
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(fontsize=9)

b_wins = int((s["diffs"] > 0).sum())
a_wins = int((s["diffs"] < 0).sum())
ties   = int((s["diffs"] == 0).sum())
st.caption(f"Query-level wins — B > A: **{b_wins}** · A > B: **{a_wins}** · Tied: **{ties}**")

plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.divider()

# ── SECTION 5: Raw results table ─────────────────────────────────────────────
st.subheader("Raw Trial Data")

col_f1, col_f2, col_f3 = st.columns(3)
arm_filter   = col_f1.selectbox("Filter by arm", ["All", "A (control)", "B (treatment)"])
rec_filter   = col_f2.selectbox("Filter by recommended", ["All", "Yes", "No"])
query_filter = col_f3.selectbox("Filter by query", ["All"] + sorted({r["query_id"] for r in results}))

filtered = results
if arm_filter != "All":
    arm_code = "A" if "A" in arm_filter else "B"
    filtered = [r for r in filtered if r["arm"] == arm_code]
if rec_filter == "Yes":
    filtered = [r for r in filtered if r["recommended"] == "1"]
elif rec_filter == "No":
    filtered = [r for r in filtered if r["recommended"] == "0"]
if query_filter != "All":
    filtered = [r for r in filtered if r["query_id"] == query_filter]

st.caption(f"Showing {len(filtered)} of {len(results)} trials")
st.dataframe(
    filtered,
    use_container_width=True,
    column_order=["query_id", "arm", "repeat", "recommended", "position", "accepted", "reasoning"],
    height=400,
)

# ── SECTION 6: Honest caveats ─────────────────────────────────────────────────
st.divider()
with st.expander("Honest caveats about this experiment"):
    st.markdown(f"""
**What is real:**
- LLM calls (answer generation, LLM-as-judge)
- Retrieval via cosine similarity over real embeddings
- Two-layer statistical testing (z-test + paired t-test)
- Power analysis with a pre-registered MDE

**What is synthetic / simulated:**
- Queries (`data/queries.json`) — generated, not from real users
- Corpus — fictional company docs, not real competitor content
- `accepted` field — defined as `recommended AND position == 1`; does not reflect real user behaviour

**What limits the conclusion:**
- Pilot sample of {n_queries} queries/arm has only ~{pilot_power:.0%} power
- A powered study would need ~{req_n} queries/arm
- Single answer engine (our RAG); real engines (ChatGPT, Gemini) may behave differently

This is a **methodology demonstration**, not a claim of real-world lift.
""")
