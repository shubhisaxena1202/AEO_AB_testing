# CLAUDE.md — LLM Eval & Experimentation Harness (demo: AEO)

> Standing context for Claude Code. This project is a **reusable evaluation + experimentation harness for LLM features**. Answer Engine Optimization (AEO) is the *first task* plugged into it — the showcase demo, not the point. Build the kitchen, not the meal.

---

## 0. Learning mode (read first)

This is a learning project — I want to understand how RAG, evals, and agents work under the hood.

- **Before writing any component**, explain the concept and the reasoning behind the design choice in plain language (analogies welcome).
- **After each component**, append a dated entry to `docs/LEARNING_LOG.md` explaining the concept it taught me.
- **Log every architectural decision** in `docs/DECISIONS.md`: the decision, options considered, why we chose one.
- We build **one component at a time**. You explain before you code; I approve each step.

## 1. One-line pitch

A lightweight platform to evaluate LLM features: define a metric, run A/B variants of a task, score outputs with an **LLM-as-judge**, get **statistical significance + power analysis**, and view results in a dashboard. Demoed on AEO (does optimized content get recommended more by an AI answer engine?).

## 2. Why this design (the role it's built for)

This maps to three things a platform-minded data org cares about:
1. **Experimentation platform** — infra that lets *anyone* run a clean A/B test, not a one-off analysis.
2. **AI eval framework** — measuring whether an LLM feature actually works (LLM-as-judge + rigorous stats).
3. **LLM analytics** — understanding interaction quality (we include a simulated acceptance-style metric; flagged as synthetic).

The architecture is therefore **generic and pluggable**: tasks, evaluators, and metrics are interfaces. AEO is one implementation. Summarization or code-gen could drop in without touching the core.

## 3. Core abstractions (the platform spine)

Build these as interfaces first; AEO is a concrete implementation of each.

- **`Task`** — given an `input` (e.g. a user query) and a `variant` (control/treatment config), produce an `output` (the LLM's answer). AEO's task = RAG answer over a corpus containing the variant doc.
- **`Variant`** — a named arm (A/control, B/treatment). For AEO: which content version sits in the corpus.
- **`Evaluator`** — given `(input, output)`, return structured scores. Implemented via **LLM-as-judge**. AEO evaluator returns `{recommended: bool, position: int|None, accepted: bool, reasoning: str}`.
- **`Metric`** — reduces evaluator outputs to a number per arm (e.g. recommendation rate, acceptance rate).
- **`ExperimentRunner`** — loops `inputs × variants × repeats`, calls Task + Evaluator, writes `results.csv`.
- **`Analysis`** — significance tests, lift + CI, power analysis over `results.csv`.
- **`Dashboard`** — reads `results.csv`, shows per-arm metrics, lift with CI, and the chart.

Keeping these generic is the whole point — it signals platform thinking, not a single script.

## 4. The demo task: AEO (parallel-arms design)

We can't A/B test ChatGPT/Gemini's real ranker — we don't own it. So we build a RAG answer engine we **do** own and test content variants inside it (a "wind tunnel").

- **Variant A (control):** normal marketing-style content for our product.
- **Variant B (treatment):** AEO-optimized — answer-shaped sentences, FAQ structure, concrete stats, chunk-friendly formatting.
- **Parallel arms / between-corpus:**
  - `Corpus_A` = [variant A doc] + [N fixed competitor distractor docs]
  - `Corpus_B` = [variant B doc] + [the *same* N distractors]
  - Only the treatment doc differs → any change in recommendation rate is attributable to the optimization.
- **Outcome (binary):** was our product recommended in the answer? (clean to judge).
- **Domain:** SaaS tool comparison — **project management tools**. Our product is the test entity; 4–6 fixed competitor docs are distractors.

**Honest caveats (put in README):** real = LLM calls, retrieval, content, judging, stats. Synthetic = queries, corpus, and the acceptance signal. This is a methodology demo, not a claim of real-world lift.

## 5. Experimental rigor (the credibility center)

- **Trial = (query × arm × repeat).** Repeats (temperature ~0.7) handle LLM stochasticity.
- **Two layers of stats:**
  1. *Headline:* per-trial two-proportion z-test → p_a, p_b, lift, CI.
  2. *Correct version:* same queries run both arms → **paired** design. Aggregate repeats to a per-query rate, run a **paired t-test** (handles within-query correlation). Lead with this.
- **Power analysis:** given baseline rate (~0.3) and a minimum detectable effect (e.g. +20% relative), compute required queries. This is the strongest competence signal.
- **Pre-register** metric + n in the README *before* looking at results.
- **Scale (cheap):** ~40 queries × 5 repeats × 2 arms ≈ 400 answer + 400 judge calls on a small model.

## 6. Repo structure

```
<repo>/
├── README.md                 # platform pitch → AEO demo → results + chart → caveats
├── CLAUDE.md                 # this file
├── requirements.txt
├── .env.example
├── config.yaml               # task, domain, model, n_queries, n_repeats, temperature, top_k
├── docs/
│   ├── DECISIONS.md          # decision records (options + why)
│   └── LEARNING_LOG.md       # dated concept diary
├── data/
│   ├── content/
│   │   ├── variant_a.md
│   │   ├── variant_b.md
│   │   └── distractors/      # fixed competitor docs (same in both arms)
│   ├── queries.json
│   └── results/              # results.csv (gitignore big artifacts)
├── core/                     # the generic platform spine
│   ├── interfaces.py         # Task, Variant, Evaluator, Metric (abstract)
│   ├── runner.py             # ExperimentRunner
│   ├── stats.py              # two-proportion, paired test, lift CI, power
│   └── store.py              # write/read results.csv with run metadata
├── tasks/
│   └── aeo/                  # AEO implementation of the interfaces
│       ├── generate_content.py   # GenAI: A/B variants from a brief
│       ├── generate_queries.py   # GenAI: synthetic queries
│       ├── corpus.py             # chunk + embed + build per-arm index
│       ├── answer_engine.py      # RAG Task: retrieve top-k + LLM answer
│       └── judge.py              # AEO Evaluator (LLM-as-judge)
├── dashboard/
│   └── app.py                # Streamlit (or static HTML) over results.csv
├── scripts/
│   └── run_experiment.sh
└── tests/
    └── test_stats.py         # unit-test stats math vs known values
```

## 7. Tech defaults (swap freely)

- **LLM + embeddings:** OpenAI `gpt-4o-mini` (answers + judge), `text-embedding-3-small`. One provider to start.
- **Vector store:** in-memory cosine over numpy first (no infra). FAISS/Chroma only if showcasing.
- **Stats:** `statsmodels` + `scipy`. **Dashboard:** Streamlit. **Config-driven:** every knob in `config.yaml`.

## 8. `core/stats.py` — stubbed core

```python
import numpy as np
from scipy import stats as sp
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower


def two_proportion_test(cited_a, n_a, cited_b, n_b):
    """Per-trial headline test: is B's rate different from A's?"""
    z, p = proportions_ztest([cited_b, cited_a], [n_b, n_a])
    p_a, p_b = cited_a / n_a, cited_b / n_b
    lift = (p_b - p_a) / p_a if p_a > 0 else float("nan")
    return {"p_a": p_a, "p_b": p_b, "z": z, "p_value": p, "lift": lift}


def lift_ci_bootstrap(arm_a, arm_b, iters=10000):
    """Bootstrap 95% CI on relative lift. arm_x = 1/0 outcomes."""
    rng = np.random.default_rng(0)
    lifts = []
    for _ in range(iters):
        a = rng.choice(arm_a, len(arm_a), replace=True).mean()
        b = rng.choice(arm_b, len(arm_b), replace=True).mean()
        if a > 0:
            lifts.append((b - a) / a)
    lo, hi = np.percentile(lifts, [2.5, 97.5])
    return {"lift_ci_low": lo, "lift_ci_high": hi}


def paired_query_test(rate_a_per_query, rate_b_per_query):
    """Rigor version: same queries both arms => paired t-test on per-query rates."""
    t, p = sp.ttest_rel(rate_b_per_query, rate_a_per_query)
    return {"t": t, "p_value": p,
            "mean_a": float(np.mean(rate_a_per_query)),
            "mean_b": float(np.mean(rate_b_per_query))}


def required_n(baseline_rate, mde_relative, alpha=0.05, power=0.8):
    """Queries per arm to detect a relative lift of `mde_relative`."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde_relative)
    effect = proportion_effectsize(p2, p1)
    n = NormalIndPower().solve_power(effect_size=effect, alpha=alpha,
                                     power=power, alternative="two-sided")
    return int(np.ceil(n))
```

`tests/test_stats.py` checks `required_n` and `two_proportion_test` against hand-computed values.

## 9. Component contracts

- `tasks/aeo/generate_content.py` → product brief + AEO techniques ⇒ `variant_a.md`, `variant_b.md` (differ only in optimization, not facts).
- `tasks/aeo/generate_queries.py` → `queries.json` (realistic PM-tool questions).
- `tasks/aeo/corpus.py` → `build_corpus(treatment_doc, distractors) -> index`.
- `tasks/aeo/answer_engine.py` → implements `Task`: `answer(query, index, top_k) -> {answer_text, retrieved}`.
- `tasks/aeo/judge.py` → implements `Evaluator`: strict-JSON rubric ⇒ `{recommended, position, accepted, reasoning}`.
- `core/runner.py` → loops queries × {A,B} × repeats ⇒ `results.csv` columns: `query_id, arm, repeat, recommended, position, accepted`.
- `core/stats.py` / `dashboard/app.py` → consume `results.csv`.

## 10. Weekend sequence

**Saturday — platform spine + AEO task**
1. Scaffold repo (section 6) + `config.yaml` + `.env` + `docs/`. Git init.
2. `core/interfaces.py` — define Task/Variant/Evaluator/Metric. (Explain why interfaces = platform thinking.)
3. `core/stats.py` + `tests/test_stats.py` (get the math provably right early).
4. AEO content: brief → `variant_a.md`, `variant_b.md` + 4–6 distractors.
5. `generate_queries.py` → ~40 PM-tool queries.
6. `corpus.py` + `answer_engine.py` → single query → answer end to end.
7. `judge.py` → verify on an obvious recommend vs non-recommend case.

**Sunday — run, analyze, ship**
8. `core/runner.py` → full loop → `results.csv` (watch cost/rate limits).
9. Analysis: lift, two-proportion + paired test, power analysis. Run tests.
10. `dashboard/app.py` → per-arm rates, lift+CI, chart.
11. `README.md`: platform pitch → AEO demo → results table → chart → honest caveats.
12. LinkedIn draft.

## 11. Risks & mitigations

- **Judge unreliability** → strict JSON rubric, validate on hand-labeled cases first, log reasoning.
- **Correlated repeats inflating significance** → that's why the paired per-query test leads.
- **Variants differing in facts not form** → keep claims identical; only structure/phrasing changes.
- **Cost creep** → small model + modest n; one number in `config.yaml`.
- **Overclaiming pillar 3** → acceptance metric is simulated; say so.

## 12. LinkedIn / outreach framing

- **Hook:** "I built a lightweight eval + experimentation harness for LLM features."
- **What:** define a metric → run A/B variants → score with LLM-as-judge → significance + power → dashboard.
- **Demo:** Answer Engine Optimization — did optimized content get recommended more? "X% lift, p = …, 95% CI …" + chart.
- **Platform angle:** tasks/evaluators are pluggable — AEO today, summarization or code-gen tomorrow.
- **Honest footer:** synthetic traffic, single answer engine, simulated acceptance — methodology, repo linked.
- **CTA:** repo's open.

## 13. Stretch (only if time)

- Plug in a second task (summarization eval) to prove the platform is generic — the single highest-signal stretch.
- Ablate which AEO technique drives lift (FAQ vs stats vs answer-shaping) as separate arms.
- Multi-model comparison (test 2–3 answer engines).