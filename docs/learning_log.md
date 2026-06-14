# Learning Log

Each entry is appended after a component is built. The goal is to explain
the concept the component taught — not what the code does (the code does that),
but *why* the design is what it is and what mental model to carry forward.

---

## Entry 0 — Project setup and experimental design

**Date:** 2026-06-12  
**Component:** Repo scaffold, config, docs

### What is RAG, really?

RAG stands for Retrieval-Augmented Generation. The core insight is that LLMs
have a fixed knowledge cutoff and a limited context window — you can't stuff an
entire knowledge base into a prompt. So instead you:

1. **Store** your documents as dense vector representations (embeddings) in an
   index.
2. At query time, **retrieve** the k most relevant chunks by vector similarity.
3. **Inject** those chunks into the LLM prompt alongside the question, so the
   model answers from your documents rather than from its parametric memory.

The "retrieval" step is what makes this different from plain prompting. It lets
you answer questions over large, updateable, private corpora without fine-tuning.

### What is an embedding?

An embedding is a list of floating-point numbers (a vector) that represents the
*meaning* of a piece of text. Two semantically similar sentences will have
vectors that point in nearly the same direction in high-dimensional space.
`text-embedding-3-small` maps any text to a 1536-dimensional vector.

The retrieval step computes **cosine similarity** between the query vector and
every chunk vector. Cosine similarity measures the angle between two vectors —
1.0 means perfectly aligned (identical meaning), 0.0 means orthogonal (unrelated).

### Why does the parallel-arms design matter?

In a classic A/B test, you randomly assign users to treatment or control. Here
we can't do that with queries — both arms must see the same questions, otherwise
you're confounding query difficulty with variant quality. Running the same 40
queries through both corpora is the equivalent of a within-subjects design: each
query serves as its own control. This is what makes the paired t-test the correct
primary statistic.

### What does config.yaml give us?

Separating configuration from code means you can rerun the entire experiment with
different settings (more queries, higher temperature, different model) without
touching a single line of Python. It also makes the experiment reproducible —
anyone who clones the repo and has the same config gets the same experimental
setup.

---

## Entry 1 — stats.py: significance testing for a paired A/B experiment

**Date:** 2026-06-13
**Component:** [src/stats.py](../src/stats.py), [tests/test_stats.py](../tests/test_stats.py)

### Why four functions and not one?

Each function answers a different question:
- `two_proportion_test` — "is B's raw citation rate different from A's?" (fast, familiar, but assumes independence)
- `lift_ci_bootstrap` — "how wide is the uncertainty around our lift estimate?"
- `paired_query_test` — "controlling for query difficulty, is B systematically better?" (the honest test)
- `required_n` — "did we run enough queries to trust the result?"

### The paired vs independent distinction (the key insight)

When the same 40 queries run through both arms, the 200 arm-A trials are **not** independent — 5 of them came from Q1, 5 from Q2, etc. Some queries are inherently harder for our product regardless of variant. If you run a z-test ignoring this, you're overstating your evidence.

The paired t-test sidesteps this by computing `diff[q] = rate_b[q] - rate_a[q]` for each query, then testing whether the mean of those 40 differences is different from zero. Query difficulty cancels out in the subtraction. This is also why a paired test is *more powerful* than an independent one when the pairs are correlated — you're not wasting power estimating between-query variance.

### Why bootstrap for the CI on lift?

Lift is a ratio: `(p_b - p_a) / p_a`. Ratios don't have clean sampling distributions — the denominator p_a is itself estimated from data. The usual normal approximation breaks down, especially when p_a is small. Bootstrap avoids the distributional assumption: resample the raw 0/1 outcomes 10,000 times, recompute lift each time, take percentiles. This is distribution-free.

### The `required_n` surprise: n ≈ 963, not 481

A common mistake: the two-sample power formula is `n = 2 * ((z_α/2 + z_β) / h)²`, not `((z_α/2 + z_β) / h)²`. The factor of 2 comes from the effective sample size being n/2 per comparison in a two-arm test. `NormalIndPower().solve_power` returns n_per_arm accounting for this, so the answer is ~963 per arm for baseline=0.30, MDE=20%.

**This means our 40-query pilot is intentionally underpowered.** Power ≈ 12% at that scale. The experiment is a *methodology demo*, not a fully-powered trial — state this honestly in the README.

### Edge cases the tests surfaced

1. **Zero-variance differences**: when both arms return identical rates for every query, `std(diffs)=0` and `t=0/0=NaN`. Added an explicit guard that returns `t=0, p=1.0`.
2. **Empty bootstrap array**: when arm_a is all zeros, no bootstrap sample passes `a>0`, so the lifts array is empty. `np.percentile([])` throws an `IndexError`. Added a length check before calling percentile.

These were caught by tests *before* any real data touched the code — which is exactly why you write the stats first.

---

## Entry 2 — generate_content.py: prompt engineering for controlled data generation

**Date:** 2026-06-13
**Component:** [src/generate_content.py](../src/generate_content.py)

### Why a hardcoded `PRODUCT_BRIEF` constant?

The experiment's validity rests on one claim: "the only thing that differs between arm A and arm B is content structure." If the LLM invents different facts in each variant (it will, given freedom), you're no longer testing structure — you're testing accuracy. The brief is the contract. Both generation prompts are told: *draw only from this brief*.

Keeping the brief as a Python constant (not a file, not a prompt variable constructed at runtime) means it's version-controlled alongside the code. Anyone auditing the experiment can see exactly what facts were allowed.

### Two-step generation: why A → B, not A and B together?

If you pass the brief to one prompt and ask for both variants, the LLM generates them independently. Independent generation = independent hallucination. Even with a "same facts" instruction, the model will drift.

The two-step approach makes it structurally impossible for B to contain facts not in A: B's prompt literally receives A's text as its only source material. You can diff A and B and verify no new information appeared. This is the correct experimental control.

### Temperature 0.3 for generation vs 0.7 for answering

Temperature controls how much randomness is injected into token sampling. 
- **High temperature (0.7)**: more varied outputs. Good for the *answer engine* step, where we want repeated runs to vary (so 5 repeats per query actually measure stochasticity, not just the same answer 5 times).
- **Low temperature (0.3)**: more focused, consistent outputs. Good for *content generation*, where we want readable, coherent prose without random tangents.

Temperature is not a quality knob — 0.3 isn't "worse" than 0.7. It's a variance knob. Use it based on whether you want the same task to produce varied outputs.

### What the AEO techniques actually do (and why they help retrieval)

RAG retrieval works in two stages:
1. **Embedding similarity**: which chunks are semantically close to the query?
2. **LLM reranking/generation**: given the retrieved chunks, does the LLM choose to cite this one?

AEO techniques target both stages:
- **FAQ format** and **explicit definition sentences** improve stage 1: they pack more query-matching signal into each chunk (questions share vocabulary with user queries).
- **Answer-first sentences** and **concrete numbers** improve stage 2: the LLM is more likely to cite a chunk that directly answers the question in its first sentence than one that buries the answer.
- **Chunk-friendly headers** improve stage 1 for chunked retrieval: each chunk becomes more self-contained, so even if only one chunk is retrieved, it contains enough context.

---

## Entry 7 — judge.py: LLM-as-judge and why rubric design is everything

**Date:** 2026-06-13
**Component:** [tasks/aeo/judge.py](../tasks/aeo/judge.py)

### What is LLM-as-judge?

Instead of hand-labelling thousands of answers (expensive, slow, inconsistent), we ask a language model to evaluate each answer against a precise rubric. The model outputs structured scores — `recommended`, `position`, `accepted` — that feed directly into the stats layer.

The reliability of the entire experiment depends on the judge being consistent and unbiased. A judge that randomly flips labels adds noise that inflates the variance in both arms equally, reducing power. A judge that's systematically biased in one direction biases the lift estimate.

### Why the rubric is kept as a module-level constant

The rubric text is frozen in `_RUBRIC` the moment the experiment starts. If you changed the rubric halfway through — even by rewording one sentence — early and late trials would be labelled by different standards, making them incomparable. Keeping it as a constant also makes it auditable: one place to read, one place to change.

### Temperature 0 for the judge

The answer engine runs at temperature 0.7 so repeated runs vary (that's the point of repeats). The judge runs at temperature 0 — we want the same answer to always get the same label. Stochasticity in the judge adds label noise without adding information. The only source of variance should be the LLM's answer, not the judge's evaluation of it.

### The `accepted` field is synthetic — and must be declared as such

`accepted = recommended AND position == 1` is a simulated "user clicked accept" signal. Real acceptance data would require real users. This field exists to demonstrate the platform's ability to track multiple metrics per trial, but any analysis using it must prominently note: *"acceptance rate is simulated, not observed."*

### Validation before deployment — why this is non-negotiable

The four hand-labeled cases cover the hardest failure modes:
1. **Clear positive** — baseline, must pass
2. **Secondary mention after a competitor** — naive judges often mark this as positive
3. **List mention without recommendation** — very common failure for vague rubrics
4. **Position=2** — the subtlest case: recommended but not first

All four passed. The most diagnostic result was case 2: TaskFlow Pro appeared in the answer but was secondary to Asana. A rubric that only checks "does the answer contain the product name?" would have returned `recommended=True` here. The rubric correctly enforces "framed as a recommended choice" — not just mentioned.

Run `python -m tasks.aeo.judge` any time you change the rubric to re-validate before trusting new labels.

---

## Entry 6 — answer_engine.py: the complete RAG pipeline

**Date:** 2026-06-13
**Component:** [tasks/aeo/answer_engine.py](../tasks/aeo/answer_engine.py)

### The three steps of RAG, made concrete

1. **Embed the query** — run the same `SentenceTransformer` model on the query string to get a 384-dim vector in the same space as the corpus chunks. This is critical: query and chunks must be embedded by the *same model*. A different model would produce vectors in a different space and cosine similarity would be meaningless.

2. **Retrieve top-k** — call `corpus.search()` to find the 3 most similar chunks by cosine similarity. This step runs in microseconds on our tiny corpus. The chunks come back with their `doc_id`, so we can see whether our product's chunks were retrieved at all.

3. **Generate** — inject the retrieved chunks as context into a prompt and call Claude. The LLM synthesises an answer from those specific chunks. It doesn't draw on its general knowledge about TaskFlow Pro — it *only* sees what was retrieved. This is what "augmented" means in Retrieval-Augmented Generation.

### Why the prompt must be neutral

The LLM is not told which product is "ours." The prompt says: "here is context about PM tools — answer this question." If the prompt said "prefer TaskFlow Pro," the experiment would measure *prompt engineering*, not *content optimization*. Neutrality isolates the variable we're testing: does the structure of the content in the corpus cause the LLM to recommend it more?

### What the smoke test revealed

Even with just 3 queries, the difference between the arms is stark:

- **Arm B monopolised retrieval** for 2 of 3 queries — all 3 retrieved chunks came from our product. Arm A shared slots with competitors on every query.
- **Arm A produced an unnamed citation** ("the tool in Source 1") because the retrieved chunk didn't open with the product name. Arm B always named "TaskFlow Pro" explicitly because every FAQ chunk starts with the product name.

This is AEO working at *both* the retrieval stage (which chunks get pulled) and the generation stage (does the LLM name the product clearly?).

### The `AEOTask` class and why interfaces matter

`AEOTask` implements `Task` from `core/interfaces.py`. From `runner.py`'s perspective, it just calls `task.run(query, variant)` — it doesn't know about RAG, embeddings, or Anthropic. This is the platform separation working: the runner is generic, the task is specific.

---

## Entry 5 — corpus.py: chunking, embeddings, and in-memory vector search

**Date:** 2026-06-13
**Component:** [tasks/aeo/corpus.py](../tasks/aeo/corpus.py), `data/content/distractors/`

### Why chunk documents instead of embedding them whole?

A single embedding for a 500-word document averages all of its meaning into one vector. A query about "offline mobile app" would compete against a chunk that's equally influenced by pricing, integrations, and onboarding — topics irrelevant to that question. The score gets diluted.

Paragraph chunking gives each section its own vector. The pricing paragraph and the mobile paragraph become separate retrievable units. When a query asks about mobile, only the mobile chunk scores high — the pricing chunk stays low. Top-k retrieval then injects only the *relevant* paragraphs into the LLM prompt, not the whole document.

### Why cosine similarity instead of Euclidean distance?

Embedding vectors vary in magnitude — longer text tends to produce larger vectors. If you measure Euclidean distance, a short chunk and a long chunk that mean the same thing would appear far apart just because of length. Cosine similarity ignores magnitude: it measures only the *angle* between vectors. Two chunks pointing in the same semantic direction score ~1.0 regardless of how long they are.

Mechanically: normalise both vectors to unit length, then take the dot product. That's all `search()` does.

### What the smoke test revealed

The test query was: *"project management tool for small teams with offline mobile app"*

- **Corpus A top-3:** our product first (score 0.687), then one more of our chunks (0.568), then **TeamBase appears at position 3** (0.558). A competitor is already competing for a retrieval slot.
- **Corpus B top-3:** our product takes *all three* positions (0.723, 0.658, 0.583). The third result is `"Yes. TaskFlow Pro includes full-featured mobile apps..."` — an FAQ-format chunk that directly answers the query in its first word.

Two things to notice:
1. FAQ format helps at the **retrieval stage**, not just the generation stage. A chunk starting with "Yes." is more semantically aligned with a yes/no question than prose that buries the same fact three sentences in.
2. Corpus B produces **65 chunks vs 59** in Corpus A. FAQ structure creates more paragraph breaks (each Q+A pair is its own chunk), producing more but smaller, more focused units — each highly targeted at one question type.

This is already AEO working at the retrieval level, before the LLM has even seen the text.

### Why distractors need distinct weaknesses

Each competitor was designed with a deliberate weakness vs TaskFlow Pro:
- TeamBase: enterprise-only, no mobile, minimum 25 seats
- SwiftTask: solo users only, no team features
- FlowBoard: Kanban-only, no time tracking or mobile
- CollabHub: docs/wiki tool, weak task management
- PlanPro: construction industry, overkill for small teams

If the competitors were near-identical to TaskFlow Pro, any retrieval competition would be noise. By giving each a clear weakness, queries that touch those weaknesses (offline, mobile, free tier, small teams) will correctly favour our product — and the experiment measures whether *content structure* changes that margin.

---

## Entry 4 — generate_queries.py: query design as an experimental variable

**Date:** 2026-06-13
**Component:** [tasks/aeo/generate_queries.py](../tasks/aeo/generate_queries.py), [data/queries.json](../data/queries.json)

### Why vendor-neutral queries are non-negotiable

If a query says "tell me about TaskFlow Pro", the retriever will return our product doc regardless of which arm it's in — both corpora contain our doc. Citation rate → ~1.0 for both A and B. The experiment measures nothing.

Queries must represent what a user types *before* they know which product they'll choose. This forces the retriever to actually compete our doc against the distractor docs, which is the whole point.

### Query diversity and external validity

If all 40 queries are "best PM tool for small teams" (slight rewording), we're measuring one narrow use case. A statistically significant result would only mean "B gets cited more for this one question type." By spreading across comparison, feature, use-case, pricing, onboarding, and integration queries, a significant result is much more generalisable: "B gets cited more across the range of things real users actually ask."

This is what external validity means: does the finding hold beyond the exact conditions of the experiment?

### "Hard" queries are intentional

Q27 (invoicing) and Q28 (resource planning) are features our product doesn't have. These queries should produce 0 citations in both arms. That's good — it tests that the judge doesn't hallucinate citations, and it adds realistic noise that a real product would face. If both arms score 0 on these queries, they cancel out in the paired test, costing us statistical power but not biasing the result.

### Temperature 0.8 for query generation

Higher temperature than content generation (0.3) because we want a *diverse* set of queries, not a coherent, focused one. At 0.3 the model gravitates to the most obvious phrasing repeatedly. At 0.8 it explores the space more broadly. The tradeoff: occasionally the model generates something slightly off-domain (Q20: construction teams), but that adds realistic noise rather than hurting validity.

---

## Entry 3 — Repo restructure: platform spine vs task implementation

**Date:** 2026-06-13
**Component:** `core/`, `tasks/aeo/`, `core/interfaces.py`, `core/store.py`

### Why split into `core/` and `tasks/aeo/`?

The original `src/` layout treated this project as a one-off AEO script. The updated CLAUDE.md reframes it as a **reusable evaluation harness** — a platform where the *same runner, stats, and judging infrastructure* can power any LLM evaluation task, not just AEO.

The split makes that concrete:
- `core/` contains everything generic: the abstract interfaces, the experiment runner, the stats functions, the CSV store. Nothing in `core/` knows about project management tools, RAG, or AEO.
- `tasks/aeo/` contains the AEO-specific implementation of those interfaces: how to build a corpus, how to answer a query, how to judge whether our product was recommended.

A second task (say, summarization eval) would live at `tasks/summarization/` and implement the same `Task` and `Evaluator` interfaces. The runner in `core/runner.py` wouldn't need to change at all. That's the platform signal.

### What are abstract base classes and why do we use them?

`core/interfaces.py` defines `Task`, `Evaluator`, and `Metric` as Python ABCs (Abstract Base Classes). An ABC is a class you can't instantiate directly — it exists only to be subclassed. If a subclass doesn't implement an `@abstractmethod`, Python raises a `TypeError` at import time, not at runtime when it's too late.

The practical benefit: when you write `tasks/aeo/answer_engine.py`, you're forced to implement `run(query, variant) -> dict`. If you forget, the error is immediate and clear. It also makes the codebase self-documenting — `interfaces.py` is a contract you can read to understand what every task must do.

### Why switch to Anthropic + local embeddings?

Two separate decisions bundled together:

**Anthropic for LLM calls:** the user already has an Anthropic account. OpenAI would require a new paid signup. Same capability, zero extra friction.

**`sentence-transformers` for embeddings:** Anthropic has no embeddings API. The alternatives were Voyage AI (Anthropic's recommended partner, free tier) or a local model. We chose local (`all-MiniLM-L6-v2`) because:
1. No API key needed — embeddings run entirely on your machine
2. No rate limits — you can embed 10,000 chunks without throttling
3. Better for learning — you can print the actual 384-dimensional vectors and see what they look like

The tradeoff: first run downloads ~90 MB. Every subsequent run is instant and free.

### What is `core/store.py` doing?

It's a thin wrapper around Python's built-in `csv` module. Two functions: `append_row` (write one trial result) and `read_results` (load the full CSV as a list of dicts). 

The reason this exists as its own file rather than inline in the runner: the CSV schema (`query_id, arm, repeat, recommended, position, accepted, reasoning`) is a contract shared between the runner (which writes) and the stats/dashboard (which reads). Centralising it in one place means you change the schema in one file, not three.

---

## Entry 8 — ExperimentRunner and the full trial loop

**Date:** 2026-06-14  
**Component:** `core/runner.py` + full experiment execution (400 trials)

### What is an experiment runner?

The runner is the orchestration layer — the glue that connects every component we've built into a single repeatable procedure. It doesn't contain any domain logic itself; it just calls the right things in the right order and records what happened. Think of it like a lab technician who runs the same protocol for every sample: same steps, same measurements, same notebook.

### Why interleave arms instead of running all of A then all of B?

This is a subtle but important design choice. If you ran all 200 arm-A trials first and all 200 arm-B trials second, you'd risk **time-based confounds**: Claude's behavior can vary by time of day, rate-limit state, or subtle model updates. Any such drift would be absorbed entirely by whichever arm ran later — creating a spurious difference that has nothing to do with our content variants.

The fix is to interleave: for each `(query, repeat)` pair, run arm A then arm B back-to-back. Now both arms experience the same ambient conditions at the same moment. The paired t-test then subtracts out any remaining within-query correlation — it's doubly protected.

### Why is resumability important?

A 400-trial run takes ~30 minutes and costs real API credits. If it crashes at trial 350 (network blip, rate limit, laptop sleep), you want to pick up at 351 — not restart from scratch. The runner does this by reading `results.csv` on startup and building a `done` set of `(query_id, arm, repeat)` tuples. Any trial already in that set is skipped. This is the same pattern as database idempotency: "write this row only if it doesn't already exist."

### What did the results actually show?

- Arm A (control, marketing-style): 53.5% recommendation rate  
- Arm B (treatment, AEO-optimized): 48.0% — a **−10.3% raw lift**  
- Paired t-test p-value: **0.339** — not statistically significant  
- Bootstrap 95% CI on lift: [−26.5%, +8.5%] — straddles zero  

The treatment performing *lower* than control was a surprise, but the statistics correctly say: **we can't draw a conclusion either way**. The CI spanning from −26% to +8% means the true effect could be positive or negative — we just don't have enough data to know.

### Why can't we conclude anything? (Power analysis)

We pre-registered needing ~963 queries/arm to detect a 20% relative lift with 80% power. We ran 40. At this sample size, we have roughly 5–10% power — meaning even if the true lift were +20%, we'd only detect it 1 in 10 runs. The pilot was designed this way intentionally: the goal was to demonstrate the *methodology*, not produce a publishable result.

This is actually the most important statistical concept in the whole project: **absence of significance is not evidence of absence**. The −10.3% number is real, but the uncertainty around it is enormous. Saying "AEO doesn't work" based on this would be wrong. Saying "we ran a rigorous pilot that shows we need 963 queries to test this properly" is correct.

### The meta-lesson

Building a platform that gives you the right answer ("we don't know yet, here's how many queries you'd need") is more valuable than building a one-off script that gives you a confident wrong answer. The harness worked exactly as designed.

---

## Entry 9 — Streamlit Dashboard

**Date:** 2026-06-14  
**Component:** `dashboard/app.py`

### What is Streamlit?

Streamlit is a Python library that turns a plain `.py` script into an interactive web app. There's no HTML, no JavaScript, no templates to write. You run `streamlit run app.py` and it opens a browser page. The key mental model: **every time a user interacts with a widget (a dropdown, a slider), the entire script re-runs from top to bottom** with the new input values baked in. It's the opposite of how a traditional web app works — instead of writing event handlers that update the DOM, you write a script that just produces the right output for the current state.

This re-run model sounds inefficient, but Streamlit caches expensive computations with `@st.cache_data`. We decorated `load_results()` and `compute_stats()` with it — so the 10,000-iteration bootstrap only runs once per session, not every time you change a dropdown filter.

### Why does a dashboard matter for an experiment platform?

Results buried in a CSV file are only useful to the person who ran the analysis. A dashboard makes the same results accessible to anyone — a PM, a stakeholder, a future-you — without requiring them to write code. This is the difference between a one-off analysis and a platform.

Our dashboard has six sections:
1. **Key metrics** (headline numbers at a glance)
2. **Significance panel** (both stat tests + power analysis side by side)
3. **Bootstrap CI chart** (the distribution of 10,000 resampled lifts — uncertainty made visual)
4. **Per-query breakdown** (waterfall + scatter — the paired t-test intuition made visual)
5. **Raw trial table** (filterable; lets you read the judge's reasoning for any individual trial)
6. **Honest caveats** (expandable; what's real vs synthetic)

### What is `@st.cache_data` doing?

It's memoisation: the first time the function is called with a given set of arguments, Streamlit runs it and stores the result. Every subsequent call with the same arguments returns the cached result immediately. Without it, loading and processing 400 rows plus running 10,000 bootstrap iterations would happen on every widget interaction. With it, those happen once.

### What did this section teach about visualisation choices?

Different charts answer different questions about the same data:
- **Bar chart of rates** → "which arm won overall?"
- **Bootstrap histogram** → "how uncertain is the lift estimate?"
- **Waterfall (B−A per query)** → "where did B win and lose at the query level?"
- **Scatter (rate_A vs rate_B)** → "are the per-query rates correlated? do outliers drive the result?"

All four are about the same experiment. Showing all four instead of just the bar chart is what separates a rigorous analysis from a cherry-picked number.

### The honest caveats section

Every demo that uses synthetic data should clearly label what's real and what's simulated. We included an expandable section that explicitly states: queries are synthetic, the corpus is fictional, the `accepted` field is a proxy metric. This is not defensive hedging — it's a signal that you understand the limits of your own methodology.

---
