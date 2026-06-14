# Architectural Decisions

Each entry records a choice made during the build, the alternatives considered,
and the reasoning. The goal is a living record so future-you (or a reviewer)
understands *why*, not just *what*.

---

## ADR-001 — In-memory vector store over a dedicated vector DB

**Decision:** Use numpy cosine similarity on in-process arrays, not Chroma/FAISS/Pinecone.

**Options considered:**
- Chroma (local, persistent, easy API)
- FAISS (fast, battle-tested, but C++ dependency)
- Numpy in-memory cosine similarity

**Chosen:** Numpy in-memory.

**Reasoning:** Our corpus is tiny (~50 chunks across both arms). Any vector DB would
add infra complexity with zero performance benefit at this scale. The numpy approach
forces us to understand *what a vector store actually does* — store dense float
vectors and compute nearest neighbors — without framework magic hiding the mechanics.
Upgrade path to FAISS or Chroma is one function swap in `corpus.py`.

---

## ADR-002 — Parallel-arms with frozen distractors

**Decision:** Both corpora share the exact same 4–6 distractor docs; only the
treatment doc differs.

**Alternatives:**
- Different distractors per arm (confounds the retriever — we'd be testing corpus
  composition, not content quality)
- No distractors (the retriever would always return our doc; citation rate → 1.0
  for both arms; useless experiment)

**Chosen:** Frozen distractors.

**Reasoning:** Confound isolation. The only variable that differs between Corpus_A
and Corpus_B is variant_a vs variant_b. Any change in citation rate must be
attributable to how the treatment doc is written, not to what else is around it.

---

## ADR-003 — Paired t-test as primary result, z-test as secondary

**Decision:** Lead with the paired-query t-test. Present the two-proportion z-test
as a "naive comparison" rather than the headline result.

**Reasoning:** The same 40 queries run through both arms. Some queries are
inherently easier for our product regardless of which variant is in the corpus.
The z-test ignores this correlation — it treats 200 A-arm trials as IID, which
they are not. The paired t-test aggregates to per-query citation rates first, then
tests whether B's rates are systematically higher, correctly handling within-query
correlation.

---

## ADR-004 — Two-step content generation (A first, then B from A)

**Decision:** `generate_content.py` generates variant_a first, then prompts the
model to produce variant_b by restructuring variant_a — explicitly forbidding
new factual claims.

**Alternative:** Generate both in a single prompt.

**Reasoning:** A single prompt risks the model "improving" B's facts along with
its structure, making the two variants differ in content, not just form. The
two-step approach keeps A as the ground truth and treats B as a pure formatting
transformation, which is exactly the causal claim we want to test.

---

## ADR-005 — LLM provider: Anthropic Claude Haiku + local sentence-transformers

**Decision:** Use Anthropic `claude-haiku-4-5-20251001` for answering and judging;
`all-MiniLM-L6-v2` via `sentence-transformers` for embeddings (runs locally).

**Alternatives:**
- OpenAI gpt-4o-mini (original plan — requires separate paid API key)
- Anthropic Claude + Voyage AI embeddings (Anthropic's recommended partner, has free tier, but adds a second API key)
- Local model via Ollama for LLM (free, no API key, but weaker judge quality)

**Chosen:** Anthropic for LLM calls, local sentence-transformers for embeddings.

**Reasoning:** The user has an existing Anthropic paid account. OpenAI requires a
separate key and payment setup. Using sentence-transformers locally means zero
embedding cost, no rate limits, and no second provider — embeddings run on CPU
in-process. The `all-MiniLM-L6-v2` model (~90 MB) is fast enough for our ~50-chunk
corpus and teaches the concept clearly (you can inspect the actual vectors).

---

## ADR-006 — Repo structure: core/ + tasks/aeo/ instead of flat src/

**Decision:** Split the codebase into `core/` (generic platform spine) and
`tasks/aeo/` (AEO-specific implementation).

**Alternative:** Keep everything flat in `src/` as originally scaffolded.

**Chosen:** Layered structure.

**Reasoning:** The CLAUDE.md spec was updated to frame this as a *reusable
evaluation harness*, not a one-off AEO script. The `core/` layer (interfaces,
runner, stats, store) is genuinely generic — a second task like summarization
eval could plug in at `tasks/summarization/` without touching `core/`. The flat
`src/` layout would have required touching everything to add a second task.
The architectural signal matters for the portfolio angle of this project.

---
