import os

import anthropic
import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from core.interfaces import Task, Variant
from tasks.aeo.corpus import Index, build_corpora, search

load_dotenv()


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def answer(
    query: str,
    index: Index,
    embed_model: SentenceTransformer,
    llm_client: anthropic.Anthropic,
    config: dict,
) -> dict:
    """Core RAG function: embed query → retrieve chunks → generate answer.

    Returns answer_text plus the retrieved chunks so the judge (and later,
    debugging) can see exactly what context the LLM was given.
    """
    # Step 1: embed the query into the same vector space as the corpus chunks
    q_emb = embed_model.encode([query])[0]

    # Step 2: retrieve top-k chunks by cosine similarity
    top_k = config.get("top_k", 3)
    results = search(index, q_emb, top_k=top_k)

    # Step 3: build context string — chunks separated by a clear delimiter
    context_parts = []
    for i, (chunk, score) in enumerate(results, 1):
        context_parts.append(f"[Source {i}]\n{chunk.text}")
    context = "\n\n".join(context_parts)

    # The prompt is intentionally neutral: no mention of which product is "ours,"
    # no instruction to prefer any vendor. The LLM must answer from context alone.
    prompt = f"""You are a knowledgeable assistant helping someone choose a project management tool.

Use the information in the sources below to answer the question. Synthesize across sources where relevant. Be specific: name the product(s) and explain why they fit the use case. If no source is relevant, say so.

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    message = llm_client.messages.create(
        model=config["answer_model"],
        max_tokens=512,
        # Temperature from config (~0.7) so repeated runs on the same query
        # produce varied answers — this is what makes n_repeats meaningful.
        temperature=config["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer_text": message.content[0].text.strip(),
        "retrieved_chunks": [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "score": round(score, 4),
            }
            for chunk, score in results
        ],
    }


class AEOTask(Task):
    """Implements the generic Task interface for the AEO experiment.

    Holds both indices and shared models so runner.py can call .run()
    without knowing anything about RAG internals.
    """

    def __init__(
        self,
        index_a: Index,
        index_b: Index,
        embed_model: SentenceTransformer,
        config: dict,
    ):
        self.index_a = index_a
        self.index_b = index_b
        self.embed_model = embed_model
        self.config = config
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def run(self, query: str, variant: Variant) -> dict:
        index = self.index_a if variant.name == "control" else self.index_b
        return answer(query, index, self.embed_model, self.client, self.config)


if __name__ == "__main__":
    config = _load_config()

    print("Building corpora (uses cached distractor docs, re-embeds)...")
    index_a, index_b, embed_model = build_corpora()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    queries = [
        "best project management tool for small remote teams",
        "project management tool that works offline on mobile",
        "affordable PM tool with free guest access for clients",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")

        for label, index in [("ARM A (control)", index_a), ("ARM B (treatment)", index_b)]:
            print(f"\n--- {label} ---")
            result = answer(q, index, embed_model, client, config)
            print(f"Retrieved: {[c['doc_id'] for c in result['retrieved_chunks']]}")
            print(f"Answer: {result['answer_text'][:300]}...")
