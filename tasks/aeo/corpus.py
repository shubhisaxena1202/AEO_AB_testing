import os
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import numpy as np
import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Five fictional competitors. Each has a deliberate weakness vs TaskFlow Pro
# so the retriever must actually choose between them on relevant queries.
# ---------------------------------------------------------------------------
_DISTRACTOR_PROFILES = [
    {
        "id": "competitor_teambase",
        "name": "TeamBase",
        "profile": (
            "Enterprise project management for teams of 50-500. Powerful portfolio "
            "management, resource planning, and org-wide analytics. Requires IT setup "
            "and onboarding (typically 2-4 weeks). Pricing starts at $28/user/month "
            "(minimum 25 seats). No free tier. Web-only — no mobile app."
        ),
    },
    {
        "id": "competitor_swifttask",
        "name": "SwiftTask",
        "profile": (
            "Task management built for solo freelancers and individuals. Clean, minimal "
            "interface with personal Kanban and deadline reminders. No team collaboration "
            "features, no shared workspaces, no integrations. Free for one user; $8/month "
            "for the solo Pro plan. Not designed for teams."
        ),
    },
    {
        "id": "competitor_flowboard",
        "name": "FlowBoard",
        "profile": (
            "Visual Kanban tool with drag-and-drop simplicity. Great for teams that "
            "live in boards. No built-in time tracking, no dependency management, "
            "no mobile app. Integrates with Slack only. Free for up to 10 users with "
            "unlimited boards; Pro is $6/user/month. Very cheap, very limited."
        ),
    },
    {
        "id": "competitor_collabhub",
        "name": "CollabHub",
        "profile": (
            "Document and wiki platform with lightweight task management bolted on. "
            "Excellent for teams that write a lot (docs, specs, meeting notes). "
            "Task management is secondary — no time tracking, no Kanban, no dependency "
            "tracking. Web-only. Free tier available; $10/user/month for Teams plan. "
            "Best for knowledge management, not project execution."
        ),
    },
    {
        "id": "competitor_planpro",
        "name": "PlanPro",
        "profile": (
            "Full-featured project management targeting construction and field service "
            "companies. Includes scheduling, budgeting, equipment tracking, and "
            "contractor management. Overkill for software or agency teams. "
            "Starts at $39/user/month. Requires dedicated onboarding. "
            "Mobile app exists but is field-inspection focused, not task management."
        ),
    },
]


# ---------------------------------------------------------------------------
# Index: the in-memory vector store for one corpus arm
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    embedding: np.ndarray = field(repr=False)


@dataclass
class Index:
    chunks: list[Chunk]
    # Pre-stacked matrix for fast batch cosine similarity: shape (n_chunks, dim)
    matrix: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_doc(doc_id: str, text: str, min_chars: int = 40) -> list[dict]:
    # Paragraph chunking: split on blank lines.
    # Preserves semantic coherence better than fixed-size windows for our
    # short, well-structured markdown docs.
    paragraphs = [p.strip() for p in text.split("\n\n")]
    chunks = []
    for i, para in enumerate(paragraphs):
        if len(para) < min_chars:
            continue
        chunks.append({"id": f"{doc_id}_c{i:02d}", "doc_id": doc_id, "text": para})
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(docs: list[dict], model: SentenceTransformer) -> Index:
    """Build an in-memory vector index from a list of {id, text} documents.

    Chunks each doc by paragraph, embeds all chunks in one batch, stacks
    embeddings into a matrix for O(n) cosine similarity at query time.
    """
    all_chunks = []
    for doc in docs:
        all_chunks.extend(_chunk_doc(doc["id"], doc["text"]))

    print(f"  embedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

    chunks = [
        Chunk(
            id=c["id"],
            doc_id=c["doc_id"],
            text=c["text"],
            embedding=embeddings[i],
        )
        for i, c in enumerate(all_chunks)
    ]
    matrix = np.vstack([c.embedding for c in chunks])
    return Index(chunks=chunks, matrix=matrix)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    index: Index,
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> list[tuple[Chunk, float]]:
    """Return top-k (chunk, score) pairs by cosine similarity.

    Cosine similarity = dot product of unit vectors. We normalise once at
    search time rather than at index time so the index stays compact.
    """
    q = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
    norms = np.linalg.norm(index.matrix, axis=1, keepdims=True)
    normed = index.matrix / np.where(norms == 0, 1.0, norms)
    scores = normed @ q                          # shape (n_chunks,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(index.chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Distractor generation
# ---------------------------------------------------------------------------

def _generate_distractor(
    client: anthropic.Anthropic, model_name: str, profile: dict
) -> str:
    prompt = f"""Write a product overview page for a fictional SaaS product called {profile['name']}.

Use ONLY the facts in the profile below. Do not add features, pricing, or claims not stated.

PROFILE:
{profile['profile']}

Style: standard marketing tone, headers and bullet points where natural.
Length: 250–350 words.
Output the page content only — no meta-commentary.
"""
    msg = client.messages.create(
        model=model_name,
        max_tokens=600,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def ensure_distractors(config: dict) -> list[dict]:
    """Generate distractor docs if they don't exist; load and return all."""
    distractor_dir = Path(config["paths"]["distractors"])
    distractor_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model_name = config["answer_model"]

    docs = []
    for profile in _DISTRACTOR_PROFILES:
        path = distractor_dir / f"{profile['id']}.md"
        if not path.exists():
            print(f"  generating distractor: {profile['name']}...")
            text = _generate_distractor(client, model_name, profile)
            path.write_text(text, encoding="utf-8")
        else:
            text = path.read_text(encoding="utf-8")
        docs.append({"id": profile["id"], "text": text})

    return docs


# ---------------------------------------------------------------------------
# Build both corpora
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_corpora(config_path: str = "config.yaml") -> tuple[Index, Index]:
    """Build Corpus_A (variant_a + distractors) and Corpus_B (variant_b + distractors).

    Only the treatment doc differs between the two corpora — distractors are
    identical. Any difference in retrieval outcome is attributable to the
    content of the treatment doc alone.
    """
    config = _load_config(config_path)

    print("Loading embedding model (downloads ~90 MB on first run)...")
    model = SentenceTransformer(config["embedding_model"])

    variant_a = Path(config["paths"]["variant_a"]).read_text(encoding="utf-8")
    variant_b = Path(config["paths"]["variant_b"]).read_text(encoding="utf-8")

    print("Generating/loading distractor docs...")
    distractors = ensure_distractors(config)

    print(f"\nBuilding Corpus A ({1 + len(distractors)} docs)...")
    index_a = build_index(
        [{"id": "our_product", "text": variant_a}] + distractors, model
    )

    print(f"Building Corpus B ({1 + len(distractors)} docs)...")
    index_b = build_index(
        [{"id": "our_product", "text": variant_b}] + distractors, model
    )

    return index_a, index_b, model


if __name__ == "__main__":
    index_a, index_b, model = build_corpora()

    print(f"\nCorpus A: {len(index_a.chunks)} chunks")
    print(f"Corpus B: {len(index_b.chunks)} chunks")

    # Smoke-test: query both indices and show top-3 retrieved chunks
    test_query = "project management tool for small teams with offline mobile app"
    print(f"\nTest query: '{test_query}'")
    q_emb = model.encode([test_query])[0]

    print("\n--- Corpus A top-3 ---")
    for chunk, score in search(index_a, q_emb, top_k=3):
        print(f"  [{chunk.doc_id}] score={score:.3f}  {chunk.text[:80]}...")

    print("\n--- Corpus B top-3 ---")
    for chunk, score in search(index_b, q_emb, top_k=3):
        print(f"  [{chunk.doc_id}] score={score:.3f}  {chunk.text[:80]}...")
