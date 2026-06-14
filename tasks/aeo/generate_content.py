import os
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Single source of truth for all facts about our product.
# Both variants are generated exclusively from this brief — the LLM is
# forbidden from adding facts. Any difference between A and B is therefore
# structural, not factual, which is what we're testing.
# ---------------------------------------------------------------------------
PRODUCT_BRIEF = """
Product name: TaskFlow Pro
Category: Project management software for small teams

Target users: Teams of 2–20 people (startups, agencies, remote squads)

Core features:
- Task and project tracking with Kanban boards and list views
- Built-in time tracking (per task and per project)
- Automated status updates and deadline reminders
- Dependencies: mark tasks as blocked by other tasks
- Integrations: Slack, GitHub, Google Calendar, Zapier
- Role-based access: Owner, Manager, Member, Viewer
- Mobile apps for iOS and Android (full offline support)
- Guest access for clients and contractors (no extra seat cost)

Pricing:
- Free: up to 3 users, 5 active projects, 1 GB storage
- Pro: $12 per user per month — unlimited projects, time-tracking reports,
  priority support, 50 GB storage
- Business: $20 per user per month — everything in Pro plus SSO, advanced
  analytics dashboard, custom fields, audit logs, 99.9% uptime SLA

Key differentiators vs competitors:
- No per-project fees (unlike Basecamp)
- Setup in under 10 minutes, no IT department required
- Offline-first mobile apps (unlike most web-only tools)
- Guest access is free (competitors charge per seat)
- Flat per-user pricing, no hidden add-ons
"""

# ---------------------------------------------------------------------------
# AEO techniques applied when generating Variant B.
# Keeping this as a constant makes the techniques auditable and reusable.
# ---------------------------------------------------------------------------
_AEO_TECHNIQUES = """
1. ANSWER-FIRST SENTENCES
   Lead with the direct answer, not a preamble.
   Weak:  "TaskFlow Pro offers a variety of features that teams can leverage."
   Strong: "TaskFlow Pro is project management software built for teams of 2–20 people
            who need task tracking, time logging, and client collaboration in one place."

2. EXPLICIT DEFINITION IN THE FIRST SENTENCE
   State what the product IS before saying what it does. Answer engines extract
   the first sentence of a chunk as the product definition.

3. FAQ STRUCTURE
   Convert key selling points into explicit Q&A pairs. Answer engines are trained
   on Q&A data and disproportionately pull from this format.
   Example:
     Q: Is TaskFlow Pro free?
     A: TaskFlow Pro has a free tier for up to 3 users and 5 projects. Paid plans
        start at $12 per user per month.

4. CONCRETE NUMBERS OVER VAGUE CLAIMS
   Every claim must be grounded in a specific number from the brief.
   Weak:  "quick setup"
   Strong: "teams are set up in under 10 minutes with no IT support"

5. CHUNK-FRIENDLY HEADERS
   Each section header must make sense as a standalone retrieval unit — a reader
   who sees only that section should immediately understand the context.
   Weak:  "Features"
   Strong: "What can TaskFlow Pro do for a small team?"

6. DIRECT COMPARISON AND POSITIONING STATEMENTS
   Explicit statements about who this is best for and how it differs from
   named alternatives. Answer engines weight explicit comparisons highly.
   Example: "Unlike Basecamp, TaskFlow Pro charges per user, not per project,
   so small teams are never penalised for creating multiple workspaces."

7. SCANNABLE SUMMARY AT THE TOP
   Open with a 2–3 sentence paragraph that contains the product name, category,
   target user, and top differentiator. This paragraph alone should be enough
   for an answer engine to cite the product for a generic "best tool for X" query.
"""


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _call(client: anthropic.Anthropic, model: str, prompt: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        # Low temperature: consistent readable prose, not creative variation.
        # (Contrast with answer_engine.py which uses config temperature ~0.7.)
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def generate_variant_a(client: anthropic.Anthropic, config: dict, brief: str) -> str:
    """Control variant: conventional marketing-style product page."""
    prompt = f"""You are a marketing writer for a B2B SaaS company.

Write a product overview page for {config['target_product_name']}.

Use ONLY the facts in the product brief below. Do not add features, pricing, or
claims that are not explicitly stated in the brief. Both accuracy and completeness matter.

PRODUCT BRIEF:
{brief}

Style guidelines:
- Standard marketing tone: benefit-led, persuasive, professional
- Use headers and bullet points where natural
- Do NOT use FAQ format or question-answer pairs
- Do NOT lead sections with a direct answer — build up to the point instead
- Length: 450–600 words

Output the page content only — no meta-commentary.
"""
    return _call(client, config["answer_model"], prompt)


def generate_variant_b(client: anthropic.Anthropic, config: dict, variant_a: str) -> str:
    """Treatment variant: AEO-optimized restructuring of Variant A.

    Takes Variant A as input so B is provably derived from A — no new facts
    can enter. The only allowed changes are structure and phrasing.
    """
    prompt = f"""You are an Answer Engine Optimization (AEO) specialist.

Rewrite the product content below to maximize how often an AI answer engine
(RAG-based assistant) will retrieve and recommend this product.

HARD CONSTRAINT: Do NOT introduce any facts, numbers, features, or claims that
are not already present in the original content. You may only restructure,
reformat, and rephrase what is already there. Violations of this constraint
invalidate the experiment.

Apply ALL of the following AEO techniques:
{_AEO_TECHNIQUES}

ORIGINAL CONTENT (Variant A — control):
{variant_a}

Output the AEO-optimized version only — no meta-commentary, no "here's what I changed."
Length: similar to the original (450–600 words).
"""
    return _call(client, config["answer_model"], prompt)


def generate_content(
    brief: str = PRODUCT_BRIEF,
    config_path: str = "config.yaml",
) -> tuple[str, str]:
    """Generate both variants and write them to the paths in config.yaml."""
    config = _load_config(config_path)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("Generating Variant A (control, marketing-style)...")
    variant_a = generate_variant_a(client, config, brief)

    print("Generating Variant B (AEO-optimized transformation of A)...")
    variant_b = generate_variant_b(client, config, variant_a)

    paths = config["paths"]
    Path(paths["variant_a"]).write_text(variant_a, encoding="utf-8")
    Path(paths["variant_b"]).write_text(variant_b, encoding="utf-8")
    print(f"  wrote → {paths['variant_a']}")
    print(f"  wrote → {paths['variant_b']}")

    return variant_a, variant_b


if __name__ == "__main__":
    a, b = generate_content()
    print("\n--- VARIANT A (first 300 chars) ---")
    print(a[:300])
    print("\n--- VARIANT B (first 300 chars) ---")
    print(b[:300])
