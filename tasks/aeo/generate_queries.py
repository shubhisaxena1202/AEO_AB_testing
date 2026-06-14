import json
import os
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_queries(config_path: str = "config.yaml") -> list[dict]:
    config = _load_config(config_path)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    n = config["n_queries"]
    domain = config["domain"]

    prompt = f"""Generate {n} realistic queries that someone would type into an AI assistant when \
looking for {domain}.

Requirements:
- VENDOR-NEUTRAL: do not mention any specific product names. Queries must reflect what a user \
asks *before* they know which product they'll choose.
- VARIED INTENT: mix these types roughly equally —
    * Comparison: "best X for Y", "X vs Y", "top tools for Z"
    * Feature-specific: "which tools support offline mobile", "PM tool with built-in time tracking"
    * Use-case: "project management for remote agencies", "tool for client-facing projects"
    * Pricing: "free project management for small teams", "affordable PM tools under $15/user"
    * Setup/onboarding: "easiest project management tool to set up", "PM tool no IT required"
    * Integration: "project management that integrates with Slack and GitHub"
- VARIED SPECIFICITY: some broad ("best PM tool for startups"), some narrow \
("project management with free guest access for clients")
- NATURAL LANGUAGE: phrased as a real user would ask, not as keywords

Return ONLY a valid JSON array of {n} query strings. No explanation, no numbering, no markdown \
fences. Example format:
["query one", "query two", ...]
"""

    message = client.messages.create(
        model=config["answer_model"],
        max_tokens=2048,
        # Higher temperature than content generation — we want a diverse query set,
        # not a focused one. Diversity = better external validity for the experiment.
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw[raw.index("["):]
    if raw.endswith("```"):
        raw = raw[:raw.rindex("]") + 1]

    queries_list = json.loads(raw)
    queries = [{"id": f"q{i+1:02d}", "text": q} for i, q in enumerate(queries_list)]

    path = config["paths"]["queries"]
    Path(path).write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(queries)} queries → {path}")

    return queries


if __name__ == "__main__":
    queries = generate_queries()
    print("\nSample queries:")
    for q in queries[:8]:
        print(f"  [{q['id']}] {q['text']}")
