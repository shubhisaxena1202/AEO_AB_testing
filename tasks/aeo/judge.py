import json
import os
import re

import anthropic
import yaml
from dotenv import load_dotenv

from core.interfaces import Evaluator, Variant

load_dotenv()


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core rubric — keeping it as a constant makes it auditable and consistent
# across every call. If you change the rubric mid-experiment you invalidate
# comparability between early and late trials.
# ---------------------------------------------------------------------------
_RUBRIC = """You are evaluating whether an AI assistant answer recommends a specific product.

TARGET PRODUCT: {target_product_name}

STRICT CRITERIA for recommended=true (ALL must hold):
  1. The answer explicitly names "{target_product_name}" by its exact name.
  2. The answer frames it as a recommended choice, good fit, or best option for the user's query.
     Examples that count: "I recommend TaskFlow Pro", "TaskFlow Pro is well-suited for...",
     "TaskFlow Pro is the best choice", "TaskFlow Pro is an excellent fit".

CRITERIA for recommended=false (ANY of these → false):
  - The target product is not mentioned at all.
  - The target product is mentioned only as a passing reference or in a list without recommendation.
  - The answer primarily recommends a different product and only mentions the target secondarily.
  - The answer says "I don't have enough information" or declines to recommend.

position: If recommended=true, what ordinal position is the target product first recommended?
  1 = it is the first (or only) product recommended.
  2 = a different product is recommended first, then the target product second.
  null = not recommended.

accepted: Simulated signal — true ONLY if recommended=true AND position=1.
  (Represents a user "accepting" the top recommendation. This field is synthetic; it does
  not reflect real user behaviour. Report it as such in any analysis.)

Return ONLY valid JSON with no surrounding text or markdown:
{{
  "recommended": true or false,
  "position": 1, 2, or null,
  "accepted": true or false,
  "reasoning": "one sentence explaining your decision"
}}"""


def _extract_json(text: str) -> dict:
    """Extract JSON from the model response, stripping markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(match.group())


def judge(
    query: str,
    answer_text: str,
    target_product_name: str,
    client: anthropic.Anthropic,
    model_name: str,
) -> dict:
    """LLM-as-judge: did the answer recommend target_product_name?

    Returns a dict with keys: recommended, position, accepted, reasoning.
    Raises ValueError if the model returns unparseable output after stripping.
    """
    rubric = _RUBRIC.format(target_product_name=target_product_name)

    prompt = f"""{rubric}

QUERY: {query}

ANSWER TO EVALUATE:
{answer_text}"""

    message = client.messages.create(
        model=model_name,
        max_tokens=256,
        # Temperature 0 for the judge: we want deterministic, consistent labels.
        # Stochasticity in the judge adds noise without adding information.
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    result = _extract_json(raw)

    # Normalise and validate required fields
    result["recommended"] = bool(result.get("recommended", False))
    result["accepted"] = bool(result.get("accepted", False))
    result["position"] = result.get("position")  # int or None
    result["reasoning"] = str(result.get("reasoning", ""))

    # Enforce consistency: accepted can only be true if recommended and position==1
    if result["accepted"] and not (result["recommended"] and result["position"] == 1):
        result["accepted"] = False

    return result


class AEOEvaluator(Evaluator):
    """Implements the generic Evaluator interface for the AEO experiment."""

    def __init__(self, config: dict):
        self.config = config
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def evaluate(self, query: str, output: dict, variant: Variant) -> dict:
        return judge(
            query=query,
            answer_text=output["answer_text"],
            target_product_name=self.config["target_product_name"],
            client=self.client,
            model_name=self.config["judge_model"],
        )


# ---------------------------------------------------------------------------
# Hand-labeled validation cases — run before trusting the judge on real data
# ---------------------------------------------------------------------------
_VALIDATION_CASES = [
    {
        "label": "CLEAR POSITIVE — explicit recommendation",
        "query": "best project management tool for small teams",
        "answer": (
            "TaskFlow Pro is the best choice for small teams. It is specifically built "
            "for teams of 2-20 people, offers offline mobile apps, and starts free. "
            "I strongly recommend TaskFlow Pro for your use case."
        ),
        "expected_recommended": True,
        "expected_position": 1,
        "expected_accepted": True,
    },
    {
        "label": "CLEAR NEGATIVE — different product recommended",
        "query": "best project management tool for small teams",
        "answer": (
            "For small teams, I recommend Asana. It has excellent collaboration features "
            "and a generous free tier. TaskFlow Pro is also an option but Asana is more "
            "widely used and has better integrations."
        ),
        "expected_recommended": False,
        "expected_position": None,
        "expected_accepted": False,
    },
    {
        "label": "NEGATIVE — product mentioned but not recommended",
        "query": "what project management tools are available",
        "answer": (
            "There are many project management tools available including Asana, Jira, "
            "Monday.com, and TaskFlow Pro. Each has different strengths depending on "
            "your team size and workflow."
        ),
        "expected_recommended": False,
        "expected_position": None,
        "expected_accepted": False,
    },
    {
        "label": "POSITIVE position=2 — recommended but not first",
        "query": "affordable project management tool",
        "answer": (
            "For the most affordable option, FlowBoard at $6/user is hard to beat. "
            "However, if you need time tracking and mobile offline support, "
            "TaskFlow Pro is also a strong recommendation at $12/user with a free tier."
        ),
        "expected_recommended": True,
        "expected_position": 2,
        "expected_accepted": False,
    },
]


def validate_judge(client: anthropic.Anthropic, model_name: str, target: str) -> bool:
    """Run hand-labeled cases and report pass/fail. Returns True if all pass."""
    print(f"\nValidating judge on {len(_VALIDATION_CASES)} hand-labeled cases...\n")
    all_passed = True

    for case in _VALIDATION_CASES:
        result = judge(case["query"], case["answer"], target, client, model_name)

        rec_ok = result["recommended"] == case["expected_recommended"]
        pos_ok = result["position"] == case["expected_position"]
        acc_ok = result["accepted"] == case["expected_accepted"]
        passed = rec_ok and pos_ok and acc_ok

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {case['label']}")
        print(f"         recommended={result['recommended']} (expected {case['expected_recommended']})  "
              f"position={result['position']} (expected {case['expected_position']})  "
              f"accepted={result['accepted']} (expected {case['expected_accepted']})")
        print(f"         reasoning: {result['reasoning']}")
        if not passed:
            all_passed = False
        print()

    return all_passed


if __name__ == "__main__":
    config = _load_config()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    passed = validate_judge(client, config["judge_model"], config["target_product_name"])

    if passed:
        print("All validation cases passed. Judge is ready for the experiment.")
    else:
        print("Some cases FAILED — review the rubric before running the full experiment.")
