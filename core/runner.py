import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from core.interfaces import Variant
from core.store import append_row, read_results
from tasks.aeo.answer_engine import AEOTask
from tasks.aeo.corpus import build_corpora
from tasks.aeo.judge import AEOEvaluator

load_dotenv()


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _already_done(done: set, query_id: str, arm: str, repeat: int) -> bool:
    return (query_id, arm, str(repeat)) in done


def _done_set(results: list[dict]) -> set:
    return {(r["query_id"], r["arm"], r["repeat"]) for r in results}


def _run_trial(
    query_text: str,
    variant: Variant,
    task: AEOTask,
    evaluator: AEOEvaluator,
) -> dict:
    """One trial: answer + judge. Retries up to 3 times on transient errors."""
    for attempt in range(3):
        try:
            output = task.run(query_text, variant)
            scores = evaluator.evaluate(query_text, output, variant)
            return output, scores
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            tqdm.write(f"    retry {attempt + 1}/3 after error: {e} (waiting {wait}s)")
            time.sleep(wait)


def run_experiment(config_path: str = "config.yaml") -> None:
    config = _load_config(config_path)
    results_path = config["paths"]["results"]
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    queries = json.load(open(config["paths"]["queries"]))
    n_repeats = config["n_repeats"]
    variants = [
        Variant(name="control",   config={"arm": "A"}),
        Variant(name="treatment", config={"arm": "B"}),
    ]

    # Check existing results for resumability
    existing = []
    if Path(results_path).exists() and Path(results_path).stat().st_size > 0:
        existing = read_results(results_path)
    done = _done_set(existing)

    total = len(queries) * len(variants) * n_repeats
    todo  = total - len(done)

    # Each trial = 1 answer call + 1 judge call ≈ 2 LLM calls
    # Haiku is very cheap; 800 calls ≈ a few cents to ~$1 depending on answer length
    print("\n" + "=" * 55)
    print("EXPERIMENT PLAN")
    print("=" * 55)
    print(f"  Queries   : {len(queries)}")
    print(f"  Arms      : {len(variants)}  (control A + treatment B)")
    print(f"  Repeats   : {n_repeats}")
    print(f"  Total     : {total} trials  ({total * 2} LLM calls)")
    print(f"  Done      : {len(done)}  |  Remaining: {todo}")
    print(f"  Model     : {config['answer_model']}  (answer + judge)")
    print(f"  Approx cost: < $2 at Haiku rates for a full run")
    print("=" * 55 + "\n")

    if todo == 0:
        print("All trials already complete — check results.csv.")
        _print_summary(results_path)
        return

    print("Building corpora (loads embedding model + distractor docs)...")
    index_a, index_b, embed_model = build_corpora(config_path)

    task      = AEOTask(index_a, index_b, embed_model, config)
    evaluator = AEOEvaluator(config)

    print(f"\nRunning {todo} trials...\n")

    completed = 0
    errors    = 0

    # Interleave arms per (query, repeat) to prevent time-based confounds:
    # if Claude drifts or rate-limits during a long run, both arms are affected
    # equally rather than one arm absorbing all the drift.
    for q in tqdm(queries, desc="Queries", unit="q"):
        for repeat in range(1, n_repeats + 1):
            for variant in variants:
                arm = "A" if variant.name == "control" else "B"

                if _already_done(done, q["id"], arm, repeat):
                    continue

                try:
                    output, scores = _run_trial(q["text"], variant, task, evaluator)

                    row = {
                        "query_id"   : q["id"],
                        "arm"        : arm,
                        "repeat"     : repeat,
                        "recommended": int(scores["recommended"]),
                        "position"   : scores["position"] if scores["position"] else "",
                        "accepted"   : int(scores["accepted"]),
                        "reasoning"  : scores["reasoning"].replace("\n", " "),
                    }
                    append_row(results_path, row)
                    done.add((q["id"], arm, str(repeat)))
                    completed += 1

                    rec_flag = "✓" if scores["recommended"] else "·"
                    tqdm.write(
                        f"  [{rec_flag}] {q['id']} arm={arm} rep={repeat}  "
                        f"recommended={scores['recommended']}"
                    )

                    # Small pause — keeps us well inside Claude Haiku's rate limits
                    time.sleep(0.25)

                except Exception as e:
                    errors += 1
                    tqdm.write(f"  [ERROR] {q['id']} arm={arm} rep={repeat}: {e}")
                    time.sleep(3)

    print(f"\nFinished: {completed} new trials written, {errors} errors.")
    _print_summary(results_path)


def _print_summary(results_path: str) -> None:
    results = read_results(results_path)
    if not results:
        return

    a = [r for r in results if r["arm"] == "A"]
    b = [r for r in results if r["arm"] == "B"]

    def rate(rows):
        return sum(int(r["recommended"]) for r in rows) / len(rows) if rows else 0

    p_a, p_b = rate(a), rate(b)
    lift = (p_b - p_a) / p_a if p_a > 0 else float("nan")

    print("\n" + "=" * 55)
    print("PRELIMINARY RESULTS  (run stats.py for significance)")
    print("=" * 55)
    print(f"  Arm A (control)  : {p_a:.1%}  ({int(p_a*len(a))}/{len(a)} recommended)")
    print(f"  Arm B (treatment): {p_b:.1%}  ({int(p_b*len(b))}/{len(b)} recommended)")
    print(f"  Raw lift         : {lift:+.1%}")
    print("=" * 55)


if __name__ == "__main__":
    run_experiment()
