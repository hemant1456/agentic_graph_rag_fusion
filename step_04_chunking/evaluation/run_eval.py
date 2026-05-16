import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from step_01_baseline_rag.evaluation.run_eval import print_result, score
from step_04_chunking.implementation.pipeline import Step04RAG


def run_evaluation() -> dict:
    print("=== Step 04: Format-aware Chunking Evaluation ===\n")

    rag = Step04RAG(k=10)
    rag.build()

    all_results = []
    grade_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    for q in GOLDEN_QUESTIONS:
        result = rag.query(q.question)
        scoring = score(result, q)
        grade_counts[scoring["grade"]] += 1
        print_result(q, result, scoring)

        all_results.append({
            "id": q.id,
            "type": q.type,
            "question": q.question,
            "answer": result.answer,
            "grade": scoring["grade"],
            "expected_outcome": q.expected_outcome,
            "matched_expected": scoring["grade"] == q.expected_outcome,
            "required_hits": scoring["required_hits"],
            "required_missing": scoring["required_missing"],
            "sources_retrieved": [c.source for c in result.retrieved_chunks],
            "source_similarities": [round(c.similarity, 3) for c in result.retrieved_chunks],
            "context_chars": result.context_chars,
            "retrieval_latency_ms": round(result.retrieval_latency_ms, 1),
            "generation_latency_ms": round(result.generation_latency_ms, 1),
            "provider": result.provider,
        })

    total = len(GOLDEN_QUESTIONS)
    pass_rate = grade_counts["PASS"] / total
    pass_partial = (grade_counts["PASS"] + grade_counts["PARTIAL"]) / total

    summary = {
        "step": "04_chunking",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": total,
        "grade_counts": grade_counts,
        "pass_rate": round(pass_rate, 2),
        "pass_or_partial_rate": round(pass_partial, 2),
        "results": all_results,
    }

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {grade_counts['PASS']} PASS | {grade_counts['PARTIAL']} PARTIAL | {grade_counts['FAIL']} FAIL")
    print(f"Pass rate: {pass_rate:.0%}  |  Pass+Partial: {pass_partial:.0%}")
    print(f"{'=' * 50}\n")

    out_path = Path(__file__).parent.parent / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
