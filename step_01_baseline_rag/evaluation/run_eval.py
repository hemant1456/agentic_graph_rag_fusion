import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS, GoldenQuestion
from step_01_baseline_rag.implementation.pipeline import BaselineRAG, RAGResult


def score(result: RAGResult, question: GoldenQuestion) -> dict:
    """
    FAIL   — any disqualifier found in answer (wrong answer caught)
    PASS   — all required_facts found (and no disqualifiers)
    PARTIAL— some required_facts missing, but partial_facts found
    FAIL   — nothing useful found
    """
    answer_lower = result.answer.lower()

    disqualifier_hits = [d for d in question.disqualifiers if d.lower() in answer_lower]
    required_hits = [f for f in question.required_facts if f.lower() in answer_lower]
    partial_hits = [f for f in question.partial_facts if f.lower() in answer_lower]

    all_required = len(required_hits) == len(question.required_facts)

    if disqualifier_hits:
        grade = "FAIL"
    elif all_required:
        grade = "PASS"
    elif required_hits or partial_hits:
        grade = "PARTIAL"
    else:
        grade = "FAIL"

    return {
        "grade": grade,
        "required_hits": required_hits,
        "required_missing": [f for f in question.required_facts if f not in required_hits],
        "partial_hits": partial_hits,
        "disqualifier_hits": disqualifier_hits,
    }


GRADE_SYMBOL = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}
GRADE_COLOR  = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}
RESET = "\033[0m"


def print_result(q: GoldenQuestion, result: RAGResult, scoring: dict) -> None:
    grade = scoring["grade"]
    color = GRADE_COLOR[grade]
    sym = GRADE_SYMBOL[grade]
    print(f"\n{color}[{sym}] {q.id} — {q.type}{RESET}")
    print(f"  Q: {q.question}")
    print(f"  A: {result.answer[:300]}{'...' if len(result.answer) > 300 else ''}")
    print(f"  Sources: {[c.source for c in result.retrieved_chunks]}")
    print(f"  Required hits: {scoring['required_hits']} | Missing: {scoring['required_missing']}")
    print(f"  Latency: retrieval={result.retrieval_latency_ms:.0f}ms  gen={result.generation_latency_ms:.0f}ms")
    print(f"  Expected: {q.expected_outcome} → Got: {grade}")


def run_evaluation() -> dict:
    print("=== Step 01: Baseline RAG Evaluation ===\n")

    rag = BaselineRAG(k=5)
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
        "step": "01_baseline_rag",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": total,
        "grade_counts": grade_counts,
        "pass_rate": round(pass_rate, 2),
        "pass_or_partial_rate": round(pass_partial, 2),
        "results": all_results,
    }

    print(f"\n{'='*50}")
    print(f"RESULTS: {grade_counts['PASS']} PASS | {grade_counts['PARTIAL']} PARTIAL | {grade_counts['FAIL']} FAIL")
    print(f"Pass rate: {pass_rate:.0%}  |  Pass+Partial: {pass_partial:.0%}")
    print(f"{'='*50}\n")

    out_path = Path(__file__).parent.parent / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
