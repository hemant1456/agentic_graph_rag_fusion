from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from step_01_baseline_rag.evaluation.run_eval import print_result, score
from step_08_context_engineering.implementation.pipeline import Step08RAG


def run_evaluation() -> dict:
    print("=== Step 10: Context Engineering Evaluation ===\n")
    print("Techniques: CrossEncoder reranking → deduplication → extractive compression → XML formatting\n")

    rag = Step08RAG(k=5, rerank_k=8, compress_ratio=0.60).build()

    all_results = []
    grade_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    ce_totals = {"raw_chars": 0, "engineered_chars": 0, "chunks_before": 0, "chunks_final": 0}

    for q in GOLDEN_QUESTIONS:
        ext = rag.query_extended(q.question)
        result = ext.rag_result
        ce = ext.ce_metrics

        scoring = score(result, q)
        grade_counts[scoring["grade"]] += 1
        print_result(q, result, scoring)
        print(
            f"  CE: raw={ce['raw_chars']:,} chars → engineered={ce['engineered_chars']:,} chars "
            f"({ce['compression_ratio']:.0%}) | chunks {ce['chunks_before']}→{ce['chunks_final']}"
        )

        for k in ce_totals:
            ce_totals[k] += ce.get(k, 0)

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
            "context_chars": result.context_chars,
            "retrieval_latency_ms": round(result.retrieval_latency_ms, 1),
            "generation_latency_ms": round(result.generation_latency_ms, 1),
            "provider": result.provider,
            "ce_raw_chars": ce["raw_chars"],
            "ce_engineered_chars": ce["engineered_chars"],
            "ce_compression_ratio": ce["compression_ratio"],
            "ce_chunks_before": ce["chunks_before"],
            "ce_chunks_after_dedup": ce["chunks_after_dedup"],
            "ce_chunks_final": ce["chunks_final"],
        })

    n = len(GOLDEN_QUESTIONS)
    pass_rate   = grade_counts["PASS"] / n
    pass_partial = (grade_counts["PASS"] + grade_counts["PARTIAL"]) / n
    avg_compression = ce_totals["engineered_chars"] / max(ce_totals["raw_chars"], 1)
    avg_chunks_before = ce_totals["chunks_before"] / n
    avg_chunks_final  = ce_totals["chunks_final"] / n

    summary = {
        "step": "10_context_engineering",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": n,
        "grade_counts": grade_counts,
        "pass_rate": round(pass_rate, 2),
        "pass_or_partial_rate": round(pass_partial, 2),
        "ce_summary": {
            "avg_compression_ratio": round(avg_compression, 3),
            "avg_chunks_before": round(avg_chunks_before, 1),
            "avg_chunks_final": round(avg_chunks_final, 1),
            "total_raw_chars": ce_totals["raw_chars"],
            "total_engineered_chars": ce_totals["engineered_chars"],
        },
        "results": all_results,
    }

    print(f"\n{'='*55}")
    print(f"RESULTS: {grade_counts['PASS']} PASS | {grade_counts['PARTIAL']} PARTIAL | {grade_counts['FAIL']} FAIL")
    print(f"Pass rate: {pass_rate:.0%}  |  Pass+Partial: {pass_partial:.0%}")
    print(f"Context engineering: avg compression {avg_compression:.0%}  "
          f"| avg chunks {avg_chunks_before:.0f} → {avg_chunks_final:.0f}")
    print(f"{'='*55}\n")

    out_path = Path(__file__).parent.parent / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")
    return summary


if __name__ == "__main__":
    run_evaluation()
