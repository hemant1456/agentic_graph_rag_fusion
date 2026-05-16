"""
Evaluation runner for Step 12 — Production Hardening.

In addition to standard grade tallying, this runner records:
  - confidence_score / confidence_label  — answer quality heuristic
  - from_cache                           — whether the semantic cache was hit
  - health snapshot at the end           — p50/p95 latency, SLO compliance

Usage:
    uv run python step_12_production/evaluation/run_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from step_01_baseline_rag.evaluation.run_eval import print_result, score
from step_12_production.implementation.pipeline import Step12RAG


def run_evaluation() -> dict:
    print("=== Step 12: Production Hardening Evaluation ===\n")
    print("Each question: semantic cache -> retry -> fallback -> confidence -> health monitor\n")

    rag = Step12RAG(k=5).build()

    all_results = []
    grade_counts: dict[str, int] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    cache_hits = 0

    for q in GOLDEN_QUESTIONS:
        ext = rag.query_extended(q.question)
        result = ext.rag_result
        scoring = score(result, q)
        grade_counts[scoring["grade"]] += 1
        if ext.from_cache:
            cache_hits += 1

        print_result(q, result, scoring)
        print(
            f"  Slice: {ext.slice_name} | Conf: {ext.confidence_label} ({ext.confidence_score:.2f}) | "
            f"Cache: {'HIT' if ext.from_cache else 'miss'} | "
            f"Latency: {result.retrieval_latency_ms:.0f}ms"
        )

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
            "slice_used": ext.slice_name,
            "router_confidence": round(ext.router_confidence, 3),
            "confidence_score": ext.confidence_score,
            "confidence_label": ext.confidence_label,
            "from_cache": ext.from_cache,
            "ce_raw_chars": ext.ce_metrics.get("raw_chars", 0),
            "ce_engineered_chars": ext.ce_metrics.get("engineered_chars", 0),
            "ce_compression_ratio": ext.ce_metrics.get("compression_ratio", 1.0),
        })

    n = len(GOLDEN_QUESTIONS)
    pass_rate    = grade_counts["PASS"] / n
    pass_partial = (grade_counts["PASS"] + grade_counts["PARTIAL"]) / n

    health = rag._monitor.snapshot() if rag._monitor else {}
    cache_stats = rag._cache.stats if rag._cache else {}

    summary = {
        "step": "12_production",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": n,
        "grade_counts": grade_counts,
        "pass_rate": round(pass_rate, 2),
        "pass_or_partial_rate": round(pass_partial, 2),
        "production_metrics": {
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / n, 2),
            "health": health,
            "cache_stats": cache_stats,
        },
        "results": all_results,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {grade_counts['PASS']} PASS | {grade_counts['PARTIAL']} PARTIAL | {grade_counts['FAIL']} FAIL")
    print(f"Pass rate: {pass_rate:.0%}  |  Pass+Partial: {pass_partial:.0%}")
    print(f"Cache hits: {cache_hits}/{n}  |  Health: {health.get('status','?')}")
    print(f"p50 latency: {health.get('p50_latency_ms','?')}ms  |  p95: {health.get('p95_latency_ms','?')}ms")
    print(f"SLO compliance (< 10s): {health.get('slo_compliance', 0):.0%}")
    print(f"{'='*60}\n")

    out_path = Path(__file__).parent.parent / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved -> {out_path}")
    return summary


if __name__ == "__main__":
    run_evaluation()
