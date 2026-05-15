"""
Step 02 — Observability: traced evaluation run.

Re-runs all 22 golden questions through the traced pipeline.
Identical scores to Step 01 (same retrieval + generation) — this run adds
per-query observability: token counts, cost, exact sources retrieved.

Output:
  step_02_observability/results/traces.jsonl  — one trace per question
  step_02_observability/results/eval_report.json — scored results + cost summary

Usage:
    # JSONL traces only (no external server needed):
    uv run python step_02_observability/evaluation/run_traced_eval.py

    # JSONL + Arize Phoenix UI at http://localhost:6006:
    uv run python step_02_observability/evaluation/run_traced_eval.py --phoenix
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from step_01_baseline_rag.evaluation.run_eval import score
from step_01_baseline_rag.implementation.pipeline import BaselineRAG
from step_02_observability.implementation.report import print_summary
from step_02_observability.implementation.traced_pipeline import TracedRAG
from step_02_observability.implementation.tracer import TraceStore

TRACE_FILE = Path(__file__).parent.parent / "results" / "traces.jsonl"
REPORT_FILE = Path(__file__).parent.parent / "results" / "eval_report.json"

GRADE_COLOR  = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}
GRADE_SYMBOL = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}
RESET = "\033[0m"


def run_traced_evaluation(use_phoenix: bool = False) -> dict:
    print("=== Step 02: Traced RAG Evaluation ===\n")

    otel_tracer = None
    if use_phoenix:
        from step_02_observability.implementation.phoenix_exporter import PhoenixExporter
        exporter = PhoenixExporter()
        otel_tracer = exporter.start()
        print(f"Traces visible at {exporter.url}\n")

    rag = BaselineRAG(k=5).build()
    store = TraceStore(TRACE_FILE)
    store.clear()   # fresh trace file on each run

    traced = TracedRAG(rag, store, otel_tracer=otel_tracer)

    grade_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    all_results = []

    print(f"{'ID':<6} {'GRADE':<8} {'MS':>6} {'TOK':>6} {'COST':>10}  TOP SOURCES")
    print("-" * 72)

    for q in GOLDEN_QUESTIONS:
        result, trace = traced.query(q.question)
        scoring = score(result, q)
        grade = scoring["grade"]
        grade_counts[grade] += 1

        tokens = trace.generation.total_tokens
        sources_str = ", ".join(trace.retrieval.unique_sources[:2])
        if len(trace.retrieval.unique_sources) > 2:
            sources_str += f" (+{len(trace.retrieval.unique_sources) - 2})"

        color, sym = GRADE_COLOR[grade], GRADE_SYMBOL[grade]
        print(
            f"{q.id:<6} "
            f"{color}[{sym}] {grade:<5}{RESET} "
            f"{trace.total_latency_ms:>5.0f}ms "
            f"{tokens:>6} "
            f"  ${trace.generation.estimated_cost_usd:.6f}"
            f"  {sources_str}"
        )

        all_results.append({
            "id": q.id,
            "type": q.type,
            "question": q.question,
            "grade": grade,
            "expected_outcome": q.expected_outcome,
            "matched_expected": grade == q.expected_outcome,
            "trace_id": trace.trace_id,
            "total_latency_ms": trace.total_latency_ms,
            "retrieval_ms": trace.retrieval.duration_ms,
            "generation_ms": trace.generation.duration_ms,
            "prompt_tokens": trace.generation.prompt_tokens,
            "completion_tokens": trace.generation.completion_tokens,
            "total_tokens": tokens,
            "estimated_cost_usd": trace.generation.estimated_cost_usd,
            "provider": trace.generation.provider,
            "top_source": trace.retrieval.top_source,
            "unique_sources": trace.retrieval.unique_sources,
            "required_hits": scoring["required_hits"],
            "required_missing": scoring["required_missing"],
        })

    total = len(GOLDEN_QUESTIONS)
    pass_rate = grade_counts["PASS"] / total
    total_tokens = sum(r["total_tokens"] for r in all_results)
    total_cost = sum(r["estimated_cost_usd"] for r in all_results)
    avg_ms = sum(r["total_latency_ms"] for r in all_results) / total

    print(f"\n{'='*60}")
    print(f"RESULTS : {grade_counts['PASS']} PASS | {grade_counts['PARTIAL']} PARTIAL | {grade_counts['FAIL']} FAIL")
    print(f"Pass rate: {pass_rate:.0%}")
    print(f"Tokens   : {total_tokens:,}  ({total_tokens / total:.0f}/query avg)")
    print(f"Cost     : ${total_cost:.5f}  (${total_cost / total:.6f}/query avg)")
    print(f"Latency  : {avg_ms:.0f}ms avg")
    print(f"{'='*60}")

    print(f"\nTraces saved → {TRACE_FILE}")
    print(f"\nInspect a trace:")
    first_id = all_results[0]["trace_id"]
    print(f"  uv run python step_02_observability/implementation/report.py --trace-id {first_id}")
    print(f"  uv run python step_02_observability/implementation/report.py --sources {first_id}")
    print(f"  uv run python step_02_observability/implementation/report.py --summary")

    report = {
        "step": "02_observability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": total,
        "grade_counts": grade_counts,
        "pass_rate": round(pass_rate, 2),
        "total_tokens": total_tokens,
        "avg_tokens_per_query": round(total_tokens / total),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query_usd": round(total_cost / total, 7),
        "avg_latency_ms": round(avg_ms, 1),
        "trace_file": str(TRACE_FILE),
        "results": all_results,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {REPORT_FILE}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 02: traced RAG evaluation")
    parser.add_argument(
        "--phoenix", action="store_true",
        help="Launch Arize Phoenix UI and send spans (requires step-02 extras)",
    )
    args = parser.parse_args()
    run_traced_evaluation(use_phoenix=args.phoenix)
