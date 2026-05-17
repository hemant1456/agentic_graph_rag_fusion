from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from step_01_baseline_rag.evaluation.run_eval import print_result, score
from step_09_vsa.implementation.pipeline import Step09RAG
from step_09_vsa.implementation.router import all_slice_configs


def run_evaluation() -> dict:
    print("=== Step 11: Vertical Slice Architecture Evaluation ===\n")
    print("Routing each question to a domain slice before synthesis.\n")

    rag = Step09RAG(k=5).build()

    print("Registered slices:")
    for cfg in all_slice_configs():
        owns = ", ".join(cfg.owns_questions) if cfg.owns_questions else "(auto-routed)"
        print(f"  [{cfg.name}] {cfg.display_name} — owns {owns}")
    print()

    all_results = []
    grade_counts: dict[str, int]        = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    slice_grades: dict[str, dict]       = defaultdict(lambda: {"PASS": 0, "PARTIAL": 0, "FAIL": 0})
    slice_routing: dict[str, list[str]] = defaultdict(list)

    for q in GOLDEN_QUESTIONS:
        ext = rag.query_extended(q.question)
        result  = ext.rag_result
        ce      = ext.ce_metrics
        slice_n = ext.slice_name
        conf    = ext.router_confidence

        scoring = score(result, q)
        grade_counts[scoring["grade"]] += 1
        slice_grades[slice_n][scoring["grade"]] += 1
        slice_routing[slice_n].append(q.id)

        print_result(q, result, scoring)
        print(
            f"  Slice: {slice_n} (conf={conf:.2f}) | "
            f"CE: {ce.get('raw_chars', 0):,} → {ce.get('engineered_chars', 0):,} chars "
            f"({ce.get('compression_ratio', 1.0):.0%})"
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
            "slice_used": slice_n,
            "router_confidence": round(conf, 3),
            "ce_raw_chars": ce.get("raw_chars", 0),
            "ce_engineered_chars": ce.get("engineered_chars", 0),
            "ce_compression_ratio": ce.get("compression_ratio", 1.0),
            "ce_chunks_before": ce.get("chunks_before", 0),
            "ce_chunks_final": ce.get("chunks_final", 0),
        })

    n = len(GOLDEN_QUESTIONS)
    pass_rate    = grade_counts["PASS"] / n
    pass_partial = (grade_counts["PASS"] + grade_counts["PARTIAL"]) / n

    slice_summary = {}
    for sl, gc in slice_grades.items():
        total_sl = sum(gc.values())
        slice_summary[sl] = {
            "questions_handled": slice_routing[sl],
            "grade_counts": gc,
            "pass_rate": round(gc["PASS"] / total_sl, 2) if total_sl else 0,
        }

    summary = {
        "step": "11_vsa",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": n,
        "grade_counts": grade_counts,
        "pass_rate": round(pass_rate, 2),
        "pass_or_partial_rate": round(pass_partial, 2),
        "slice_summary": slice_summary,
        "results": all_results,
    }

    print(f"\n{'='*55}")
    print(f"RESULTS: {grade_counts['PASS']} PASS | {grade_counts['PARTIAL']} PARTIAL | {grade_counts['FAIL']} FAIL")
    print(f"Pass rate: {pass_rate:.0%}  |  Pass+Partial: {pass_partial:.0%}")
    print("\nPer-slice breakdown:")
    for sl, info in slice_summary.items():
        gc = info["grade_counts"]
        qs = ", ".join(info["questions_handled"])
        print(f"  [{sl}] {gc['PASS']}P/{gc['PARTIAL']}~/{gc['FAIL']}F  ({info['pass_rate']:.0%})  — {qs}")
    print(f"{'='*55}\n")

    out_path = Path(__file__).parent.parent / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")
    return summary


if __name__ == "__main__":
    run_evaluation()
