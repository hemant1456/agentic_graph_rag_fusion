"""
Evaluation runner for Step 05 — CSV Tool Calling.

Usage:
    uv run python step_03_tools/evaluation/run_eval.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from step_01_baseline_rag.evaluation.run_eval import print_result, score
from step_03_tools.implementation.pipeline import Step03ToolsRAG


def run_evaluation() -> dict:
    print("=== Step 05: CSV Tool Calling ===\n")

    rag = Step03ToolsRAG(k=10).build()

    all_results = []
    grade_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    for q in GOLDEN_QUESTIONS:
        result = rag.query(q.question)
        scoring = score(result, q)
        grade_counts[scoring["grade"]] += 1
        all_results.append({"question_id": q.id, **scoring})
        print_result(q, result, scoring)

    total = len(GOLDEN_QUESTIONS)
    summary = {
        "step": "step_03_tools",
        "pass": grade_counts["PASS"],
        "partial": grade_counts["PARTIAL"],
        "fail": grade_counts["FAIL"],
        "total": total,
        "pass_rate": grade_counts["PASS"] / total,
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = Path(__file__).parent.parent / "results" / "eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 50}")
    print(f"PASS: {grade_counts['PASS']}  PARTIAL: {grade_counts['PARTIAL']}  FAIL: {grade_counts['FAIL']}")
    print(f"Pass rate: {grade_counts['PASS']}/{total} ({grade_counts['PASS']/total:.0%})")
    print(f"Results saved to {out_path}")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    run_evaluation()
