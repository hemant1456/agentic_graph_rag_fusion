"""Backward-compat string-match scorer.

The primary eval framework is now RAGAS-based at evaluation/run_eval.py.
This module is retained as a thin substring-match helper used by dashboard.py
and observability/evaluation/run_traced_eval.py for ad-hoc single-question grading.
"""
from __future__ import annotations

from step_01_baseline_rag.evaluation.golden_questions import GoldenQuestion
from step_01_baseline_rag.implementation.pipeline import RAGResult


def score(result: RAGResult, question: GoldenQuestion) -> dict:
    """Quick substring grading. Not the official scorer — RAGAS is."""
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
