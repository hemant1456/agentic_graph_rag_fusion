"""RAGAS-based evaluation for the 10-step RAG progression.

Structure (deliberately flat and readable):

  1. Per-step `answer_step_NN(question)` functions — each builds the step's
     pipeline (cached) and returns {"answer": str, "contexts": list[str]}.
  2. `evaluate_step(step_name)` — loops over GOLDEN_QUESTIONS, calls the
     step's answer function, builds a RAGAS EvaluationDataset, scores it
     with RAGAS answer_correctness, writes results to
     <step>/results/eval_results.json.
  3. CLI: `--step <name>` or `--all` or `--list`.

LLM judge calls go through llm_gatewayV2 (free-tier providers).
Embeddings use HuggingFace all-MiniLM-L6-v2 (local, no API).

Usage:
    uv run python evaluation/run_eval.py --list
    uv run python evaluation/run_eval.py --step step_04_hybrid_retrieval
    uv run python evaluation/run_eval.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from evaluation.judge_llm import build_judge_llm
from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS


# ── Per-step answer functions ────────────────────────────────────────────────
# Each returns {"answer": str, "contexts": list[str]}.
# Pipelines are cached at module level so we build once per step.

_pipelines: dict[str, Any] = {}


def _cached(step_name: str, builder: Callable[[], Any]) -> Any:
    if step_name not in _pipelines:
        _pipelines[step_name] = builder()
    return _pipelines[step_name]


def _result_to_sample(r) -> dict:
    return {"answer": r.answer, "contexts": [c.text for c in r.retrieved_chunks]}


def answer_step_01(question: str) -> dict:
    from step_01_baseline_rag.implementation.pipeline import BaselineRAG
    rag = _cached("step_01", lambda: BaselineRAG(k=5).build())
    return _result_to_sample(rag.query(question))


def answer_step_02(question: str) -> dict:
    from step_02_chunking.implementation.pipeline import Step02RAG
    rag = _cached("step_02", lambda: Step02RAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_03(question: str) -> dict:
    from step_03_tools.implementation.pipeline import Step03ToolsRAG
    rag = _cached("step_03", lambda: Step03ToolsRAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_04(question: str) -> dict:
    from step_04_hybrid_retrieval.implementation.pipeline import Step04HybridRAG
    rag = _cached("step_04", lambda: Step04HybridRAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_05(question: str) -> dict:
    from step_05_knowledge_graph.implementation.pipeline import Step05RAG
    rag = _cached("step_05", lambda: Step05RAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_06(question: str) -> dict:
    from step_06_graph_rag.implementation.pipeline import Step06RAG
    rag = _cached("step_06", lambda: Step06RAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_07(question: str) -> dict:
    from step_07_multi_agent.implementation.pipeline import Step07RAG
    rag = _cached("step_07", lambda: Step07RAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_08(question: str) -> dict:
    from step_08_context_engineering.implementation.pipeline import Step08RAG
    rag = _cached("step_08", lambda: Step08RAG(k=5, rerank_k=8, compress_ratio=0.60).build())
    return _result_to_sample(rag.query(question))


def answer_step_09(question: str) -> dict:
    from step_09_vsa.implementation.pipeline import Step09RAG
    rag = _cached("step_09", lambda: Step09RAG(k=5).build())
    return _result_to_sample(rag.query(question))


def answer_step_10(question: str) -> dict:
    from step_10_production.implementation.pipeline import Step10RAG
    rag = _cached("step_10", lambda: Step10RAG(k=5).build())
    return _result_to_sample(rag.query(question))


STEPS: dict[str, Callable[[str], dict]] = {
    "step_01_baseline_rag":         answer_step_01,
    "step_02_chunking":             answer_step_02,
    "step_03_tools":                answer_step_03,
    "step_04_hybrid_retrieval":     answer_step_04,
    "step_05_knowledge_graph":      answer_step_05,
    "step_06_graph_rag":            answer_step_06,
    "step_07_multi_agent":          answer_step_07,
    "step_08_context_engineering":  answer_step_08,
    "step_09_vsa":                  answer_step_09,
    "step_10_production":           answer_step_10,
}


# ── RAGAS scoring ─────────────────────────────────────────────────────────────


def _reference_from_question(q) -> str:
    """Return the natural-language gold answer for RAGAS to compare against.

    Falls back to a synthetic "must contain X" string for questions without an
    explicit reference_answer (used only as a safety net — every question in
    golden_questions.py should have one).
    """
    if getattr(q, "reference_answer", "").strip():
        return q.reference_answer
    facts = ", ".join(q.required_facts) if q.required_facts else "(no specific facts required)"
    return f"The correct answer must contain these facts: {facts}. {q.explanation}"


def _build_ragas_objects() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    llm = LangchainLLMWrapper(build_judge_llm(temperature=0.0, max_tokens=1024))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return llm, embeddings


def evaluate_step(step_name: str) -> dict:
    if step_name not in STEPS:
        raise SystemExit(f"Unknown step '{step_name}'. Available: {', '.join(STEPS)}")

    answer_fn = STEPS[step_name]
    print(f"\n=== {step_name} — collecting answers for {len(GOLDEN_QUESTIONS)} questions ===\n")

    samples: list[dict] = []
    for q in GOLDEN_QUESTIONS:
        t0 = time.perf_counter()
        out = answer_fn(q.question)
        latency_ms = (time.perf_counter() - t0) * 1000
        samples.append({
            "question_id": q.id,
            "user_input": q.question,
            "response": out["answer"],
            "retrieved_contexts": out["contexts"],
            "reference": _reference_from_question(q),
            "_latency_ms": round(latency_ms, 1),
            "_required_facts": q.required_facts,
        })
        print(f"  [{q.id}] {q.question[:70]}... ({latency_ms:.0f}ms)")

    print(f"\n=== Scoring with RAGAS (5 metrics) ===\n")
    llm, embeddings = _build_ragas_objects()
    ragas_inputs = [
        {k: s[k] for k in ("user_input", "response", "retrieved_contexts", "reference")}
        for s in samples
    ]
    dataset = Dataset.from_list(ragas_inputs)

    # Judge runs via llm_gatewayV2 with JUDGE_PROVIDERS=groq,gemini.
    # Groq is fast (~200ms/call) so max_workers=4 batches without burning 30 RPM.
    run_cfg = RunConfig(timeout=60, max_retries=3, max_workers=4)
    metrics = [
        AnswerCorrectness(),
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_cfg,
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()  # type: ignore[attr-defined]

    def _safe(col: str, i: int) -> float:
        v = df.iloc[i].get(col, 0) if i < len(df) else 0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return f if f == f else 0.0  # NaN guard

    # Overall grade is based on answer_correctness (the most direct factual check).
    grades: list[str] = []
    rows: list[dict] = []
    for i, s in enumerate(samples):
        ac = _safe("answer_correctness", i)
        fa = _safe("faithfulness", i)
        ar = _safe("answer_relevancy", i)
        cp = _safe("context_precision", i)
        cr = _safe("context_recall", i)
        if ac >= 0.7:
            grade = "PASS"
        elif ac >= 0.4:
            grade = "PARTIAL"
        else:
            grade = "FAIL"
        grades.append(grade)
        rows.append({
            "id": s["question_id"],
            "question": s["user_input"],
            "answer": s["response"],
            "grade": grade,
            "answer_correctness": round(ac, 3),
            "faithfulness": round(fa, 3),
            "answer_relevancy": round(ar, 3),
            "context_precision": round(cp, 3),
            "context_recall": round(cr, 3),
            "required_facts": s["_required_facts"],
            "retrieval_latency_ms": s["_latency_ms"],
            "contexts_count": len(s["retrieved_contexts"]),
        })

    grade_counts = {g: grades.count(g) for g in ("PASS", "PARTIAL", "FAIL")}
    n = len(samples)
    pass_rate = grade_counts["PASS"] / n if n else 0
    avg_ac = sum(r["answer_correctness"] for r in rows) / n if n else 0
    avg_fa = sum(r["faithfulness"] for r in rows) / n if n else 0
    avg_ar = sum(r["answer_relevancy"] for r in rows) / n if n else 0
    avg_cp = sum(r["context_precision"] for r in rows) / n if n else 0
    avg_cr = sum(r["context_recall"] for r in rows) / n if n else 0

    summary = {
        "step": step_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": n,
        "grade_counts": grade_counts,
        "pass_rate": round(pass_rate, 2),
        "ragas_averages": {
            "answer_correctness": round(avg_ac, 3),
            "faithfulness":       round(avg_fa, 3),
            "answer_relevancy":   round(avg_ar, 3),
            "context_precision":  round(avg_cp, 3),
            "context_recall":     round(avg_cr, 3),
        },
        "thresholds": {"PASS": "answer_correctness>=0.7", "PARTIAL": ">=0.4", "FAIL": "<0.4"},
        "results": rows,
    }

    out_path = ROOT / step_name / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 55}")
    print(f"{step_name}")
    print(f"  PASS={grade_counts['PASS']}  PARTIAL={grade_counts['PARTIAL']}  FAIL={grade_counts['FAIL']}  ({pass_rate:.0%})")
    print(f"  answer_correctness: {avg_ac:.3f}  faithfulness: {avg_fa:.3f}  answer_relevancy: {avg_ar:.3f}")
    print(f"  context_precision:  {avg_cp:.3f}  context_recall: {avg_cr:.3f}")
    print(f"  results → {out_path}")
    print("=" * 55)
    return summary


def main():
    p = argparse.ArgumentParser(description="RAGAS-based RAG step evaluation.")
    p.add_argument("--step", choices=list(STEPS), help="Run a single step.")
    p.add_argument("--all", action="store_true", help="Run every step in order.")
    p.add_argument("--list", action="store_true", help="List available steps and exit.")
    args = p.parse_args()

    if args.list:
        for name in STEPS:
            print(name)
        return
    if args.all:
        for name in STEPS:
            print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
            evaluate_step(name)
        return
    if args.step:
        evaluate_step(args.step)
        return
    p.print_help()


if __name__ == "__main__":
    main()
