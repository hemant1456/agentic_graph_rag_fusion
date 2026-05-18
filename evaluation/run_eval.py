"""RAGAS-based evaluation for the 7-step RAG progression.

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
    uv run python evaluation/run_eval.py --step step_03_hybrid_retrieval
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

import json as _json
import re as _re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage

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
    rag = _cached("step_01", lambda: BaselineRAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_02(question: str) -> dict:
    from step_02_tools.implementation.pipeline import Step02ToolsRAG
    rag = _cached("step_02", lambda: Step02ToolsRAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_03(question: str) -> dict:
    from step_03_hybrid_retrieval.implementation.pipeline import Step03HybridRAG
    rag = _cached("step_03", lambda: Step03HybridRAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_04(question: str) -> dict:
    from step_04_knowledge_graph.implementation.pipeline import Step04RAG
    rag = _cached("step_04", lambda: Step04RAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_05(question: str) -> dict:
    from step_05_multi_agent.implementation.pipeline import Step05RAG
    rag = _cached("step_05", lambda: Step05RAG(k=10).build())
    return _result_to_sample(rag.query(question))


def answer_step_06(question: str) -> dict:
    from step_06_context_engineering.implementation.pipeline import Step06RAG
    rag = _cached("step_06", lambda: Step06RAG(k=5).build())
    return _result_to_sample(rag.query(question))


def answer_step_07(question: str) -> dict:
    from step_07_production.implementation.pipeline import Step07RAG
    rag = _cached("step_07", lambda: Step07RAG(k=5).build())
    return _result_to_sample(rag.query(question))


STEPS: dict[str, Callable[[str], dict]] = {
    "step_01_baseline_rag":         answer_step_01,
    "step_02_tools":                answer_step_02,
    "step_03_hybrid_retrieval":     answer_step_03,
    "step_04_knowledge_graph":      answer_step_04,
    "step_05_multi_agent":          answer_step_05,
    "step_06_context_engineering":  answer_step_06,
    "step_07_production":           answer_step_07,
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


_JUDGE_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system. Score the actual answer along 5 RAGAS-equivalent dimensions, EACH scored 0.0 to 1.0. Be format-tolerant — "June 30, 2024" and "2024-06-30" mean the same date; "Daniel Osei" and "Daniel Osei, Security Lead" name the same person.

QUESTION: {question}

REFERENCE ANSWER (gold): {reference}

ACTUAL ANSWER (under test): {answer}

RETRIEVED CONTEXTS (the chunks the system gave the LLM):
{contexts}

Score these 5 metrics:

1. answer_correctness — factual + semantic match against the reference. Penalize missing facts, wrong values, or contradictions. (1.0 = fully correct; 0.5 = partially right but missing facts; 0.0 = wrong)
2. faithfulness — every claim in the actual answer must be supported by the retrieved contexts (no hallucination). (1.0 = all claims grounded; 0.0 = invented facts)
3. answer_relevancy — does the answer directly address the question? Penalize evasive/off-topic answers. (1.0 = on-topic; 0.0 = evasive)
4. context_precision — of the retrieved contexts shown, what fraction were actually relevant to answering this question? (1.0 = every chunk relevant; 0.0 = noise)
5. context_recall — do the retrieved contexts collectively contain every fact in the reference answer? (1.0 = nothing missing; 0.0 = required facts missing from contexts)

If the answer says "I don't know" or "the documents don't contain this", treat it as: faithfulness 1.0, answer_relevancy ~0.3, answer_correctness 0.0 (unless the reference says "no info", in which case correctness 1.0).

Return ONE JSON object, no prose, no markdown fences, with exactly these keys:
{{"answer_correctness": <0-1>, "faithfulness": <0-1>, "answer_relevancy": <0-1>, "context_precision": <0-1>, "context_recall": <0-1>, "reasoning": "<one short sentence>"}}"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
        if m:
            text = m.group(1)
    else:
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            text = m.group(0)
    return _json.loads(text)


def _score_one(judge, sample: dict) -> dict:
    """Single LLM call → all 5 metrics for one sample."""
    contexts = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(sample["retrieved_contexts"]))
    prompt = _JUDGE_PROMPT.format(
        question=sample["user_input"],
        reference=sample["reference"],
        answer=sample["response"],
        contexts=contexts[:8000],  # keep prompt under judge context limit
    )
    try:
        out = judge.invoke([HumanMessage(content=prompt)]).content
        parsed = _extract_json(out)
    except Exception as e:
        return {
            "answer_correctness": 0.0, "faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0,
            "reasoning": f"judge failure: {type(e).__name__}: {str(e)[:120]}",
        }

    def _f(k):
        v = parsed.get(k, 0)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f if f == f else 0.0))

    return {
        "answer_correctness": _f("answer_correctness"),
        "faithfulness":       _f("faithfulness"),
        "answer_relevancy":   _f("answer_relevancy"),
        "context_precision":  _f("context_precision"),
        "context_recall":     _f("context_recall"),
        "reasoning":          str(parsed.get("reasoning", ""))[:300],
    }


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

    print(f"\n=== Scoring (1 judge call per question → 5 metrics in one JSON) ===\n")
    judge = build_judge_llm(temperature=0.0, max_tokens=512)

    scores: list[dict] = [None] * len(samples)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_score_one, judge, s): i for i, s in enumerate(samples)}
        done = 0
        for fut in futures:
            i = futures[fut]
            scores[i] = fut.result()
            done += 1
            s = samples[i]
            print(f"  [{s['question_id']}] correctness={scores[i]['answer_correctness']:.2f}  "
                  f"faith={scores[i]['faithfulness']:.2f}  recall={scores[i]['context_recall']:.2f}  "
                  f"prec={scores[i]['context_precision']:.2f}  relev={scores[i]['answer_relevancy']:.2f}  "
                  f"({done}/{len(samples)})")

    grades: list[str] = []
    rows: list[dict] = []
    for s, sc in zip(samples, scores):
        ac = sc["answer_correctness"]
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
            "faithfulness":       round(sc["faithfulness"], 3),
            "answer_relevancy":   round(sc["answer_relevancy"], 3),
            "context_precision":  round(sc["context_precision"], 3),
            "context_recall":     round(sc["context_recall"], 3),
            "judge_reasoning":    sc["reasoning"],
            "required_facts":     s["_required_facts"],
            "retrieval_latency_ms": s["_latency_ms"],
            "contexts_count":     len(s["retrieved_contexts"]),
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
