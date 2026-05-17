import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from step_01_baseline_rag.implementation.pipeline import BaselineRAG
from evaluation.implementation.evaluator import Evaluator

RESULTS_FILE = Path(__file__).parent.parent / "results" / "metric_scores.json"

GRADE_COLOR  = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}
GRADE_SYMBOL = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}
RESET = "\033[0m"


def run_evaluation() -> dict:
    print("=== Step 03: RAGAS-style Evaluation ===\n")
    print("Building RAG pipeline …")
    rag = BaselineRAG(k=5).build()
    evaluator = Evaluator()

    all_rows: list[dict] = []
    grade_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    print(f"\n{'ID':<6} {'GRADE':<9} {'FAITH':>6} {'RELEV':>6} {'PREC':>6} {'RCALL':>6}")
    print("-" * 52)

    for q in GOLDEN_QUESTIONS:
        result = rag.query(q.question)
        record = evaluator.evaluate(q, result)
        row = evaluator.to_dict(record)
        all_rows.append(row)
        grade_counts[record.grade] += 1

        color = GRADE_COLOR[record.grade]
        sym   = GRADE_SYMBOL[record.grade]
        print(
            f"{q.id:<6} "
            f"{color}[{sym}] {record.grade:<5}{RESET} "
            f"{record.faithfulness.score:>6.2f} "
            f"{record.answer_relevance.score:>6.2f} "
            f"{record.context_precision.score:>6.2f} "
            f"{record.context_recall.score:>6.2f}"
        )

    total = len(GOLDEN_QUESTIONS)

    def _avg(key: str) -> float:
        vals = [r[key] for r in all_rows if isinstance(r[key], float) and r[key] >= 0]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    mh_rows = [r for r in all_rows if r["multihop_score"] >= 0]
    mh_avg  = round(sum(r["multihop_score"] for r in mh_rows) / len(mh_rows), 3) if mh_rows else None

    summary = {
        "pass_rate":         round(grade_counts["PASS"] / total, 2),
        "faithfulness":      _avg("faithfulness"),
        "answer_relevance":  _avg("answer_relevance"),
        "context_precision": _avg("context_precision"),
        "context_recall":    _avg("context_recall"),
        "multihop_success":  mh_avg,
    }

    print(f"\n{'=' * 52}")
    print(f"RESULTS : {grade_counts['PASS']} PASS | {grade_counts['PARTIAL']} PARTIAL | {grade_counts['FAIL']} FAIL")
    print(f"\nAGGREGATE METRIC SCORES")
    print(f"  Pass rate         : {summary['pass_rate']:.0%}")
    print(f"  Faithfulness      : {summary['faithfulness']:.3f}")
    print(f"  Answer relevance  : {summary['answer_relevance']:.3f}")
    print(f"  Context precision : {summary['context_precision']:.3f}")
    print(f"  Context recall    : {summary['context_recall']:.3f}")
    if mh_avg is not None:
        print(f"  Multi-hop success : {mh_avg:.3f}")
    print(f"{'=' * 52}\n")

    report = {
        "step": "03_evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": total,
        "grade_counts": grade_counts,
        "summary": summary,
        "results": all_rows,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(report, indent=2))
    print(f"Report saved → {RESULTS_FILE}")

    return report


if __name__ == "__main__":
    run_evaluation()
