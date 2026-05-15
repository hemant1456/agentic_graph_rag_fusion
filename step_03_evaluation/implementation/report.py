"""
Step 03 — Evaluation Framework: report CLI.

Reads step_03_evaluation/results/metric_scores.json and presents it in a
human-readable format.

Usage:
    # Aggregate scores + per-question table
    uv run python step_03_evaluation/implementation/report.py --summary

    # Full detail for one question (metrics + reasoning + answer)
    uv run python step_03_evaluation/implementation/report.py --question Q01

    # All questions ranked by a specific metric (worst first)
    uv run python step_03_evaluation/implementation/report.py --worst faithfulness
    uv run python step_03_evaluation/implementation/report.py --worst context_recall
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_FILE = Path(__file__).parent.parent / "results" / "metric_scores.json"

GRADE_COLOR  = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}
RESET = "\033[0m"
METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


def _load() -> dict:
    if not RESULTS_FILE.exists():
        print(f"No results found at {RESULTS_FILE}")
        print("Run:  uv run python step_03_evaluation/evaluation/run_eval.py")
        sys.exit(1)
    return json.loads(RESULTS_FILE.read_text())


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.2f}"


def cmd_summary(data: dict) -> None:
    s = data["summary"]
    ts = data.get("timestamp", "unknown")
    total = data["total_questions"]
    gc = data["grade_counts"]

    print(f"\n=== Step 03 Evaluation — {ts} ===\n")
    print(f"Questions : {total}   ({gc['PASS']} PASS / {gc['PARTIAL']} PARTIAL / {gc['FAIL']} FAIL)")
    print(f"Pass rate : {s['pass_rate']:.0%}\n")
    print("METRIC SCORES")
    for m in METRICS:
        print(f"  {m:<22} {_bar(s[m])}")
    if s.get("multihop_success") is not None:
        print(f"  {'multihop_success':<22} {_bar(s['multihop_success'])}")

    print(f"\n{'ID':<6} {'GRADE':<9} {'FAITH':>6} {'RELEV':>6} {'PREC':>6} {'RCALL':>6}")
    print("-" * 52)
    for r in data["results"]:
        color = GRADE_COLOR.get(r["grade"], "")
        mh = f"{r['multihop_score']:>5.2f}" if r["multihop_score"] >= 0 else "  N/A"
        print(
            f"{r['id']:<6} "
            f"{color}{r['grade']:<9}{RESET}"
            f"{r['faithfulness']:>6.2f} "
            f"{r['answer_relevance']:>6.2f} "
            f"{r['context_precision']:>6.2f} "
            f"{r['context_recall']:>6.2f}"
        )


def cmd_question(data: dict, qid: str) -> None:
    rows = [r for r in data["results"] if r["id"].upper() == qid.upper()]
    if not rows:
        print(f"Question {qid!r} not found. Available: {[r['id'] for r in data['results']]}")
        sys.exit(1)
    r = rows[0]
    color = GRADE_COLOR.get(r["grade"], "")

    print(f"\n=== {r['id']} — {r['type']} ===")
    print(f"Grade  : {color}{r['grade']}{RESET}")
    print(f"\nQuestion: {r['question']}")
    print(f"\nAnswer  : {r['answer']}\n")

    print("METRIC SCORES")
    for m in METRICS:
        print(f"  {m:<22} {_bar(r[m])}")
        print(f"    → {r[m + '_reasoning']}")
    if r["multihop_score"] >= 0:
        print(f"  {'multihop_success':<22} {_bar(r['multihop_score'])}")


def cmd_worst(data: dict, metric: str) -> None:
    valid = METRICS + ["multihop_success"]
    if metric not in valid:
        print(f"Unknown metric {metric!r}. Choose from: {valid}")
        sys.exit(1)

    key = "multihop_score" if metric == "multihop_success" else metric
    rows = [r for r in data["results"] if r[key] >= 0]
    rows.sort(key=lambda r: r[key])

    print(f"\n=== Questions ranked by {metric} (worst first) ===\n")
    print(f"{'ID':<6} {'GRADE':<9} {metric.upper():>7}  REASONING")
    print("-" * 80)
    for r in rows:
        color = GRADE_COLOR.get(r["grade"], "")
        reasoning = r.get(f"{metric}_reasoning", r.get("multihop_reasoning", ""))
        print(
            f"{r['id']:<6} {color}{r['grade']:<9}{RESET}"
            f"{r[key]:>7.2f}  {reasoning[:60]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 03 evaluation report")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--summary",  action="store_true", help="Aggregate scores + per-question table")
    group.add_argument("--question", metavar="ID",        help="Full detail for one question (e.g. Q01)")
    group.add_argument("--worst",    metavar="METRIC",    help="Questions ranked by metric, worst first")
    args = parser.parse_args()

    data = _load()
    if args.summary:
        cmd_summary(data)
    elif args.question:
        cmd_question(data, args.question)
    elif args.worst:
        cmd_worst(data, args.worst)


if __name__ == "__main__":
    main()
