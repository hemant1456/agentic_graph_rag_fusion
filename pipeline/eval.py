import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from ingest import load_vector_store
from pipeline import ask

PASS_SYMBOL  = "\033[92m✓\033[0m"
PARTIAL_SYMBOL = "\033[93m~\033[0m"
FAIL_SYMBOL  = "\033[91m✗\033[0m"
SYMBOLS = {"PASS": PASS_SYMBOL, "PARTIAL": PARTIAL_SYMBOL, "FAIL": FAIL_SYMBOL}


def score(answer: str, question) -> tuple[str, list, list]:
    answer_lower = answer.lower()
    disqualifier_hits = [d for d in question.disqualifiers if d.lower() in answer_lower]
    required_hits = [f for f in question.required_facts if f.lower() in answer_lower]
    partial_hits = [f for f in question.partial_facts if f.lower() in answer_lower]

    if disqualifier_hits:
        grade = "FAIL"
    elif len(required_hits) == len(question.required_facts):
        grade = "PASS"
    elif required_hits or partial_hits:
        grade = "PARTIAL"
    else:
        grade = "FAIL"

    missing = [f for f in question.required_facts if f not in required_hits]
    return grade, missing, disqualifier_hits


if __name__ == "__main__":
    vs = load_vector_store()
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    for q in GOLDEN_QUESTIONS:
        answer = ask(q.question, vs)
        grade, missing, disqualifiers = score(answer, q)
        counts[grade] += 1

        sym = SYMBOLS[grade]
        print(f"{sym} {q.id} [{q.type}]")
        print(f"   Q: {q.question}")
        print(f"   A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        if missing:
            print(f"   Missing facts: {missing}")
        if disqualifiers:
            print(f"   Disqualifiers hit: {disqualifiers}")
        print()

    total = len(GOLDEN_QUESTIONS)
    print("=" * 50)
    print(f"PASS: {counts['PASS']}  PARTIAL: {counts['PARTIAL']}  FAIL: {counts['FAIL']}")
    print(f"Pass rate: {counts['PASS']}/{total} ({counts['PASS']/total:.0%})")
    print("=" * 50)
