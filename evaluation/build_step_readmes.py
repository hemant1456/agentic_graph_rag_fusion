"""Build per-step README sections with per-question eval detail.

Reads <step>/results/eval_results.json and golden_questions metadata, generates
a markdown block per step that lives between the markers
<!-- RESULTS_DETAIL_START --> and <!-- RESULTS_DETAIL_END -->, and writes it
into each step's README.md.

Usage:
    uv run python evaluation/build_step_readmes.py            # rewrite in place
    uv run python evaluation/build_step_readmes.py --dry-run  # print to stdout
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS

# Map step name → (display name, tier number that step "unlocks")
STEP_INFO = {
    "step_01_baseline_rag":         ("Step 01 — Baseline RAG + Format-aware Chunking", 1),
    "step_02_tools":                ("Step 02 — CSV Tool Calling", 2),
    "step_03_hybrid_retrieval":     ("Step 03 — BM25 + Dense Hybrid", 3),
    "step_04_knowledge_graph":      ("Step 04 — Knowledge Graph + Graph RAG", 4),
    "step_05_multi_agent":          ("Step 05 — Multi-Agent System", 5),
    "step_06_context_engineering":  ("Step 06 — Context Engineering + VSA", 6),
    "step_07_production":           ("Step 07 — Production Hardening", 7),
}

# fixed_by_step → tier number it implies for "expected to pass starting at this step"
STEP_TO_TIER = {
    "step_01_baseline_rag":         1,
    "step_02_tools":                2,
    "step_03_hybrid_retrieval":     3,
    "step_04_knowledge_graph":      4,
    "step_05_multi_agent":          5,
    "step_06_context_engineering":  6,
    "step_07_production":           7,
}

MARKER_START = "<!-- RESULTS_DETAIL_START -->"
MARKER_END = "<!-- RESULTS_DETAIL_END -->"


def _truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _question_meta() -> dict:
    return {q.id: q for q in GOLDEN_QUESTIONS}


def _analysis_blurb(qid: str, q, row: dict, step_tier: int) -> str:
    """One-line analysis: was this expected, and why did it happen?"""
    expected_tier = STEP_TO_TIER.get(q.fixed_by_step, 99)
    grade = row["grade"]
    judge = row.get("judge_reasoning", "") or ""

    # 1) Expected to pass starting at this step
    if expected_tier == step_tier:
        if grade == "PASS":
            return f"Pass-tier hits as designed — the step's new capability surfaces the required fact(s)."
        elif grade == "PARTIAL":
            return f"Should PASS at this tier but only PARTIAL. {_truncate(judge, 180)}"
        else:
            return f"Should PASS at this tier but FAILED — diagnose. {_truncate(judge, 180)}"

    # 2) Should have already passed at an earlier step
    if expected_tier < step_tier:
        if grade == "PASS":
            return "Continues to PASS from an earlier tier — capability still works."
        elif grade == "PARTIAL":
            return f"Regression: was solvable at step {expected_tier}; now only PARTIAL. {_truncate(judge, 160)}"
        else:
            return f"Regression: was solvable at step {expected_tier}; now FAILED. {_truncate(judge, 160)}"

    # 3) Question's required capability hasn't been introduced yet
    if expected_tier > step_tier:
        if grade == "PASS":
            return (
                f"Unexpected PASS — question targets step {expected_tier}'s capability, "
                "but retrieved context happened to contain enough signal."
            )
        elif grade == "PARTIAL":
            return (
                f"Expected — capability arrives at step {expected_tier}. "
                f"PARTIAL means retrieval brought some related context. {_truncate(judge, 120)}"
            )
        else:
            return f"Expected FAIL — required capability arrives at step {expected_tier}."

    return _truncate(judge, 180)


def build_step_block(step_name: str) -> str:
    label, step_tier = STEP_INFO[step_name]
    json_path = ROOT / step_name / "results" / "eval_results.json"
    if not json_path.exists():
        return f"{MARKER_START}\n\n_No eval results yet for {step_name}. Run `uv run python evaluation/run_eval.py --step {step_name}`._\n\n{MARKER_END}"

    data = json.loads(json_path.read_text())
    rows = {r["id"]: r for r in data.get("results", [])}
    gc = data.get("grade_counts", {})
    avg = data.get("ragas_averages", {})
    n = data.get("total_questions", len(rows))

    meta = _question_meta()
    lines: list[str] = [MARKER_START, ""]

    # Overall stats
    lines.append("## Eval results")
    lines.append("")
    lines.append(
        f"**Run summary** — {gc.get('PASS', 0)} PASS · "
        f"{gc.get('PARTIAL', 0)} PARTIAL · {gc.get('FAIL', 0)} FAIL out of {n} questions "
        f"({(gc.get('PASS', 0) / n * 100) if n else 0:.0f}% pass rate)."
    )
    lines.append("")
    lines.append("RAGAS averages:")
    lines.append("")
    lines.append("| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |")
    lines.append("|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {avg.get('answer_correctness', 0):.3f} "
        f"| {avg.get('faithfulness', 0):.3f} "
        f"| {avg.get('answer_relevancy', 0):.3f} "
        f"| {avg.get('context_precision', 0):.3f} "
        f"| {avg.get('context_recall', 0):.3f} |"
    )
    lines.append("")

    # Per-question table
    lines.append("### Per-question detail")
    lines.append("")
    lines.append("| ID | Grade | correctness | Fixed-by step | Notes |")
    lines.append("|---|---|---:|---|---|")
    for qid in sorted(rows.keys()):
        row = rows[qid]
        q = meta.get(qid)
        if not q:
            continue
        fixed_by = q.fixed_by_step.replace("step_", "S").split("_")[0]
        blurb = _analysis_blurb(qid, q, row, step_tier)
        # escape pipes in blurb
        blurb = blurb.replace("|", "\\|")
        question_short = _truncate(q.question, 90).replace("|", "\\|")
        lines.append(
            f"| **{qid}** | {row['grade']} | {row['answer_correctness']:.2f} "
            f"| `{q.fixed_by_step}` | {blurb} |"
        )
    lines.append("")
    lines.append("> Each question's text + reference answer lives in "
                 "`step_01_baseline_rag/evaluation/golden_questions.py`. The full per-question "
                 "JSON (including the judge's reasoning) is in `results/eval_results.json`.")
    lines.append("")
    lines.append(MARKER_END)
    return "\n".join(lines)


def apply_to_readme(step_name: str, dry_run: bool = False) -> None:
    readme = ROOT / step_name / "README.md"
    if not readme.exists():
        print(f"  WARN: {readme} missing — skipping")
        return
    block = build_step_block(step_name)
    text = readme.read_text()

    if MARKER_START in text and MARKER_END in text:
        pattern = re.compile(
            re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
            re.DOTALL,
        )
        new_text = pattern.sub(block, text)
    else:
        # Append a Results section if no markers exist yet
        new_text = text.rstrip() + "\n\n" + block + "\n"

    if dry_run:
        print(f"=== {step_name}/README.md ===")
        print(new_text)
        print()
    else:
        if new_text != text:
            readme.write_text(new_text)
            print(f"  updated {readme.relative_to(ROOT)}")
        else:
            print(f"  unchanged {readme.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for step in STEP_INFO:
        # Skip step_07 — user requested not to refresh its README in this pass.
        if step == "step_07_production":
            print(f"  skipped {step}/README.md (per instruction)")
            continue
        apply_to_readme(step, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
