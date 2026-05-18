"""Rebuild the README "Latest eval results" table from per-step JSONs.

Usage:
    uv run python evaluation/build_results_table.py            # print to stdout
    uv run python evaluation/build_results_table.py --inplace  # rewrite README
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

STEPS = [
    "step_01_baseline_rag",
    "step_02_tools",
    "step_03_hybrid_retrieval",
    "step_04_knowledge_graph",
    "step_05_multi_agent",
    "step_06_context_engineering",
    "step_07_vsa",
    "step_08_production",
]

TABLE_START = "<!-- RESULTS_TABLE_START -->"
TABLE_END = "<!-- RESULTS_TABLE_END -->"


def _load(step: str) -> dict | None:
    p = ROOT / step / "results" / "eval_results.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    # Sanity check: must have grade_counts + ragas_averages
    if "grade_counts" not in data or "ragas_averages" not in data:
        return None
    return data


def _row(step: str, data: dict | None) -> str:
    if data is None:
        return f"| {step} | _pending_ | | | | | | | |"
    gc = data["grade_counts"]
    avg = data["ragas_averages"]
    return (
        f"| {step} | {gc['PASS']} | {gc['PARTIAL']} | {gc['FAIL']} "
        f"| {avg['answer_correctness']:.3f} | {avg['faithfulness']:.3f} "
        f"| {avg['context_recall']:.3f} | {avg['context_precision']:.3f} "
        f"| {avg['answer_relevancy']:.3f} |"
    )


def build_table() -> str:
    lines = [
        TABLE_START,
        "",
        "| Step | PASS | PART | FAIL | answer_correctness | faithfulness | context_recall | context_precision | answer_relevancy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in STEPS:
        lines.append(_row(step, _load(step)))
    lines.append("")
    lines.append(TABLE_END)
    return "\n".join(lines)


def update_readme(readme_path: Path) -> bool:
    text = readme_path.read_text()
    new_table = build_table()

    if TABLE_START in text and TABLE_END in text:
        pattern = re.compile(
            re.escape(TABLE_START) + r".*?" + re.escape(TABLE_END),
            re.DOTALL,
        )
        new_text = pattern.sub(new_table, text)
    else:
        # Insert markers around an existing manually-built table — replace from
        # the table heading down to the next blank line after the rows.
        marker_heading = "| Step | PASS | PART | FAIL |"
        if marker_heading in text:
            start = text.index(marker_heading)
            # Walk back to the start of the line above (blank line before table)
            prev_blank = text.rfind("\n\n", 0, start) + 2
            # Walk forward past the table to the next blank line
            end = text.index("\n\n", start) if "\n\n" in text[start:] else len(text)
            new_text = text[:prev_blank] + new_table + "\n\n" + text[end + 2:]
        else:
            print("ERROR: cannot find table location in README", file=sys.stderr)
            return False

    if new_text == text:
        return False
    readme_path.write_text(new_text)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inplace", action="store_true", help="rewrite README.md")
    args = ap.parse_args()

    if args.inplace:
        changed = update_readme(ROOT / "README.md")
        print("README.md updated" if changed else "README.md unchanged")
    else:
        print(build_table())


if __name__ == "__main__":
    main()
