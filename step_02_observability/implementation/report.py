"""
Step 02 — Observability: trace inspector CLI.

Answers the core observability question for any past query:
  "What chunks drove that answer, how many tokens did it cost, and how long did it take?"

Usage:
    # Full detail for one trace (query, sources, answer, cost):
    uv run python step_02_observability/implementation/report.py --trace-id <id>

    # Show exactly which chunks were retrieved and their similarities:
    uv run python step_02_observability/implementation/report.py --sources <id>

    # Summary table across all stored traces:
    uv run python step_02_observability/implementation/report.py --summary

    # Read from a different trace file:
    uv run python step_02_observability/implementation/report.py --summary --file path/to/traces.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TRACE_FILE = Path(__file__).parent.parent / "results" / "traces.jsonl"


def load_traces(path: Path = DEFAULT_TRACE_FILE) -> list[dict]:
    if not path.exists():
        print(f"No trace file at {path}")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_trace(t: dict) -> None:
    """Full trace detail — everything we know about one query."""
    r, g = t["retrieval"], t["generation"]
    w = 64

    print(f"\n{'='*w}")
    print(f"Trace  : {t['trace_id']}  |  {t['timestamp']}  |  step: {t['step']}")
    print(f"Query  : {t['query']}")
    print(f"{'='*w}")

    print(f"\n── RETRIEVAL  {r['duration_ms']:.0f}ms ──")
    for c in r["chunks"]:
        print(f"  #{c['rank']}  sim={c['similarity']:.3f}  {c['source']}  ({c['department']})")
        print(f"       {c['text_preview'][:120]}{'...' if len(c['text_preview']) >= 120 else ''}")

    print(f"\n── GENERATION  {g['duration_ms']:.0f}ms  |  {g['provider']} / {g['model']} ──")
    print(f"  Context : {g['context_chars']:,} chars  |  {g['context_chunk_count']} chunks")
    print(f"  Tokens  : {g['prompt_tokens']} prompt + {g['completion_tokens']} completion"
          f"  = {g['prompt_tokens'] + g['completion_tokens']} total")
    print(f"  Cost    : ${g['estimated_cost_usd']:.6f}")

    print(f"\n── ANSWER ──")
    answer = t["answer"]
    print(f"  {answer[:500]}{'...' if len(answer) > 500 else ''}")

    print(f"\n── TOTAL  {t['total_latency_ms']:.0f}ms ──\n")


def print_sources(t: dict) -> None:
    """Retrieval-only view — answers 'what drove this answer?'."""
    r = t["retrieval"]
    print(f"\nSources for trace {t['trace_id']}")
    print(f"Query : {t['query']}\n")
    for c in r["chunks"]:
        bar = "█" * int(c["similarity"] * 20)
        print(f"  #{c['rank']}  {bar:<20}  {c['similarity']:.3f}  {c['source']}")
        print(f"       dept={c['department']}  chars={c['char_count']}")
        print(f"       {c['text_preview'][:160]}")
        print()


def print_summary(traces: list[dict]) -> None:
    """Summary table: one row per trace, sorted by ID."""
    if not traces:
        print("No traces found.")
        return

    total_cost = sum(t["generation"]["estimated_cost_usd"] for t in traces)
    total_tokens = sum(
        t["generation"]["prompt_tokens"] + t["generation"]["completion_tokens"]
        for t in traces
    )
    avg_ms = sum(t["total_latency_ms"] for t in traces) / len(traces)

    col = 80
    print(f"\n{'='*col}")
    print(f"{'ID':<10} {'MS':>6} {'PROMPT':>7} {'COMP':>6} {'COST':>10}  TOP SOURCE")
    print(f"{'-'*col}")
    for t in traces:
        g = t["generation"]
        top = t["retrieval"]["chunks"][0]["source"] if t["retrieval"]["chunks"] else "-"
        print(
            f"{t['trace_id']:<10}"
            f"{t['total_latency_ms']:>5.0f}ms"
            f"{g['prompt_tokens']:>8}"
            f"{g['completion_tokens']:>7}"
            f"  ${g['estimated_cost_usd']:>8.6f}"
            f"  {top}"
        )
    print(f"{'-'*col}")
    print(
        f"{'TOTAL':<10}  avg={avg_ms:.0f}ms"
        f"  tokens={total_tokens:,}"
        f"  cost=${total_cost:.5f}"
        f"  ({len(traces)} traces)"
    )
    print(f"{'='*col}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step 02: inspect RAG traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trace-id", metavar="ID", help="Full detail for one trace")
    parser.add_argument("--sources", metavar="ID", help="Retrieval sources for one trace")
    parser.add_argument("--summary", action="store_true", help="Summary table of all traces")
    parser.add_argument(
        "--file", metavar="PATH",
        help=f"Trace JSONL file (default: {DEFAULT_TRACE_FILE})",
    )
    args = parser.parse_args(argv)

    path = Path(args.file) if args.file else DEFAULT_TRACE_FILE
    traces = load_traces(path)

    if args.trace_id:
        match = next((t for t in traces if t["trace_id"] == args.trace_id), None)
        if match:
            print_trace(match)
        else:
            print(f"No trace with id '{args.trace_id}'")
            sys.exit(1)
    elif args.sources:
        match = next((t for t in traces if t["trace_id"] == args.sources), None)
        if match:
            print_sources(match)
        else:
            print(f"No trace with id '{args.sources}'")
            sys.exit(1)
    elif args.summary:
        print_summary(traces)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
