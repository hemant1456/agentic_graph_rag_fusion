"""
Interactive trace explorer — boots Arize Phoenix UI and lets you query the RAG
pipeline one question at a time. Every query shows up as a span in the browser.

Usage:
    uv run python step_02_observability/trace_explorer.py

Then open http://localhost:6006 in a browser and type questions at the prompt.
Each query produces:
  - a printed answer + trace summary in the terminal
  - a CHAIN → RETRIEVER + LLM span tree in the Phoenix UI
  - a JSONL record appended to step_02_observability/results/traces.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from step_01_baseline_rag.implementation.pipeline import BaselineRAG
from step_02_observability.implementation.phoenix_exporter import PhoenixExporter
from step_02_observability.implementation.traced_pipeline import TracedRAG
from step_02_observability.implementation.tracer import TraceStore

TRACE_FILE = Path(__file__).parent / "results" / "traces.jsonl"


def main() -> None:
    print("Booting Arize Phoenix …")
    exporter = PhoenixExporter()
    otel_tracer = exporter.start()
    print(f"Phoenix UI → {exporter.url}   (open in browser)\n")

    rag = BaselineRAG(k=5).build()
    store = TraceStore(TRACE_FILE)
    traced = TracedRAG(rag, store, otel_tracer=otel_tracer)

    print("RAG pipeline ready. Type a question and press Enter.")
    print("Commands:  'quit' or Ctrl-C to exit\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        result, trace = traced.query(question)

        print(f"\nAnswer : {result.answer}")
        print(f"────────────────────────────────────────────────────────")
        print(f"Trace  : {trace.trace_id}")
        print(f"Latency: {trace.total_latency_ms:.0f}ms  "
              f"(retrieval {trace.retrieval.duration_ms:.0f}ms  "
              f"+ generation {trace.generation.duration_ms:.0f}ms)")
        print(f"Tokens : {trace.generation.total_tokens}  "
              f"(prompt {trace.generation.prompt_tokens}  "
              f"+ completion {trace.generation.completion_tokens})")
        print(f"Cost   : ${trace.generation.estimated_cost_usd:.6f}")
        print(f"Sources: {', '.join(trace.retrieval.unique_sources)}")
        print()


if __name__ == "__main__":
    main()
