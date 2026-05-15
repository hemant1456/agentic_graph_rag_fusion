"""
Step 02 — Observability: TracedRAG.

Wraps Step 01's BaselineRAG with transparent trace emission.
The retrieval and generation logic are identical — we just record everything.

Usage:
    rag = BaselineRAG(k=5).build()
    store = TraceStore(Path("step_02_observability/results/traces.jsonl"))
    traced = TracedRAG(rag, store)
    result, trace = traced.query("What is the data retention policy?")
    print(trace.generation.total_tokens)
    print(trace.retrieval.unique_sources)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.implementation.pipeline import BaselineRAG, RAGResult
from step_01_baseline_rag.implementation.retrieve import format_context, retrieve

from .generate_traced import GenerationOutput, generate_with_usage
from .tracer import (
    ChunkTrace,
    GenerationSpan,
    QueryTrace,
    RetrievalSpan,
    TraceStore,
    estimate_cost,
    new_trace_id,
)


class TracedRAG:
    """
    Drop-in wrapper around BaselineRAG that emits a QueryTrace per query.

    Returns (RAGResult, QueryTrace) — RAGResult is backward-compatible with
    the Step 01 scorer; QueryTrace is the new observability record.
    """

    def __init__(self, rag: BaselineRAG, store: TraceStore) -> None:
        self.rag = rag
        self.store = store

    def query(self, question: str) -> tuple[RAGResult, QueryTrace]:
        trace_id = new_trace_id()
        t_start = time.perf_counter()

        # ── Retrieval ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = retrieve(question, self.rag.collection, k=self.rag.k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        context = format_context(chunks)

        retrieval_span = RetrievalSpan(
            duration_ms=round(retrieval_ms, 1),
            chunks=[
                ChunkTrace(
                    rank=i + 1,
                    source=c.source,
                    department=c.department,
                    similarity=round(c.similarity, 4),
                    char_count=len(c.text),
                    text_preview=c.text[:200].replace("\n", " "),
                )
                for i, c in enumerate(chunks)
            ],
        )

        # ── Generation ────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        gen_out: GenerationOutput = generate_with_usage(context, question)
        generation_ms = (time.perf_counter() - t1) * 1000

        generation_span = GenerationSpan(
            duration_ms=round(generation_ms, 1),
            provider=gen_out.provider,
            model=gen_out.model,
            context_chars=len(context),
            context_chunk_count=len(chunks),
            prompt_tokens=gen_out.prompt_tokens,
            completion_tokens=gen_out.completion_tokens,
            estimated_cost_usd=round(
                estimate_cost(gen_out.provider, gen_out.prompt_tokens, gen_out.completion_tokens),
                6,
            ),
        )

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        trace = QueryTrace(
            trace_id=trace_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            step="01_baseline_rag",
            query=question,
            answer=gen_out.answer,
            retrieval=retrieval_span,
            generation=generation_span,
            total_latency_ms=total_ms,
        )
        self.store.write(trace)

        # RAGResult for backward compatibility with the Step 01 scorer
        result = RAGResult(
            question=question,
            answer=gen_out.answer,
            provider=gen_out.provider,
            retrieved_chunks=chunks,
            context_sent=context,
            context_chars=len(context),
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )

        return result, trace
