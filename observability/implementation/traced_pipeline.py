import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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

    Returns (RAGResult, QueryTrace):
    - RAGResult: backward-compatible with the Step 01 scorer
    - QueryTrace: full observability record (JSONL + optional OTel/Phoenix)

    When otel_tracer is None (default): JSONL trace only.
    When otel_tracer is set: JSONL trace + OTel spans sent to Phoenix.
    OTel spans are created *around* the actual work so timings are accurate.
    """

    def __init__(
        self,
        rag: BaselineRAG,
        store: TraceStore,
        otel_tracer: Any = None,    # opentelemetry.trace.Tracer | None
    ) -> None:
        self.rag = rag
        self.store = store
        self._otel = otel_tracer    # None = JSONL only; set = also emit to Phoenix

    def query(self, question: str) -> tuple[RAGResult, QueryTrace]:
        trace_id = new_trace_id()
        t = self._otel

        chain_ctx = t.start_as_current_span("rag_query") if t else nullcontext()
        with chain_ctx as chain_span:
            if chain_span:
                chain_span.set_attribute("openinference.span.kind", "CHAIN")
                chain_span.set_attribute("input.value", question)
                chain_span.set_attribute("trace_id", trace_id)

            t_start = time.perf_counter()

            r_ctx = t.start_as_current_span("retrieval") if t else nullcontext()
            with r_ctx as r_span:
                t0 = time.perf_counter()
                chunks = retrieve(question, self.rag.collection, k=self.rag.k)
                retrieval_ms = (time.perf_counter() - t0) * 1000
                context = format_context(chunks)

                if r_span:
                    r_span.set_attribute("openinference.span.kind", "RETRIEVER")
                    r_span.set_attribute("input.value", question)
                    r_span.set_attribute("retrieval.num_chunks", len(chunks))
                    for i, c in enumerate(chunks):
                        r_span.set_attribute(
                            f"retrieval.documents.{i}.document.content",
                            c.text[:500],
                        )
                        r_span.set_attribute(
                            f"retrieval.documents.{i}.document.metadata",
                            json.dumps({
                                "source": c.source,
                                "department": c.department,
                                "similarity": round(c.similarity, 4),
                            }),
                        )

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

            g_ctx = t.start_as_current_span("llm_generation") if t else nullcontext()
            with g_ctx as g_span:
                t1 = time.perf_counter()
                gen_out: GenerationOutput = generate_with_usage(context, question)
                generation_ms = (time.perf_counter() - t1) * 1000

                if g_span:
                    g_span.set_attribute("openinference.span.kind", "LLM")
                    g_span.set_attribute("llm.model_name", gen_out.model)
                    g_span.set_attribute("llm.provider", gen_out.provider)
                    g_span.set_attribute("input.value", question)
                    g_span.set_attribute("output.value", gen_out.answer)
                    g_span.set_attribute("llm.token_count.prompt", gen_out.prompt_tokens)
                    g_span.set_attribute("llm.token_count.completion", gen_out.completion_tokens)
                    g_span.set_attribute(
                        "llm.token_count.total",
                        gen_out.prompt_tokens + gen_out.completion_tokens,
                    )
                    g_span.set_attribute("context.chars", len(context))

            generation_span = GenerationSpan(
                duration_ms=round(generation_ms, 1),
                provider=gen_out.provider,
                model=gen_out.model,
                context_chars=len(context),
                context_chunk_count=len(chunks),
                prompt_tokens=gen_out.prompt_tokens,
                completion_tokens=gen_out.completion_tokens,
                estimated_cost_usd=round(
                    estimate_cost(
                        gen_out.provider, gen_out.prompt_tokens, gen_out.completion_tokens
                    ),
                    6,
                ),
            )

            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            if chain_span:
                chain_span.set_attribute("output.value", gen_out.answer)
                chain_span.set_attribute("total_latency_ms", total_ms)

            query_trace = QueryTrace(
                trace_id=trace_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                step="01_baseline_rag",
                query=question,
                answer=gen_out.answer,
                retrieval=retrieval_span,
                generation=generation_span,
                total_latency_ms=total_ms,
            )
            self.store.write(query_trace)

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

        return result, query_trace
