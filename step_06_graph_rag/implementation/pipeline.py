"""
Step 08 — Graph RAG pipeline (alias resolution + BFS dependency traversal).

Adds over Step 07:
  - Alias/keyword entity resolution (e.g. "LENS" → InsightLens)
  - Full blast-radius BFS traversal of the dependency graph
  - Richer graph context extracted per retrieved chunk

Retains from Step 07: BM25 + dense RRF retrieval, CSV tool calling, knowledge graph.
"""

import time
from pathlib import Path

from step_01_baseline_rag.implementation.generate import generate_answer
from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_01_baseline_rag.implementation.retrieve import format_context
from step_03_tools.implementation.csv_tool import detect_intent, run_query
from step_05_knowledge_graph.implementation.pipeline import Step05RAG

CORPUS_PATH = Path(__file__).parent.parent.parent / "dataset" / "company_data"
GRAPH_PATH  = Path(__file__).parent.parent.parent / "step_05_knowledge_graph" / "results" / "graph.json"


class Step06RAG(Step05RAG):
    """
    Extends Step05RAG with alias-resolved graph context and BFS dependency expansion.

    Usage:
        rag = Step06RAG(k=10).build()
        result = rag.query("If NexusFlow goes down, what services are affected?")
    """

    def query(self, question: str) -> RAGResult:
        if self.collection is None or self.bm25 is None or self.graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        vector_ctx = format_context(chunks)
        csv_intent = detect_intent(question)
        csv_ctx = run_query(csv_intent) if csv_intent else ""

        from step_06_graph_rag.implementation.graph_query import build_graph_context
        graph_ctx = build_graph_context(question, [c.text for c in chunks], self.graph)

        parts = []
        if csv_ctx:
            parts.append(csv_ctx)
        if graph_ctx:
            parts.append(graph_ctx)
        parts.append(vector_ctx)
        context = "\n\n".join(parts)

        t1 = time.perf_counter()
        answer, provider = generate_answer(context, question)
        generation_ms = (time.perf_counter() - t1) * 1000

        return RAGResult(
            question=question,
            answer=answer,
            provider=provider,
            retrieved_chunks=chunks,
            context_sent=context,
            context_chars=len(context),
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )
