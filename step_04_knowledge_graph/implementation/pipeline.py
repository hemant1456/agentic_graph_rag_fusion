"""
Step 04 — Knowledge Graph RAG pipeline.

Adds over Step 03:
  - Entity knowledge graph built from structured CSVs (org chart, API deps, etc.)
  - Multi-hop graph traversal appended to the retrieval context

Retains from Step 03: BM25 + dense RRF retrieval, CSV tool calling.
"""

import time
from pathlib import Path

from step_01_baseline_rag.implementation.generate import generate_answer
from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_01_baseline_rag.implementation.retrieve import format_context
from step_02_tools.implementation.csv_tool import detect_intent, run_query
from step_03_hybrid_retrieval.implementation.pipeline import Step03HybridRAG

CORPUS_PATH = Path(__file__).parent.parent.parent / "dataset" / "company_data"
GRAPH_PATH  = Path(__file__).parent.parent / "results" / "graph.json"


class Step04RAG(Step03HybridRAG):
    """
    Extends Step03HybridRAG with entity knowledge graph traversal.

    Usage:
        rag = Step04RAG(k=10).build()
        result = rag.query("Who does Aisha Johnson report to?")
    """

    def __init__(self, k: int = 10) -> None:
        super().__init__(k=k)
        self.graph = None

    def build(self, reset_graph: bool = False) -> "Step04RAG":
        super().build()
        if reset_graph and GRAPH_PATH.exists():
            GRAPH_PATH.unlink()
        from step_04_knowledge_graph.implementation.graph_store import load_or_build
        self.graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def query(self, question: str) -> RAGResult:
        if self.collection is None or self.bm25 is None or self.graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        vector_ctx = format_context(chunks)
        csv_intent = detect_intent(question)
        csv_ctx = run_query(csv_intent) if csv_intent else ""

        from step_04_knowledge_graph.implementation.graph_query import build_graph_context
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
