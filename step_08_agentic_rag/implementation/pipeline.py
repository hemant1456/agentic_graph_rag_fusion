"""
Step 08 RAG pipeline — Agentic RAG with tool-calling LLM.

Upgrades over Step 07:
  - LLM agent (Claude Haiku) plans which tools to call and iterates
  - Multi-query disambiguation: asks targeted follow-up questions until satisfied
  - Handles "two things with the same name" by searching for each separately
  - Fixes Q18: Project Phoenix (completed) + Phoenix Corp (signed)

Reuses: Step 07 BM25+dense retriever, Step 06 graph, Step 07 CSV tool.
"""

import time
from pathlib import Path

from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_07_rag_fusion.implementation.pipeline import Step07RAG
from step_08_agentic_rag.implementation.agent import run_agent

CORPUS_PATH = Path(__file__).parent.parent.parent / "step_00_dataset" / "company_data"
GRAPH_PATH  = Path(__file__).parent.parent.parent / "step_05_knowledge_graph" / "results" / "graph.json"


class Step08RAG:
    """
    Agentic RAG pipeline.

    The agent receives each question, calls vector_search / graph_query /
    csv_query as needed (up to 5 rounds), then generates a final answer.

    Usage:
        rag = Step08RAG(k=5).build()
        result = rag.query("Vertexia has two efforts sharing the same name…")
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._retriever: Step07RAG | None = None
        self._graph = None

    def build(self) -> "Step08RAG":
        self._retriever = Step07RAG(k=self.k).build()
        from step_05_knowledge_graph.implementation.graph_store import load_or_build
        self._graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def query(self, question: str) -> RAGResult:
        if self._retriever is None or self._graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        answer, provider = run_agent(question, self._retriever, self._graph)
        total_ms = (time.perf_counter() - t0) * 1000

        # Fetch a representative set of chunks for the result record
        # (used by dashboard source-list display, not for generation)
        chunks = self._retriever.retrieve(question, k=self.k)

        return RAGResult(
            question=question,
            answer=answer,
            provider=provider,
            retrieved_chunks=chunks,
            context_sent="[agent-managed context — see tool calls]",
            context_chars=len(answer),
            retrieval_latency_ms=total_ms,
            generation_latency_ms=0.0,
        )
