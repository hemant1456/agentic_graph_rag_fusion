from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_04_hybrid_retrieval.implementation.pipeline import Step04HybridRAG
from step_07_multi_agent.implementation.orchestrator import run as orchestrate

CORPUS_PATH = _PROJECT_ROOT / "dataset" / "company_data"
GRAPH_PATH  = _PROJECT_ROOT / "step_05_knowledge_graph" / "results" / "graph.json"


class Step07RAG:
    """
    Multi-Agent RAG pipeline.

    Usage:
        rag = Step07RAG(k=5).build()
        result = rag.query("Vertexia has two efforts sharing the same name…")
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._retriever: Step04HybridRAG | None = None
        self._graph = None

    def build(self) -> "Step07RAG":
        self._retriever = Step04HybridRAG(k=self.k).build()
        from step_05_knowledge_graph.implementation.graph_store import load_or_build
        self._graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def query(self, question: str) -> RAGResult:
        if self._retriever is None or self._graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        answer, provider, traces = orchestrate(question, self._retriever, self._graph)
        total_ms = (time.perf_counter() - t0) * 1000

        # Representative chunk set for dashboard source display
        chunks = self._retriever.retrieve(question, k=self.k)

        return RAGResult(
            question=question,
            answer=answer,
            provider=f"multi-agent:{provider}",
            retrieved_chunks=chunks,
            context_sent="[multi-agent managed context — see orchestrator traces]",
            context_chars=len(answer),
            retrieval_latency_ms=total_ms,
            generation_latency_ms=0.0,
        )
