from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from dataclasses import dataclass

import networkx as nx

from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_04_hybrid_retrieval.implementation.pipeline import Step04HybridRAG
from step_09_vsa.implementation.router import dispatch as vsa_dispatch

CORPUS_PATH = _PROJECT_ROOT / "dataset"  / "company_data"
GRAPH_PATH  = _PROJECT_ROOT / "step_05_knowledge_graph" / "results" / "graph.json"


@dataclass
class Step09Result:
    """Extended result that carries routing metadata."""
    rag_result: RAGResult
    slice_name: str
    router_confidence: float
    ce_metrics: dict


class Step09RAG:
    """
    VSA-aware RAG pipeline.

    Usage:
        rag = Step09RAG(k=5).build()
        ext = rag.query_extended("What is the total ARR?")   # → Step09Result
        res = rag.query("What is the total ARR?")             # → RAGResult
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._retriever: Step04HybridRAG | None = None
        self._graph: nx.DiGraph | None = None

    def build(self) -> "Step09RAG":
        # Wide candidate set — reranker will select the best 8
        self._retriever = Step04HybridRAG(k=20).build()
        from step_05_knowledge_graph.implementation.graph_store import load_or_build
        self._graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def query_extended(self, question: str) -> Step09Result:
        if self._retriever is None or self._graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        answer, provider, ce_metrics, slice_name, confidence = vsa_dispatch(
            question, self._retriever, self._graph
        )
        total_ms = (time.perf_counter() - t0) * 1000

        display_chunks = self._retriever.retrieve(question, k=self.k)

        rag_result = RAGResult(
            question=question,
            answer=answer,
            provider=f"vsa:{slice_name}:{provider}",
            retrieved_chunks=display_chunks,
            context_sent=f"[VSA slice={slice_name} conf={confidence:.2f}]",
            context_chars=ce_metrics.get("engineered_chars", 0),
            retrieval_latency_ms=total_ms,
            generation_latency_ms=0.0,
        )
        return Step09Result(
            rag_result=rag_result,
            slice_name=slice_name,
            router_confidence=confidence,
            ce_metrics=ce_metrics,
        )

    def query(self, question: str) -> RAGResult:
        return self.query_extended(question).rag_result
