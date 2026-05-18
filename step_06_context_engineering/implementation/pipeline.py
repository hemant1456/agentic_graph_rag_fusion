from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import networkx as nx

from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_03_hybrid_retrieval.implementation.pipeline import Step03HybridRAG
from step_06_context_engineering.implementation.router import dispatch as vsa_dispatch

CORPUS_PATH = _PROJECT_ROOT / "dataset" / "company_data"
GRAPH_PATH  = _PROJECT_ROOT / "step_04_knowledge_graph" / "results" / "graph.json"


@dataclass
class Step06Result:
    """Extended result that carries context-engineering + VSA routing metadata."""
    rag_result: RAGResult
    slice_name: str
    router_confidence: float
    ce_metrics: dict


class Step06RAG:
    """
    Context-Engineered RAG pipeline with VSA domain routing.

    Each question is routed to a domain slice (Finance / HR / Engineering / General)
    that owns its own system prompt, retrieval augmentation, rerank_k, and
    compression ratio. The underlying context-engineering stack (rerank → dedup →
    compress → XML budget) is shared across slices.

    Usage:
        rag = Step06RAG(k=5).build()
        ext = rag.query_extended("What is the total ARR?")   # → Step06Result
        res = rag.query("What is the total ARR?")             # → RAGResult
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._retriever: Step03HybridRAG | None = None
        self._graph: nx.DiGraph | None = None

    def build(self) -> "Step06RAG":
        # Wide candidate set — reranker selects the best top-k inside each slice
        self._retriever = Step03HybridRAG(k=20).build()
        from step_04_knowledge_graph.implementation.graph_store import load_or_build
        self._graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def query_extended(self, question: str) -> Step06Result:
        if self._retriever is None or self._graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        (
            answer,
            provider,
            ce_metrics,
            slice_name,
            confidence,
            context_xml,
            display_chunks,
        ) = vsa_dispatch(question, self._retriever, self._graph)
        total_ms = (time.perf_counter() - t0) * 1000

        # display_chunks come from engineer_context (post-rerank+dedup). No
        # second retrieval pass — the dashboard's "sources" panel renders the
        # exact chunks the answer is grounded on, capped at self.k for display.
        display_chunks = display_chunks[: self.k]

        rag_result = RAGResult(
            question=question,
            answer=answer,
            provider=f"vsa:{slice_name}:{provider}",
            retrieved_chunks=display_chunks,
            # Carry the full engineered context (CSV tool output + graph + reranked
            # chunks) so the eval judge can verify grounded vs hallucinated claims.
            context_sent=f"[VSA slice={slice_name} conf={confidence:.2f}]\n\n{context_xml}",
            context_chars=ce_metrics.get("engineered_chars", 0),
            retrieval_latency_ms=total_ms,
            generation_latency_ms=0.0,
        )
        return Step06Result(
            rag_result=rag_result,
            slice_name=slice_name,
            router_confidence=confidence,
            ce_metrics=ce_metrics,
        )

    def query(self, question: str) -> RAGResult:
        """Eval-compatible entry point — returns plain RAGResult."""
        return self.query_extended(question).rag_result
