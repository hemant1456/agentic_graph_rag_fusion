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
from step_05_multi_agent.implementation.agents import (
    critic,
    graph_navigator,
    query_analyst,
    retrieval_specialist,
    structured_data,
    synthesis,
)
from step_06_context_engineering.implementation.context_engineer import engineer_context

CORPUS_PATH = _PROJECT_ROOT / "dataset" / "company_data"
GRAPH_PATH  = _PROJECT_ROOT / "step_04_knowledge_graph" / "results" / "graph.json"


@dataclass
class Step06Result:
    """Extended RAGResult that carries context engineering metrics."""
    rag_result: RAGResult
    ce_metrics: dict  # raw_chars, engineered_chars, compression_ratio, etc.


class Step06RAG:
    """
    Context-Engineered RAG pipeline.

    Usage:
        rag = Step06RAG(k=5).build()
        result = rag.query_extended("Who is the CEO?")   # → Step06Result
        result = rag.query("Who is the CEO?")             # → RAGResult (eval-compatible)
    """

    def __init__(self, k: int = 5, rerank_k: int = 8, compress_ratio: float = 0.60) -> None:
        self.k = k
        self.rerank_k = rerank_k
        self.compress_ratio = compress_ratio
        self._retriever: Step03HybridRAG | None = None
        self._graph: nx.DiGraph | None = None

    def build(self) -> "Step06RAG":
        # Retrieve wide candidate set (k=20) for the reranker to select from
        self._retriever = Step03HybridRAG(k=20).build()
        from step_04_knowledge_graph.implementation.graph_store import load_or_build
        self._graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def query_extended(self, question: str) -> Step06Result:
        """Full query returning RAGResult + CE metrics."""
        if self._retriever is None or self._graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()

        analysis = query_analyst.analyze(question)

        ret = retrieval_specialist.retrieve(question, self._retriever, k=20)
        raw_chunks = list(ret.chunks)

        # Sub-question retrieval for compound queries
        for sub_q in analysis.sub_questions[:4]:
            sub_ret = retrieval_specialist.retrieve(sub_q, self._retriever, k=10)
            raw_chunks.extend(sub_ret.chunks)

        # Graph + CSV contexts (not compressed — exact data must survive).
        # Always run graph navigation using retrieved chunks as seeds, mirroring step 07.
        csv_data = ""
        graph_seeds = [c.text for c in raw_chunks] if raw_chunks else analysis.primary_entities
        graph_res = graph_navigator.navigate(question, graph_seeds, self._graph)
        graph_ctx = graph_res.context if graph_res.success else ""
        # Always run structured CSV query — mirrors step 07's unconditional detect_intent() → run_query().
        csv_res = structured_data.query(question)
        if csv_res.success:
            csv_data = csv_res.data

        context_xml, ce_metrics = engineer_context(
            question=question,
            raw_chunks=raw_chunks,
            csv_data=csv_data,
            graph_context=graph_ctx,
            rerank_k=self.rerank_k,
            compress_ratio=self.compress_ratio,
        )

        synth = synthesis.synthesize(
            question, {"Engineered Context": context_xml}, analysis.query_type
        )
        critic_res = critic.review(
            question, synth.answer, {"Engineered Context": context_xml}
        )

        total_ms = (time.perf_counter() - t0) * 1000

        # Representative retrieval set for dashboard display
        display_chunks = self._retriever.retrieve(question, k=self.k)

        rag_result = RAGResult(
            question=question,
            answer=critic_res.answer,
            provider=f"ce:{synth.provider}",
            retrieved_chunks=display_chunks,
            context_sent=context_xml[:2000] + ("…" if len(context_xml) > 2000 else ""),
            context_chars=ce_metrics["engineered_chars"],
            retrieval_latency_ms=total_ms,
            generation_latency_ms=0.0,
        )
        return Step06Result(rag_result=rag_result, ce_metrics=ce_metrics)

    def query(self, question: str) -> RAGResult:
        """Eval-compatible entry point — returns plain RAGResult."""
        return self.query_extended(question).rag_result
