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

from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_07_vsa.implementation.pipeline import Step07RAG, Step07Result
from step_08_production.implementation.confidence import score_answer
from step_08_production.implementation.graceful_degradation import extractive_fallback
from step_08_production.implementation.health_monitor import HealthMonitor
from step_08_production.implementation.retry import with_retry
from step_08_production.implementation.semantic_cache import SemanticCache

CORPUS_PATH = _PROJECT_ROOT / "dataset" / "company_data"
GRAPH_PATH  = _PROJECT_ROOT / "step_04_knowledge_graph" / "results" / "graph.json"

# Module-level singletons so build() calls across the session share state
_global_cache: SemanticCache | None = None
_global_monitor: HealthMonitor | None = None


@dataclass
class Step08Result:
    """Extended result with all production metadata."""
    rag_result: RAGResult
    slice_name: str
    router_confidence: float
    ce_metrics: dict
    confidence_score: float
    confidence_label: str
    from_cache: bool
    cache_stats: dict
    health_snapshot: dict


class Step08RAG:
    """
    Production-hardened RAG pipeline.
    """

    def __init__(self, k: int = 5, cache_threshold: float = 0.92) -> None:
        self.k = k
        self.cache_threshold = cache_threshold
        self._inner: Step07RAG | None = None
        self._cache: SemanticCache | None = None
        self._monitor: HealthMonitor | None = None

    def build(self) -> "Step08RAG":
        global _global_cache, _global_monitor
        self._inner = Step07RAG(k=self.k).build()
        if _global_cache is None:
            _global_cache = SemanticCache(threshold=self.cache_threshold)
        if _global_monitor is None:
            _global_monitor = HealthMonitor(window=100)
        self._cache = _global_cache
        self._monitor = _global_monitor
        return self

    def query_extended(self, question: str) -> Step08Result:
        if self._inner is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()

        cached = self._cache.get(question)
        if cached is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            conf = score_answer(question, cached.answer)
            self._monitor.record(
                latency_ms=latency_ms,
                grade="PASS",
                from_cache=True,
                slice_name="cache",
                confidence_label=conf["label"],
            )
            rag_result = RAGResult(
                question=question,
                answer=cached.answer,
                provider=f"cache:{cached.provider}",
                retrieved_chunks=[],
                context_sent="[from semantic cache]",
                context_chars=cached.ce_metrics.get("engineered_chars", 0),
                retrieval_latency_ms=latency_ms,
                generation_latency_ms=0.0,
            )
            return Step08Result(
                rag_result=rag_result,
                slice_name="cache",
                router_confidence=1.0,
                ce_metrics=cached.ce_metrics,
                confidence_score=conf["score"],
                confidence_label=conf["label"],
                from_cache=True,
                cache_stats=self._cache.stats,
                health_snapshot=self._monitor.snapshot(),
            )

        answer, provider, ce_metrics, slice_name, router_confidence = (
            None, "error", {}, "unknown", 0.0
        )
        display_chunks = []

        @with_retry(max_attempts=3, base_delay=0.5, exceptions=(Exception,))
        def _run_pipeline():
            nonlocal answer, provider, ce_metrics, slice_name, router_confidence, display_chunks
            ext: Step07Result = self._inner.query_extended(question)
            answer = ext.rag_result.answer
            provider = ext.rag_result.provider
            ce_metrics = ext.ce_metrics
            slice_name = ext.slice_name
            router_confidence = ext.router_confidence
            display_chunks = ext.rag_result.retrieved_chunks

        try:
            _run_pipeline()
        except Exception:
            raw_chunks = self._inner._retriever.retrieve(question, k=self.k) if self._inner._retriever else []
            answer, provider = extractive_fallback(question, raw_chunks)
            display_chunks = raw_chunks
            ce_metrics = {"engineered_chars": 0, "raw_chars": 0}
            slice_name = "fallback"
            router_confidence = 0.0

        conf = score_answer(question, answer or "")
        if conf["label"] == "low" and not answer.startswith("[Extractive]"):
            # Append uncertainty signal to the answer
            answer = answer + "\n\n[Note: low confidence — verify against source documents]"

        latency_ms = (time.perf_counter() - t0) * 1000

        self._monitor.record(
            latency_ms=latency_ms,
            grade="PASS" if conf["label"] in ("high", "medium") else "FAIL",
            from_cache=False,
            slice_name=slice_name,
            confidence_label=conf["label"],
        )

        self._cache.put(question, answer, provider, ce_metrics)

        rag_result = RAGResult(
            question=question,
            answer=answer,
            provider=f"prod:{provider}",
            retrieved_chunks=display_chunks,
            context_sent=f"[VSA slice={slice_name} conf={router_confidence:.2f}]",
            context_chars=ce_metrics.get("engineered_chars", 0),
            retrieval_latency_ms=latency_ms,
            generation_latency_ms=0.0,
        )
        return Step08Result(
            rag_result=rag_result,
            slice_name=slice_name,
            router_confidence=router_confidence,
            ce_metrics=ce_metrics,
            confidence_score=conf["score"],
            confidence_label=conf["label"],
            from_cache=False,
            cache_stats=self._cache.stats,
            health_snapshot=self._monitor.snapshot(),
        )

    def query(self, question: str) -> RAGResult:
        return self.query_extended(question).rag_result
