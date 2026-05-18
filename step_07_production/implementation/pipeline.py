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
from step_06_context_engineering.implementation.pipeline import Step06RAG, Step06Result
from step_07_production.implementation.confidence import score_answer
from step_07_production.implementation.graceful_degradation import extractive_fallback
from step_07_production.implementation.health_monitor import HealthMonitor
from step_07_production.implementation.semantic_cache import SemanticCache

CORPUS_PATH = _PROJECT_ROOT / "dataset" / "company_data"
GRAPH_PATH  = _PROJECT_ROOT / "step_04_knowledge_graph" / "results" / "graph.json"

# Module-level singletons so build() calls across the session share state
_global_cache: SemanticCache | None = None
_global_monitor: HealthMonitor | None = None


@dataclass
class Step07Result:
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


class Step07RAG:
    """
    Production-hardened RAG pipeline.

    Composition:
      - Inner: Step06RAG (VSA routing → CE stack → gateway LLM).
      - Semantic cache (cosine-similarity over question embeddings).
      - Multi-signal confidence (critic + retrieval + answer sanity).
      - Health monitor (latency, error rate, SLO).
      - Graceful degradation (extractive fallback if the inner pipeline
        throws on a non-transient failure).

    Retry of transient LLM failures lives inside `llm_gatewayV2.client.LLM`
    (HTTP-level retries on 5xx / connect errors). The earlier "retry the
    whole pipeline" approach is gone — see AUDIT_FIXES.md #13.
    """

    def __init__(self, k: int = 5, cache_threshold: float = 0.92) -> None:
        self.k = k
        self.cache_threshold = cache_threshold
        self._inner: Step06RAG | None = None
        self._cache: SemanticCache | None = None
        self._monitor: HealthMonitor | None = None

    def build(self) -> "Step07RAG":
        global _global_cache, _global_monitor
        self._inner = Step06RAG(k=self.k).build()
        if _global_cache is None:
            _global_cache = SemanticCache(threshold=self.cache_threshold)
        if _global_monitor is None:
            _global_monitor = HealthMonitor(window=100)
        self._cache = _global_cache
        self._monitor = _global_monitor
        return self

    def query_extended(self, question: str) -> Step07Result:
        if self._inner is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()

        # ── Cache hit path ──────────────────────────────────────────────────
        cached = self._cache.get(question)
        if cached is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            conf = score_answer(
                question,
                cached.answer,
                critic_approved=cached.critic_approved,
                critic_notes=cached.critic_notes,
                context_chars=cached.ce_metrics.get("engineered_chars", 0),
                chunks_used=cached.ce_metrics.get("chunks_final", 0),
            )
            self._monitor.record(
                latency_ms=latency_ms,
                grade="PASS" if conf["label"] in ("high", "medium") else "FAIL",
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
                critic_approved=cached.critic_approved,
                critic_notes=cached.critic_notes,
            )
            return Step07Result(
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

        # ── Miss path — run inner pipeline, fall back to extractive ─────────
        answer: str | None = None
        provider = "error"
        ce_metrics: dict = {}
        slice_name = "unknown"
        router_confidence = 0.0
        display_chunks: list = []
        critic_approved: bool | None = None
        critic_notes = ""

        try:
            ext: Step06Result = self._inner.query_extended(question)
            answer = ext.rag_result.answer
            provider = ext.rag_result.provider
            ce_metrics = ext.ce_metrics
            slice_name = ext.slice_name
            router_confidence = ext.router_confidence
            display_chunks = ext.rag_result.retrieved_chunks
            critic_approved = ext.rag_result.critic_approved
            critic_notes = ext.rag_result.critic_notes
        except Exception:
            # The gateway client owns transient-LLM retries; reaching this
            # branch means a non-transient pipeline failure (Chroma down,
            # graph file missing, etc.). Fall back to deterministic
            # extractive answer over the top-k retrieved chunks so the user
            # gets something traceable instead of an error.
            raw_chunks = (
                self._inner._retriever.retrieve(question, k=self.k)
                if self._inner._retriever
                else []
            )
            answer, provider = extractive_fallback(question, raw_chunks)
            display_chunks = raw_chunks
            ce_metrics = {"engineered_chars": 0, "raw_chars": 0, "chunks_final": len(raw_chunks)}
            slice_name = "fallback"
            router_confidence = 0.0
            critic_approved = None  # no critic ran on the extractive answer
            critic_notes = "extractive fallback — no synthesis"

        # ── Score, annotate, and record ─────────────────────────────────────
        conf = score_answer(
            question,
            answer or "",
            critic_approved=critic_approved,
            critic_notes=critic_notes,
            context_chars=ce_metrics.get("engineered_chars", 0),
            chunks_used=ce_metrics.get("chunks_final", len(display_chunks)),
        )

        if conf["label"] == "low" and answer and not answer.startswith("[Extractive]"):
            answer = answer + (
                f"\n\n[Note: low-confidence answer — {conf['reason']}. "
                "Verify against source documents.]"
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        self._monitor.record(
            latency_ms=latency_ms,
            grade="PASS" if conf["label"] in ("high", "medium") else "FAIL",
            from_cache=False,
            slice_name=slice_name,
            confidence_label=conf["label"],
        )

        self._cache.put(
            question,
            answer,
            provider,
            ce_metrics,
            critic_approved=critic_approved,
            critic_notes=critic_notes,
        )

        rag_result = RAGResult(
            question=question,
            answer=answer,
            provider=f"prod:{provider}",
            retrieved_chunks=display_chunks,
            context_sent=f"[VSA slice={slice_name} conf={router_confidence:.2f}]",
            context_chars=ce_metrics.get("engineered_chars", 0),
            retrieval_latency_ms=latency_ms,
            generation_latency_ms=0.0,
            critic_approved=critic_approved,
            critic_notes=critic_notes,
        )
        return Step07Result(
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
