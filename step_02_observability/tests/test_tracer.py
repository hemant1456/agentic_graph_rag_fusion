"""
Tests for Step 02 — Observability.

Unit tests for the trace data model, JSONL store, and cost estimator.
Integration test verifies TracedRAG produces a valid trace for a real query.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_02_observability.implementation.tracer import (
    ChunkTrace,
    GenerationSpan,
    QueryTrace,
    RetrievalSpan,
    TraceStore,
    estimate_cost,
    new_trace_id,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_trace(trace_id: str = "abc12345") -> QueryTrace:
    return QueryTrace(
        trace_id=trace_id,
        timestamp="2025-01-01T12:00:00",
        step="01_baseline_rag",
        query="What is the data retention policy?",
        answer="Hot storage: 90 days. Cold storage: S3 Glacier (1 year).",
        retrieval=RetrievalSpan(
            duration_ms=150.0,
            chunks=[
                ChunkTrace(
                    rank=1,
                    source="hr/onboarding_handbook.txt",
                    department="hr",
                    similarity=0.921,
                    char_count=800,
                    text_preview="Data retention: 90-day hot storage...",
                ),
                ChunkTrace(
                    rank=2,
                    source="legal/data_processing_agreement_template.txt",
                    department="legal",
                    similarity=0.874,
                    char_count=600,
                    text_preview="Customer data retained for 90 days in hot tier...",
                ),
            ],
        ),
        generation=GenerationSpan(
            duration_ms=900.0,
            provider="gemini",
            model="gemini-flash",
            context_chars=1400,
            context_chunk_count=5,
            prompt_tokens=350,
            completion_tokens=80,
            estimated_cost_usd=0.000051,
        ),
        total_latency_ms=1060.0,
    )


# ── trace_id ──────────────────────────────────────────────────────────────────

def test_new_trace_id_length():
    assert len(new_trace_id()) == 8


def test_new_trace_id_unique():
    ids = {new_trace_id() for _ in range(100)}
    assert len(ids) == 100


# ── TraceStore ────────────────────────────────────────────────────────────────

def test_store_write_and_read(tmp_path: Path):
    store = TraceStore(tmp_path / "traces.jsonl")
    trace = _make_trace()
    store.write(trace)
    records = store.read_all()
    assert len(records) == 1
    assert records[0]["trace_id"] == trace.trace_id
    assert records[0]["query"] == trace.query


def test_store_multiple_traces(tmp_path: Path):
    store = TraceStore(tmp_path / "traces.jsonl")
    for i in range(5):
        store.write(_make_trace(f"id{i:05d}"))
    assert store.count() == 5


def test_store_count_empty(tmp_path: Path):
    store = TraceStore(tmp_path / "traces.jsonl")
    assert store.count() == 0


def test_store_roundtrip_nested(tmp_path: Path):
    store = TraceStore(tmp_path / "traces.jsonl")
    trace = _make_trace()
    store.write(trace)
    records = store.read_all()
    r = records[0]
    assert r["retrieval"]["chunks"][0]["similarity"] == 0.921
    assert r["retrieval"]["chunks"][1]["source"] == "legal/data_processing_agreement_template.txt"
    assert r["generation"]["prompt_tokens"] == 350
    assert r["generation"]["estimated_cost_usd"] == 0.000051


def test_store_get_by_id(tmp_path: Path):
    store = TraceStore(tmp_path / "traces.jsonl")
    store.write(_make_trace("aaa11111"))
    store.write(_make_trace("bbb22222"))
    found = store.get("bbb22222")
    assert found is not None
    assert found["trace_id"] == "bbb22222"


def test_store_get_missing(tmp_path: Path):
    store = TraceStore(tmp_path / "traces.jsonl")
    store.write(_make_trace())
    assert store.get("notexist") is None


def test_store_clear(tmp_path: Path):
    store = TraceStore(tmp_path / "traces.jsonl")
    store.write(_make_trace())
    store.write(_make_trace("other001"))
    store.clear()
    assert store.count() == 0


def test_store_creates_parent_dir(tmp_path: Path):
    deep_path = tmp_path / "a" / "b" / "c" / "traces.jsonl"
    store = TraceStore(deep_path)
    store.write(_make_trace())
    assert deep_path.exists()


# ── Cost estimation ───────────────────────────────────────────────────────────

def test_cost_gemini_one_million_tokens():
    cost = estimate_cost("gemini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(cost - 0.375) < 0.001   # $0.075 + $0.30


def test_cost_anthropic_one_million_tokens():
    cost = estimate_cost("anthropic", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(cost - 0.48) < 0.001    # $0.08 + $0.40


def test_cost_zero_tokens():
    assert estimate_cost("gemini", 0, 0) == 0.0


def test_cost_unknown_provider():
    assert estimate_cost("unknown_model", 100_000, 10_000) == 0.0


def test_cost_small_query():
    cost = estimate_cost("gemini", prompt_tokens=400, completion_tokens=100)
    assert 0.0 < cost < 0.001


# ── RetrievalSpan properties ──────────────────────────────────────────────────

def test_retrieval_top_source():
    trace = _make_trace()
    assert trace.retrieval.top_source == "hr/onboarding_handbook.txt"


def test_retrieval_unique_sources():
    trace = _make_trace()
    sources = trace.retrieval.unique_sources
    assert sources[0] == "hr/onboarding_handbook.txt"
    assert sources[1] == "legal/data_processing_agreement_template.txt"
    assert len(sources) == 2


def test_retrieval_unique_sources_deduplicates():
    span = RetrievalSpan(
        duration_ms=100.0,
        chunks=[
            ChunkTrace(1, "doc/a.txt", "eng", 0.9, 100, "..."),
            ChunkTrace(2, "doc/a.txt", "eng", 0.85, 100, "..."),   # same source
            ChunkTrace(3, "doc/b.txt", "hr",  0.8, 100, "..."),
        ],
    )
    assert span.unique_sources == ["doc/a.txt", "doc/b.txt"]


# ── GenerationSpan properties ─────────────────────────────────────────────────

def test_generation_total_tokens():
    trace = _make_trace()
    assert trace.generation.total_tokens == 350 + 80


# ── QueryTrace serialization ──────────────────────────────────────────────────

def test_trace_as_dict_keys():
    trace = _make_trace()
    d = trace.as_dict()
    assert set(d.keys()) == {
        "trace_id", "timestamp", "step", "query", "answer",
        "retrieval", "generation", "total_latency_ms",
    }


def test_trace_as_dict_is_json_serializable():
    import json
    trace = _make_trace()
    raw = json.dumps(trace.as_dict())
    parsed = json.loads(raw)
    assert parsed["trace_id"] == trace.trace_id


# ── Integration: TracedRAG produces a valid trace ─────────────────────────────

@pytest.mark.integration
def test_traced_rag_produces_trace(tmp_path: Path):
    """
    Full pipeline integration test — requires ChromaDB index and API keys.
    Skipped automatically if the index doesn't exist.
    """
    from step_01_baseline_rag.implementation.pipeline import BaselineRAG
    from step_01_baseline_rag.implementation.ingest import get_chroma_collection

    db_path = Path(__file__).parent.parent.parent / "step_01_baseline_rag" / "results" / "chroma_db"
    if not db_path.exists():
        pytest.skip("ChromaDB index not built — run ingest.py first")

    rag = BaselineRAG(k=5).build()
    store = TraceStore(tmp_path / "traces.jsonl")

    from step_02_observability.implementation.traced_pipeline import TracedRAG
    traced = TracedRAG(rag, store)
    result, trace = traced.query("What is Vertexia's data retention policy?")

    # Structural checks
    assert len(trace.trace_id) == 8
    assert trace.query == "What is Vertexia's data retention policy?"
    assert len(trace.answer) > 10
    assert trace.retrieval.duration_ms > 0
    assert trace.generation.duration_ms > 0
    assert len(trace.retrieval.chunks) == 5
    assert all(0.0 <= c.similarity <= 1.0 for c in trace.retrieval.chunks)
    assert trace.generation.context_chars > 0
    assert trace.generation.total_tokens > 0
    assert trace.generation.estimated_cost_usd >= 0
    assert store.count() == 1
