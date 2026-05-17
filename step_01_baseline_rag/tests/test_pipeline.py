"""
Sanity tests for Step 01 components.

These are smoke tests — they verify the pipeline is wired correctly,
not that the answers are good (that's what run_eval.py measures).

Run:
    uv run pytest step_01_baseline_rag/tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Ingest tests ──────────────────────────────────────────────────────────────

def test_load_and_chunk_produces_chunks():
    from step_01_baseline_rag.implementation.ingest import load_and_chunk
    corpus = Path(__file__).parent.parent.parent / "dataset" / "company_data"
    chunks = load_and_chunk(corpus)
    assert len(chunks) > 50, f"Expected 50+ chunks, got {len(chunks)}"


def test_chunks_have_required_fields():
    from step_01_baseline_rag.implementation.ingest import load_and_chunk
    corpus = Path(__file__).parent.parent.parent / "dataset" / "company_data"
    chunks = load_and_chunk(corpus)
    for c in chunks[:5]:
        assert c.text.strip(), "Chunk text must not be empty"
        assert c.source, "Chunk must have a source"
        assert c.department, "Chunk must have a department"
        assert c.chunk_id, "Chunk must have a unique ID"


def test_csv_chunking_creates_row_chunks():
    from step_01_baseline_rag.implementation.ingest import load_and_chunk
    corpus = Path(__file__).parent.parent.parent / "dataset" / "company_data"
    chunks = load_and_chunk(corpus)
    csv_chunks = [c for c in chunks if c.format == "csv"]
    assert len(csv_chunks) > 20, "Expected many CSV row chunks"
    # Each CSV chunk should contain key=value pairs
    for c in csv_chunks[:3]:
        assert ":" in c.text or "|" in c.text, "CSV chunk should have key-value format"


def test_chunk_ids_are_unique():
    from step_01_baseline_rag.implementation.ingest import load_and_chunk
    corpus = Path(__file__).parent.parent.parent / "dataset" / "company_data"
    chunks = load_and_chunk(corpus)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_sizes_within_bound():
    from step_01_baseline_rag.implementation.ingest import CHUNK_SIZE_CHARS, load_and_chunk
    corpus = Path(__file__).parent.parent.parent / "dataset" / "company_data"
    chunks = load_and_chunk(corpus)
    # Allow 10% overshoot for edge cases (large single paragraphs)
    oversized = [c for c in chunks if len(c.text) > CHUNK_SIZE_CHARS * 1.1]
    assert len(oversized) == 0, f"{len(oversized)} chunks exceed size limit"


# ── Pipeline integration test (requires API key, marked slow) ─────────────────

@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "step_01_baseline_rag" / "results" / "chroma_db").exists(),
    reason="ChromaDB index not built yet — run ingest.py first",
)
def test_pipeline_returns_result():
    from step_01_baseline_rag.implementation.pipeline import BaselineRAG
    rag = BaselineRAG(k=3)
    rag.build()
    result = rag.query("What is Vertexia's data retention policy?")
    assert result.answer, "Answer should not be empty"
    assert len(result.retrieved_chunks) > 0, "Should retrieve at least one chunk"
    assert result.context_chars > 0


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "step_01_baseline_rag" / "results" / "chroma_db").exists(),
    reason="ChromaDB index not built yet",
)
def test_retrieval_returns_relevant_source_for_simple_query():
    """The onboarding handbook should be in top-5 for a data retention question."""
    from step_01_baseline_rag.implementation.pipeline import BaselineRAG
    rag = BaselineRAG(k=5)
    rag.build()
    result = rag.query("What is the data retention policy for customer data?")
    sources = [c.source for c in result.retrieved_chunks]
    assert any("onboarding" in s for s in sources), (
        f"Expected onboarding_handbook.txt in top-5, got: {sources}"
    )
