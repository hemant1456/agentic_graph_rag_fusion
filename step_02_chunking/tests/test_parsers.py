"""
Unit tests for Step 04 parsers and SmartChunk.

No API calls or filesystem I/O required (except for markdown/text parsers
which use a tmp_path fixture). All CSV tests use inline test data.
"""

import hashlib
import textwrap
from pathlib import Path

import pytest

from step_02_chunking.implementation.parsers.csv_parser import (
    _build_aggregate_text,
    _classify_columns,
    parse_csv,
)
from step_02_chunking.implementation.parsers.markdown_parser import parse_markdown
from step_02_chunking.implementation.parsers.text_parser import parse_text
from step_02_chunking.implementation.types import SmartChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content)
    return p


def _write_file(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content)
    return p


# ── Test 1: CSV aggregate — numeric totals appear ─────────────────────────────

def test_csv_aggregate_customer_list(tmp_path):
    """Aggregate chunk must contain correct sum of arr_usd for a 3-row mock CSV."""
    csv_content = textwrap.dedent("""\
        company_name,segment,arr_usd,contract_start
        Acme Corp,enterprise,1000000,2022-01-01
        Beta Ltd,mid-market,500000,2023-03-15
        Gamma Inc,smb,250000,2023-08-01
    """)
    csv_path = _write_csv(tmp_path, "mock_customers.csv", csv_content)
    chunks = parse_csv(csv_path, "sales/mock_customers.csv", "sales")

    # Find the aggregate chunk
    agg = next(c for c in chunks if c.chunk_type == "aggregate")

    # Total should be 1,750,000
    assert "1,750,000" in agg.text, f"Expected total 1,750,000 in:\n{agg.text}"
    # The table header should appear
    assert "mock_customers.csv" in agg.text
    assert "AGGREGATE SUMMARY" in agg.text


def test_csv_aggregate_section_counts(tmp_path):
    """Categorical breakdown must appear with correct group counts."""
    csv_content = textwrap.dedent("""\
        company_name,segment,arr_usd,contract_start
        Acme Corp,enterprise,1000000,2022-01-01
        Beta Ltd,mid-market,500000,2023-03-15
        Gamma Inc,smb,250000,2023-08-01
        Delta Co,enterprise,800000,2023-07-01
        Echo Sys,mid-market,300000,2022-11-01
    """)
    csv_path = _write_csv(tmp_path, "mock_customers.csv", csv_content)
    chunks = parse_csv(csv_path, "sales/mock_customers.csv", "sales")

    agg = next(c for c in chunks if c.chunk_type == "aggregate")

    # "segment breakdown:" section should appear
    assert "segment breakdown:" in agg.text.lower() or "segment" in agg.text

    # enterprise group should show 2 rows
    assert "enterprise: 2 rows" in agg.text, (
        f"Expected 'enterprise: 2 rows' in:\n{agg.text}"
    )


def test_csv_row_chunk_count(tmp_path):
    """Verify correct number of row chunks (N) + 1 aggregate chunk."""
    csv_content = textwrap.dedent("""\
        id,value,category
        A,100,X
        B,200,Y
        C,300,X
        D,400,Y
        E,500,X
    """)
    csv_path = _write_csv(tmp_path, "data.csv", csv_content)
    chunks = parse_csv(csv_path, "data.csv", "test")

    row_chunks = [c for c in chunks if c.chunk_type == "row"]
    agg_chunks = [c for c in chunks if c.chunk_type == "aggregate"]

    assert len(row_chunks) == 5, f"Expected 5 row chunks, got {len(row_chunks)}"
    assert len(agg_chunks) == 1, f"Expected 1 aggregate chunk, got {len(agg_chunks)}"
    assert len(chunks) == 6


# ── Test 4: Markdown section split ───────────────────────────────────────────

def test_markdown_section_split(tmp_path):
    """Markdown with 3 H2 headers should produce 3 section chunks (plus optional preamble)."""
    md_content = textwrap.dedent("""\
        ## Overview

        This is the overview section with some content.

        ## Architecture

        This is the architecture section with more content.

        ## Deployment

        This is the deployment section with deployment steps.
    """)
    md_path = _write_file(tmp_path, "doc.md", md_content)
    chunks = parse_markdown(md_path, "engineering/doc.md", "engineering")

    section_chunks = [c for c in chunks if c.chunk_type == "section"]

    # Should have at least 3 sections (one per heading)
    assert len(section_chunks) >= 3, (
        f"Expected >= 3 section chunks, got {len(section_chunks)}: "
        f"{[c.extra.get('section_title') for c in section_chunks]}"
    )

    titles = [c.extra.get("section_title", "") for c in section_chunks]
    assert any("Overview" in t for t in titles), f"No Overview section in {titles}"
    assert any("Architecture" in t for t in titles), f"No Architecture section in {titles}"
    assert any("Deployment" in t for t in titles), f"No Deployment section in {titles}"


def test_markdown_large_section_split(tmp_path):
    """A section exceeding 3000 chars must be sub-chunked into multiple chunks."""
    # Build a section > 3000 chars
    repeated_para = "This is a paragraph with content. " * 20  # ~680 chars per para
    long_section = "\n\n".join([repeated_para] * 6)  # ~4080 chars

    md_content = f"## Big Section\n\n{long_section}\n\n## Small Section\n\nShort content.\n"
    md_path = _write_file(tmp_path, "big_doc.md", md_content)
    chunks = parse_markdown(md_path, "engineering/big_doc.md", "engineering")

    section_chunks = [c for c in chunks if c.chunk_type == "section"]

    # The large section must have been split into 2+ chunks
    big_section_chunks = [
        c for c in section_chunks if "Big Section" in c.extra.get("section_title", "")
    ]
    assert len(big_section_chunks) >= 2, (
        f"Expected large section to be sub-chunked into >= 2 chunks, "
        f"got {len(big_section_chunks)}"
    )

    # Each chunk should be <= 3000 chars (the fallback chunk size is 2000)
    for c in big_section_chunks:
        assert len(c.text) <= 3000, f"Chunk too large: {len(c.text)} chars"


# ── Test 6: Text parser paragraph chunking ───────────────────────────────────

def test_text_parser_paragraph(tmp_path):
    """Text file without structure should be chunked paragraph-by-paragraph."""
    # 3 small distinct paragraphs that fit in one 2000-char chunk
    text_content = textwrap.dedent("""\
        Vertexia was founded in 2019.

        The company builds data pipelines.

        NexusFlow is their flagship product.
    """)
    txt_path = _write_file(tmp_path, "notes.txt", text_content)
    chunks = parse_text(txt_path, "executive/notes.txt", "executive")

    # Should produce at least 1 chunk
    assert len(chunks) >= 1

    # All chunks should be prose or section type
    for c in chunks:
        assert c.chunk_type in ("prose", "section"), (
            f"Unexpected chunk_type '{c.chunk_type}'"
        )

    # Content should appear somewhere in the chunks
    all_text = " ".join(c.text for c in chunks)
    assert "Vertexia" in all_text
    assert "NexusFlow" in all_text


# ── Test 7: SmartChunk ID stability ──────────────────────────────────────────

def test_smart_chunk_id_stable():
    """Same source + chunk_type + chunk_index must always produce the same chunk_id."""
    c1 = SmartChunk(
        text="hello world",
        source="sales/customers.csv",
        department="sales",
        format="csv",
        chunk_type="aggregate",
        chunk_index=20,
    )
    c2 = SmartChunk(
        text="completely different text",
        source="sales/customers.csv",
        department="sales",
        format="csv",
        chunk_type="aggregate",
        chunk_index=20,
    )

    # IDs should be identical — content doesn't affect the ID
    assert c1.chunk_id == c2.chunk_id, (
        f"Expected same IDs: {c1.chunk_id!r} vs {c2.chunk_id!r}"
    )

    # Different chunk_index → different ID
    c3 = SmartChunk(
        text="hello world",
        source="sales/customers.csv",
        department="sales",
        format="csv",
        chunk_type="aggregate",
        chunk_index=21,
    )
    assert c1.chunk_id != c3.chunk_id


def test_smart_chunk_id_formula():
    """chunk_id must be derived from source::chunk_type::chunk_index via MD5."""
    source = "finance/budget.csv"
    chunk_type = "row"
    chunk_index = 3

    key = f"{source}::{chunk_type}::{chunk_index}"
    expected_digest = hashlib.md5(key.encode()).hexdigest()[:8]

    c = SmartChunk(
        text="any text",
        source=source,
        department="finance",
        format="csv",
        chunk_type=chunk_type,
        chunk_index=chunk_index,
    )

    assert expected_digest in c.chunk_id, (
        f"Expected digest {expected_digest!r} in chunk_id {c.chunk_id!r}"
    )


# ── Test 8: SmartChunk metadata — only scalar values ─────────────────────────

def test_smart_chunk_metadata_scalars_only():
    """to_metadata() must return only str/int/float/bool values (ChromaDB constraint)."""
    c = SmartChunk(
        text="some text",
        source="hr/employees.csv",
        department="hr",
        format="csv",
        chunk_type="aggregate",
        chunk_index=5,
        extra={
            "row_count": 48,               # int — allowed
            "headers": "id, name, dept",   # str — allowed
            "nested_dict": {"foo": "bar"},  # dict — NOT allowed, should be excluded
            "a_list": [1, 2, 3],           # list — NOT allowed, should be excluded
            "is_summary": True,            # bool — allowed
            "ratio": 1.23,                 # float — allowed
        },
    )

    meta = c.to_metadata()

    # Check that no non-scalar values are present
    for k, v in meta.items():
        assert isinstance(v, (str, int, float, bool)), (
            f"Metadata key '{k}' has non-scalar value: {type(v).__name__} = {v!r}"
        )

    # Scalars from extra should be present
    assert meta["row_count"] == 48
    assert meta["headers"] == "id, name, dept"
    assert meta["is_summary"] is True
    assert meta["ratio"] == 1.23

    # Non-scalars from extra should be excluded
    assert "nested_dict" not in meta
    assert "a_list" not in meta

    # Core fields should always be present
    assert meta["source"] == "hr/employees.csv"
    assert meta["department"] == "hr"
    assert meta["format"] == "csv"
    assert meta["chunk_type"] == "aggregate"
    assert meta["chunk_index"] == 5


# ── Test 9: Aggregate date-based aggregations ─────────────────────────────────

def test_csv_aggregate_date_periods(tmp_path):
    """Aggregate chunk must contain quarter/half-year aggregations for date columns."""
    csv_content = textwrap.dedent("""\
        month,revenue
        2023-01,100000
        2023-04,110000
        2023-07,120000
        2023-10,130000
    """)
    csv_path = _write_csv(tmp_path, "revenue.csv", csv_content)
    chunks = parse_csv(csv_path, "finance/revenue.csv", "finance")

    agg = next(c for c in chunks if c.chunk_type == "aggregate")

    # Should contain quarter labels
    assert "Q1 2023" in agg.text, f"Expected Q1 2023 in:\n{agg.text}"
    assert "Q3 2023" in agg.text, f"Expected Q3 2023 in:\n{agg.text}"
    # Should contain half-year labels
    assert "H1 2023" in agg.text, f"Expected H1 2023 in:\n{agg.text}"
    assert "H2 2023" in agg.text, f"Expected H2 2023 in:\n{agg.text}"


# ── Test 10: Per-row ratio in aggregate ──────────────────────────────────────

def test_csv_aggregate_per_row_ratio(tmp_path):
    """Aggregate chunk must compute and display per-row numeric ratios."""
    csv_content = textwrap.dedent("""\
        department,budget,headcount
        Engineering,9000000,45
        Marketing,3000000,15
        Sales,6000000,30
    """)
    csv_path = _write_csv(tmp_path, "budget.csv", csv_content)
    chunks = parse_csv(csv_path, "finance/budget.csv", "finance")

    agg = next(c for c in chunks if c.chunk_type == "aggregate")

    # Should contain ratio section
    assert "Per-row" in agg.text and "ratio" in agg.text.lower(), (
        f"Expected ratio section in:\n{agg.text}"
    )
    # Engineering has the highest budget/headcount: 9M/45 = 200,000/unit
    assert "Engineering" in agg.text
    assert "Highest" in agg.text
