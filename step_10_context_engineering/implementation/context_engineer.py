"""
ContextEngineer — the Step 10 context engineering pipeline.

Sits between retrieval and synthesis.  Takes the raw outputs from the Step 09
multi-agent retrieval layer and applies four transformations in sequence:

  1. Rerank    — CrossEncoder scores all candidate chunks; keep top-k
  2. Deduplicate — remove near-duplicate passages (6-gram Jaccard > 0.72)
  3. Compress  — extractive sentence filtering per passage (60% retention)
  4. Format    — structured XML with source attribution + token budget cap

The result is a smaller, higher-signal context that the synthesis LLM can
attend to more precisely — demonstrating "same quality, less noise, lower cost".

Usage:
    from step_10_context_engineering.implementation.context_engineer import engineer_context

    ctx, metrics = engineer_context(question, raw_chunks, csv_data, graph_ctx)
    # metrics: raw_chars, engineered_chars, compression_ratio, chunks_before/after
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from step_10_context_engineering.implementation import (
    compressor,
    deduplicator,
    formatter,
    reranker,
)

if TYPE_CHECKING:
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk


def engineer_context(
    question: str,
    raw_chunks: "list[RetrievedChunk]",
    csv_data: str = "",
    graph_context: str = "",
    rerank_k: int = 8,
    compress_ratio: float = 0.60,
    budget_chars: int = 24_000,
) -> tuple[str, dict]:
    """
    Full context engineering pipeline.  Returns (context_xml, metrics).

    metrics:
      raw_chars          — chars before any engineering
      engineered_chars   — chars of the final XML context
      compression_ratio  — engineered / raw
      chunks_before      — candidate chunks entering the pipeline
      chunks_after_dedup — after near-duplicate removal
      chunks_final       — passages in the XML (≤ rerank_k)
    """
    chunks_before = len(raw_chunks)

    # ── 1. Rerank ──────────────────────────────────────────────────────────────
    scored = reranker.rerank(question, raw_chunks, k=rerank_k)

    # ── 2. Deduplicate ─────────────────────────────────────────────────────────
    deduped = deduplicator.deduplicate(scored)
    chunks_after_dedup = len(deduped)

    # ── 3. Compress each passage (vector chunks only; CSV/Graph skipped) ───────
    compressed: list[tuple[float, "RetrievedChunk"]] = []
    for score, chunk in deduped:
        # Shallow-copy the chunk so we don't mutate the shared retriever cache
        import copy
        c = copy.copy(chunk)
        c.text = compressor.compress(question, chunk.text, ratio=compress_ratio)
        compressed.append((score, c))

    # ── 4. Format + budget ─────────────────────────────────────────────────────
    raw_passage_chars = sum(len(c.text) for c in raw_chunks)
    raw_extra_chars   = len(csv_data) + len(graph_context)
    raw_total_chars   = raw_passage_chars + raw_extra_chars

    context_xml, fmt_stats = formatter.format_context(
        question=question,
        scored_chunks=compressed,
        csv_data=csv_data,
        graph_context=graph_context,
        max_chars=budget_chars,
    )

    metrics = {
        "raw_chars": raw_total_chars,
        "engineered_chars": fmt_stats["engineered_chars"],
        "compression_ratio": round(fmt_stats["engineered_chars"] / raw_total_chars, 3)
        if raw_total_chars > 0 else 1.0,
        "chunks_before": chunks_before,
        "chunks_after_dedup": chunks_after_dedup,
        "chunks_final": fmt_stats["chunks_used"],
    }

    return context_xml, metrics
