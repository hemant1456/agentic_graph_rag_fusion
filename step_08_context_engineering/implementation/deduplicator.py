from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk


def _jaccard_6gram(a: str, b: str) -> float:
    def ng(s: str) -> set[str]:
        s = s.lower()
        return {s[i : i + 6] for i in range(max(0, len(s) - 5))}

    a_ng, b_ng = ng(a), ng(b)
    if not a_ng or not b_ng:
        return 0.0
    return len(a_ng & b_ng) / len(a_ng | b_ng)


def deduplicate(
    scored_chunks: "list[tuple[float, RetrievedChunk]]",
    threshold: float = 0.72,
) -> "list[tuple[float, RetrievedChunk]]":
    """
    Input:  list of (rerank_score, chunk) already sorted best-first.
    Output: same format, with near-duplicates removed.
            First occurrence (highest score) is always kept.
    """
    kept: list[tuple[float, "RetrievedChunk"]] = []
    for score, chunk in scored_chunks:
        is_dup = any(
            _jaccard_6gram(chunk.text, kept_chunk.text) > threshold
            for _, kept_chunk in kept
        )
        if not is_dup:
            kept.append((score, chunk))
    return kept
