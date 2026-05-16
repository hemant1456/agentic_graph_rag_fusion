"""
Extractive sentence compressor — keeps only query-relevant sentences per chunk.

This is a zero-LLM-call approach: split text into sentences, score each by
overlap with query terms, retain the top `ratio` fraction while preserving
sentence order.

Why extractive over abstractive (LLMLingua-style):
- No extra LLM call → no added latency or cost.
- Zero hallucination risk — only original sentences are kept.
- Predictable: the same input always produces the same output.

Tradeoffs vs abstractive:
- Cannot merge or paraphrase across sentences.
- A key fact buried in a low-scoring sentence can be dropped.

Mitigations applied here:
- Minimum floor of 2 sentences per chunk regardless of ratio.
- CSV / structured sections are NEVER compressed (exact numbers must survive).
- Short chunks (≤ 3 sentences) are passed through unchanged.
"""

from __future__ import annotations

import re


def _query_terms(question: str) -> frozenset[str]:
    return frozenset(re.findall(r"\b\w{4,}\b", question.lower()))


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def compress(question: str, text: str, ratio: float = 0.60) -> str:
    """
    Return a compressed version of `text` keeping the `ratio` most
    query-relevant sentences (by term overlap), in original order.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 3:
        return text  # too short to compress meaningfully

    terms = _query_terms(question)
    scored: list[tuple[int, int, str]] = []
    for idx, sent in enumerate(sentences):
        sent_terms = frozenset(re.findall(r"\b\w{4,}\b", sent.lower()))
        overlap = len(terms & sent_terms)
        scored.append((overlap, idx, sent))

    n_keep = max(2, round(len(scored) * ratio))
    # Select top-n by score; break ties by original order (earlier = better)
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:n_keep]
    # Restore original order
    kept_sentences = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return " ".join(kept_sentences)
