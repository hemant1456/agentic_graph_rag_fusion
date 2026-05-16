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
