"""
CrossEncoder reranker — scores (query, passage) pairs and returns the top-k.

Retrieves a wide candidate set upstream (k=20) then uses a cross-encoder to
select the k most relevant passages, discarding noise before synthesis.

Why this wins over similarity-only ordering:
- Bi-encoder embeddings rank by approximate semantic direction; they can miss
  key passages whose surface phrasing diverges from the query.
- A cross-encoder sees the query AND passage together, capturing fine-grained
  relevance that bi-encoders can't — at the cost of higher per-query compute.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~67 MB, runs on CPU, ~30ms per batch)

Falls back to embedding-score ordering if the model isn't available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None  # module-level singleton — loaded once per process


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(
    question: str,
    chunks: "list[RetrievedChunk]",
    k: int = 8,
) -> "list[tuple[float, RetrievedChunk]]":
    """
    Score every chunk against the question and return the top-k as
    (rerank_score, chunk) tuples, sorted descending.

    Falls back to (similarity, chunk) ordering if CrossEncoder unavailable.
    """
    if not chunks:
        return []

    k = min(k, len(chunks))

    try:
        model = _get_model()
        # CrossEncoder expects short strings; truncate at 512 chars to stay fast
        pairs = [(question, c.text[:512]) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        scored = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
        return [(float(s), c) for s, c in scored[:k]]
    except Exception as exc:
        logging.warning("CrossEncoder unavailable (%s); using similarity ordering.", exc)
        scored = sorted(chunks, key=lambda c: c.similarity, reverse=True)
        return [(c.similarity, c) for c in scored[:k]]
