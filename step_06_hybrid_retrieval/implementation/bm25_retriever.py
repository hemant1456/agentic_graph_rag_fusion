"""
BM25 keyword retriever for Step 07.

Loads all chunks from a ChromaDB collection at build time, builds a BM25Okapi
index over them, and exposes a search method that returns ranked (text, meta, score)
tuples — parallel to ChromaDB's dense retrieval output.
"""

import re

import chromadb
from rank_bm25 import BM25Okapi

_STRIP = re.compile(r"[^\w\-./]")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation (keep hyphens/dots for IDs like RFC-001, v2.1), split."""
    return [t for t in _STRIP.sub(" ", text.lower()).split() if t]


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._docs: list[str] = []
        self._metas: list[dict] = []

    def build(self, collection: chromadb.Collection) -> "BM25Index":
        result = collection.get(include=["documents", "metadatas"])
        self._docs = result["documents"] or []
        self._metas = [dict(m) for m in (result["metadatas"] or [])]
        tokenized = [_tokenize(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built: {len(self._docs)} documents")
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, dict, float]]:
        """Return top-k (text, meta, bm25_score) sorted by score descending."""
        if self._bm25 is None:
            raise RuntimeError("Call .build() first")
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._docs[i], self._metas[i], float(scores[i])) for i in top_indices]
