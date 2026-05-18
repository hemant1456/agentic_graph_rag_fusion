"""
BM25 keyword retriever for Step 03.

Loads all chunks from a ChromaDB collection at build time, builds a BM25Okapi
index over them, and exposes a search method that returns ranked (text, meta, score)
tuples — parallel to ChromaDB's dense retrieval output.

The built index is pickled to disk next to the ChromaDB so warm starts skip the
tokenization + BM25Okapi construction cost. The cache key includes the
collection's document count so it auto-invalidates when chunks are added or
removed.
"""

import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

_STRIP = re.compile(r"[^\w\-./]")
_INDEX_VERSION = 1  # bump on any tokenizer / payload change


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation (keep hyphens/dots for IDs like RFC-001, v2.1), split."""
    return [t for t in _STRIP.sub(" ", text.lower()).split() if t]


_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CACHE_DIR = _PROJECT_ROOT / "chroma_db"
_CACHE_PATH = _CACHE_DIR / "bm25_index.pkl"


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._docs: list[str] = []
        self._metas: list[dict] = []

    def build(self, collection: chromadb.Collection) -> "BM25Index":
        doc_count = collection.count()
        cached = self._try_load_cache(doc_count)
        if cached is not None:
            self._bm25, self._docs, self._metas = cached
            print(f"BM25 index loaded from cache: {len(self._docs)} documents")
            return self

        result = collection.get(include=["documents", "metadatas"])
        self._docs = result["documents"] or []
        self._metas = [dict(m) for m in (result["metadatas"] or [])]
        tokenized = [_tokenize(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built: {len(self._docs)} documents")
        self._save_cache(doc_count)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, dict, float]]:
        """Return top-k (text, meta, bm25_score) sorted by score descending."""
        if self._bm25 is None:
            raise RuntimeError("Call .build() first")
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._docs[i], self._metas[i], float(scores[i])) for i in top_indices]

    # ── persistence ──────────────────────────────────────────────────────────

    def _try_load_cache(self, doc_count: int):
        if not _CACHE_PATH.exists():
            return None
        try:
            with _CACHE_PATH.open("rb") as f:
                blob = pickle.load(f)
        except Exception:
            return None
        if not isinstance(blob, dict):
            return None
        if blob.get("version") != _INDEX_VERSION:
            return None
        if blob.get("doc_count") != doc_count:
            # Chunks added/removed; rebuild.
            return None
        return blob["bm25"], blob["docs"], blob["metas"]

    def _save_cache(self, doc_count: int) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = _CACHE_PATH.with_suffix(".pkl.tmp")
            with tmp.open("wb") as f:
                pickle.dump(
                    {
                        "version": _INDEX_VERSION,
                        "doc_count": doc_count,
                        "bm25": self._bm25,
                        "docs": self._docs,
                        "metas": self._metas,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp.replace(_CACHE_PATH)
        except Exception as e:
            print(f"BM25 cache save failed (non-fatal): {e}")
