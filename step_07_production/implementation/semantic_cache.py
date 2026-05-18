from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None  # lazy-loaded singleton


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    answer: str
    provider: str
    ce_metrics: dict
    timestamp: float
    hits: int = 0
    # Critic verdict carried with the cached answer so re-scoring confidence
    # on a cache hit uses the same signals as the original miss path.
    critic_approved: bool | None = None
    critic_notes: str = ""


class SemanticCache:
    def __init__(self, threshold: float = 0.92, max_size: int = 200):
        self.threshold = threshold
        self.max_size = max_size
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, question: str) -> Optional[CacheEntry]:
        emb = _get_model().encode(question, normalize_embeddings=True)
        with self._lock:
            best_score, best_entry = 0.0, None
            for entry in self._entries:
                s = self._cosine_sim(emb, entry.embedding)
                if s > best_score:
                    best_score, best_entry = s, entry
            if best_score >= self.threshold and best_entry is not None:
                best_entry.hits += 1
                self._total_hits += 1
                return best_entry
            self._total_misses += 1
            return None

    def put(
        self,
        question: str,
        answer: str,
        provider: str,
        ce_metrics: dict,
        *,
        critic_approved: bool | None = None,
        critic_notes: str = "",
    ) -> None:
        emb = _get_model().encode(question, normalize_embeddings=True)
        entry = CacheEntry(
            query=question, embedding=emb, answer=answer,
            provider=provider, ce_metrics=ce_metrics, timestamp=time.time(),
            critic_approved=critic_approved, critic_notes=critic_notes,
        )
        with self._lock:
            if len(self._entries) >= self.max_size:
                # Evict LRU: sort by (hits, timestamp), remove the least-used oldest
                self._entries.sort(key=lambda e: (e.hits, e.timestamp))
                self._entries.pop(0)
            self._entries.append(entry)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._entries),
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate": round(self._total_hits / max(1, self._total_hits + self._total_misses), 3),
            }
