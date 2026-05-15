"""
End-to-end Baseline RAG pipeline.

RAGResult captures everything needed for evaluation and later observability work:
query, retrieved chunks (with metadata), context sent to LLM, answer, latency.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .generate import generate_answer
from .ingest import build_index, get_chroma_collection
from .retrieve import RetrievedChunk, format_context, retrieve

CORPUS_PATH = Path(__file__).parent.parent.parent / "step_00_dataset" / "company_data"
DB_PATH = Path(__file__).parent.parent / "results" / "chroma_db"


@dataclass
class RAGResult:
    question: str
    answer: str
    provider: str             # which LLM answered
    retrieved_chunks: list[RetrievedChunk]
    context_sent: str         # exact context block sent to LLM
    context_chars: int        # size of context window used
    retrieval_latency_ms: float
    generation_latency_ms: float

    @property
    def total_latency_ms(self) -> float:
        return self.retrieval_latency_ms + self.generation_latency_ms

    def sources(self) -> list[str]:
        return [c.source for c in self.retrieved_chunks]


class BaselineRAG:
    """
    Naive RAG pipeline:
      query → embed → top-5 cosine search → format context → LLM → answer

    Design choices (intentionally simple):
    - k=5 retrieved chunks, no reranking
    - No query preprocessing or expansion
    - No metadata filtering
    - No document-level deduplication
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.collection: chromadb.Collection | None = None

    def build(self, reset: bool = False) -> "BaselineRAG":
        """Build or load the ChromaDB index."""
        if DB_PATH.exists() and not reset:
            self.collection = get_chroma_collection(DB_PATH)
            if self.collection.count() > 0:
                print(f"Loaded existing index: {self.collection.count()} chunks")
                return self
        self.collection = build_index(CORPUS_PATH, DB_PATH, reset=reset)
        return self

    def query(self, question: str) -> RAGResult:
        if self.collection is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = retrieve(question, self.collection, k=self.k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        context = format_context(chunks)

        t1 = time.perf_counter()
        answer, provider = generate_answer(context, question)
        generation_ms = (time.perf_counter() - t1) * 1000

        return RAGResult(
            question=question,
            answer=answer,
            provider=provider,
            retrieved_chunks=chunks,
            context_sent=context,
            context_chars=len(context),
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )
