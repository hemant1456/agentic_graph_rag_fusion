"""
Step 04 RAG pipeline — format-aware chunking with aggregate summaries.

Same interface as BaselineRAG (.build() / .query()) but backed by
the step04 ChromaDB collection built from SmartChunks.

Key improvement over baseline:
  - CSV files get an "aggregate" chunk with pre-computed sums, breakdowns,
    and date-period aggregations → aggregation queries answered from 1 chunk
  - Markdown files use section-aware chunking (better precision)
  - Text files use structure-aware chunking with section detection
  - k=10 by default (up from 5) to improve recall; aggregate chunks are
    compact enough that larger k doesn't balloon context size
"""

import time
from pathlib import Path

import chromadb

from step_01_baseline_rag.implementation.generate import generate_answer
from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context
from step_04_chunking.implementation.ingestor import (
    build_index,
    embed_query,
    get_chroma_collection,
)

CORPUS_PATH = Path(__file__).parent.parent.parent / "step_00_dataset" / "company_data"
DB_PATH = Path(__file__).parent.parent / "results" / "chroma_db"


def _retrieve(
    query: str,
    collection: chromadb.Collection,
    k: int = 10,
) -> list[RetrievedChunk]:
    """Embed query → cosine search → return top-k chunks as RetrievedChunk objects."""
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append(RetrievedChunk(
            text=doc,
            source=meta.get("source", ""),
            department=meta.get("department", ""),
            format=meta.get("format", ""),
            chunk_index=meta.get("chunk_index", 0),
            distance=dist,
        ))

    return chunks


class Step04RAG:
    """
    RAG pipeline using format-aware SmartChunks.

    Usage:
        rag = Step04RAG(k=10).build()
        result = rag.query("What is the total ARR?")
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.collection: chromadb.Collection | None = None

    def build(self, reset: bool = False) -> "Step04RAG":
        """Build or load the step04 ChromaDB index."""
        if DB_PATH.exists() and not reset:
            self.collection = get_chroma_collection(DB_PATH)
            if self.collection.count() > 0:
                print(f"Loaded existing step04 index: {self.collection.count()} chunks")
                return self
        self.collection = build_index(CORPUS_PATH, DB_PATH, reset=reset)
        return self

    def query(self, question: str) -> RAGResult:
        if self.collection is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = _retrieve(question, self.collection, k=self.k)
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
