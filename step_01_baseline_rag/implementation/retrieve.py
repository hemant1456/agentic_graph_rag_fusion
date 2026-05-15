"""
Retrieval layer for Step 01 — Baseline Vector RAG.

Straightforward cosine similarity search in ChromaDB.
No reranking, no filtering, no hybrid search — just top-k nearest neighbors.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from .ingest import embed_query, get_chroma_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    department: str
    format: str
    chunk_index: int
    distance: float   # cosine distance: 0 = identical, 2 = opposite

    @property
    def similarity(self) -> float:
        """Convert cosine distance to similarity score [0, 1]."""
        return 1.0 - (self.distance / 2.0)


def retrieve(
    query: str,
    collection: chromadb.Collection,
    k: int = 5,
) -> list[RetrievedChunk]:
    """Embed query → cosine search → return top-k chunks."""
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


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into the context block sent to the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk.source} | dept: {chunk.department} | "
            f"similarity: {chunk.similarity:.2f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)
