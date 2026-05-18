"""
Step 06 — Hybrid Retrieval pipeline (BM25 + dense vector, fused via RRF).

Adds over Step 05:
  - BM25 keyword retrieval alongside dense vector retrieval
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists
  - Fixes keyword-exact questions (version strings, vendor names, exact dates)

Retains from Step 05: CSV tool calling for aggregate questions.
"""

import time
from pathlib import Path

import chromadb

from step_01_baseline_rag.implementation.generate import generate_answer
from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context
from step_01_baseline_rag.implementation.ingest import embed_query, get_chroma_collection
from step_02_tools.implementation.csv_tool import detect_intent, run_query
from step_03_hybrid_retrieval.implementation.bm25_retriever import BM25Index

STEP02_DB = Path(__file__).parent.parent.parent / "chroma_db"


def _rrf_fuse(
    dense: list[tuple[str, dict, float]],
    bm25: list[tuple[str, dict, float]],
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[tuple[str, dict]]:
    scores: dict[str, tuple[dict, float]] = {}
    for rank, (text, meta, _) in enumerate(dense):
        score = 1.0 / (k_rrf + rank + 1)
        scores[text] = (meta, scores.get(text, (meta, 0.0))[1] + score)
    for rank, (text, meta, _) in enumerate(bm25):
        score = 1.0 / (k_rrf + rank + 1)
        scores[text] = (meta, scores.get(text, (meta, 0.0))[1] + score)
    return [(text, meta) for text, (meta, _) in sorted(scores.items(), key=lambda x: -x[1][1])[:top_k]]


class Step03HybridRAG:
    """
    Hybrid retrieval: BM25 + dense vector merged via RRF, plus CSV tool calling.

    Usage:
        rag = Step03HybridRAG(k=10).build()
        result = rag.query("What is Vertexia's annual AWS spend?")
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.collection: chromadb.Collection | None = None
        self.bm25: BM25Index | None = None

    def build(self) -> "Step03HybridRAG":
        self.collection = get_chroma_collection(STEP02_DB)
        print(f"Loaded step04 index: {self.collection.count()} chunks")
        self.bm25 = BM25Index().build(self.collection)
        return self

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        if self.collection is None or self.bm25 is None:
            raise RuntimeError("Call .build() first")
        k = k or self.k
        qvec = embed_query(question)
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=min(k * 2, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense = [
            (doc, dict(meta), dist)
            for doc, meta, dist in zip(
                (res["documents"] or [[]])[0],
                (res["metadatas"] or [[]])[0],
                (res["distances"] or [[]])[0],
            )
        ]
        bm25_hits = [(t, dict(m), s) for t, m, s in self.bm25.search(question, k=k * 2)]
        fused = _rrf_fuse(dense, bm25_hits, top_k=k)
        return [
            RetrievedChunk(
                text=text,
                source=str(meta.get("source", "")),
                department=str(meta.get("department", "")),
                format=str(meta.get("format", "")),
                chunk_index=int(str(meta.get("chunk_index") or 0)),
                distance=0.0,
            )
            for text, meta in fused
        ]

    def query(self, question: str) -> RAGResult:
        if self.collection is None or self.bm25 is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        vector_ctx = format_context(chunks)
        csv_intent = detect_intent(question)
        csv_ctx = run_query(csv_intent) if csv_intent else ""

        parts = []
        if csv_ctx:
            parts.append(csv_ctx)
        parts.append(vector_ctx)
        context = "\n\n".join(parts)

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
