"""
Step 07 RAG pipeline — RAG Fusion (BM25 + dense) with structured CSV query tool.

Upgrades over Step 06:
  - BM25 keyword retrieval alongside dense vector retrieval
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists
  - Structured CSV query tool for exact arithmetic (fixes Q20 revenue sum)

Reuses: Step 04 ChromaDB index, Step 05 knowledge graph, Step 06 graph context.
"""

import time
from pathlib import Path

import chromadb

from step_01_baseline_rag.implementation.generate import generate_answer
from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context
from step_04_chunking.implementation.ingestor import embed_query, get_chroma_collection
from step_06_graph_rag.implementation.graph_query import build_graph_context
from step_07_rag_fusion.implementation.bm25_retriever import BM25Index
from step_07_rag_fusion.implementation.csv_tool import detect_intent, run_query

CORPUS_PATH = Path(__file__).parent.parent.parent / "step_00_dataset" / "company_data"
STEP04_DB   = Path(__file__).parent.parent.parent / "step_04_chunking" / "results" / "chroma_db"
GRAPH_PATH  = Path(__file__).parent.parent.parent / "step_05_knowledge_graph" / "results" / "graph.json"


def _rrf_fuse(
    dense: list[tuple[str, dict, float]],
    bm25: list[tuple[str, dict, float]],
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[tuple[str, dict]]:
    """Reciprocal Rank Fusion over dense + BM25 results. Returns top-k (text, meta)."""
    scores: dict[str, tuple[dict, float]] = {}
    for rank, (text, meta, _) in enumerate(dense):
        score = 1.0 / (k_rrf + rank + 1)
        if text in scores:
            scores[text] = (meta, scores[text][1] + score)
        else:
            scores[text] = (meta, score)
    for rank, (text, meta, _) in enumerate(bm25):
        score = 1.0 / (k_rrf + rank + 1)
        if text in scores:
            scores[text] = (meta, scores[text][1] + score)
        else:
            scores[text] = (meta, score)
    sorted_items = sorted(scores.items(), key=lambda x: -x[1][1])
    return [(text, meta) for text, (meta, _) in sorted_items[:top_k]]


class Step07RAG:
    """
    RAG Fusion pipeline: BM25 + dense vector retrieval merged via RRF,
    plus Step 06 graph context and a structured CSV query tool.

    Usage:
        rag = Step07RAG(k=10).build()
        result = rag.query("What was Q3 2023 total revenue?")
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.collection: chromadb.Collection | None = None
        self.bm25: BM25Index | None = None
        self.graph = None

    def build(self) -> "Step07RAG":
        self.collection = get_chroma_collection(STEP04_DB)
        print(f"Loaded step04 index: {self.collection.count()} chunks")
        self.bm25 = BM25Index().build(self.collection)
        from step_05_knowledge_graph.implementation.graph_store import load_or_build
        self.graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        """Public method so Step 08 agent can call it directly."""
        if self.collection is None or self.bm25 is None:
            raise RuntimeError("Call .build() first")
        k = k or self.k

        qvec = embed_query(question)
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=min(k * 2, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense_results: list[tuple[str, dict, float]] = [
            (doc, dict(meta), dist)
            for doc, meta, dist in zip(
                (res["documents"] or [[]])[0],
                (res["metadatas"] or [[]])[0],
                (res["distances"] or [[]])[0],
            )
        ]

        bm25_results: list[tuple[str, dict, float]] = [
            (text, dict(meta), score) for text, meta, score in self.bm25.search(question, k=k * 2)
        ]

        fused = _rrf_fuse(dense_results, bm25_results, top_k=k)
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
        if self.collection is None or self.bm25 is None or self.graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        vector_ctx = format_context(chunks)
        graph_ctx = build_graph_context(question, [c.text for c in chunks], self.graph)

        csv_intent = detect_intent(question)
        csv_ctx = run_query(csv_intent) if csv_intent else ""

        # Structured facts first — LLM anchors on exact data before reading fuzzy docs
        parts = []
        if csv_ctx:
            parts.append(csv_ctx)
        if graph_ctx:
            parts.append(graph_ctx)
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
