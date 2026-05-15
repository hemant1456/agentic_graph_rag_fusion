"""
Step 05 RAG pipeline — vector retrieval augmented by knowledge graph traversal.

Build: reuses the Step 04 ChromaDB index (already embedded).
Query:
  1. Vector retrieve top-k chunks (Gemini embeddings, same as Step 04)
  2. Extract entity mentions from question + retrieved chunks
  3. Traverse the knowledge graph for each entity (1–2 hops)
  4. Append graph context to vector context → LLM generation
"""

import time
from pathlib import Path

import chromadb

from step_01_baseline_rag.implementation.generate import generate_answer
from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context
from step_04_chunking.implementation.ingestor import embed_query, get_chroma_collection

CORPUS_PATH = Path(__file__).parent.parent.parent / "step_00_dataset" / "company_data"
STEP04_DB   = Path(__file__).parent.parent.parent / "step_04_chunking" / "results" / "chroma_db"
GRAPH_PATH  = Path(__file__).parent.parent / "results" / "graph.json"


def _retrieve(query: str, collection: chromadb.Collection, k: int) -> list[RetrievedChunk]:
    qvec = embed_query(query)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks: list[RetrievedChunk] = []
    for doc, meta, dist in zip(
        (res["documents"] or [[]])[0],
        (res["metadatas"] or [[]])[0],
        (res["distances"] or [[]])[0],
    ):
        chunks.append(RetrievedChunk(
            text=doc,
            source=str(meta.get("source", "")),
            department=str(meta.get("department", "")),
            format=str(meta.get("format", "")),
            chunk_index=int(str(meta.get("chunk_index") or 0)),
            distance=dist,
        ))
    return chunks


class Step05RAG:
    """
    RAG pipeline with knowledge graph augmentation.

    Usage:
        rag = Step05RAG(k=10).build()
        result = rag.query("Who is Adrian Blake's manager?")
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.collection: chromadb.Collection | None = None
        self.graph = None

    def build(self, reset_graph: bool = False) -> "Step05RAG":
        self.collection = get_chroma_collection(STEP04_DB)
        print(f"Loaded step04 index: {self.collection.count()} chunks")
        if reset_graph and GRAPH_PATH.exists():
            GRAPH_PATH.unlink()
        from step_05_knowledge_graph.implementation.graph_store import load_or_build
        self.graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def query(self, question: str) -> RAGResult:
        if self.collection is None or self.graph is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = _retrieve(question, self.collection, k=self.k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        vector_ctx = format_context(chunks)

        from step_05_knowledge_graph.implementation.query import get_graph_context
        graph_ctx = get_graph_context(question, [c.text for c in chunks], self.graph)
        context = vector_ctx + ("\n\n" + graph_ctx if graph_ctx else "")

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
