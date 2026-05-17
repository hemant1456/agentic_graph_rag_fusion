"""
Step 05 — CSV Tool pipeline.

Upgrades over Step 04:
  - Detects aggregate questions (total ARR, Q3 revenue, headcount, etc.)
  - Runs exact Pandas queries and injects the result into LLM context
  - Fixes all Tier 2 questions that dense retrieval cannot answer

Reuses: Step 04 ChromaDB index and dense retrieval.
"""

import time
from pathlib import Path

from step_01_baseline_rag.implementation.generate import generate_answer
from step_01_baseline_rag.implementation.pipeline import RAGResult
from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context
from step_04_chunking.implementation.ingestor import embed_query, get_chroma_collection
from step_05_tools.implementation.csv_tool import detect_intent, run_query

STEP04_DB = Path(__file__).parent.parent.parent / "step_04_chunking" / "results" / "chroma_db"


class Step05ToolsRAG:
    """
    Retrieval pipeline that augments dense search with structured CSV tool calls.

    Usage:
        rag = Step05ToolsRAG(k=10).build()
        result = rag.query("What is the total ARR across all customers?")
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.collection = None

    def build(self) -> "Step05ToolsRAG":
        self.collection = get_chroma_collection(STEP04_DB)
        print(f"Loaded step04 index: {self.collection.count()} chunks")
        return self

    def query(self, question: str) -> RAGResult:
        if self.collection is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        qvec = embed_query(question)
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = [
            RetrievedChunk(
                text=doc,
                source=str(meta.get("source", "")),
                department=str(meta.get("department", "")),
                format=str(meta.get("format", "")),
                chunk_index=int(str(meta.get("chunk_index") or 0)),
                distance=float(dist),
            )
            for doc, meta, dist in zip(
                (res["documents"] or [[]])[0],
                (res["metadatas"] or [[]])[0],
                (res["distances"] or [[]])[0],
            )
        ]
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
