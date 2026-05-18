"""
Step 03 — Hybrid Retrieval pipeline (BM25 + dense vector, fused via RRF).

Adds over Step 02:
  - BM25 keyword retrieval alongside dense vector retrieval
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists
  - Fixes keyword-exact questions (version strings, vendor names, exact dates)

Retains from Step 02: CSV tool calling for aggregate questions. Step 02's
`build_context_sections` adds the "csv" section automatically through the
inheritance chain.
"""

from step_01_baseline_rag.implementation.ingest import embed_query
from step_01_baseline_rag.implementation.retrieve import RetrievedChunk
from step_02_tools.implementation.pipeline import Step02ToolsRAG
from step_03_hybrid_retrieval.implementation.bm25_retriever import BM25Index


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


class Step03HybridRAG(Step02ToolsRAG):
    """BM25 + dense → RRF fusion + CSV tool calling.

    Usage:
        rag = Step03HybridRAG(k=10).build()
        result = rag.query("What is Vertexia's annual AWS spend?")
    """

    def __init__(self, k: int = 10) -> None:
        super().__init__(k=k)
        self.bm25: BM25Index | None = None

    def build(self, reset: bool = False) -> "Step03HybridRAG":
        super().build(reset=reset)
        if self.collection is None:
            raise RuntimeError("BaselineRAG.build() did not initialize the collection")
        self.bm25 = BM25Index().build(self.collection)
        return self

    def retrieve_chunks(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
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

    # `retrieve()` kept as an alias for older callers (step_05/06 use it).
    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.retrieve_chunks(question, k=k)
