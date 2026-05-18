"""
Step 04 — Knowledge Graph RAG pipeline.

Adds over Step 03:
  - Entity knowledge graph built from structured CSVs (org chart, API deps, etc.)
  - Multi-hop graph traversal appended to the retrieval context

Retains from Step 03: BM25 + dense RRF retrieval, CSV tool calling — both
inherited automatically through `build_context_sections` super() chain.
"""

from pathlib import Path

from step_01_baseline_rag.implementation.retrieve import RetrievedChunk
from step_03_hybrid_retrieval.implementation.pipeline import Step03HybridRAG

CORPUS_PATH = Path(__file__).parent.parent.parent / "dataset" / "company_data"
GRAPH_PATH  = Path(__file__).parent.parent / "results" / "graph.json"


class Step04RAG(Step03HybridRAG):
    """Hybrid retrieval + CSV tool + knowledge-graph multi-hop traversal.

    Usage:
        rag = Step04RAG(k=10).build()
        result = rag.query("Who does Aisha Johnson report to?")
    """

    def __init__(self, k: int = 10) -> None:
        super().__init__(k=k)
        self.graph = None

    def build(self, reset: bool = False, reset_graph: bool = False) -> "Step04RAG":
        super().build(reset=reset)
        if reset_graph and GRAPH_PATH.exists():
            GRAPH_PATH.unlink()
        from step_04_knowledge_graph.implementation.graph_store import load_or_build
        self.graph = load_or_build(CORPUS_PATH, GRAPH_PATH)
        return self

    def build_context_sections(
        self, chunks: list[RetrievedChunk], question: str
    ) -> dict[str, str]:
        sections = super().build_context_sections(chunks, question)
        if self.graph is None:
            raise RuntimeError("Call .build() before .query()")
        from step_04_knowledge_graph.implementation.graph_query import build_graph_context
        graph_ctx = build_graph_context(question, [c.text for c in chunks], self.graph)
        if graph_ctx:
            sections["graph"] = graph_ctx
        return sections
