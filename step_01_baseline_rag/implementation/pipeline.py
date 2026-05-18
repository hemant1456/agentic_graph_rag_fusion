import time
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .generate import generate_answer
from .ingest import build_index, get_chroma_collection
from .retrieve import RetrievedChunk, format_context, retrieve

CORPUS_PATH = Path(__file__).parent.parent.parent / "dataset" / "company_data"
DB_PATH = Path(__file__).parent.parent.parent / "chroma_db"


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
    # Optional faithfulness verdict from steps that run a critic (05+).
    # None means no critic was consulted — the baseline RAG leaves this empty.
    critic_approved: bool | None = None
    critic_notes: str = ""

    @property
    def total_latency_ms(self) -> float:
        return self.retrieval_latency_ms + self.generation_latency_ms

    def sources(self) -> list[str]:
        return [c.source for c in self.retrieved_chunks]


class BaselineRAG:
    """
    Baseline RAG pipeline with format-aware chunking:
      load → format-aware chunk (markdown by H1/H2, CSV by row, etc.)
      → embed → top-k cosine search → format context → LLM → answer

    Design choices:
    - k=10 retrieved chunks, no reranking
    - Format-aware SmartChunks (markdown sections, CSV rows, paragraph prose)
    - No query preprocessing or expansion
    - No metadata filtering
    - No document-level deduplication

    Extension points for subclasses (steps 02–04 use these, step 05+ replace
    `query()` entirely with their multi-agent orchestrator):

      - `retrieve_chunks(question, k)`: override to change retrieval strategy
        (step 03 swaps in BM25+dense+RRF).
      - `build_context_sections(chunks, question)`: override to add new context
        sources (step 02 adds "csv", step 04 adds "graph"). Returns a dict
        keyed by section name; CONTEXT_PRIORITY defines the assembly order.

    The base `query()` is a template method — subclasses generally don't
    override it, they just contribute new sections or a new retriever.
    """

    # Higher-priority sections come first in the assembled context. The LLM
    # is more likely to attend to the top of the context window, so we put
    # authoritative tool output (csv) above structural (graph) above textual
    # (vector). Subclasses can extend this tuple if they introduce new keys.
    CONTEXT_PRIORITY: tuple[str, ...] = ("csv", "graph", "vector")

    def __init__(self, k: int = 10) -> None:
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

    def retrieve_chunks(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        """Default dense top-k cosine retrieval. Steps 03+ override this."""
        if self.collection is None:
            raise RuntimeError("Call .build() before .query()")
        return retrieve(question, self.collection, k=k or self.k)

    def build_context_sections(
        self, chunks: list[RetrievedChunk], question: str
    ) -> dict[str, str]:
        """Return named, ordered context sections. Subclasses contribute new
        keys (step 02 adds "csv", step 04 adds "graph") via super() + augment.
        `question` is part of the contract for subclasses but unused in the
        base — see Step02ToolsRAG / Step04RAG for usage.
        """
        del question
        return {"vector": format_context(chunks)}

    def _assemble_context(self, sections: dict[str, str]) -> str:
        return "\n\n".join(sections[k] for k in self.CONTEXT_PRIORITY if sections.get(k))

    def query(self, question: str) -> RAGResult:
        if self.collection is None:
            raise RuntimeError("Call .build() before .query()")

        t0 = time.perf_counter()
        chunks = self.retrieve_chunks(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        sections = self.build_context_sections(chunks, question)
        context = self._assemble_context(sections)

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
