"""
Step 02 — Observability: trace data model and JSONL store.

Every RAG query produces a QueryTrace capturing:
  - What was asked and answered
  - What was retrieved: rank, source, similarity score, text preview
  - What was sent to the LLM: context size, token count
  - Cost estimate and latency per phase

Storage: append-only JSONL file — one JSON object per line, easy to grep and parse.
"""

import json
import uuid
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ChunkTrace:
    rank: int
    source: str
    department: str
    similarity: float
    char_count: int
    text_preview: str    # first 200 chars of the chunk


@dataclass
class RetrievalSpan:
    duration_ms: float
    chunks: list[ChunkTrace]

    @property
    def top_source(self) -> str:
        return self.chunks[0].source if self.chunks else ""

    @property
    def unique_sources(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen


@dataclass
class GenerationSpan:
    duration_ms: float
    provider: str
    model: str
    context_chars: int
    context_chunk_count: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class QueryTrace:
    trace_id: str
    timestamp: str
    step: str
    query: str
    answer: str
    retrieval: RetrievalSpan
    generation: GenerationSpan
    total_latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def new_trace_id() -> str:
    """Short 8-char hex ID — readable in CLI output."""
    return str(uuid.uuid4())[:8]


# ── Cost tables ───────────────────────────────────────────────────────────────
# (input $/1M tokens, output $/1M tokens)
COST_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini":    (0.075, 0.300),   # gemini flash variants
    "anthropic": (0.080, 0.400),   # claude haiku
}


def estimate_cost(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = COST_PER_MILLION.get(provider, (0.0, 0.0))
    return (
        (prompt_tokens / 1_000_000) * input_price
        + (completion_tokens / 1_000_000) * output_price
    )


# ── Trace store ───────────────────────────────────────────────────────────────

class TraceStore:
    """Append-only JSONL trace store. One JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: QueryTrace) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(trace.as_dict()) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def count(self) -> int:
        return len(self.read_all())

    def get(self, trace_id: str) -> dict | None:
        return next((t for t in self.read_all() if t["trace_id"] == trace_id), None)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
