"""
Typed contracts for all Step 09 agents.

Every agent accepts and returns these dataclasses — no implicit coupling,
no string-to-string handoffs. If an agent changes its output shape, callers
get a type error at the boundary rather than a silent downstream failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Query classification ───────────────────────────────────────────────────────

@dataclass
class QueryAnalysis:
    """Output of the QueryAnalystAgent."""
    query_type: str          # simple_lookup | aggregation | multi_hop | comparative | graph | mixed
    needs_vector: bool       # free-text document search needed
    needs_graph: bool        # entity-relationship traversal needed
    needs_csv: bool          # exact numerical aggregation needed
    sub_questions: list[str] # decomposed sub-questions for compound/multi-hop queries
    primary_entities: list[str]  # named entities mentioned in the question


# ── Per-agent results ──────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Output of the RetrievalSpecialistAgent."""
    chunks: list              # list[RetrievedChunk] — typed at import time to avoid circular
    context: str              # formatted context string ready for synthesis
    strategy: str             # e.g. "bm25+dense+rrf"


@dataclass
class GraphResult:
    """Output of the GraphNavigatorAgent."""
    context: str
    entities_found: list[str]
    success: bool


@dataclass
class CSVResult:
    """Output of the StructuredDataAgent."""
    data: str
    success: bool
    intent_matched: str | None   # the matched intent key, or None


@dataclass
class SynthesisResult:
    """Output of the SynthesisAgent."""
    answer: str
    provider: str             # gateway:gemini, gemini-direct, etc.


@dataclass
class CriticResult:
    """Output of the CriticAgent."""
    approved: bool
    answer: str               # possibly revised answer
    confidence: str           # high | medium | low
    notes: str                # brief critic note (empty string if approved cleanly)


# ── Observability ─────────────────────────────────────────────────────────────

@dataclass
class AgentTrace:
    """One record per agent invocation — collected by the Orchestrator."""
    agent_id: str
    input_summary: str
    output_summary: str
    latency_ms: float
    status: str = "ok"       # ok | error | skipped
