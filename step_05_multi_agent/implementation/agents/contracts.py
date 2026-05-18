from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryAnalysis:
    """Output of the QueryAnalystAgent.

    Note: earlier revisions also exposed needs_vector / needs_graph / needs_csv
    routing flags. They were always set but never consulted — the orchestrator
    runs vector, graph, and CSV branches unconditionally because empirically
    every branch is cheap and aggressive routing was costing recall. The flags
    were removed to stop the prompt and parser pretending to do work that the
    orchestrator ignores.
    """
    query_type: str          # simple_lookup | aggregation | multi_hop | comparative | graph | mixed
    sub_questions: list[str] # decomposed sub-questions for compound/multi-hop queries
    primary_entities: list[str]  # named entities mentioned in the question


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


@dataclass
class AgentTrace:
    """One record per agent invocation — collected by the Orchestrator."""
    agent_id: str
    input_summary: str
    output_summary: str
    latency_ms: float
    status: str = "ok"       # ok | error | skipped


@dataclass
class OrchestratorResult:
    """Aggregated output of the multi-agent orchestrator. Carries everything the
    pipeline needs to construct a RAGResult plus the critic verdict so callers
    (e.g. step_07 confidence scoring) can route on it."""
    answer: str
    provider: str
    traces: list[AgentTrace] = field(default_factory=list)
    context_text: str = ""
    critic_approved: bool = True
    critic_notes: str = ""
