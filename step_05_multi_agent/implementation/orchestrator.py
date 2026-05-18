from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import networkx as nx

from step_05_multi_agent.implementation.agents import (
    critic,
    graph_navigator,
    query_analyst,
    retrieval_specialist,
    structured_data,
    synthesis,
)
from step_05_multi_agent.implementation.agents.contracts import (
    AgentTrace,
    OrchestratorResult,
    QueryAnalysis,
)

if TYPE_CHECKING:
    from step_03_hybrid_retrieval.implementation.pipeline import Step03HybridRAG


def run(
    question: str,
    retriever: "Step03HybridRAG",
    graph: nx.DiGraph,
) -> OrchestratorResult:
    """
    Run the full multi-agent pipeline.

    Returns OrchestratorResult — answer, provider, per-agent traces, the full
    context the LLM saw, and the critic verdict (approved + notes) so callers
    can route on it (e.g. step_07 confidence scoring).
    """
    traces: list[AgentTrace] = []

    t0 = time.perf_counter()
    analysis: QueryAnalysis = query_analyst.analyze(question)
    traces.append(AgentTrace(
        agent_id="query_analyst",
        input_summary=question[:120],
        output_summary=(
            f"type={analysis.query_type} sub_qs={len(analysis.sub_questions)} "
            f"entities={len(analysis.primary_entities)}"
        ),
        latency_ms=(time.perf_counter() - t0) * 1000,
    ))

    contexts: dict[str, str] = {}

    t0 = time.perf_counter()
    ret = retrieval_specialist.retrieve(question, retriever, k=10)
    contexts["Vector"] = ret.context
    traces.append(AgentTrace(
        agent_id="retrieval_specialist",
        input_summary=f"q={question[:80]} k=10",
        output_summary=f"{len(ret.chunks)} chunks, {len(ret.context)} chars",
        latency_ms=(time.perf_counter() - t0) * 1000,
    ))

    # Sub-question retrieval for compound/multi-hop queries.
    #
    # Sub-Qs are independent — each one is a separate BM25 + dense + RRF lookup
    # against the shared retriever. Running them in parallel is a 3-4x wall-clock
    # win on multi-hop questions. The retriever's underlying structures (Chroma
    # collection, prebuilt BM25Okapi index) are read-only at query time, so
    # concurrent .retrieve() calls are safe.
    sub_qs = analysis.sub_questions[:4]
    if sub_qs:
        def _run_sub(sub_q: str):
            t = time.perf_counter()
            r = retrieval_specialist.retrieve(sub_q, retriever, k=5)
            return r, (time.perf_counter() - t) * 1000

        with ThreadPoolExecutor(max_workers=len(sub_qs)) as pool:
            results = list(pool.map(_run_sub, sub_qs))

        for i, (sub_q, (sub_ret, lat_ms)) in enumerate(zip(sub_qs, results)):
            if sub_ret.context and sub_ret.context != "[No relevant passages found]":
                contexts[f"Sub-Q{i+1} ({sub_q[:40]}…)"] = sub_ret.context
            traces.append(AgentTrace(
                agent_id=f"retrieval_specialist/sub_q{i+1}",
                input_summary=sub_q[:80],
                output_summary=f"{len(sub_ret.chunks)} chunks",
                latency_ms=lat_ms,
            ))

    # Always run graph navigation — pass retrieved chunk texts as seeds so traversal
    # finds relevant nodes regardless of question type. Aggressive routing was
    # empirically costing recall, so every branch runs unconditionally.
    t0 = time.perf_counter()
    graph_seeds = [c.text for c in ret.chunks] if ret.chunks else analysis.primary_entities
    graph_res = graph_navigator.navigate(question, graph_seeds, graph)
    if graph_res.success:
        contexts["Graph"] = graph_res.context
    traces.append(AgentTrace(
        agent_id="graph_navigator",
        input_summary=f"seeds={len(graph_seeds)} chunks, entities={analysis.primary_entities[:4]}",
        output_summary=f"success={graph_res.success}, {len(graph_res.context)} chars",
        latency_ms=(time.perf_counter() - t0) * 1000,
        status="ok" if graph_res.success else "skipped",
    ))

    # Always run structured CSV query — unconditional detect_intent() → run_query().
    t0 = time.perf_counter()
    csv_res = structured_data.query(question)
    if csv_res.success:
        contexts["CSV"] = csv_res.data
    traces.append(AgentTrace(
        agent_id="structured_data",
        input_summary=question[:80],
        output_summary=f"intent={csv_res.intent_matched} success={csv_res.success}",
        latency_ms=(time.perf_counter() - t0) * 1000,
        status="ok" if csv_res.success else "error",
    ))

    t0 = time.perf_counter()
    synth = synthesis.synthesize(question, contexts, analysis.query_type)
    traces.append(AgentTrace(
        agent_id="synthesis",
        input_summary=f"{len(contexts)} context sections",
        output_summary=f"provider={synth.provider}, {len(synth.answer)} chars",
        latency_ms=(time.perf_counter() - t0) * 1000,
        status="ok" if synth.provider != "error" else "error",
    ))

    t0 = time.perf_counter()
    critic_res = critic.review(question, synth.answer, contexts)
    traces.append(AgentTrace(
        agent_id="critic",
        input_summary=f"answer={len(synth.answer)} chars",
        output_summary=f"approved={critic_res.approved} confidence={critic_res.confidence}",
        latency_ms=(time.perf_counter() - t0) * 1000,
    ))

    # Assemble the contexts the LLM actually saw, so callers (the pipeline + the
    # eval judge contract) can verify which facts were grounded vs hallucinated.
    context_text = "\n\n".join(f"[{label}]\n{body}" for label, body in contexts.items())

    return OrchestratorResult(
        answer=critic_res.answer,
        provider=synth.provider,
        traces=traces,
        context_text=context_text,
        critic_approved=critic_res.approved,
        critic_notes=critic_res.notes,
    )
