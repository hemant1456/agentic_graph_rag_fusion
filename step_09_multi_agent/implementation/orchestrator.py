from __future__ import annotations

import time
from typing import TYPE_CHECKING

import networkx as nx

from step_09_multi_agent.implementation.agents import (
    critic,
    graph_navigator,
    query_analyst,
    retrieval_specialist,
    structured_data,
    synthesis,
)
from step_09_multi_agent.implementation.agents.contracts import AgentTrace, QueryAnalysis

if TYPE_CHECKING:
    from step_07_rag_fusion.implementation.pipeline import Step07RAG


def run(
    question: str,
    retriever: "Step07RAG",
    graph: nx.DiGraph,
) -> tuple[str, str, list[AgentTrace]]:
    """
    Run the full multi-agent pipeline.

    Returns:
        answer   (str)          — final answer text
        provider (str)          — provider tag from synthesis
        traces   (list[AgentTrace]) — per-agent observability records
    """
    traces: list[AgentTrace] = []

    t0 = time.perf_counter()
    analysis: QueryAnalysis = query_analyst.analyze(question)
    traces.append(AgentTrace(
        agent_id="query_analyst",
        input_summary=question[:120],
        output_summary=(
            f"type={analysis.query_type} vec={analysis.needs_vector} "
            f"graph={analysis.needs_graph} csv={analysis.needs_csv} "
            f"sub_qs={len(analysis.sub_questions)}"
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

    # Sub-question retrieval for compound/multi-hop queries
    for i, sub_q in enumerate(analysis.sub_questions[:4]):
        t0 = time.perf_counter()
        sub_ret = retrieval_specialist.retrieve(sub_q, retriever, k=5)
        if sub_ret.context and sub_ret.context != "[No relevant passages found]":
            contexts[f"Sub-Q{i+1} ({sub_q[:40]}…)"] = sub_ret.context
        traces.append(AgentTrace(
            agent_id=f"retrieval_specialist/sub_q{i+1}",
            input_summary=sub_q[:80],
            output_summary=f"{len(sub_ret.chunks)} chunks",
            latency_ms=(time.perf_counter() - t0) * 1000,
        ))

    if analysis.needs_graph:
        t0 = time.perf_counter()
        graph_res = graph_navigator.navigate(question, analysis.primary_entities, graph)
        if graph_res.success:
            contexts["Graph"] = graph_res.context
        traces.append(AgentTrace(
            agent_id="graph_navigator",
            input_summary=f"entities={analysis.primary_entities[:4]}",
            output_summary=f"success={graph_res.success}, {len(graph_res.context)} chars",
            latency_ms=(time.perf_counter() - t0) * 1000,
            status="ok" if graph_res.success else "skipped",
        ))

    if analysis.needs_csv:
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

    return critic_res.answer, synth.provider, traces
