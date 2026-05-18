from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    import networkx as nx
    from step_03_hybrid_retrieval.implementation.pipeline import Step03HybridRAG

from llm_gatewayV2.client import LLM

_GATEWAY_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")


@dataclass
class SliceConfig:
    """Everything a slice needs to describe itself and influence execution.

    Note: `force_csv` / `force_graph` were removed — graph and CSV branches are
    always run (the orchestrator-style "unconditional fan-out" pattern), so
    those flags had no effect. `owns_questions` was also removed: it was set
    on every slice but read nowhere, and the IDs referenced an obsolete 27Q
    golden set.
    """
    name: str                        # slug used in routing logs and eval output
    display_name: str                # human-readable label
    system_prompt: str               # domain-tuned system prompt sent to synthesis LLM
    keywords: list[str]              # routing vocabulary — matched against the query
    query_augmentation: str = ""     # extra terms appended to the retrieval query
    rerank_k: int = 8                # top-k for CrossEncoder reranking
    compress_ratio: float = 0.60     # extractive compression retention ratio
    floor_confidence: float = 0.0    # minimum confidence floor (general slice → 0.15)

    def can_handle(self, question: str) -> float:
        """Confidence [0..1] that this slice should handle the question.

        Lifted from the four slice files which all implemented the same
        keyword-density formula. Normalize by question length to avoid length
        bias. The `floor_confidence` lets the general slice stay competitive
        even when no keyword matches.
        """
        q = question.lower()
        hits = sum(1 for kw in self.keywords if kw in q)
        words = max(len(q.split()), 1)
        return max(min(hits / words * 4.0, 1.0), self.floor_confidence)


def run_with_config(
    question: str,
    config: SliceConfig,
    retriever: "Step03HybridRAG",
    graph: "nx.DiGraph",
) -> tuple[str, str, dict, str, list]:
    """
    Execute the full CE + synthesis pipeline with slice-specific overrides.

    Returns:
        answer         (str)   — final answer text
        provider       (str)   — e.g. "gateway:gemini"
        ce_metrics     (dict)  — context engineering stats
        context_xml    (str)   — the full assembled context the LLM saw, so the
                                 eval judge can verify which facts were grounded
        display_chunks (list[RetrievedChunk]) — post-rerank chunks the answer
                                 is grounded on. Returned here so the pipeline
                                 doesn't need a second retrieval pass.
    """
    from step_05_multi_agent.implementation.agents import (
        critic,
        graph_navigator,
        query_analyst,
        retrieval_specialist,
        structured_data,
    )
    from step_06_context_engineering.implementation.context_engineer import engineer_context

    analysis = query_analyst.analyze(question)

    retrieval_q = question
    if config.query_augmentation:
        retrieval_q = f"{question} {config.query_augmentation}"

    ret = retrieval_specialist.retrieve(retrieval_q, retriever, k=20)
    raw_chunks = list(ret.chunks)

    for sub_q in analysis.sub_questions[:4]:
        sub_ret = retrieval_specialist.retrieve(sub_q, retriever, k=10)
        raw_chunks.extend(sub_ret.chunks)

    # Always run graph navigation using retrieved chunks as seeds. Aggressive
    # routing was empirically costing recall, so every branch runs.
    csv_data  = ""
    graph_seeds = [c.text for c in raw_chunks] if raw_chunks else analysis.primary_entities
    graph_res = graph_navigator.navigate(question, graph_seeds, graph)
    graph_ctx = graph_res.context if graph_res.success else ""
    # Always run structured CSV query — unconditional detect_intent() → run_query().
    csv_res = structured_data.query(question)
    if csv_res.success:
        csv_data = csv_res.data

    context_xml, ce_metrics, display_chunks = engineer_context(
        question=question,
        raw_chunks=raw_chunks,
        csv_data=csv_data,
        graph_context=graph_ctx,
        rerank_k=config.rerank_k,
        compress_ratio=config.compress_ratio,
    )

    user_msg = f"RETRIEVED CONTEXT:\n{context_xml}\n\nQUESTION: {question}"

    # Single call through the gateway: it owns multi-provider fallback,
    # rate-limit cooldowns, and retries. The previous direct google.genai
    # fallback duplicated this responsibility (and hardcoded a non-existent
    # Gemini preview model that 404'd on every fallback attempt).
    llm = LLM(base_url=_GATEWAY_URL, timeout=120)
    result = llm.chat(
        messages=[{"role": "user", "content": user_msg}],
        system=config.system_prompt,
        max_tokens=1024,
        temperature=0.0,
    )
    answer   = result["text"]
    provider = f"gateway:{result.get('provider', '?')}"

    critic_res = critic.review(question, answer, {"Context": context_xml})
    return critic_res.answer, provider, ce_metrics, context_xml, display_chunks
