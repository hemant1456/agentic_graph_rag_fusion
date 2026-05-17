from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    import networkx as nx
    from step_07_rag_fusion.implementation.pipeline import Step07RAG

_GATEWAY_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")
_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


@dataclass
class SliceConfig:
    """Everything a slice needs to describe itself and influence execution."""
    name: str                        # slug used in routing logs and eval output
    display_name: str                # human-readable label
    system_prompt: str               # domain-tuned system prompt sent to synthesis LLM
    keywords: list[str]              # routing vocabulary — matched against the query
    force_csv: bool = False          # always activate CSV tool regardless of query_analyst
    force_graph: bool = False        # always activate graph tool regardless of query_analyst
    query_augmentation: str = ""     # extra terms appended to the retrieval query
    rerank_k: int = 8                # top-k for CrossEncoder reranking
    compress_ratio: float = 0.60     # extractive compression retention ratio
    # Subset of golden question IDs that belong to this slice (for eval breakdown)
    owns_questions: list[str] = field(default_factory=list)


def run_with_config(
    question: str,
    config: SliceConfig,
    retriever: "Step07RAG",
    graph: "nx.DiGraph",
) -> tuple[str, str, dict]:
    """
    Execute the full CE + synthesis pipeline with slice-specific overrides.

    Returns:
        answer        (str)   — final answer text
        provider      (str)   — e.g. "gateway:gemini"
        ce_metrics    (dict)  — context engineering stats
    """
    from step_09_multi_agent.implementation.agents import (
        critic,
        graph_navigator,
        query_analyst,
        retrieval_specialist,
        structured_data,
    )
    from step_10_context_engineering.implementation.context_engineer import engineer_context

    analysis = query_analyst.analyze(question)

    retrieval_q = question
    if config.query_augmentation:
        retrieval_q = f"{question} {config.query_augmentation}"

    ret = retrieval_specialist.retrieve(retrieval_q, retriever, k=20)
    raw_chunks = list(ret.chunks)

    for sub_q in analysis.sub_questions[:4]:
        sub_ret = retrieval_specialist.retrieve(sub_q, retriever, k=10)
        raw_chunks.extend(sub_ret.chunks)

    # Always run graph navigation using retrieved chunks as seeds — mirrors step 07.
    # force_graph still used by slice config; this makes graph a floor, not an opt-in.
    csv_data  = ""
    graph_seeds = [c.text for c in raw_chunks] if raw_chunks else analysis.primary_entities
    graph_res = graph_navigator.navigate(question, graph_seeds, graph)
    graph_ctx = graph_res.context if graph_res.success else ""
    # Always run structured CSV query — mirrors step 07's unconditional detect_intent() → run_query().
    csv_res = structured_data.query(question)
    if csv_res.success:
        csv_data = csv_res.data

    context_xml, ce_metrics = engineer_context(
        question=question,
        raw_chunks=raw_chunks,
        csv_data=csv_data,
        graph_context=graph_ctx,
        rerank_k=config.rerank_k,
        compress_ratio=config.compress_ratio,
    )

    user_msg = f"RETRIEVED CONTEXT:\n{context_xml}\n\nQUESTION: {question}"
    answer   = ""
    provider = "error"

    try:
        from llm_gatewayV2.client import LLM
        llm = LLM(base_url=_GATEWAY_URL, timeout=120)
        result = llm.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=config.system_prompt,
            max_tokens=1024,
            temperature=0.0,
        )
        answer   = result["text"]
        provider = f"gateway:{result.get('provider', 'gemini')}"
    except Exception:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=config.system_prompt,
                max_output_tokens=1024,
                temperature=0.0,
            ),
        )
        answer   = resp.text or ""
        provider = "gemini-direct"

    critic_res = critic.review(question, answer, {"Context": context_xml})
    return critic_res.answer, provider, ce_metrics
