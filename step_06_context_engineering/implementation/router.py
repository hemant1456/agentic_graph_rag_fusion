from __future__ import annotations

from step_06_context_engineering.implementation.slices import (
    engineering_slice,
    finance_slice,
    general_slice,
    hr_slice,
)
from step_06_context_engineering.implementation.slices.base import SliceConfig, run_with_config

_SLICE_MODULES = [
    finance_slice,
    hr_slice,
    engineering_slice,
    general_slice,   # keep last: general is the fallback
]


def route(question: str) -> tuple[str, float, SliceConfig]:
    """
    Return (slice_name, confidence, config) for the best-fit slice.

    The returned config is passed directly to run_with_config().
    """
    scores: list[tuple[float, str, SliceConfig]] = []
    for mod in _SLICE_MODULES:
        conf = mod.CONFIG.can_handle(question)
        scores.append((conf, mod.CONFIG.name, mod.CONFIG))

    scores.sort(key=lambda x: -x[0])
    best_score, best_name, best_config = scores[0]
    return best_name, best_score, best_config


def dispatch(
    question: str,
    retriever,
    graph,
) -> tuple[str, str, dict, str, float, str, list]:
    """
    Route + execute.

    Returns:
        answer            (str)
        provider          (str)
        ce_metrics        (dict)
        slice_name        (str)
        router_confidence (float)
        context_xml       (str)   — assembled context the LLM saw, for eval grounding
        display_chunks    (list[RetrievedChunk]) — post-rerank sources for dashboard
    """
    slice_name, confidence, config = route(question)
    answer, provider, ce_metrics, context_xml, display_chunks = run_with_config(
        question, config, retriever, graph
    )
    return answer, provider, ce_metrics, slice_name, confidence, context_xml, display_chunks
