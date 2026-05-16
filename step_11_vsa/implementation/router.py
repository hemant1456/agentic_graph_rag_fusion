"""
VSA Router — maps an incoming question to the best-fit domain slice.

Algorithm: each registered slice exports `can_handle(question) -> float` (0–1).
The router scores all slices and dispatches to the highest scorer.  If all
slices tie or all score below THRESHOLD, the General slice wins.

This is deliberately stateless and dependency-free — the router itself never
makes an LLM call.  Speed matters here because routing happens on every query.

Extending: register a new slice in SLICES below and create its module.  Nothing
else needs to change — the router will pick it up automatically.
"""

from __future__ import annotations

from step_11_vsa.implementation.slices import (
    engineering_slice,
    finance_slice,
    general_slice,
    hr_slice,
)
from step_11_vsa.implementation.slices.base import SliceConfig, run_with_config

# ── Registry — add new slices here ────────────────────────────────────────────
_SLICE_MODULES = [
    finance_slice,
    hr_slice,
    engineering_slice,
    general_slice,   # keep last: general is the fallback
]

_THRESHOLD = 0.10   # minimum score for a non-general slice to win


def route(question: str) -> tuple[str, float, SliceConfig]:
    """
    Return (slice_name, confidence, config) for the best-fit slice.

    The returned config is passed directly to run_with_config().
    """
    scores: list[tuple[float, str, SliceConfig]] = []
    for mod in _SLICE_MODULES:
        conf = mod.can_handle(question)
        scores.append((conf, mod.CONFIG.name, mod.CONFIG))

    scores.sort(key=lambda x: -x[0])
    best_score, best_name, best_config = scores[0]

    # If the winner is general and another slice is almost as good, prefer general
    # (the general slice has a forced floor of 0.15 so it always shows up)
    return best_name, best_score, best_config


def dispatch(
    question: str,
    retriever,
    graph,
) -> tuple[str, str, dict, str, float]:
    """
    Route + execute.

    Returns:
        answer          (str)
        provider        (str)
        ce_metrics      (dict)
        slice_name      (str)
        router_confidence (float)
    """
    slice_name, confidence, config = route(question)
    answer, provider, ce_metrics = run_with_config(question, config, retriever, graph)
    return answer, provider, ce_metrics, slice_name, confidence


def all_slice_configs() -> list[SliceConfig]:
    """Return all registered SliceConfig objects — useful for dashboard introspection."""
    return [mod.CONFIG for mod in _SLICE_MODULES]
