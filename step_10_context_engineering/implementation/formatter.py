"""
Structured context formatter — wraps retrieved content in labelled XML sections.

Structured context vs raw concatenation:
- LLMs attend more precisely when sources are clearly demarcated.
- XML tags give the LLM named anchors ("in <csv_data>…</csv_data>") it can
  reference when quoting facts, reducing cross-source confusion.
- The `src` attribute on each passage lets the model attribute claims without
  hallucinating file names.

Budget enforcement: if total char length exceeds `max_chars`, sections are
truncated in priority order — CSV first (exact), Graph second, passages last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk

_DEFAULT_BUDGET = 24_000  # chars (~6 000 tokens)
_PRIORITY = ["csv_data", "graph_context", "passages"]


def format_context(
    question: str,
    scored_chunks: "list[tuple[float, RetrievedChunk]]",
    csv_data: str = "",
    graph_context: str = "",
    max_chars: int = _DEFAULT_BUDGET,
) -> tuple[str, dict]:
    """
    Build a structured XML context string and return (context_str, stats).

    stats keys:
      raw_chars          — total chars of all inputs before budget enforcement
      engineered_chars   — total chars of the returned context string
      compression_ratio  — engineered / raw  (< 1.0 means we trimmed)
      chunks_used        — number of passages included
    """
    # ── Assemble sections ────────────────────────────────────────────────────
    sections: dict[str, str] = {}

    if csv_data and csv_data.strip():
        sections["csv_data"] = csv_data.strip()

    if graph_context and graph_context.strip():
        sections["graph_context"] = graph_context.strip()

    if scored_chunks:
        passage_parts: list[str] = []
        for rank, (score, chunk) in enumerate(scored_chunks, 1):
            src = getattr(chunk, "source", "unknown")
            passage_parts.append(
                f'  <passage rank="{rank}" src="{src}" score="{score:.3f}">\n'
                f"  {chunk.text.strip()}\n"
                f"  </passage>"
            )
        sections["passages"] = "\n".join(passage_parts)

    # ── Raw size (before budget) ──────────────────────────────────────────────
    raw_chars = sum(len(v) for v in sections.values())

    # ── Budget enforcement ────────────────────────────────────────────────────
    remaining = max_chars
    trimmed: dict[str, str] = {}
    for key in _PRIORITY:
        if key not in sections:
            continue
        text = sections[key]
        if len(text) <= remaining:
            trimmed[key] = text
            remaining -= len(text)
        elif remaining > 200:
            trimmed[key] = text[:remaining] + "\n  … [truncated — token budget reached]"
            remaining = 0
        # else: skip the section entirely

    # ── Build XML ─────────────────────────────────────────────────────────────
    xml_parts = [f'<context query="{question[:120]}">']
    for key, content in trimmed.items():
        xml_parts.append(f"  <{key}>\n{content}\n  </{key}>")
    xml_parts.append("</context>")
    context_str = "\n".join(xml_parts)

    engineered_chars = len(context_str)
    return context_str, {
        "raw_chars": raw_chars,
        "engineered_chars": engineered_chars,
        "compression_ratio": round(engineered_chars / raw_chars, 3) if raw_chars > 0 else 1.0,
        "chunks_used": len(scored_chunks),
    }
