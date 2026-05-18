"""
Step 02 — CSV Tool pipeline.

Upgrades over Step 01 baseline:
  - Detects aggregate questions (total ARR, Q3 revenue, headcount, etc.)
  - Runs exact Pandas queries and injects the result into LLM context
  - Fixes all Tier 2 questions that dense retrieval cannot answer

Now inherits from BaselineRAG and contributes a "csv" context section. The
inherited `query()` template handles retrieval, context assembly, and
generation. See AUDIT_FIXES.md #14 for why the prior compose-by-import
implementation was replaced.
"""

from step_01_baseline_rag.implementation.pipeline import BaselineRAG
from step_01_baseline_rag.implementation.retrieve import RetrievedChunk
from step_02_tools.implementation.csv_tool import detect_intent, run_query


class Step02ToolsRAG(BaselineRAG):
    """Baseline retrieval + exact CSV tool calls for aggregate questions.

    Usage:
        rag = Step02ToolsRAG(k=10).build()
        result = rag.query("What is the total ARR across all customers?")
    """

    def build_context_sections(
        self, chunks: list[RetrievedChunk], question: str
    ) -> dict[str, str]:
        sections = super().build_context_sections(chunks, question)
        intent = detect_intent(question)
        if intent:
            sections["csv"] = run_query(intent)
        return sections
