"""
General / Company Slice — catch-all for company history, leadership, legal, and
any question that doesn't clearly belong to another domain.

No force-overrides; relies on the query analyst's dynamic classification.
System prompt is identical to Step 10's proven baseline — works well across
all question types.

Owns: Q01, Q02, Q03, Q04, Q13 (company history + cross-domain)
"""

from step_11_vsa.implementation.slices.base import SliceConfig

_SYSTEM = """\
You are a precise research assistant for Vertexia Inc.

You are given RETRIEVED CONTEXT assembled from Vertexia's knowledge base.
Answer the question using ONLY information present in the context.

## Rules:
- Use EXACT field values from source data. Departure type: "voluntary" (never "voluntarily").
  Other exact values: "completed", "signed", "closed-won", "active", "departed".
- When a product is referenced by alias ("analytics dashboard"), name the actual product
  (InsightLens, NexusFlow, PulseConnect) explicitly in your answer.
- For "two efforts with the same name" questions: identify BOTH named things and state
  the outcome of EACH.
- NEVER use numbered bullet lists (1. 2. 3.) for counts — use plain prose instead.
- When identifying an on-call engineer for a specific date or week, state ONLY that
  engineer's name and week. Do NOT list other engineers on neighboring weeks.
- For deal/contract status questions: look for the exact word "signed" in the source
  data (e.g., "signed in June 2022").
- Keep the answer concise and directly responsive to the question asked.
"""

CONFIG = SliceConfig(
    name="general",
    display_name="General / Company",
    system_prompt=_SYSTEM,
    keywords=[
        "vertexia", "founded", "year", "ceo", "cto", "coo", "co-founder",
        "acquisition", "datacraft", "history", "policy", "retention", "legal",
        "data", "company", "who is", "when was", "what is",
    ],
    force_csv=False,
    force_graph=False,
    rerank_k=8,
    compress_ratio=0.60,
    owns_questions=["Q01", "Q02", "Q03", "Q04"],
)


def can_handle(question: str) -> float:
    """General slice has a constant low confidence — it accepts anything no other slice claims."""
    q = question.lower()
    hits = sum(1 for kw in CONFIG.keywords if kw in q)
    words = max(len(q.split()), 1)
    base = min(hits / words * 4.0, 1.0)
    # Slight boost so general always competes rather than being pure fallback
    return max(base, 0.15)
