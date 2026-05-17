from step_11_vsa.implementation.slices.base import SliceConfig

_SYSTEM = """\
You are Vertexia's Finance & Contracts analyst.

You are given RETRIEVED CONTEXT including CSV data with exact figures. Answer the
question using ONLY information present in the context.

## Finance-specific rules:
- ALL monetary figures must use exact comma-formatted values (e.g., $4,234,500 not $4.2M).
- Deal / contract status is exactly one of: "closed-won", "closed-lost", "in-progress",
  "signed", "active". Never paraphrase these labels.
- When reporting ARR (Annual Recurring Revenue): always say "ARR" in full on first use.
- For percentage questions: compute to one decimal place (e.g., 34.2%, not "about a third").
- Q3 2023 = July, August, September 2023.  H2 2023 = July – December 2023.
- Headcount figures come from CSV employee records — use them exactly.
- Keep the answer concise: number first, then supporting context.
"""

CONFIG = SliceConfig(
    name="finance",
    display_name="Finance & Contracts",
    system_prompt=_SYSTEM,
    keywords=[
        "arr", "annual recurring revenue", "revenue", "spend", "vendor",
        "deal", "contract", "closed", "won", "q1", "q2", "q3", "q4",
        "h1", "h2", "quarter", "budget", "allocation", "headcount", "hire",
        "subscription", "price", "total", "cost", "financial", "percentage",
        "enterprise", "segment", "invoice", "billing",
    ],
    force_csv=True,
    force_graph=False,
    rerank_k=8,
    compress_ratio=0.60,
    owns_questions=["Q07", "Q08", "Q10", "Q11", "Q12", "Q15", "Q16", "Q21", "Q22"],
)


def can_handle(question: str) -> float:
    """Return a confidence score [0..1] for routing this question to the Finance slice."""
    q = question.lower()
    hits = sum(1 for kw in CONFIG.keywords if kw in q)
    # Normalize by number of question words to avoid length bias
    words = max(len(q.split()), 1)
    return min(hits / words * 4.0, 1.0)
