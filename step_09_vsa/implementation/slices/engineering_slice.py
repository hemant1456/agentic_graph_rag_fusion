from step_09_vsa.implementation.slices.base import SliceConfig

_SYSTEM = """\
You are Vertexia's Engineering and Product Knowledge analyst.

You are given RETRIEVED CONTEXT including architecture documents, on-call schedules,
and product records. Answer the question using ONLY information present in the context.

## Engineering-specific rules:
- Service names are proper nouns — capitalize exactly: NexusFlow, InsightLens, PulseConnect.
- "analytics dashboard" is an alias for InsightLens — always use the canonical name.
- For blast-radius / dependency questions: list every downstream service that depends
  on the failed component. Trace the full chain, not just the first hop.
- SLO values must include the exact percentage AND the measurement window (e.g., "99.9% monthly").
- For "two things with the same name" (disambiguation): name BOTH things and state the
  outcome of EACH separately. Never merge them into a single answer.
- On-call schedule: for a specific date or incident, state ONLY that engineer and week.
  Do NOT list the full rotation roster.
- Message queue / infrastructure technology: use exact names from the architecture doc.
- Keep the answer technically precise and directly responsive.
"""

CONFIG = SliceConfig(
    name="engineering",
    display_name="Engineering & Product",
    system_prompt=_SYSTEM,
    keywords=[
        "nexusflow", "insightlens", "pulseconnect", "api", "depend", "service",
        "message queue", "infrastructure", "slo", "availability", "blast radius",
        "outage", "incident", "integration", "kafka", "pipeline", "architecture",
        "analytics dashboard", "ingestion", "failure", "on-call", "oncall",
        "downtime", "latency", "throughput", "microservice", "platform",
        "message", "queue", "broker", "stream", "product", "feature",
        "launch", "roadmap", "two", "same name", "disambiguation",
    ],
    force_csv=False,
    force_graph=True,
    query_augmentation="Project Phoenix",
    rerank_k=8,
    compress_ratio=0.80,
    owns_questions=["Q05", "Q06", "Q17", "Q18", "Q20", "Q23", "Q24"],
)


def can_handle(question: str) -> float:
    q = question.lower()
    hits = sum(1 for kw in CONFIG.keywords if kw in q)
    words = max(len(q.split()), 1)
    return min(hits / words * 4.0, 1.0)
