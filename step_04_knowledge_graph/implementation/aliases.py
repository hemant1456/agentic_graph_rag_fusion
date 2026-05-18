"""
Entity alias table for Step 04 Graph RAG.

Maps graph node IDs → list of natural-language phrases that describe them.
Used for alias-based entity resolution: when a query mentions "analytics dashboard",
we resolve it to the InsightLens graph node even though the exact name isn't present.

Augments the strict name-matching in query.py with phrase-based resolution.
"""

# node_id → list of lowercase keyword phrases
ENTITY_ALIASES: dict[str, list[str]] = {
    # ── Internal products ─────────────────────────────────────────────────────
    "NexusFlow": [
        "nexusflow", "data pipeline", "pipeline platform",
        "event streaming", "streaming platform", "pipeline",
    ],
    "InsightLens": [
        "insightlens", "analytics dashboard", "analytics product",
        "business intelligence", "bi dashboard", "dashboard product",
        "analytics ui", "reporting dashboard", "analytics tool",
    ],
    "PulseConnect": [
        "pulseconnect", "pulse connect", "customer communication",
        "notification platform", "email platform", "communication platform",
    ],
    "DataCraft": [
        "datacraft", "legacy pipeline", "migration pipeline",
        "datacraft integration", "acquired pipeline",
    ],
    # ── External services ─────────────────────────────────────────────────────
    "external_pulsar": [
        "pulsar", "message queue", "primary message queue",
        "message broker", "message queue infrastructure",
        "queue infrastructure", "event queue",
    ],
    "external_snowflake": [
        "snowflake data warehouse", "snowflake connector",
    ],
    "external_aws_s3": [
        "aws s3", "s3 bucket",
    ],
    "external_sendgrid": [
        "sendgrid",
    ],
    "external_twilio": [
        "twilio",
    ],
}

# Build reverse index: phrase → node_id (longest phrase first to avoid partial matches)
_ALIAS_INDEX: dict[str, str] = {}
for _nid, _phrases in ENTITY_ALIASES.items():
    for _phrase in _phrases:
        _ALIAS_INDEX[_phrase.lower()] = _nid

# Sorted by phrase length descending so we match "analytics dashboard" before "dashboard"
ALIAS_LOOKUP: list[tuple[str, str]] = sorted(
    _ALIAS_INDEX.items(), key=lambda x: -len(x[0])
)


def resolve_aliases(text: str) -> list[str]:
    """Return node IDs whose alias phrases appear in the given text (case-insensitive)."""
    text_lower = text.lower()
    found: dict[str, str] = {}  # nid → phrase  (dedup by nid)
    for phrase, nid in ALIAS_LOOKUP:
        if phrase in text_lower and nid not in found:
            found[nid] = phrase
    return list(found.keys())
