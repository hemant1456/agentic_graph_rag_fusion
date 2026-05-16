"""
Confidence scoring for Step 12 production hardening.

Heuristic: key-term recall (how many important question terms appear in the answer)
combined with a length signal.  Fast, zero LLM calls.

Thresholds (empirically tuned on the 22 golden questions):
  high   >= 0.65  -- answer is probably correct
  medium >= 0.40  -- answer exists but may be incomplete
  low    < 0.40   -- answer is thin; consider "I don't know" fallback
"""
from __future__ import annotations
import re

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can cannot what who which when "
    "where why how of in on at to for by with from about into over after".split()
)

def _key_terms(text: str) -> frozenset[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return frozenset(t for t in tokens if t not in _STOP)

def score_answer(question: str, answer: str) -> dict:
    """Return {"score": float, "label": str, "matched_terms": int, "total_terms": int}"""
    q_terms = _key_terms(question)
    a_terms = _key_terms(answer)
    if not q_terms:
        return {"score": 0.5, "label": "medium", "matched_terms": 0, "total_terms": 0}

    overlap = len(q_terms & a_terms) / len(q_terms)

    # Length signal: penalise very short answers (< 30 chars)
    length_bonus = min(1.0, len(answer) / 150)

    raw = 0.70 * overlap + 0.30 * length_bonus
    score = round(min(1.0, raw), 3)

    if score >= 0.65:
        label = "high"
    elif score >= 0.40:
        label = "medium"
    else:
        label = "low"

    return {
        "score": score,
        "label": label,
        "matched_terms": len(q_terms & a_terms),
        "total_terms": len(q_terms),
    }
