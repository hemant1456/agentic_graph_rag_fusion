"""Multi-signal answer confidence scoring.

The earlier version was `0.7 * question_term_overlap + 0.3 * length_bonus`.
That measured "answer repeats the question's words" — a copy-back attack would
ace it. It said nothing about whether the answer was grounded in the retrieved
context.

This rewrite uses signals that actually correlate with faithfulness:

  1. **Critic verdict** (when available). The step_05 critic is an LLM that
     reads (question, context, answer) and returns approved/notes. This is by
     far the strongest signal — when the critic says "approved: false" with a
     reason, we trust it and cap confidence low. When the critic approves, we
     trust that too.

  2. **Retrieval evidence**. If the engineered context is empty or tiny, the
     answer cannot be grounded in anything we showed the LLM. Low confidence.

  3. **Answer sanity**. Empty answers, single-line refusals, "I don't know"
     responses all imply the model couldn't ground an answer. Low confidence.

  4. **Term overlap** stays in the mix but as a weak tiebreaker, not the
     dominant signal.
"""
from __future__ import annotations

import re

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can cannot what who which when "
    "where why how of in on at to for by with from about into over after".split()
)

# Substrings that strongly imply the model refused or hedged.
_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "no information",
    "not enough information",
    "not present in the context",
    "context does not contain",
    "cannot answer",
    "unable to answer",
)


def _key_terms(text: str) -> frozenset[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return frozenset(t for t in tokens if t not in _STOP)


def _label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def score_answer(
    question: str,
    answer: str,
    *,
    critic_approved: bool | None = None,
    critic_notes: str = "",
    context_chars: int = 0,
    chunks_used: int = 0,
) -> dict:
    """Return a confidence breakdown.

    Keyword args are all optional so callers from older paths still work,
    but supplying critic_approved + context_chars gives a meaningful score.

    Returns:
        {
          "score": float in [0, 1],
          "label": "high" | "medium" | "low",
          "signals": {critic, retrieval, answer_sanity, overlap},
          "reason": short human-readable explanation,
        }
    """
    answer_clean = (answer or "").strip()
    answer_lower = answer_clean.lower()
    answer_len = len(answer_clean)

    # ── Hard floors: refuse early when the answer or context clearly fails ──
    if answer_len < 20:
        return {
            "score": 0.20,
            "label": "low",
            "signals": {"critic": critic_approved, "retrieval": 0, "answer_sanity": 0, "overlap": 0.0},
            "reason": "answer too short to be informative",
        }

    if any(marker in answer_lower[:200] for marker in _REFUSAL_MARKERS):
        return {
            "score": 0.30,
            "label": "low",
            "signals": {"critic": critic_approved, "retrieval": 0, "answer_sanity": 0, "overlap": 0.0},
            "reason": "model explicitly refused or hedged",
        }

    if critic_approved is False:
        # The critic actively rejected. Trust it — cap confidence regardless
        # of the other signals.
        return {
            "score": 0.30,
            "label": "low",
            "signals": {"critic": False, "retrieval": 1, "answer_sanity": 1, "overlap": 0.0},
            "reason": f"critic rejected: {critic_notes or 'unspecified'}",
        }

    # ── Soft signals: combine for a graded score ────────────────────────────
    q_terms = _key_terms(question)
    a_terms = _key_terms(answer)
    overlap = len(q_terms & a_terms) / max(1, len(q_terms))

    has_real_context = context_chars >= 200 or chunks_used >= 1
    sane_length = 30 <= answer_len <= 1500

    base = 0.50

    # The critic explicitly approved is the strongest positive signal.
    if critic_approved is True:
        base += 0.25

    # We retrieved something to ground in.
    if has_real_context:
        base += 0.10

    # Answer is in the reasonable length window.
    if sane_length:
        base += 0.05

    # Term overlap as a small tiebreaker (capped contribution).
    base += 0.10 * overlap

    score = round(min(1.0, base), 3)
    return {
        "score": score,
        "label": _label(score),
        "signals": {
            "critic": critic_approved,
            "retrieval": 1 if has_real_context else 0,
            "answer_sanity": 1 if sane_length else 0,
            "overlap": round(overlap, 3),
        },
        "reason": "critic-grounded" if critic_approved is True else "no critic signal",
    }
