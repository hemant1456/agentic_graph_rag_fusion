"""
Graceful degradation for Step 12 production hardening.

If synthesis completely fails (exception or empty answer), this module produces
an extractive fallback answer from the best retrieved chunk rather than returning
a blank or error string to the user.
"""
from __future__ import annotations
import re

def extractive_fallback(question: str, chunks) -> tuple[str, str]:
    """
    Return (answer, provider) using the top-scored retrieved chunk.

    chunks: list of RetrievedChunk (with .text, .score or .source)
    """
    if not chunks:
        return "I don't have enough information to answer this question.", "fallback:empty"

    # Pick best chunk (highest .score if available, else first)
    best = chunks[0]
    for c in chunks[1:]:
        if getattr(c, "score", 0) > getattr(best, "score", 0):
            best = c

    # Take first 3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", best.text.strip())
    answer = " ".join(sentences[:3]).strip()
    if not answer:
        answer = best.text[:300].strip()

    return f"[Extractive] {answer}", "fallback:extractive"
