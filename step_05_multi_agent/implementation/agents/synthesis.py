from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_gatewayV2.client import LLM
from step_05_multi_agent.implementation.agents.contracts import SynthesisResult

_GATEWAY_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")

_SYSTEM = """\
You are a precise research assistant for Vertexia Inc.

You are given RETRIEVED CONTEXT assembled by specialized agents. Answer the question
using ONLY information present in the context. Be concise and directly responsive.

## Rules:
- Use EXACT field values from source data. Departure type: "voluntary" (never "voluntarily").
  Other exact values: "completed", "signed", "closed-won", "active", "departed".
- Dates from CSV source records must be reproduced in ISO format (e.g., 2023-07-01, not "July 1, 2023").
- When a product is referenced by alias ("analytics dashboard"), name the actual product
  (InsightLens, NexusFlow, PulseConnect) explicitly in your answer.
- For "two efforts with the same name" questions: identify BOTH named things and state
  the outcome of EACH.
- NEVER use numbered bullet lists (1. 2. 3.) for counts — use plain prose instead.
- When identifying an on-call engineer for a specific date or week, state ONLY that
  engineer's name and week. Do NOT list other engineers on neighboring weeks.
- For deal/contract status questions: look for the exact word "signed" in the source
  data when reporting contract status (e.g. "signed in June 2022").
- Keep the answer concise and directly responsive to the question asked.
"""


def _build_context_block(contexts: dict[str, str]) -> str:
    parts = []
    for label, text in contexts.items():
        if text and text.strip():
            parts.append(f"### {label}\n{text.strip()}")
    return "\n\n".join(parts)


def synthesize(
    question: str,
    contexts: dict[str, str],
    query_type: str = "simple_lookup",
) -> SynthesisResult:
    context_block = _build_context_block(contexts)
    user_msg = (
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}"
    )

    # Gateway owns multi-provider fallback. The earlier direct-Gemini except
    # branch hardcoded a non-existent preview model and ran a redundant retry
    # layer below the gateway's own — see AUDIT_FIXES.md #10 and #13.
    try:
        llm = LLM(base_url=_GATEWAY_URL, timeout=120)
        result = llm.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=_SYSTEM,
            max_tokens=1024,
            temperature=0.0,
        )
        return SynthesisResult(
            answer=result["text"],
            provider=f"gateway:{result.get('provider', '?')}",
        )
    except Exception as exc:
        return SynthesisResult(
            answer=f"[Synthesis failed: {exc}]",
            provider="error",
        )
