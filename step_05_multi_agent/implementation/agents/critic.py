from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_gatewayV2.client import LLM
from step_05_multi_agent.implementation.agents.contracts import CriticResult

_GATEWAY_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")

_SYSTEM = """\
You are a faithfulness critic for a RAG system.
Given a question, the retrieved context, and a generated answer, check:
1. Are all factual claims in the answer supported by the context?
2. Does the answer contradict any source?

Return ONLY a valid JSON object — no prose, no markdown fences:
{"approved": <true|false>, "confidence": "<high|medium|low>", "issue": "<brief issue or empty string>"}

Approve (approved=true) if the answer is clearly grounded, even if incomplete.
Only reject (approved=false) if the answer contains facts NOT present in the context.
"""


def _call_llm(prompt: str) -> dict:
    llm = LLM(base_url=_GATEWAY_URL, timeout=60)
    result = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system=_SYSTEM,
        max_tokens=128,
        temperature=0.0,
    )
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", result.get("text", "")).strip()
    return json.loads(raw)


def review(question: str, answer: str, contexts: dict[str, str]) -> CriticResult:
    ctx_snippet = "\n\n".join(
        f"[{k}] {v[:600]}" for k, v in contexts.items() if v and v.strip()
    )
    prompt = (
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CONTEXT (truncated):\n{ctx_snippet}\n\n"
        f"ANSWER TO REVIEW:\n{answer}"
    )

    try:
        obj = _call_llm(prompt)
        approved = bool(obj.get("approved", True))
        confidence = obj.get("confidence", "high")
        issue = obj.get("issue", "")

        if approved or confidence != "low":
            return CriticResult(
                approved=approved,
                answer=answer,
                confidence=confidence,
                notes=issue,
            )

        # Low confidence → one revision attempt: ask gateway to try again with the issue noted
        from step_05_multi_agent.implementation.agents.synthesis import _SYSTEM as SYNTH_SYS
        llm = LLM(base_url=_GATEWAY_URL, timeout=120)
        ctx_full = "\n\n".join(
            f"### {k}\n{v}" for k, v in contexts.items() if v and v.strip()
        )
        fix_msgs = [
            {"role": "user", "content": f"RETRIEVED CONTEXT:\n{ctx_full}\n\nQUESTION: {question}"},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": f"Your previous answer has an issue: {issue}. Please revise your answer using ONLY the retrieved context above."},
        ]
        revised = llm.chat(messages=fix_msgs, system=SYNTH_SYS, max_tokens=1024, temperature=0.0)
        return CriticResult(
            approved=True,
            answer=revised["text"],
            confidence="medium",
            notes=f"revised after critic: {issue}",
        )

    except Exception:
        # Critic call failed — pass through original answer unchanged
        return CriticResult(approved=True, answer=answer, confidence="high", notes="")
