from __future__ import annotations

import os
import sys
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step_06_graph_rag.implementation.graph_query import build_graph_context
from step_07_rag_fusion.implementation.csv_tool import detect_intent, run_query
from step_07_rag_fusion.implementation.pipeline import Step07RAG
from step_08_agentic_rag.implementation.tools import TOOL_SCHEMAS, execute_tool
from step_01_baseline_rag.implementation.retrieve import format_context

_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
_GATEWAY_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")

_SYSTEM_PROMPT = """\
You are a precise research assistant for Vertexia Inc.

You are given PRE-RETRIEVED CONTEXT from Vertexia's knowledge base. Your job is to:
1. Review the pre-retrieved context carefully.
2. If it is sufficient to answer the question precisely, answer directly.
3. If it is incomplete, use tools (vector_search, graph_query, csv_query) to fill the gaps.

## Rules:
- Use EXACT field values from source data. For departure type the exact value is
  "voluntary" (never "voluntarily"). Other exact values: "completed", "signed",
  "closed-won", "active", "departed".
- When a product is referenced by alias ("analytics dashboard"), name the actual product
  (InsightLens, NexusFlow, PulseConnect) explicitly in your answer.
- For "two efforts with the same name" questions: identify BOTH named things and state
  the outcome of EACH. If the pre-retrieved context already shows them, report both.
- For employee office counts, use csv_query if the context doesn't have the exact number.
- NEVER use numbered bullet lists (1. 2. 3.) for counts — use plain prose instead.
- When identifying an on-call engineer for a specific date or week, state ONLY that
  engineer's name and week. Do NOT list other engineers on neighboring weeks.
- For deal/contract status questions: look for the exact word "signed" in the source
  data when reporting contract status (e.g. "signed in June 2022").
- Keep the answer concise and directly responsive to the question asked.
"""


def run_agent(
    question: str,
    retriever: Step07RAG,
    graph: nx.DiGraph,
    max_rounds: int = 3,
) -> tuple[str, str]:
    """
    Context-first agentic loop using free APIs only. Returns (answer_text, provider).

    Tries Gateway V2 (Gemini with tool-use) first; falls back to Gemini direct
    (pre-context only, no tool-use) if the gateway is unreachable or rate-limited.
    """
    chunks = retriever.retrieve(question, k=10)
    vector_ctx = format_context(chunks)
    graph_ctx = build_graph_context(question, [c.text for c in chunks], graph)
    csv_intent = detect_intent(question)
    csv_ctx = run_query(csv_intent) if csv_intent else ""

    ctx_parts: list[str] = []
    if csv_ctx:
        ctx_parts.append(csv_ctx)
    if graph_ctx:
        ctx_parts.append(graph_ctx)
    ctx_parts.append(vector_ctx)
    pre_context = "\n\n".join(ctx_parts)

    try:
        return _run_gateway_agent(question, pre_context, retriever, graph, max_rounds)
    except Exception as e:
        print(f"  [gateway failed: {type(e).__name__}: {str(e)[:80]}] → Gemini direct", file=sys.stderr)

    return _run_gemini_direct(question, pre_context)


def _run_gateway_agent(
    question: str,
    pre_context: str,
    retriever: Step07RAG,
    graph: nx.DiGraph,
    max_rounds: int,
) -> tuple[str, str]:
    from llm_gatewayV2.client import LLM

    llm = LLM(base_url=_GATEWAY_URL, timeout=120)

    initial_msg = {
        "role": "user",
        "content": (
            f"Please answer the following question. "
            f"PRE-RETRIEVED CONTEXT is provided below — use it as your starting point "
            f"and call tools only if the context is incomplete.\n\n"
            f"QUESTION: {question}\n\n"
            f"PRE-RETRIEVED CONTEXT:\n{pre_context}"
        ),
    }
    messages: list[dict] = [initial_msg]

    for _round in range(max_rounds):
        result = llm.chat(
            messages=messages,
            system=_SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.0,
        )

        provider_tag = f"gateway:{result.get('provider', 'gemini')}"

        if result["stop_reason"] == "end_turn":
            return result["text"], provider_tag

        if result["stop_reason"] != "tool_use":
            # max_tokens or other — return what we have, or fall through to final
            if result.get("text", "").strip():
                return result["text"], provider_tag
            break

        # Echo assistant turn back (full tool_calls preserves provider_meta for Gemini)
        tool_calls: list[dict] = result.get("tool_calls") or []
        messages.append({
            "role": "assistant",
            "content": result.get("text", ""),
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            result_text = execute_tool(tc["name"], tc.get("arguments", {}), retriever, graph)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "tool_name": tc["name"],
                "content": result_text,
            })

    messages.append({
        "role": "user",
        "content": "Based on all the information retrieved above, provide your final answer.",
    })
    final = llm.chat(
        messages=messages,
        system=_SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.0,
    )
    return final["text"], f"gateway:{final.get('provider', 'gemini')}"


def _run_gemini_direct(question: str, pre_context: str) -> tuple[str, str]:
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    user_message = (
        f"PRE-RETRIEVED CONTEXT:\n{pre_context}\n\n"
        f"QUESTION: {question}"
    )
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=user_message,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.0,
        ),
    )
    return response.text or "", "gemini-direct"
