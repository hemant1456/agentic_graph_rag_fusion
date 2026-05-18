from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step_05_multi_agent.implementation.agents.contracts import QueryAnalysis

_SYSTEM = """\
You are a query-analysis agent for Vertexia's internal knowledge base.
Classify the user's question and return ONLY a valid JSON object — no prose, no markdown fences.

JSON schema:
{
  "query_type": "<simple_lookup|aggregation|multi_hop|comparative|graph|mixed>",
  "needs_vector": <true|false>,
  "needs_graph": <true|false>,
  "needs_csv": <true|false>,
  "sub_questions": ["<sub-question 1>", ...],
  "primary_entities": ["<entity>", ...]
}

Rules:
- needs_csv=true for exact numbers/totals/counts/revenue/ARR/spend/headcount questions.
- needs_graph=true for org-chart, reporting lines, service dependencies, blast-radius questions.
- sub_questions: decompose compound questions into 2-4 independent sub-questions; empty list for simple queries.
- primary_entities: named people, products, projects, companies, teams mentioned in the query.
"""


def _heuristic_analysis(question: str) -> QueryAnalysis:
    """Rule-based fallback when LLM call fails."""
    q = question.lower()

    needs_csv = bool(re.search(
        r"total|sum|how many|count|revenue|arr|spend|q[1-4]\s*20|h[12]\s*20|headcount",
        q,
    ))
    needs_graph = bool(re.search(
        r"reports? to|depends? on|connected|org|hierarchy|blast.radius|service|integrat",
        q,
    ))

    # Detect multi-hop: question mentions two distinct named things
    entities = re.findall(r"[A-Z][a-z]+ (?:[A-Z][a-z]+ )?(?:Corp|Inc|Project|Team|Agent)?", question)
    is_multi_hop = (
        " and " in q
        and len(set(entities)) >= 2
    ) or bool(re.search(r"(two|both|each|respectively).{0,40}(project|deal|agent|team)", q))

    sub_questions: list[str] = []
    if is_multi_hop and " and " in question:
        parts = re.split(r"\s+and\s+", question, maxsplit=1)
        if len(parts) == 2 and len(parts[0]) > 10 and len(parts[1]) > 10:
            sub_questions = [parts[0].strip() + "?", parts[1].strip() + "?"]

    if needs_csv:
        query_type = "aggregation"
    elif needs_graph:
        query_type = "graph"
    elif is_multi_hop:
        query_type = "multi_hop"
    else:
        query_type = "simple_lookup"

    return QueryAnalysis(
        query_type=query_type,
        needs_vector=True,
        needs_graph=needs_graph,
        needs_csv=needs_csv,
        sub_questions=sub_questions,
        primary_entities=entities[:6],
    )


def analyze(question: str) -> QueryAnalysis:
    """Call the LLM to classify the query; fall back to heuristics on any failure."""
    try:
        from llm_gatewayV2.client import LLM
        llm = LLM(timeout=30)
        result = llm.chat(
            messages=[{"role": "user", "content": question}],
            system=_SYSTEM,
            max_tokens=256,
            temperature=0.0,
        )
        raw = result.get("text", "")
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        obj = json.loads(raw)
        return QueryAnalysis(
            query_type=obj.get("query_type", "simple_lookup"),
            needs_vector=bool(obj.get("needs_vector", True)),
            needs_graph=bool(obj.get("needs_graph", False)),
            needs_csv=bool(obj.get("needs_csv", False)),
            sub_questions=list(obj.get("sub_questions", [])),
            primary_entities=list(obj.get("primary_entities", [])),
        )
    except Exception:
        return _heuristic_analysis(question)
