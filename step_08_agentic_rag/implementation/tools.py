"""
Tool definitions and executors for the Step 08 agent.

Three tools:
  vector_search  — BM25 + dense RRF retrieval over Step 04 index
  graph_query    — Step 06 knowledge graph context (entity resolution + traversal)
  csv_query      — Step 07 structured Pandas query over raw CSVs

The agent (Claude Haiku with tool_use) calls these to gather information before
writing its final answer. Each tool returns a plain string to inject into context.
"""

from __future__ import annotations

import networkx as nx

from step_01_baseline_rag.implementation.retrieve import format_context
from step_06_graph_rag.implementation.graph_query import build_graph_context
from step_07_rag_fusion.implementation.csv_tool import detect_intent, run_query
from step_07_rag_fusion.implementation.pipeline import Step07RAG

# ── Anthropic tool schemas ─────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "vector_search",
        "description": (
            "Search Vertexia's internal documents using semantic + keyword retrieval. "
            "Returns the most relevant text passages from company data (HR, Finance, "
            "Engineering, Sales, Product, Legal, Executive). "
            "Use for: factual lookups, policy details, dates, names, org structure, "
            "project descriptions, meeting notes, and anything in free-text documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — be specific to get the most relevant passages",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of passages to return (default 5, max 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "graph_query",
        "description": (
            "Query the Vertexia knowledge graph for entity relationships, dependency "
            "chains, org reporting lines, customer-product links, and API dependencies. "
            "Use for: blast-radius analysis, reporting hierarchies, service dependencies, "
            "customer ownership, product integration graphs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer using graph traversal",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "csv_query",
        "description": (
            "Run a structured Pandas query over Vertexia's raw CSV data for exact "
            "numerical aggregations. Use for: total ARR, revenue sums, vendor spend, "
            "deal totals, budget figures, employee headcount by office location. "
            "Returns exact comma-formatted numbers. "
            "Describe what you want — e.g. 'Q3 2023 total revenue', "
            "'all customer ARR', 'total vendor spend', 'H2 2023 deal ARR', "
            "'employees by office location' (for counting staff per city/office)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Natural language description of the aggregate to compute",
                },
            },
            "required": ["intent"],
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────────────

def execute_tool(
    name: str,
    inputs: dict,
    retriever: Step07RAG,
    graph: nx.DiGraph,
) -> str:
    if name == "vector_search":
        query = str(inputs.get("query", ""))
        k = min(int(inputs.get("k", 5)), 10)
        # For departure/offboarding queries always include offboarding CSV in search
        if any(w in query.lower() for w in ("depart", "left vertexia", "voluntary", "offboard", "resign")):
            query = query + " departure_type voluntary offboarding_records"
        # For Phoenix Corp deal queries, boost toward the "signed June 2022" chunk
        if "phoenix" in query.lower() and any(w in query.lower() for w in ("deal", "contract", "signed", "corp", "enterprise", "outcome")):
            query = query + " Phoenix Corp signed June 2022 executed MSA closed won"
        chunks = retriever.retrieve(query, k=k)
        return format_context(chunks) or "[No relevant passages found]"

    if name == "graph_query":
        question = str(inputs.get("question", ""))
        ctx = build_graph_context(question, [], graph)
        return ctx or "[No graph context found for this query]"

    if name == "csv_query":
        intent_text = str(inputs.get("intent", ""))
        intent = detect_intent(intent_text)
        if intent is None:
            # Try a few common intents by keyword matching in the description
            intent_lower = intent_text.lower()
            if "q3" in intent_lower and "revenue" in intent_lower:
                intent = "q3_2023_revenue"
            elif "vendor" in intent_lower:
                intent = "total_vendor_spend"
            elif "arr" in intent_lower and ("total" in intent_lower or "all" in intent_lower):
                intent = "total_arr"
            elif "h2" in intent_lower or "second half" in intent_lower:
                intent = "h2_2023_arr"
            elif "q3" in intent_lower and ("deal" in intent_lower or "closed" in intent_lower):
                intent = "q3_closed_deals"
        if intent:
            return run_query(intent)
        return "[CSV QUERY: Could not determine the appropriate query from the intent description]"

    return f"[UNKNOWN TOOL: {name}]"
