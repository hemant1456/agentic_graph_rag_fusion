from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step_06_graph_rag.implementation.graph_query import build_graph_context
from step_09_multi_agent.implementation.agents.contracts import GraphResult


def navigate(question: str, entity_hints: list[str], graph: nx.DiGraph) -> GraphResult:
    # Pass entity hints as seed texts so graph traversal starts from the right nodes
    seed_texts = entity_hints if entity_hints else []
    context = build_graph_context(question, seed_texts, graph)
    if not context:
        return GraphResult(context="", entities_found=[], success=False)
    # Count distinct entities found as a rough proxy
    entities_found = [h for h in entity_hints if h.lower() in context.lower()]
    return GraphResult(context=context, entities_found=entities_found, success=True)
