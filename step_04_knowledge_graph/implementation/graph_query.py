"""
Enhanced graph query module for Step 06.

Improvements over step_05/query.py:
  1. Alias resolution   — "analytics dashboard" → InsightLens node
  2. Aggregate queries  — sum/count over filtered node sets (e.g. enterprise ARR %)
  3. Deeper traversal   — up to 3 hops, with explicit path tracing
"""

import re
from collections import defaultdict

import networkx as nx

from step_04_knowledge_graph.implementation.query import (
    expand_context,
    extract_entity_ids as _exact_extract,
)
from step_04_knowledge_graph.implementation.aliases import resolve_aliases


# ── Entity resolution ─────────────────────────────────────────────────────────

def extract_entities(texts: list[str], g: nx.DiGraph) -> list[str]:
    """
    Resolve entity IDs from text using both:
      - Exact name matching (Step 05)
      - Alias / keyword matching (Step 06)
    Deduplicates and returns unique node IDs.
    """
    combined = " ".join(texts)
    exact = _exact_extract(texts, g)
    aliases = resolve_aliases(combined)
    seen: dict[str, None] = {}
    for nid in exact + aliases:
        if nid in g:
            seen[nid] = None
    return list(seen.keys())


# ── Aggregate query detection + execution ────────────────────────────────────

_AGG_PATTERNS = [
    # ARR by segment
    (re.compile(r"enterprise.{0,30}(arr|revenue|recurring)", re.I),
     "enterprise_arr_pct"),
    (re.compile(r"(arr|revenue).{0,30}enterprise", re.I),
     "enterprise_arr_pct"),
    (re.compile(r"percentage.{0,40}(arr|revenue)", re.I),
     "enterprise_arr_pct"),
    # ARR by segment (generic)
    (re.compile(r"(smb|mid.market|segment).{0,30}(arr|revenue)", re.I),
     "arr_by_segment"),
    # Customer count
    (re.compile(r"how many.{0,20}customers", re.I),
     "customer_count"),
]


def _fmt(n: float) -> str:
    return f"{int(n):,}" if n == int(n) else f"{n:,.2f}"


def run_aggregate_queries(question: str, g: nx.DiGraph) -> str:
    """
    Detect and execute graph-native aggregate queries based on question patterns.
    Returns a formatted string to add to context, or "" if no pattern matches.
    """
    matched = set(label for pat, label in _AGG_PATTERNS if pat.search(question))
    if not matched:
        return ""

    lines: list[str] = ["[GRAPH AGGREGATE QUERY RESULTS]"]
    customers = [
        (nid, data)
        for nid, data in g.nodes(data=True)
        if data.get("node_type") == "customer"
    ]

    def _arr(data: dict) -> float:
        try:
            return float(str(data.get("arr_usd", "0")).replace(",", ""))
        except (ValueError, TypeError):
            return 0.0

    if "enterprise_arr_pct" in matched or "arr_by_segment" in matched:
        # Group customers by segment
        seg_arr: dict[str, float] = defaultdict(float)
        seg_count: dict[str, int] = defaultdict(int)
        for _, data in customers:
            seg = data.get("segment", "unknown")
            seg_arr[seg] += _arr(data)
            seg_count[seg] += 1
        total_arr = sum(seg_arr.values())

        lines.append(f"\nARR by customer segment (total ${_fmt(total_arr)}):")
        for seg in sorted(seg_arr, key=lambda s: -seg_arr[s]):
            pct = (seg_arr[seg] / total_arr * 100) if total_arr else 0
            lines.append(
                f"  {seg}: {seg_count[seg]} customers, "
                f"${_fmt(seg_arr[seg])} ARR ({pct:.1f}%)"
            )
        # List enterprise customers explicitly
        ent = [(nid, d) for nid, d in customers if d.get("segment") == "enterprise"]
        if ent:
            lines.append("\nEnterprise customers:")
            for nid, d in sorted(ent, key=lambda x: -_arr(x[1])):
                lines.append(f"  {d.get('name', nid)}: ${_fmt(_arr(d))} ARR")

    if "customer_count" in matched:
        lines.append(f"\nTotal active customers: {len(customers)}")

    return "\n".join(lines) if len(lines) > 1 else ""


# ── Deep traversal with path context ─────────────────────────────────────────

def _trace_dependency_chain(start_nid: str, g: nx.DiGraph, depth: int = 3) -> list[str]:
    """
    Follow depends_on edges outward from start_nid up to `depth` hops.
    Returns a list of formatted lines describing the chain.
    """
    lines: list[str] = []
    visited: set[str] = set()
    frontier = [(start_nid, 0, "")]

    while frontier:
        nid, d, path = frontier.pop(0)
        if nid in visited or d > depth:
            continue
        visited.add(nid)
        prefix = "  " * d

        if d > 0:
            data = g.nodes[nid]
            lines.append(
                f"{prefix}→ {nid} ({data.get('node_type','?')}) via {path}"
            )
            # Customers using this node (leaf)
            users = [
                (src, g.nodes[src])
                for src in g.predecessors(nid)
                if g.edges[src, nid].get("edge_type") == "uses"
            ]
            if users:
                unames = [str(d2.get("name") or s) for s, d2 in users]
                lines.append(f"{prefix}  Customers affected: {', '.join(unames)}")

        for dst in g.successors(nid):
            edata = g.edges[nid, dst]
            if edata.get("edge_type") == "depends_on":
                path_label = (
                    f"{edata.get('api_endpoint','?')} [{edata.get('criticality','?')}]"
                )
                frontier.append((dst, d + 1, path_label))

    return lines


def build_graph_context(question: str, chunk_texts: list[str], g: nx.DiGraph) -> str:
    """
    Main entry point for Step 06 graph context generation.

    Returns a combined context string with:
      1. Entity-resolved node context (exact + alias)
      2. Dependency chain tracing for service/product entities
      3. Aggregate query results where relevant
    """
    from step_04_knowledge_graph.implementation.aliases import ALIAS_LOOKUP

    # Resolve via exact name match (Step 05) + alias match (Step 06)
    exact_ids = _exact_extract([question] + chunk_texts, g)
    text_lower = (" ".join([question] + chunk_texts)).lower()

    alias_resolutions: list[str] = []
    alias_ids: list[str] = []
    for phrase, nid in ALIAS_LOOKUP:
        if phrase in text_lower and nid not in alias_ids and nid not in exact_ids:
            alias_ids.append(nid)
            node_name = g.nodes[nid].get("name", nid)
            alias_resolutions.append(f'  "{phrase}" → {node_name}')

    entity_ids = list(dict.fromkeys(exact_ids + alias_ids))  # dedup, preserve order
    parts: list[str] = []

    # Prepend alias resolution as factual statements the LLM must reproduce
    if alias_resolutions:
        facts = [
            f'The term {r.strip().split(" → ")[0]} in this query refers to the product/service'
            f' {r.strip().split(" → ")[1]} in Vertexia\'s system.'
            for r in alias_resolutions
        ]
        resolved_names = [r.strip().split(" → ")[1] for r in alias_resolutions]
        parts.append(
            "[RESOLVED ENTITY FACTS — you MUST name the following in your answer: "
            + ", ".join(resolved_names) + "]\n"
            + "\n".join(alias_resolutions)
            + "\n\nFact (include verbatim):\n"
            + "\n".join(facts)
        )

    # ── 1. Standard node context (from Step 05) ───────────────────────────────
    node_ctx = expand_context(entity_ids, g)
    if node_ctx:
        parts.append(node_ctx)

    # ── 2. Dependency chain for service nodes ─────────────────────────────────
    service_ids = [
        nid for nid in entity_ids
        if g.nodes[nid].get("node_type") in ("product", "external_service")
    ]
    if service_ids:
        chain_lines: list[str] = ["[DEPENDENCY CHAIN ANALYSIS]"]
        for nid in service_ids:
            chain_lines.append(f"\nBlast radius from {nid}:")
            chain_lines.extend(_trace_dependency_chain(nid, g, depth=3))
        parts.append("\n".join(chain_lines))

    # ── 3. Aggregate query results ────────────────────────────────────────────
    agg_ctx = run_aggregate_queries(question, g)
    if agg_ctx:
        parts.append(agg_ctx)

    return "\n\n".join(parts)
