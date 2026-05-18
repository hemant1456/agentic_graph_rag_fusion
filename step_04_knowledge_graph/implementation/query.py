import networkx as nx

_NAME_INDEX_ATTR = "_name_index"


def _build_name_index(g: nx.DiGraph) -> dict[str, str]:
    """
    Map every resolvable name string (lowercase) → node ID.

    Sources:
      - node 'name' attribute (persons, products, customers, vendors)
      - 'departed_to' attribute on Person nodes
        (so "FinDataCo" → E029 Adrian Blake, the employee who left for that company)
    """
    index: dict[str, str] = {}
    for nid, data in g.nodes(data=True):
        name: str = data.get("name", "")
        if len(name) >= 4:
            index[name.lower()] = nid
        departed_to: str = data.get("departed_to", "")
        if len(departed_to) >= 4:
            index[departed_to.lower()] = nid
    return index


def _get_name_index(g: nx.DiGraph) -> dict[str, str]:
    """Lazy-build and cache the name index on the graph itself.

    The graph is immutable after load_or_build, so this index is safe to
    memoize via networkx's graph-level attribute dict (`g.graph`). Subsequent
    queries reuse the cached map without re-iterating every node.
    """
    cached = g.graph.get(_NAME_INDEX_ATTR)
    if cached is not None:
        return cached
    index = _build_name_index(g)
    g.graph[_NAME_INDEX_ATTR] = index
    return index


def extract_entity_ids(texts: list[str], g: nx.DiGraph) -> list[str]:
    """Return unique node IDs whose name appears in any of the given texts."""
    index = _get_name_index(g)
    combined = " ".join(texts).lower()
    found: dict[str, str] = {}  # nid → matched name (dedup by nid)
    for name_lower, nid in index.items():
        if name_lower in combined and nid not in found:
            found[nid] = name_lower
    return list(found.keys())


def _fmt_person(nid: str, data: dict) -> str:
    parts = [f"{data['name']} ({nid})", f"role: {data.get('role','?')}",
             f"dept: {data.get('department','?')}", f"status: {data.get('status','?')}",
             f"location: {data.get('location','?')}"]
    if data.get("departure_type"):
        parts.append(f"departure_type: {data['departure_type']}")
    if data.get("departed_to"):
        parts.append(f"departed_to: {data['departed_to']}")
    if data.get("departure_notes"):
        parts.append(f"departure_notes: {data['departure_notes']}")
    return " | ".join(parts)


def _manager_of(nid: str, g: nx.DiGraph) -> tuple[str, dict] | None:
    for dst in g.successors(nid):
        if g.edges[nid, dst].get("edge_type") == "reports_to":
            return dst, g.nodes[dst]
    return None


def _direct_reports(nid: str, g: nx.DiGraph) -> list[tuple[str, dict]]:
    return [
        (src, g.nodes[src])
        for src in g.predecessors(nid)
        if g.edges[src, nid].get("edge_type") == "reports_to"
    ]


def expand_context(entity_ids: list[str], g: nx.DiGraph) -> str:
    if not entity_ids:
        return ""

    lines: list[str] = ["[KNOWLEDGE GRAPH CONTEXT]"]
    seen: set[str] = set()

    def _add_person(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        data = g.nodes[nid]
        lines.append(f"\nPerson: {_fmt_person(nid, data)}")
        mgr = _manager_of(nid, g)
        if mgr:
            mid, mdata = mgr
            lines.append(f"  Manager: {mdata.get('name','?')} ({mid}) — {mdata.get('role','?')}")
            seen.add(mid)
            # Manager's manager (2nd hop — needed for Q14: vendor → owner → manager)
            mgr2 = _manager_of(mid, g)
            if mgr2:
                mid2, mdata2 = mgr2
                lines.append(f"  Manager's manager: {mdata2.get('name','?')} ({mid2}) — {mdata2.get('role','?')}")
                seen.add(mid2)
        reports = _direct_reports(nid, g)
        if reports:
            rnames = [d.get("name", r) for r, d in reports[:6]]
            suffix = " (+more)" if len(reports) > 6 else ""
            lines.append(f"  Direct reports: {', '.join(rnames)}{suffix}")
        for dst in g.successors(nid):
            if g.edges[nid, dst].get("edge_type") == "owns_contract":
                vdata = g.nodes[dst]
                lines.append(
                    f"  Owns contract: {vdata.get('name','?')} "
                    f"(${vdata.get('annual_value_usd','?')}/yr, {vdata.get('category','?')})"
                )
        for dst in g.successors(nid):
            if g.edges[nid, dst].get("edge_type") == "manages_account":
                cdata = g.nodes[dst]
                lines.append(
                    f"  Manages account: {cdata.get('name','?')} "
                    f"(ARR ${cdata.get('arr_usd','?')}, {cdata.get('segment','?')})"
                )

    for nid in entity_ids:
        if nid not in g:
            continue
        data = g.nodes[nid]
        ntype = data.get("node_type", "")

        if ntype == "person":
            _add_person(nid)

        elif ntype in ("product", "external_service"):
            if nid in seen:
                continue
            seen.add(nid)
            lines.append(f"\nService: {nid} (type: {ntype})")
            for src in g.predecessors(nid):
                edata = g.edges[src, nid]
                if edata.get("edge_type") == "depends_on":
                    lines.append(
                        f"  Depended on by: {src} via {edata.get('api_endpoint','?')} "
                        f"[{edata.get('criticality','?')}]"
                    )
            for dst in g.successors(nid):
                edata = g.edges[nid, dst]
                if edata.get("edge_type") == "depends_on":
                    lines.append(
                        f"  Depends on: {dst} via {edata.get('api_endpoint','?')} "
                        f"[{edata.get('criticality','?')}]"
                    )
            users = [
                (src, g.nodes[src])
                for src in g.predecessors(nid)
                if g.edges[src, nid].get("edge_type") == "uses"
            ]
            if users:
                user_strs = [
                    f"{d.get('name',s)} (ARR ${d.get('arr_usd','?')}, {d.get('segment','?')})"
                    for s, d in users
                ]
                lines.append(f"  Customers using this: {'; '.join(user_strs)}")

        elif ntype == "customer":
            if nid in seen:
                continue
            seen.add(nid)
            lines.append(
                f"\nCustomer: {data.get('name','?')} | "
                f"ARR ${data.get('arr_usd','?')} | "
                f"segment: {data.get('segment','?')} | "
                f"industry: {data.get('industry','?')}"
            )
            prods = [
                str(g.nodes[dst].get("name") or dst)
                for dst in g.successors(nid)
                if g.edges[nid, dst].get("edge_type") == "uses"
            ]
            if prods:
                lines.append(f"  Uses products: {', '.join(prods)}")
            for src in g.predecessors(nid):
                if g.edges[src, nid].get("edge_type") == "manages_account":
                    pd = g.nodes[src]
                    status = pd.get("status", "?")
                    extra = ""
                    if "departed" in status:
                        extra = f", departed ({pd.get('departure_type','?')})"
                    lines.append(f"  CSM: {pd.get('name',src)} — status: {status}{extra}")
                    if "departed" in status:
                        lines.append(f"    Departure notes: {pd.get('departure_notes','')}")

        elif ntype == "vendor":
            if nid in seen:
                continue
            seen.add(nid)
            lines.append(
                f"\nVendor: {data.get('name','?')} | "
                f"${data.get('annual_value_usd','?')}/yr | "
                f"category: {data.get('category','?')} | "
                f"renewal: {data.get('renewal_date','?')}"
            )
            # Contract owner + their manager (covers Q14 fully)
            for src in g.predecessors(nid):
                if g.edges[src, nid].get("edge_type") == "owns_contract":
                    _add_person(src)

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
