import csv
import re
from pathlib import Path

import networkx as nx

INTERNAL_PRODUCTS = {"NexusFlow", "InsightLens", "PulseConnect", "DataCraft"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _extract_company_from_notes(notes: str) -> str:
    """Try to extract a company name from offboarding notes like 'Joined competitor FinDataCo'."""
    match = re.search(r"[Jj]oined\s+(?:competitor\s+)?([A-Z][A-Za-z0-9]+)", notes)
    return match.group(1) if match else ""


def build_graph(corpus_path: Path) -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()

    hr          = corpus_path / "hr"
    engineering = corpus_path / "engineering"
    sales       = corpus_path / "sales"
    finance     = corpus_path / "finance"

    for row in _read_csv(hr / "employee_directory.csv"):
        eid = row["employee_id"]
        g.add_node(
            eid,
            node_type="person",
            name=row["name"],
            department=row["department"],
            role=row["role"],
            manager_id=row.get("manager_id", ""),
            start_date=row["start_date"],
            status=row["status"],
            location=row["location"],
            departure_type="",
            departed_to="",
            departure_notes="",
            last_day="",
        )

    # reports_to edges (second pass so all nodes exist)
    for eid, data in list(g.nodes(data=True)):
        mgr = data.get("manager_id", "")
        if mgr and mgr in g:
            g.add_edge(eid, mgr, edge_type="reports_to")

    for row in _read_csv(hr / "offboarding_records_2023.csv"):
        eid = row["employee_id"]
        if eid in g:
            notes = row.get("notes", "")
            g.nodes[eid].update(
                departure_type=row.get("departure_type", ""),
                last_day=row.get("last_day", ""),
                departure_notes=notes,
                departed_to=_extract_company_from_notes(notes),
            )

    for row in _read_csv(engineering / "api_dependencies.csv"):
        consumer = row["consuming_service"]
        provider = row["providing_service"]
        for svc in (consumer, provider):
            if svc not in g:
                ntype = "product" if svc in INTERNAL_PRODUCTS else "external_service"
                g.add_node(svc, node_type=ntype, name=svc)
        if g.has_edge(consumer, provider):
            # Merge into existing edge: append endpoint names, escalate criticality
            edata = g.edges[consumer, provider]
            edata["api_endpoint"] = edata["api_endpoint"] + ", " + row["api_endpoint"]
            edata["purpose"] = edata["purpose"] + "; " + row["purpose"]
            if row["criticality"] == "critical":
                edata["criticality"] = "critical"
        else:
            g.add_edge(
                consumer, provider,
                edge_type="depends_on",
                api_endpoint=row["api_endpoint"],
                criticality=row["criticality"],
                purpose=row["purpose"],
            )

    name_to_id: dict[str, str] = {
        data["name"]: eid
        for eid, data in g.nodes(data=True)
        if data.get("node_type") == "person"
    }

    for row in _read_csv(sales / "customer_list.csv"):
        cname = row["company_name"]
        g.add_node(
            cname,
            node_type="customer",
            name=cname,
            industry=row.get("industry", ""),
            arr_usd=row.get("arr_usd", ""),
            segment=row.get("segment", ""),
            contract_start=row.get("contract_start", ""),
            csm_name=row.get("csm", ""),
        )
        for prod in row.get("products", "").split("+"):
            prod = prod.strip()
            if prod and prod in g:
                g.add_edge(cname, prod, edge_type="uses")
        csm_id = name_to_id.get(row.get("csm", ""))
        if csm_id:
            g.add_edge(csm_id, cname, edge_type="manages_account")

    for row in _read_csv(finance / "vendor_contracts_summary.csv"):
        vname = row["vendor"]
        vid = f"vendor:{vname}"
        g.add_node(
            vid,
            node_type="vendor",
            name=vname,
            category=row.get("category", ""),
            annual_value_usd=row.get("annual_value_usd", ""),
            renewal_date=row.get("renewal_date", ""),
            owner_name=row.get("owner", ""),
        )
        owner_id = name_to_id.get(row.get("owner", ""))
        if owner_id:
            g.add_edge(owner_id, vid, edge_type="owns_contract")

    return g
