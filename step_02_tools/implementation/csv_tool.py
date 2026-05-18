"""
Structured CSV query tools for Step 02.

Each function is a discrete Pandas query that computes an exact answer from raw
CSV data. The pipeline calls detect_intent() to choose which tool to run, then
injects the precise result into the LLM context alongside the retrieved chunks.

Replaces pre-baked aggregate chunks with live computation at query time — the
correct approach for aggregate questions.
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

CORPUS_PATH = Path(__file__).parent.parent.parent / "dataset" / "company_data"


@lru_cache(maxsize=32)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """Read a CSV once and cache the DataFrame. mtime is part of the lru_cache
    key (hash on parameters) so regenerating the corpus invalidates the entry
    automatically — it's intentionally unused inside the function body."""
    del mtime
    return pd.read_csv(path_str)


def _read_csv(path: Path) -> pd.DataFrame:
    """mtime-aware cached read. Returns a copy so callers can mutate freely."""
    return _read_csv_cached(str(path), path.stat().st_mtime).copy()

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Q09 — Q3 2023 total revenue
    (re.compile(r"(Q3|third.quarter|july.{0,20}august.{0,20}september).{0,60}(revenue|income)", re.I), "q3_2023_revenue"),
    (re.compile(r"(revenue|income).{0,60}(Q3|third.quarter|july.{0,20}august.{0,20}september)", re.I), "q3_2023_revenue"),
    # Q10 — employees by office location
    (re.compile(r"(how many|number of|count).{0,30}(employee|staff|people|headcount).{0,30}(berlin|bangalore|austin|tokyo|singapore|dublin|san francisco)", re.I), "employees_by_location"),
    (re.compile(r"(berlin|bangalore|austin|tokyo|singapore|dublin|san francisco).{0,30}(employee|staff|people|headcount|office|based)", re.I), "employees_by_location"),
    # Q12 — Q3 2023 closed-won deals
    (re.compile(r"(closed.won|closed deal|deals closed).{0,40}(Q3|2023-Q3|third quarter)", re.I), "q3_closed_deals"),
    (re.compile(r"(Q3|third quarter).{0,40}(closed.won|closed deal|deal.ARR|new ARR)", re.I), "q3_closed_deals"),
    # Q08 — total customer ARR
    (re.compile(r"total.{0,20}(arr|annual recurring revenue).{0,30}(all|across)", re.I), "total_arr"),
    (re.compile(r"(all|across).{0,30}(arr|annual recurring revenue).{0,20}total", re.I), "total_arr"),
    # Q11 — total planned headcount from budget allocation
    (re.compile(r"(total|planned).{0,30}headcount.{0,40}(budget|department|2023)", re.I), "total_headcount"),
    (re.compile(r"(budget|2023).{0,40}(total|planned).{0,30}headcount", re.I), "total_headcount"),
    # Q28 — H2 2023 contract ARR (cross-quarter, Tier 5)
    (re.compile(r"(H2|second half).{0,30}2023.{0,40}(arr|contract|deal)", re.I), "h2_2023_arr"),
    (re.compile(r"(arr|contract|deal).{0,40}(H2|second half).{0,30}2023", re.I), "h2_2023_arr"),
]


# Parametric intent: "combined ARR of customers under <Manager>'s active direct reports".
# Captures the manager name into the intent string so run_query can dispatch on the param.
# Inline (?i:...) makes the descriptor case-insensitive but keeps the name capture
# strictly title-case so it stops at the proper name (otherwise re.I would let
# words like "per" match [A-Z][a-z]+ and the capture would over-extend).
_ARR_BY_MANAGER_REPORTS_RE = re.compile(
    r"(?i:combined).{0,40}(?i:arr|annual recurring revenue)"
    r".{0,200}?(?i:direct\s+reports?\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
)


def detect_intent(question: str) -> str | None:
    # Parametric first — captures a manager name and embeds it in the intent string.
    m = _ARR_BY_MANAGER_REPORTS_RE.search(question)
    if m:
        return f"arr_by_manager_reports:{m.group(1).strip()}"
    for pat, intent in _PATTERNS:
        if pat.search(question):
            return intent
    return None


def run_query(intent: str) -> str:
    try:
        name, _, arg = intent.partition(":")
        if name == "q3_2023_revenue":
            return _q3_revenue()
        elif name == "q3_closed_deals":
            return _q3_closed_deals()
        elif name == "total_arr":
            return _total_arr()
        elif name == "h2_2023_arr":
            return _h2_2023_arr()
        elif name == "employees_by_location":
            return _employees_by_location()
        elif name == "total_headcount":
            return _total_headcount()
        elif name == "arr_by_manager_reports":
            return _arr_by_manager_reports(arg)
    except Exception as e:
        return f"[CSV QUERY ERROR: {e}]"
    return ""


# ── Query implementations ─────────────────────────────────────────────────────

def _q3_revenue() -> str:
    df = _read_csv(CORPUS_PATH / "finance" / "revenue_by_product_2023.csv")
    q3 = df[df["month"].isin(["2023-07", "2023-08", "2023-09"])].copy()
    total = int(q3["total_revenue"].sum())
    lines = [
        "[STRUCTURED CSV QUERY — AUTHORITATIVE: Q3 2023 total revenue]",
        "[Source: revenue_by_product_2023.csv rows for 2023-07, 2023-08, 2023-09]",
    ]
    for _, r in q3.iterrows():
        lines.append(
            f"  {r['month']}: NexusFlow ${r['nexusflow_revenue']:,} + "
            f"InsightLens ${r['insightlens_revenue']:,} + "
            f"PulseConnect ${r['pulseconnect_revenue']:,} = ${r['total_revenue']:,}"
        )
    lines.append(f"  Q3 2023 TOTAL (Jul+Aug+Sep): ${total:,}")
    return "\n".join(lines)


def _q3_closed_deals() -> str:
    df = _read_csv(CORPUS_PATH / "sales" / "deal_pipeline_q3_2023.csv")
    won = df[df["stage"] == "Closed-Won"]
    total = int(won["arr"].sum())
    lines = ["[STRUCTURED CSV QUERY: Q3 2023 Closed-Won Deals — deal_pipeline_q3_2023.csv]"]
    for _, r in won.iterrows():
        lines.append(f"  {r['deal_id']} {r['company_name']}: ${r['arr']:,} ARR ({r['products']})")
    lines.append(f"  Q3 2023 Closed-Won TOTAL: ${total:,}")
    return "\n".join(lines)


def _total_arr() -> str:
    df = _read_csv(CORPUS_PATH / "sales" / "customer_list.csv")
    total = int(df["arr_usd"].sum())
    lines = [
        f"[STRUCTURED CSV QUERY — AUTHORITATIVE: total ARR across all {len(df)} customers]",
        f"[Source: customer_list.csv — all {len(df)} rows summed]",
        f"  Total ARR: ${total:,}",
    ]
    return "\n".join(lines)


def _h2_2023_arr() -> str:
    q3 = _read_csv(CORPUS_PATH / "sales" / "deal_pipeline_q3_2023.csv")
    q4 = _read_csv(CORPUS_PATH / "sales" / "deal_pipeline_q4_2023.csv")
    won_q3 = q3[q3["stage"] == "Closed-Won"]["arr"].sum()
    won_q4 = q4[q4["stage"] == "Closed-Won"]["arr"].sum()
    total = int(won_q3 + won_q4)
    lines = [
        "[STRUCTURED CSV QUERY: H2 2023 Closed-Won ARR (Q3 + Q4)]",
        f"  Q3 2023 Closed-Won: ${int(won_q3):,}",
        f"  Q4 2023 Closed-Won: ${int(won_q4):,}",
        f"  H2 2023 TOTAL: ${total:,}",
    ]
    return "\n".join(lines)


def _employees_by_location() -> str:
    df = _read_csv(CORPUS_PATH / "hr" / "employee_directory.csv")
    active = df[df["status"] == "active"]
    counts = active.groupby("location").size().sort_values(ascending=False)
    lines = ["[STRUCTURED CSV QUERY: Active Employees by Office Location — employee_directory.csv]"]
    for loc, cnt in counts.items():
        lines.append(f"  {loc}: {cnt} active employees")
    lines.append(f"  Total active: {active.shape[0]}")
    return "\n".join(lines)


def _total_headcount() -> str:
    df = _read_csv(CORPUS_PATH / "finance" / "budget_allocation_2023.csv")
    total = int(df["headcount"].sum())
    lines = [
        f"[STRUCTURED CSV QUERY — AUTHORITATIVE: total planned headcount across all {len(df)} departments]",
        "[Source: budget_allocation_2023.csv]",
    ]
    for _, r in df.iterrows():
        lines.append(f"  {r['department']}: {r['headcount']}")
    lines.append(f"  TOTAL planned headcount: {total}")
    return "\n".join(lines)


def _arr_by_manager_reports(manager_name: str) -> str:
    """Compose two CSVs: resolve <manager_name>'s active direct reports in
    employee_directory.csv, then sum arr_usd over customer_list.csv where csm
    is in that set.

    This is the only intent that JOINS two CSV files — required for Tier 5
    questions where the multi-agent has to compose graph + structured aggregate.
    """
    emp = _read_csv(CORPUS_PATH / "hr" / "employee_directory.csv")
    cust = _read_csv(CORPUS_PATH / "sales" / "customer_list.csv")

    match = emp[emp["name"].str.strip().str.lower() == manager_name.strip().lower()]
    if match.empty:
        return f"[CSV QUERY ERROR: manager '{manager_name}' not found in employee_directory.csv]"
    manager_id = match.iloc[0]["employee_id"]

    reports = emp[(emp["manager_id"] == manager_id) & (emp["status"] == "active")]
    csms = reports["name"].tolist()
    if not csms:
        return (
            f"[STRUCTURED CSV QUERY: {manager_name} ({manager_id}) has no active direct "
            "reports in employee_directory.csv]"
        )

    matched = cust[cust["csm"].isin(csms)]
    total = int(matched["arr_usd"].sum())

    lines = [
        f"[STRUCTURED CSV QUERY — AUTHORITATIVE: combined ARR of customers whose CSM is an active direct report of {manager_name}]",
        "[Source: employee_directory.csv (active direct reports) + customer_list.csv (ARR sum)]",
        f"  Manager: {manager_name} ({manager_id})",
        f"  Active direct reports: {', '.join(csms)} ({len(csms)} total)",
        f"  Customers in their portfolios: {len(matched)}",
    ]
    for _, r in matched.iterrows():
        lines.append(f"    {r['company_name']} (CSM: {r['csm']}): ${int(r['arr_usd']):,}")
    lines.append(f"  COMBINED ARR: ${total:,}")
    return "\n".join(lines)
