from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step_03_tools.implementation.csv_tool import detect_intent, run_query
from step_07_multi_agent.implementation.agents.contracts import CSVResult


def _fallback_intent(question: str) -> str | None:
    q = question.lower()
    if "q3" in q and "revenue" in q:
        return "q3_2023_revenue"
    if "vendor" in q and ("spend" in q or "cost" in q):
        return "total_vendor_spend"
    if ("total" in q or "all" in q) and "arr" in q:
        return "total_arr"
    if ("h2" in q or "second half" in q) and "2023" in q:
        return "h2_2023_arr"
    if "q3" in q and ("deal" in q or "closed" in q):
        return "q3_closed_deals"
    if any(w in q for w in ("berlin", "bangalore", "austin", "tokyo", "singapore", "dublin")):
        return "employees_by_location"
    if ("headcount" in q or "head count" in q) and ("budget" in q or "department" in q or "planned" in q):
        return "total_headcount"
    return None


def query(question: str) -> CSVResult:
    intent = detect_intent(question) or _fallback_intent(question)
    if intent is None:
        return CSVResult(data="", success=False, intent_matched=None)
    try:
        data = run_query(intent)
        return CSVResult(data=data, success=True, intent_matched=intent)
    except Exception as exc:
        return CSVResult(data=f"[CSV error: {exc}]", success=False, intent_matched=intent)
