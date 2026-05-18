from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step_02_tools.implementation.csv_tool import detect_intent, run_query
from step_05_multi_agent.implementation.agents.contracts import CSVResult


def _fallback_intent(question: str) -> str | None:
    q = question.lower()
    # Parametric: "combined ARR of customers under <Manager>'s active direct reports"
    # Lets the structured-data agent recognize the multi-CSV-join pattern even
    # when detect_intent's regex misfires.
    if "combined" in q and "arr" in q and "direct report" in q:
        import re as _re
        m = _re.search(r"direct\s+reports?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", question)
        if m:
            return f"arr_by_manager_reports:{m.group(1).strip()}"
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
