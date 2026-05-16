"""
Health monitor for Step 12 production hardening.

Keeps a rolling window of the last N query results and exposes p50/p95 latency,
error rate, cache hit rate, and SLO compliance (target: 95% of queries < 10 s).
"""
from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass, field

_SLO_LATENCY_MS = 10_000   # 10-second budget

@dataclass
class QueryRecord:
    timestamp: float
    latency_ms: float
    grade: str       # PASS / PARTIAL / FAIL / ERROR
    from_cache: bool
    slice_name: str
    confidence_label: str

class HealthMonitor:
    def __init__(self, window: int = 100):
        self._window = window
        self._records: deque[QueryRecord] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(
        self,
        latency_ms: float,
        grade: str,
        from_cache: bool = False,
        slice_name: str = "unknown",
        confidence_label: str = "medium",
    ) -> None:
        rec = QueryRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            grade=grade,
            from_cache=from_cache,
            slice_name=slice_name,
            confidence_label=confidence_label,
        )
        with self._lock:
            self._records.append(rec)

    def snapshot(self) -> dict:
        with self._lock:
            recs = list(self._records)
        if not recs:
            return {"status": "no_data", "total_recorded": 0}

        latencies = sorted(r.latency_ms for r in recs)
        n = len(latencies)

        def percentile(lst, p):
            idx = max(0, int(len(lst) * p / 100) - 1)
            return round(lst[idx], 1)

        grades = [r.grade for r in recs]
        pass_ct = grades.count("PASS")
        fail_ct = grades.count("FAIL") + grades.count("ERROR")
        cache_hits = sum(1 for r in recs if r.from_cache)
        slo_ok = sum(1 for r in recs if r.latency_ms < _SLO_LATENCY_MS)

        return {
            "total_recorded": n,
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "pass_rate": round(pass_ct / n, 3),
            "error_rate": round(fail_ct / n, 3),
            "cache_hit_rate": round(cache_hits / n, 3),
            "slo_compliance": round(slo_ok / n, 3),
            "slo_target_ms": _SLO_LATENCY_MS,
            "status": "healthy" if (slo_ok / n) >= 0.95 else "degraded",
        }
