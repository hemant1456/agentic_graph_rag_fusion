# Step 12 — Production Hardening

> **Problem**: Steps 09–11 work *usually*. In production a system must work *always* — transient LLM failures, repeated queries, slow responses, and degraded providers all need explicit handling.  
> **Fix**: Wrap Step 11's VSA pipeline with five independent hardening layers. Each layer is a thin module — no changes to the core retrieval or synthesis logic.

## The Five Layers

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — SEMANTIC CACHE  (semantic_cache.py)              │
│  Encode query with all-MiniLM-L6-v2                         │
│  Cosine similarity > 0.92 → instant cached answer (<10ms)   │
│  200-entry LRU, thread-safe                                 │
└──────────────────────┬──────────────────────────────────────┘
  cache miss            │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 — RETRY / BACKOFF  (retry.py)                      │
│  @with_retry(max_attempts=3, base_delay=0.5s, exp backoff)  │
│  Wraps the entire Step11 VSA dispatch call                  │
│  Handles transient gateway timeouts and rate-limit errors   │
└──────────────────────┬──────────────────────────────────────┘
  all retries failed    │  success path
                ┌───────┘
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — GRACEFUL DEGRADATION  (graceful_degradation.py)  │
│  Picks the highest-scored retrieved chunk                   │
│  Returns its first 3 sentences as an [Extractive] answer   │
│  Zero LLM calls — always returns something                  │
└─────────────────────────────────────────────────────────────┘

On success path:
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 — CONFIDENCE SCORING  (confidence.py)              │
│  key_term_overlap(question, answer) × 0.70                  │
│  + length_signal × 0.30                                     │
│  → score 0–1, label: high (≥0.65) / medium / low           │
│  low-confidence answers get an uncertainty notice appended  │
└──────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 5 — HEALTH MONITOR  (health_monitor.py)              │
│  Ring buffer of last 100 queries                            │
│  Tracks: p50/p95 latency, error rate, cache hit rate        │
│  SLO target: 95% of queries < 10 seconds                    │
│  Status: "healthy" | "degraded"                             │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
Step12Result {
  rag_result, slice_name, router_confidence,
  ce_metrics, confidence_score, confidence_label,
  from_cache, cache_stats, health_snapshot
}
```

## Production Metrics Reported

| Metric | Source |
|--------|--------|
| `from_cache` | Semantic cache hit |
| `confidence_score` / `confidence_label` | Heuristic quality score |
| `p50_latency_ms` / `p95_latency_ms` | Health monitor ring buffer |
| `slo_compliance` | % queries under 10 s |
| `cache_hit_rate` | Hits / total queries |

## Key Files

| File | What it does |
|------|-------------|
| `implementation/semantic_cache.py` | Embedding cache with LRU eviction |
| `implementation/retry.py` | `@with_retry` exponential backoff decorator |
| `implementation/confidence.py` | `score_answer(question, answer)` → score + label |
| `implementation/health_monitor.py` | `HealthMonitor.record()` + `snapshot()` |
| `implementation/graceful_degradation.py` | `extractive_fallback(question, chunks)` |
| `implementation/pipeline.py` | `Step12RAG` — decorator over `Step11RAG` |

## Run It

```bash
uv run python step_12_production/evaluation/run_eval.py
```
