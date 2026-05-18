# Step 07 — Production Hardening

## What it adds
Wraps the Step 06 Context Engineering + VSA pipeline with five independent production layers: a semantic cache that returns prior answers for paraphrased questions, retry-with-backoff around transient LLM failures, extractive fallback when retries are exhausted, confidence scoring that appends an uncertainty note for low-confidence answers, and a rolling health monitor. No changes to the core retrieval or synthesis path.

## Design
- **Class:** `Step07RAG` in `step_07_production/implementation/pipeline.py`
- **Inherits from:** composes `Step06RAG`
- **Key components:**
  - `step_07_production/implementation/semantic_cache.py` — `SemanticCache` with a configurable cosine threshold (default `0.92`)
  - `step_07_production/implementation/retry.py` — `@with_retry(max_attempts=3, base_delay=0.5)` decorator
  - `step_07_production/implementation/graceful_degradation.py` — `extractive_fallback()` returns a chunk-derived answer when retries fail
  - `step_07_production/implementation/confidence.py` — `score_answer()` produces a numeric score plus high/medium/low label
  - `step_07_production/implementation/health_monitor.py` — `HealthMonitor` with a 100-query rolling window for latency, pass rate, cache hits, and slice distribution

## How it works
On each query, the cache is checked first: a hit returns the stored answer immediately and updates the monitor. On a miss, the Step 06 Context Engineering + VSA pipeline runs inside a retry decorator. If all retries fail, `extractive_fallback()` builds an answer directly from the top retrieved chunks. The resulting answer is passed to `score_answer()`; low-confidence answers get a `[Note: low confidence — verify against source documents]` suffix. The query is recorded in the health monitor (latency, pass/fail, slice, cache status) and the answer plus engineering metrics are written into the cache for future paraphrases. The cache and monitor are module-level singletons so state persists across `build()` calls within a session.

## Run
```bash
uv run python evaluation/run_eval.py --step step_07_production
```

## Results
See `step_07_production/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
The earlier steps optimise for accuracy on a known test set. Production traffic is different: the same question gets asked in slightly different wording dozens of times a day, the LLM provider occasionally rate-limits or times out, and operators need to know when answers are unreliable. The cache cuts latency and cost on repeat queries; retry-with-backoff absorbs transient provider errors; extractive fallback guarantees the user always gets something traceable to the corpus; confidence scoring signals when to verify; the monitor surfaces regressions. Each layer is a thin module that can be enabled or replaced without touching the retrieval stack.
