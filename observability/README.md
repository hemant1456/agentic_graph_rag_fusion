# Observability — Traces for RAG Queries

A utility folder (not a numbered step). Originally Step 02 in the 12-step plan; demoted to a utility on 2026-05-17 since it measures the pipeline rather than improving it.

## Goal
Before adding complexity to a RAG pipeline, instrument it. Every query produces a structured trace that answers: what was retrieved, what went into the LLM's context, how many tokens were used, what did it cost, and how long did each phase take.

---

## Design Decisions

### Custom JSONL trace store — no external server required
Traces are written as append-only JSONL (one JSON object per line). This is the
primary storage — no dependencies beyond the standard library, readable with any
text tool, easy to grep or feed into a notebook.

**Why not Arize Phoenix immediately?** Phoenix needs a running server and either
auto-instrumentation of the LLM SDK or manual OpenTelemetry span construction.
The JSONL store teaches the same concepts (what a trace contains, what a span is)
without the setup friction. The `arize-phoenix` extras are declared in
`pyproject.toml` for the follow-up migration.

### TracedRAG wraps BaselineRAG — no changes to Step 01
The traced pipeline imports Step 01's retrieval and generation logic directly.
`TracedRAG` intercepts the calls, records timing and token counts, builds the
`QueryTrace`, and returns `(RAGResult, QueryTrace)`. Step 01 is unchanged.

### Token counts from the API response — not estimated
Both Gemini (`response.usage_metadata`) and Anthropic (`message.usage`) return
actual token counts. `generate_traced.py` captures these from the response objects,
so the cost estimates are based on real consumption rather than character proxies.

### Cost model
| Provider | Input ($/1M tokens) | Output ($/1M tokens) |
|---|---|---|
| Gemini Flash | $0.075 | $0.30 |
| Anthropic Haiku | $0.080 | $0.40 |

---

## Evaluation Results (2025-05-15)

**Score: 8 PASS / 13 PARTIAL / 1 FAIL — 36% pass rate** (identical to Step 01 baseline)

The traces confirm the retrieval is unchanged. Observability adds zero latency overhead
beyond the time to write one line to a JSONL file.

### Per-query cost/latency profile

| Metric | Value |
|---|---|
| Total tokens (22 queries) | 34,875 |
| Avg tokens per query | 1,585 |
| Total cost (22 queries) | $0.00318 |
| Avg cost per query | $0.000144 |
| Avg total latency | ~1,970ms |
| Avg retrieval latency | ~580ms |
| Avg generation latency | ~1,200ms |

### What the traces reveal

**Anchor questions (Q01–Q06)** use 2,000–2,600 tokens — the context is rich,
multiple documents retrieved, model synthesises well. Higher token count = higher
similarity scores pulled from diverse sources.

**CSV aggregation questions (Q07, Q08, Q16)** use only 550–700 tokens — all 5
retrieved chunks come from the same CSV file (customer_list or vendor_contracts).
The model sees 5 rows of a 15–20 row table and computes an incomplete sum. Low token
count is a signal that the context is homogeneous.

**Cross-CSV multi-hop questions (Q13, Q14)** show the first hop is retrieved
correctly (offboarding, vendor CSV) but the second-hop files (employee_directory)
don't appear in top-5. The answer correctly names the intermediate entity but can't
complete the chain.

**The single FAIL (Q10)** uses only 717 tokens — 5 rows of the budget CSV, and the
model confidently names the wrong winner ($167K/head Executive vs. correct $195K/head
Platform Engineering) because it never sees the Platform Engineering row.

---

## Usage

```bash
# Run traced evaluation (re-runs all 22 golden questions with full trace capture)
uv run python observability/evaluation/run_traced_eval.py

# Summary table: latency, tokens, cost for every trace
uv run python observability/implementation/report.py --summary

# Full detail for one trace
uv run python observability/implementation/report.py --trace-id <id>

# Retrieval-only view — what drove this answer?
uv run python observability/implementation/report.py --sources <id>

# Run unit tests
uv run pytest observability/tests/ -v -k "not integration"

# Run integration test (requires built index + API keys)
uv run pytest observability/tests/ -v -m integration
```

---

## Architecture

```
observability/
  implementation/
    tracer.py           QueryTrace, RetrievalSpan, GenerationSpan dataclasses
                        TraceStore (JSONL), estimate_cost()
    generate_traced.py  generate_answer variant that captures token usage from API
    traced_pipeline.py  TracedRAG wraps BaselineRAG, emits one trace per query
    report.py           CLI: --summary | --trace-id | --sources
  evaluation/
    run_traced_eval.py  Re-runs 22 golden questions, writes traces + eval_report.json
  tests/
    test_tracer.py      21 unit tests + 1 integration test
  results/
    traces.jsonl        Append-only trace log (one JSON per line)
    eval_report.json    Scored results with per-query token/cost/latency
```

---

## What a Trace Contains

```json
{
  "trace_id": "47808ad3",
  "timestamp": "2026-05-15T16:36:05",
  "step": "01_baseline_rag",
  "query": "What is Vertexia's customer data retention policy?",
  "answer": "Hot Storage: 90 days. Cold Storage: S3 Glacier (1 year).",
  "retrieval": {
    "duration_ms": 673,
    "chunks": [
      {
        "rank": 1,
        "source": "legal/phoenix_corp_msa.txt",
        "department": "legal",
        "similarity": 0.877,
        "char_count": 1842,
        "text_preview": "..."
      }
    ]
  },
  "generation": {
    "duration_ms": 1212,
    "provider": "gemini",
    "model": "gemini-3.1-flash-lite-preview",
    "context_chars": 8658,
    "context_chunk_count": 5,
    "prompt_tokens": 2039,
    "completion_tokens": 38,
    "estimated_cost_usd": 0.000164
  },
  "total_latency_ms": 1885
}
```

---

## Key Findings from Traces

**Low token count is a retrieval quality signal.** Queries that produce <800 tokens
consistently fail — it means 5 rows from one CSV dominates the context, leaving no
room for corroborating or complementary documents.

**Generation is the dominant cost driver.** The prompt dwarfs the completion
(avg 1,500 prompt vs. 100 completion tokens). Context compression in Step 10 will
have outsized ROI.

**Retrieval from prose documents costs ~3× more than CSV retrieval.** Prose chunks
are larger (1,000–2,000 chars) vs. CSV row chunks (100–300 chars), so 5 prose chunks
fill the context window much more than 5 CSV chunks.

---

## Next Step

→ **Step 03 — Evaluation Framework**: Now that every query is traced, build automated
metrics. RAGAS dimensions: faithfulness, answer relevance, context precision, context
recall. Lock in baseline scores per dimension — not just PASS/PARTIAL/FAIL.
