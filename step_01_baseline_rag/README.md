# Step 01 — Baseline Vector RAG

## Goal
Build the simplest possible working RAG system. Measure it against 10 golden questions. Establish the floor.

---

## Design Decisions

### Chunking — Paragraph-aware, ~2000 chars (~512 tokens)
Split on blank lines, accumulate paragraphs until ~2000 chars, carry 200-char overlap.

**Why this is naive**: chunk boundaries ignore semantic meaning — a critical sentence can be split across chunks, or irrelevant paragraphs lumped together.

**CSV exception**: Each row becomes its own chunk rendered as `key: value | key: value`. CSV rows are independent facts, not continuous prose.

### Embedding — `gemini-embedding-2` (3072-dim)
Free via Google API. One call per chunk — the SDK's `embed_content` doesn't support true batching (each call returns exactly 1 embedding regardless of input list length).

### Vector Store — ChromaDB (local persistent, cosine similarity)
Zero infra, persists to disk. Fine for a 190-chunk corpus.

### Retrieval — Top-5 cosine similarity, no filtering
No metadata filtering, no query rewriting, no reranking.

### LLM — `gemini-3.1-flash-lite-preview` with Claude fallback
Temperature 0.0 for deterministic, reproducible evaluation.

---

## Evaluation Results (2025-05-15)

**Final score: 6 PASS / 4 PARTIAL / 0 FAIL — 60% pass rate**

| ID  | Type | Expected | Got | Notes |
|-----|------|----------|-----|-------|
| Q01 | simple_lookup | PASS | **PASS** | Retention policy in single doc |
| Q02 | simple_aggregation | PASS | **PASS** | "12" employees found directly |
| Q03 | csv_arithmetic | FAIL | **PASS** | Gemini retrieves both docs + reasons through 99.99−99.9=0.09 |
| Q04 | multi_hop_implicit | FAIL | **PASS** | Aug 14 CSV row retrieved; model picks Kenji Ito correctly |
| Q05 | temporal_inference | PARTIAL | **PASS** | Board meeting notes connect Sarah Chen → VP Engineering |
| Q06 | stale_reference | PARTIAL | **PARTIAL** | Finds Preet Kaur; cannot confirm she departed (cross-CSV gap) |
| Q07 | csv_aggregation | FAIL | **PARTIAL** | Retrieves only 2/5 NexusFlow customers; admits it can't total |
| Q08 | multi_format_multi_hop | FAIL | **PARTIAL** | Finds InsightLens; misses combined revenue $1.02M |
| Q09 | disambiguation_no_name | PARTIAL | **PARTIAL** | Finds Python migration + completed; misses "signed" for Phoenix Corp deal |
| Q10 | sla_breach_inference | PARTIAL | **PASS** | Retrieves postmortem + SLA; calculates 99.4% uptime correctly |

---

## Where the Model Outperformed Predictions

### Q03 — csv_arithmetic (expected FAIL → PASS)
We expected the model to retrieve both uptime numbers but skip the subtraction. Gemini surprised us by performing `99.99 − 99.9 = 0.09` and stating the gap explicitly.

**Lesson**: Strong LLMs compensate for retrieval design gaps via reasoning. Q03 would still fail for a weaker model or if either source document missed the top-5.

### Q04 — multi_hop_implicit (expected FAIL → PASS)
The on-call CSV was the *only* source retrieved (all 5 slots were CSV rows). The model correctly filtered to the Aug 14 week row and named Kenji Ito. The failure we designed for — model anchoring to "tonight" and reading the last row — didn't materialize.

**Lesson**: The trap assumed the model would confuse the query's temporal reference. Gemini's instruction following is precise enough to handle date-anchored CSV lookups when the relevant rows are retrieved.

### Q05, Q10 — temporal/arithmetic (expected PARTIAL → PASS)
Both require multi-step reasoning that we predicted would break. Gemini handled both correctly when the supporting documents landed in top-5.

**Lesson**: At 60% baseline, the genuine failures are retrieval failures (wrong chunks returned), not synthesis failures. The model is smarter than the retrieval.

---

## Genuine Failures — Why They Matter

### Q06 — stale_reference (PARTIAL)
`customer_list.csv` names Preet Kaur as CSM. `hr/offboarding_records_2023.csv` records her departure. The model finds the customer record but never retrieves the offboarding CSV — these two files share no semantic overlap in the query space.

**Fix**: Step 05 — Graph RAG. A Person node for Preet Kaur with edges to both CustomerAccount and EmploymentRecord lets us follow the relationship regardless of query semantics.

### Q07 — csv_aggregation (PARTIAL)
Only 2 of 5 NexusFlow customers appear in the retrieved context. The model correctly diagnoses incomplete data and refuses to invent a total. The sum ($3.612M) requires seeing all 5 rows.

**Fix**: Step 07 — Structured query tool. SQL-like filter on `products contains 'NexusFlow'` + sum, bypassing retrieval entirely.

### Q08 — multi_format_multi_hop (PARTIAL)
The model correctly identifies InsightLens as an indirect victim via the postmortem. But it doesn't compute the August revenue sum ($520k + $500k = $1.02M) — the revenue CSV rows don't appear in top-5 for this query.

**Fix**: Step 07 + Step 05 (graph traversal to find affected products, then structured query for their August revenue).

### Q09 — disambiguation_no_name (PARTIAL)
The query never says "Phoenix" — so cosine search has no anchor. The model retrieves engineering docs (Project Phoenix = Python migration) but not the sales deal (Phoenix Corp enterprise contract). The "signed" fact for the Phoenix Corp deal is missing.

**Fix**: Step 08 — Agentic RAG. An agent can recognize entity ambiguity, enumerate possible meanings, and run separate sub-queries for each.

---

## Latency Profile

| Metric | Value |
|--------|-------|
| Avg retrieval latency | ~580ms (1 embedding API call per query) |
| Avg generation latency | ~1,000ms |
| Index build time | ~45s (190 chunks × 1 API call each) |
| Index size on disk | ~4MB (ChromaDB) |

---

## What Naive RAG Is and Isn't Good At

**Good at**: Single-document lookup, explicit aggregation where the answer is in one chunk, arithmetic when both operands land in context.

**Fragile at**: Cross-document joins with no semantic overlap (Q06), full-table aggregation where rows don't all fit in top-k (Q07), implicit multi-hop chains where the connecting entity isn't in the query (Q09).

**Counter-intuitive finding**: A strong LLM can compensate for retrieval gaps when the required facts happen to land in top-5. The baseline score reflects *both* the retrieval system and the model's reasoning. Improvements in later steps will be most visible in the questions where retrieval structurally cannot return the right chunks.

---

## Next Step

→ **Step 02 — Observability**: Before we improve the system, instrument it. Every query should produce a trace showing exactly what was retrieved, what context was sent, what the model said, and what it cost.
