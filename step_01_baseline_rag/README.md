# Step 01 — Baseline Vector RAG

## Goal
Build the simplest possible working RAG system. Measure it against 22 golden questions spanning five failure-mode categories. Establish the floor that later steps will improve upon.

---

## Design Decisions

### Chunking — Paragraph-aware, ~2000 chars (~512 tokens)
Split on blank lines, accumulate paragraphs until ~2000 chars, carry 200-char overlap.

**Why this is naive**: chunk boundaries ignore semantic meaning — a critical sentence can be split across chunks, or irrelevant paragraphs lumped together.

**CSV exception**: Each row becomes its own chunk rendered as `key: value | key: value`. CSV rows are independent facts, not continuous prose.

### Embedding — `gemini-embedding-2` (3072-dim)
One call per chunk. The SDK's `embed_content` doesn't support true batching.

### Vector Store — ChromaDB (local persistent, cosine similarity)
Zero infra, persists to disk. 309 chunks from 48 files.

### Retrieval — Top-5 cosine similarity, no filtering
No metadata filtering, no query rewriting, no reranking.

### LLM — `gemini-2.5-flash-preview-04-17`
Temperature 0.0 for deterministic, reproducible evaluation.

---

## Evaluation Results (2025-05-15)

**Final score: 10 PASS / 11 PARTIAL / 1 FAIL — 45% pass rate**  
Corpus: 22 questions, 309 chunks, 48 source files

| ID  | Type | Expected | Got | Notes |
|-----|------|----------|-----|-------|
| Q01 | simple_lookup | PASS | **PASS** | Retention policy in single doc |
| Q02 | simple_aggregation | PASS | **PASS** | "12" DataCraft employees found directly |
| Q03 | simple_lookup | PASS | **PASS** | CEO + co-founder in founding_story |
| Q04 | simple_lookup | PASS | **PASS** | Founded 2019 |
| Q05 | simple_lookup | PASS | **PASS** | Pulsar in architecture doc |
| Q06 | simple_lookup | PASS | **PASS** | 99.9% SLO in architecture doc |
| Q07 | csv_full_aggregation | FAIL | **PARTIAL** | Sums 5/20 rows → $4.9M vs correct $11M |
| Q08 | csv_full_aggregation | FAIL | **PARTIAL** | Sums 5/15 vendor rows → $135K vs correct $956K |
| Q09 | csv_full_scan_filter | FAIL | **PARTIAL** | Finds Ravi+Noah; misses Emma Fischer and Aleksander Nowak |
| Q10 | csv_arithmetic_aggregation | FAIL | **FAIL** | Declares Executive dept ($167K/head) winner; correct is Platform Eng ($195K/head) |
| Q11 | csv_full_aggregation | FAIL | **PARTIAL** | Retrieves report total but can't enumerate all 8 deals |
| Q12 | csv_arithmetic_cross_doc | FAIL | **PARTIAL** | References $16.5M ARR from prose doc; can't compute 65% from CSV |
| Q13 | cross_csv_multi_hop | FAIL | **PARTIAL** | Finds Adrian Blake + E010; can't map E010 → Priya Nair |
| Q14 | cross_csv_multi_hop | FAIL | **PARTIAL** | Finds Priya Nair as owner; can't find Marcus Webb as manager |
| Q15 | csv_arithmetic_full_scan | FAIL | **PARTIAL** | Gets numerator ($18M) but can't compute total budget denominator |
| Q16 | csv_date_filter_aggregation | FAIL | **PARTIAL** | Finds 2 of 11 H2 2023 customers; sums $720K vs correct $3.12M |
| Q17 | stale_reference | PARTIAL | **PASS** | csm_account_history.csv surfaces both Preet Kaur and her departure |
| Q18 | disambiguation_no_name | PARTIAL | **PARTIAL** | Finds Python migration + completed; misses "signed" for Phoenix Corp deal |
| Q19 | sla_breach_inference | PARTIAL | **PASS** | Correctly computes 99.4% uptime and states breach |
| Q20 | multi_format_multi_hop | PARTIAL | **PARTIAL** | Can't connect outage → InsightLens dependency → revenue |
| Q21 | multi_hop_implicit | PARTIAL | **PASS** | Retrieves correct Aug 28-Sep 3 row; correctly names Priya Nair |
| Q22 | blast_radius_multi_hop | PARTIAL | **PASS** | Finds InsightLens, events_api, enterprise in retrieved context |

---

## Structural Failure Modes

### CSV Full-Table Aggregation (Q07, Q08, Q11, Q16) — all PARTIAL
Every aggregation question retrieves only 5 rows from CSVs with 15–20 rows, producing a confident but wrong partial sum. The model doesn't hallucinate — it just sums whatever rows appear in context.

- Q07: 5 of 20 customer rows → $4.9M (correct: $11M)
- Q08: 5 of 15 vendor rows → $135K (correct: $956K)
- Q16: 2 of 11 H2-2023 rows → $720K (correct: $3.12M)

**Fix path**: Step 07 (structured SQL-like query tool: `SUM arr_usd WHERE ...`).

### CSV Arithmetic Requiring Full Scan (Q10, Q12, Q15) — FAIL/PARTIAL
Requires dividing or comparing values across all rows. Without seeing all rows, the model either picks the wrong winner (Q10) or correctly identifies it can't compute the denominator (Q12, Q15).

- Q10: Sees Executive dept ($167K/head from 5 rows) and declares it winner; misses Platform Engineering ($195K/head) which appears later in the CSV.

**Fix path**: Step 07 (structured query with computed column + ORDER BY).

### Cross-CSV Multi-Hop with ID Joins (Q13, Q14) — PARTIAL
First hop retrieves the intermediate entity. Second hop requires matching a numeric ID (E010, E009) across two unrelated CSV rows — no semantic signal to guide vector retrieval.

- Q13: Retrieves offboarding CSV → Adrian Blake (E029) → manager_id=E010. Can't map E010 → Priya Nair.
- Q14: Retrieves vendor CSV → Priya Nair. Can't map her employee_id → Marcus Webb.

**Fix path**: Step 05 (Graph RAG: `Person→reports_to` edge traversal).

### Multi-Hop Inference Chains (Q18, Q20) — PARTIAL
- Q18: Query never contains "Phoenix" — model must recognize the ambiguity unprompted, then enumerate both meanings and state outcomes for each. Gets the engineering migration but misses the sales deal "signed" fact.
- Q20: Chain requires: postmortem → API dependency CSV → revenue CSV. The middle hop (api_dependencies) never surfaces for the query "August 2023 outage products revenue."

**Fix path**: Step 05 + Step 08 (agent: multi-query strategy for disambiguation and dependency traversal).

---

## Where the Model Outperformed Predictions

### Q17 — stale_reference (expected PARTIAL → PASS)
The new `csm_account_history.csv` directly records the Preet Kaur → Sam Rivera transition, making the cross-file join unnecessary. The model retrieved both the customer list and the history file and connected both required facts.

### Q19, Q21 — arithmetic and on-call lookup (expected PARTIAL → PASS)
The model correctly computed 263 min / 44640 min = 99.4% uptime and correctly identified the Aug 28–Sep 3 on-call row as Priya Nair. Strong retrieval + zero-temperature generation produced correct answers for these intended traps.

### Q22 — blast_radius (expected PARTIAL → PASS)
The architecture doc and postmortem together contain enough signal to name InsightLens, events_api, and enterprise as affected. The required facts landed in top-5 via strong cosine similarity on "message queue infrastructure" queries.

---

## Latency Profile

| Metric | Value |
|--------|-------|
| Avg retrieval latency | ~580ms (1 embedding API call per query) |
| Avg generation latency | ~1,200ms |
| Index build time | ~90s (309 chunks) |
| Index size on disk | ~7MB (ChromaDB) |

---

## Improvement Map for Later Steps

| Questions | Fixed by |
|-----------|----------|
| Q07, Q08, Q09, Q11, Q12, Q15, Q16 | Step 07 — Structured CSV query tool |
| Q13, Q14, Q20, Q22 | Step 05 — Graph RAG (entity relationship traversal) |
| Q10 | Step 07 — Structured query with computed column + ORDER BY |
| Q18 | Step 08 — Agentic multi-query (disambiguation) |
| Q19, Q21 | Step 10 — Context engineering (already near-PASS at baseline) |

---

## What Naive RAG Is and Isn't Good At

**Good at**: Single-document lookup, explicit facts that appear verbatim, arithmetic when both operands land in the same top-5 result.

**Fragile at**: Full-table CSV aggregation (rows > top-k=5), cross-document ID-based joins with no semantic overlap, multi-hop chains where the connecting entity is implicit in the query.

**Counter-intuitive finding**: A strong LLM compensates for retrieval gaps through reasoning. The 45% baseline reflects *both* the retrieval system and the model's capability. The remaining 55% (11 PARTIAL + 1 FAIL) represents genuine structural retrieval limits that reasoning alone cannot overcome.

---

## Next Step

→ **Step 02 — Observability**: Before we improve the system, instrument it. Every query should produce a trace showing exactly what was retrieved, what context was sent, what the model said, and what it cost.
