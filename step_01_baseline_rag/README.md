# Step 01 — Baseline Vector RAG

## Goal
Build the simplest possible working RAG system. Measure it against our 10 golden questions. Establish the floor.

---

## Design Decisions

### Chunking — Paragraph-aware, ~2000 chars (~512 tokens)
Split on blank lines, accumulate paragraphs until ~2000 chars, carry 200-char overlap.

**Why this is naive**: chunk boundaries ignore semantic meaning — a critical sentence can be split across chunks, or irrelevant paragraphs lumped together.

**CSV exception**: Each row becomes its own chunk rendered as `key: value | key: value`. This is the right call even for a naive baseline — CSV rows are independent facts, not continuous prose.

### Embedding — `gemini-embedding-2` (3072-dim)
Free via Google API. One API call per chunk (the SDK's `embed_content` doesn't support true batching — each call returns exactly 1 embedding regardless of input list length, so we call it once per text).

**Note for Step 04+**: If embedding latency matters at scale, switch to a local model (e.g., `all-MiniLM-L6-v2`, 384-dim) — no API calls, runs on CPU in milliseconds.

### Vector Store — ChromaDB (local persistent, cosine similarity)
Zero infra, persists to disk. For a 190-chunk corpus this is fine. We'll need something more scalable (Qdrant, Weaviate) when we're at 10k+ chunks.

### Retrieval — Top-5 cosine similarity, no filtering
The dumbest possible retrieval. No metadata filtering, no query rewriting, no reranking.

### LLM — `gemini-3.1-flash-lite-preview` with Claude fallback
Temperature 0.0 for deterministic, reproducible evaluation results.

---

## Evaluation Results (2025-05-15)

| ID  | Type | Question (abbreviated) | Grade | Expected | Sources Retrieved |
|-----|------|------------------------|-------|----------|-------------------|
| Q01 | simple_lookup | Data retention policy? | **PASS** | PASS | onboarding_handbook ✓ |
| Q02 | comparative | Q3 vs Q2 NexusFlow revenue? | **PARTIAL** | PARTIAL | Got Q3 ($1.6M), missed Q2 ($1.4M) |
| Q03 | multi_hop | On-call engineer Aug 2023? | **PARTIAL** | FAIL | Retrieved schedule CSV but model didn't synthesize Kenji Ito |
| Q04 | temporal | Sarah Chen's title Jan 2023? | **PASS** | PARTIAL | Q1 org chart retrieved correctly |
| Q05 | disambiguation | What is Project Phoenix? | **PARTIAL** | PARTIAL | Retrieved engineering docs + sales deal, but answered only one |
| Q06 | implicit_link | Phoenix Corp SLA > NexusFlow target? | **PASS** | FAIL | Both legal and engineering docs retrieved in top-5 |
| Q07 | contradictory | Q3 2023 total revenue? | **PASS** | PARTIAL | Retrieved both finance report and all-hands notes — surfaced both figures |
| Q08 | aggregation | DataCraft employees count? | **PASS** | PASS | Integration memo retrieved, "12" found |
| Q09 | stale_reference | VP of Customer Success? | **PASS** | PASS | Org chart retrieved, Maya Sharma returned |
| Q10 | cross_format | Was InsightLens affected by outage? | **PASS** | FAIL | Postmortem + RFC mentioned events_api |

**Final score: 7 PASS / 3 PARTIAL / 0 FAIL — 70% pass rate**

---

## Where We Beat Our Hypotheses (Surprises)

### Q03 — PARTIAL instead of FAIL
We predicted the on-call schedule CSV would not be retrieved. It *was* retrieved (top source). But the model still didn't extract Kenji Ito's name from the CSV context. The failure mode shifted: **retrieval was fine, generation/synthesis failed**. The model said "the documents don't contain info about an outage" even while showing the schedule. This is a **generation faithfulness issue**, not a retrieval issue.

**Lesson**: Bad answers aren't always caused by bad retrieval. Sometimes the model fails to synthesize across retrieved chunks.

### Q06 — PASS instead of FAIL
We predicted this would fail because it requires connecting a legal doc to an engineering doc from different departments. It passed because both `phoenix_corp_msa.txt` and `nexusflow_architecture.md` happened to land in the top-5 for the query. The key terms ("SLA", "availability", "uptime") created enough semantic overlap to pull both documents.

**Lesson**: Semantic similarity is sometimes enough for cross-document questions — when the question explicitly names the link. The graph becomes necessary when the link is *implicit* (the question doesn't contain the connecting terms).

### Q07 — PASS instead of PARTIAL
We expected RAG to return one number and hallucinate. Instead, both the finance report ($4.12M GAAP) and the all-hands notes ($4.2M bookings) were retrieved, and Gemini correctly synthesized both as different accounting methods.

**Lesson**: The model's instruction-following ("answer based only on context") combined with multiple contradictory sources can actually produce a *better* answer than a human reading one document.

### Q10 — PASS instead of FAIL
We predicted the CSV row for the `events_api` dependency would not surface. It didn't — but the `rfc_001_event_schema.md` and the postmortem *together* gave the model enough context. The RFC explicitly mentions InsightLens consuming NexusFlow's events_api.

**Lesson**: The "trap" was partially defused by the prose document (RFC) that served as a secondary source for the same fact. The CSV-only trap still holds — if the RFC hadn't mentioned it, this would have been a FAIL.

---

## Remaining Failures — Why They Matter

### Q02 — Comparative (PARTIAL)
The revenue CSV *was* retrieved, but only the September row landed in context — not the Q2 rows. The model correctly said "Q3 NexusFlow = $1.6M" but couldn't find Q2 data.

**Why**: Cosine similarity matches "Q3 NexusFlow revenue compare to Q2" to Q3 documents, not Q2 documents. The query's *subject* is Q3 so Q2 data ranks lower. Fixing this requires either: query decomposition (split into two sub-queries: Q2 and Q3), or structured retrieval (query the CSV directly by month column).

### Q05 — Disambiguation (PARTIAL)
Both "Project Phoenix" documents were retrieved, but the model answered about only the engineering migration. The sales doc was retrieved (2nd source) but under-weighted in synthesis.

**Why**: The query "What is Project Phoenix?" has strong semantic overlap with the engineering migration document (which uses the phrase "Project Phoenix" extensively). The sales document uses "Phoenix Corp" more than "Project Phoenix". The model picked the dominant retrieved signal. Fixing this requires explicit disambiguation: detect entity ambiguity → ask clarifying question, or surface both answers with confidence scores.

### Q03 — Multi-hop (PARTIAL)
The on-call schedule was retrieved but Kenji Ito wasn't extracted as the answer.

**Why**: The model was anchored to "August 2023 outage" context from the postmortem (which it also retrieved), and the postmortem says "the on-call engineer was paged" without naming them. The model appeared to weight the postmortem's vague reference over the specific CSV data, producing a confused answer that listed all on-call engineers without connecting them to the outage date.

**Fix direction**: Step 08 (Agentic RAG) — an agent can recognize it needs to cross-reference the outage date against the on-call schedule rather than letting the LLM synthesize across disconnected chunks.

---

## Latency and Cost Profile

| Metric | Value |
|--------|-------|
| Avg retrieval latency | ~630ms (1 embedding API call per query) |
| Avg generation latency | ~1,000ms |
| Avg total latency | ~1,630ms |
| Index build time | ~45s (190 chunks × 1 API call each) |
| Index size on disk | ~4MB (ChromaDB) |
| LLM calls for 10 questions | 10 (Gemini free tier) |

---

## What Naive RAG Is and Isn't Good At

**Good at**: Questions where the exact answer is in a single chunk, retrievable by semantic similarity. Simple lookup, aggregation of explicit facts, single-document reasoning.

**Fragile at**: Multi-hop chains (no mechanism to follow links), temporal disambiguation (all eras mixed together), pure CSV facts (short rows rank lower than rich prose), and questions where the connecting term isn't in the query.

**Counter-intuitive finding**: Cross-document questions sometimes work when the query explicitly names the entities in both documents. Graph RAG's advantage will emerge most clearly for *implicit* connections.

---

## Next Step

→ **Step 02 — Observability**: Before we improve the system, instrument it. Every query should produce a trace showing exactly what was retrieved, what context was sent, what the model said, and what it cost.
