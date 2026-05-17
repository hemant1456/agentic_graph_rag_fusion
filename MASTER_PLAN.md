# Agentic Graph RAG Fusion — Master Plan

> **Philosophy**: Fundamentals to periphery. Core to outer circle. Sometimes-working to always-working. Decent answer to precise answer.
>
> Every step is: **understand the floor → build → measure → understand the tradeoff → decide the next layer**.
> We never add complexity because it's cool. We add it because we can *prove* the simpler thing was insufficient.

---

## What We Are Building

A production-grade **Agentic Graph RAG Fusion** system over company-level heterogeneous data that:
- Retrieves information using **multiple strategies** (vector, graph traversal, keyword, structured query)
- **Fuses** those strategies intelligently based on query type
- Uses **agents and subagents** that coordinate via well-defined protocols and contracts
- Is **observable** at every layer (traces, spans, cost, latency, retrieval quality)
- Is **evaluatable** against a golden dataset with multiple quality dimensions
- Applies **context engineering** to maximize answer quality within token constraints
- Is designed using **Vertical Slice Architecture (VSA)** so features are self-contained and testable

---

## The Progression Model

10 numbered pipeline steps, plus 3 unnumbered utility folders that support them.

**Numbered steps — each adds one capability over the prior step:**

```
Step 01: Baseline RAG           → The floor. Paragraph chunks + MiniLM + top-5 cosine.
Step 02: Format-aware Chunking  → Markdown by section + CSV by row + contextual headers.
Step 03: CSV Tool Calling       → Pandas tool for exact aggregates (total ARR, headcount).
Step 04: BM25 Hybrid Retrieval  → BM25 + dense fused via RRF for keyword-exact queries.
Step 05: Knowledge Graph        → Entity + relationship edges from CSVs; multi-hop traversal.
Step 06: Graph RAG              → Alias resolution + BFS blast-radius queries.
Step 07: Multi-Agent System     → QueryAnalyst → specialists → Critic → Synthesis.
Step 08: Context Engineering    → CrossEncoder rerank → dedup → compress → XML budget.
Step 09: Vertical Slice Arch    → Domain slice router (Finance / HR / Engineering).
Step 10: Production Grade       → Cache + retry + confidence + health monitor.
```

**Utility folders — supporting infrastructure, not part of the progression:**

```
dataset/         Synthetic Vertexia corpus (50 files, 7 departments).
observability/   JSONL trace store + Arize Phoenix integration.
evaluation/      RAGAS framework + 15-question golden set + LLM judge.
```

Observability and evaluation were originally steps 02 and 03; they were demoted to utility folders on 2026-05-17 because they don't move pipeline accuracy — they measure it.

---

## Step Breakdown (Original 12-step design)

> **Note on numbering**: This section reflects the original plan, where observability and evaluation were numbered steps. They are now utility folders (`observability/`, `evaluation/`). Pipeline steps were renumbered into 10 capability-adding steps on 2026-05-17.
>
> Translation table from the original 12 steps to today's 10 steps + 3 utility folders:
>
> | Original | Today |
> |---|---|
> | Step 00 Foundation Dataset | `dataset/` (utility) |
> | Step 01 Baseline RAG | Step 01 (unchanged) |
> | Step 02 Observability | `observability/` (utility) |
> | Step 03 Evaluation Framework | `evaluation/` (utility, now RAGAS-based) |
> | Step 04 Parsing & Chunking | Step 02 |
> | Step 05 Knowledge Graph | Step 05 |
> | Step 06 Graph RAG | Step 06 |
> | Step 07 RAG Fusion (BM25 + dense) | Step 04 |
> | Step 08 Single Agentic RAG | dropped (redundant with multi-agent) |
> | Step 09 Multi-Agent | Step 07 |
> | Step 10 Context Engineering | Step 08 |
> | Step 11 VSA | Step 09 |
> | Step 12 Production | Step 10 |
>
> Step 03 (CSV Tool Calling) is new — it didn't exist in the original plan. The design rationale in each subsection below is preserved as the *why* behind that capability; cross-reference the table above when reading.

### STEP 00 — Foundation Dataset
**Goal**: Create a rich, realistic synthetic company corpus that will stress-test every retrieval strategy we build.

**Output**: `dataset/company_data/` — 50+ documents across 7+ formats, engineered with deliberate "traps" (temporal ambiguity, entity collision, multi-hop chains, contradictory facts, implicit relationships).

**Why it matters**: Garbage data → garbage learning. The dataset is the experiment. Every later step will either pass or fail against these documents. If the data is too clean, we learn nothing.

**Done when**: We can articulate 10 specific questions that naive RAG will *fail* on, and explain *why* it will fail on each.

---

### STEP 01 — Baseline Vector RAG
**Goal**: Build the simplest possible working RAG system. Chunk naively, embed, store, retrieve top-k, generate.

**Stack choices to make**:
- Embedding model: OpenAI `text-embedding-3-small` vs. local `all-MiniLM` — understand the tradeoff (cost vs. quality vs. latency)
- Vector store: ChromaDB (persistent, local) — no cloud overhead yet
- Chunking: Fixed 512-token chunks with 50-token overlap — intentionally naive
- LLM: Claude via Anthropic SDK

**Questions to answer**:
- What is the retrieval precision@k on our golden questions?
- Where does it fail? Why?
- What types of questions is cosine similarity fundamentally incapable of answering?

**Done when**: Baseline accuracy score established and failure modes documented.

---

### STEP 02 — Observability Foundation
**Goal**: Before adding any complexity, instrument everything. Every LLM call, every retrieval, every token consumed must be traceable.

**Stack choices to make**:
- Tracing: Arize Phoenix (open source, local) vs. LangSmith (hosted)
- What to capture: query, retrieved chunks, context sent to LLM, response, latency, token count, cost estimate

**Key concepts**:
- Spans and traces in the context of RAG pipelines
- What "retrieval quality" means observationally (not just output quality)
- The difference between *what the model said* and *what was in context*

**Done when**: Every query produces a trace we can inspect. We can answer "what chunks drove that answer?" for any response.

---

### STEP 03 — Evaluation Framework
**Goal**: Build a golden dataset of 30+ question-answer pairs across different difficulty levels. Establish automated metrics.

**Evaluation dimensions**:
- **Faithfulness**: Is the answer grounded in the retrieved context? (No hallucination)
- **Answer Relevance**: Does the answer actually address the question?
- **Context Precision**: Of the retrieved chunks, how many were actually needed?
- **Context Recall**: Did we retrieve all chunks needed to answer correctly?
- **Multi-hop Success Rate**: Specific metric for questions requiring chained reasoning

**Question taxonomy**:
```
Type 1 — Simple Lookup:       "What is the company's data retention policy?"
Type 2 — Comparative:         "How did Q3 2023 revenue compare to Q3 2022?"
Type 3 — Multi-hop:           "Who led the team responsible for the August outage?"
Type 4 — Aggregation:         "Which department had the highest headcount growth in 2023?"
Type 5 — Temporal:            "What was Sarah Chen's title before the restructuring?"
Type 6 — Implicit:            "Which product SLA was at risk after the Phoenix deal signed?"
Type 7 — Contradictory data:  "What was Q3 revenue?" (two sources give different numbers)
```

**Done when**: RAGAS (or equivalent) runs automatically on every system change. Baseline scores locked in.

---

### STEP 04 — Parsing and Chunking Strategies
**Goal**: Replace naive fixed chunking with format-aware, metadata-preserving, semantically-coherent chunking.

**Concepts to learn**:
- **Semantic chunking**: Split on meaning boundaries, not token counts
- **Hierarchical chunking**: Parent-child chunks (small retrieval, large context)
- **Format-aware parsing**: CSV → structured data, DOCX → sections, tables extracted separately
- **Metadata enrichment**: Document type, date, author, department, entities → attached to every chunk
- **Late chunking**: Chunk after embedding (ColBERT-style) vs. chunk before embedding

**Questions to answer**:
- Does better chunking improve our evaluation scores? By how much?
- What chunk size maximizes precision@3?
- What metadata filters have the highest query-time signal?

**Done when**: Evaluation scores improve measurably over Step 01 baseline, and we understand *why* each chunking decision helped.

---

### STEP 05 — Knowledge Graph Construction
**Goal**: Extract a rich knowledge graph from the corpus: entities (people, projects, teams, products, events) and relationships between them.

**Key design decisions**:
- **Extraction method**: LLM-based extraction vs. NER models vs. rule-based
- **Graph schema**: What node types? What edge types? What properties?
- **Storage**: Start with NetworkX (in-memory, no infra) → migrate to Neo4j when scale demands it
- **Entity resolution**: "Sarah Chen" == "S. Chen" == "the CTO" — how do we deduplicate?

**The graph schema for Vertexia Inc.**:
```
Nodes: Person, Team, Department, Product, Project, Document, Event, Customer, Vendor
Edges: REPORTS_TO, WORKS_ON, OWNS, REFERENCES, CAUSED, SUPERSEDES, PARTICIPATED_IN, LEADS
```

**Done when**: Graph is queryable via Cypher/NetworkX. We can answer "show me all people connected to the August 2023 outage within 2 hops."

---

### STEP 06 — Graph RAG
**Goal**: Augment vector retrieval with graph traversal. Understand when the graph wins, when it loses, and why.

**Retrieval strategies to compare**:
- Pure vector (Step 01 baseline)
- Pure graph traversal (BFS/DFS from entity mentions in query)
- Vector + graph (union the results, rank by relevance)
- Graph-guided vector (use graph to identify relevant node neighborhoods, then do vector search within them)

**Expected wins for graph**: Multi-hop questions, relationship questions, "who/what is connected to X" questions
**Expected losses for graph**: Semantic similarity questions, questions where the answer is in a chunk with no graph anchor

**Done when**: Evaluation scores show clear improvement on multi-hop questions. We have a routing heuristic for when to use which strategy.

---

### STEP 07 — RAG Fusion
**Goal**: Combine all retrieval strategies and fuse their results intelligently.

**Strategies to fuse**:
1. Dense vector retrieval (ChromaDB)
2. Graph traversal retrieval (Step 06)
3. Sparse keyword retrieval (BM25)
4. Structured query retrieval (for CSV/tabular data: convert to SQL/Pandas query)

**Fusion methods**:
- **Reciprocal Rank Fusion (RRF)**: Simple, no learned parameters, often surprisingly effective
- **Learned fusion**: Train a small reranker to weight strategies by query type
- **Query decomposition**: Break compound questions into sub-questions, retrieve for each

**The "fusion insight"**: Why is fusion better than picking the best single strategy? Because different queries have different optimal strategies, and *we don't always know which at query time*.

**Done when**: Fusion system outperforms best single strategy on our full evaluation set by a measurable margin.

---

### STEP 08 — Single Agentic RAG
**Goal**: Replace the fixed retrieval pipeline with a reasoning agent that *decides* what to retrieve, how, and whether to retrieve again.

**Agent design**:
- **Architecture**: ReAct (Reason + Act) loop
- **Tools available**:
  - `vector_search(query, k, filters)` — dense retrieval
  - `graph_query(entity, relationship_types, hops)` — graph traversal
  - `structured_query(table, filters, aggregation)` — CSV/tabular data
  - `keyword_search(terms, doc_type_filter)` — BM25
  - `document_fetch(doc_id)` — full document retrieval
  - `clarify_question(ambiguity)` — ask for clarification (for ambiguous queries)

**Key concepts**:
- Tool schemas and contracts: what the agent can expect from each tool
- Thought-action-observation loop
- When to stop retrieving (diminishing returns)
- Hallucination vs. "I don't know" boundary

**Done when**: Agent consistently uses the right tool for each question type. Scores exceed Step 07 on ambiguous/complex queries.

---

### STEP 09 — Multi-Agent Architecture
**Goal**: Decompose the monolithic agent into specialized subagents with a coordinating orchestrator.

**Agent topology**:
```
Orchestrator Agent
├── Query Analyst Agent         (classifies query type, decomposes if needed)
├── Retrieval Specialist Agent  (decides retrieval strategy, executes fusion)
├── Graph Navigator Agent       (graph traversal, entity resolution)
├── Structured Data Agent       (handles CSV/tabular queries, SQL generation)
├── Synthesis Agent             (assembles final answer from sub-answers)
└── Critic Agent                (checks answer for faithfulness before returning)
```

**Contracts between agents**:
- What format does each agent accept as input?
- What does it guarantee about its output? (schema, null handling, uncertainty signaling)
- How does the orchestrator route? (static routing vs. dynamic LLM-based routing)
- How do agents signal failure vs. uncertainty vs. "data not found"?

**Protocols**:
- Agent-to-agent message format
- Context passing (what context travels with the task?)
- Error propagation
- Timeout and fallback behavior

**Done when**: Each agent is independently testable. The orchestrator can route correctly 95%+ of the time. The system degrades gracefully when one subagent fails.

---

### STEP 10 — Context Engineering
**Goal**: Maximize answer quality by being surgical about what goes into the LLM's context window.

**Techniques**:
- **Reranking**: Cross-encoder reranker to select top-k from larger retrieved set
- **Context compression**: LLMLingua or similar — compress retrieved text without losing key facts
- **Structured context**: Format retrieved chunks as structured context (XML tags, role labels) vs. raw text
- **Context deduplication**: Remove redundant chunks before sending
- **Instruction placement**: System prompt engineering, where to put retrieved context, how to format citations
- **Prompt chaining**: When a single context window isn't enough, chain multiple calls

**The core tradeoff**: More context → more recall, more cost, more noise, slower. Less context → faster, cheaper, risk of missing key facts.

**Done when**: Context engineering improvements are measurable in evaluation scores and cost/latency metrics.

---

### STEP 11 — Vertical Slice Architecture (VSA) Refactor
**Goal**: Restructure the entire codebase so that each "feature" (e.g., "answer a multi-hop HR question") owns its full vertical stack — from data access to agent logic to evaluation.

**What VSA means here**:
- Instead of organizing by *technical layer* (retrieval/, agents/, evaluation/)...
- Organize by *capability slice* (hr_queries/, finance_queries/, incident_analysis/)
- Each slice has its own: retrieval strategy, agent config, evaluation suite, observability hooks
- Slices share infrastructure (vector DB, graph DB) but not logic

**Why**: As the system grows, horizontal layers become monolithic. A change to "how we chunk HR documents" should not require touching the finance query slice. VSA gives each slice autonomy.

**Done when**: Adding a new query slice requires touching only one directory. Slices can be enabled/disabled independently.

---

### STEP 12 — Production Hardening
**Goal**: The system works *always*, not *usually*. Every failure mode is handled.

**Areas**:
- **Reliability**: Retries, fallbacks, circuit breakers, graceful degradation
- **Latency budgets**: Define SLOs. Streaming responses. Parallel retrieval.
- **Cost control**: Token budgets per query tier. Caching (semantic cache for repeated queries).
- **Always-right guardrails**: Confidence scoring, uncertainty propagation, "I don't know" threshold
- **Freshness**: How does the system handle new documents added to the corpus?

**Done when**: System can handle 100 diverse queries with <5% hard failures, latency SLO met, cost per query within budget.

---

## Technology Stack (Evolving)

| Concern | Start Simple | Graduate To |
|---|---|---|
| LLM | Claude (Anthropic SDK) | Claude + prompt caching |
| Embeddings | OpenAI text-embedding-3-small | Local model when cost matters |
| Vector Store | ChromaDB local | Qdrant (production) |
| Graph Store | NetworkX (in-memory) | Neo4j (when persistence matters) |
| Observability | Print + structured logs | Arize Phoenix (open source) |
| Evaluation | Manual + RAGAS | Custom eval harness |
| Orchestration | Pure Python | LangGraph (when state machines needed) |
| Parsing | PyMuPDF + python-docx | Unstructured.io |
| Reranking | CrossEncoder (sentence-transformers) | Cohere Rerank API |

---

## Evaluation Scorecard

**Framework**: RAGAS via `llm_gatewayV2` (groq llama-3.3-70b → gemini fallback). Five metrics per question: `answer_correctness`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. See `evaluation/README.md`.

**Golden set**: 15 questions across 6 tiers, each tier requires the next capability to PASS:

| Tier | IDs | Type | Step that unlocks it |
|---|---|---|---|
| 1 | Q01–Q02 | Simple retrieval | Step 01 baseline |
| 2 | Q03–Q04 | Format-aware chunking | Step 02 |
| 3 | Q05–Q07 | CSV aggregates | Step 03 |
| 4 | Q08–Q10 | BM25 keyword-exact | Step 04 |
| 5 | Q11–Q13 | Knowledge-graph multi-hop | Step 05 |
| 6 | Q14–Q15 | Cross-document reasoning | Step 07 |

**Grading**: `answer_correctness ≥ 0.7` = PASS, `≥ 0.4` = PARTIAL, else FAIL.

**Live results** are written to `<step>/results/eval_results.json` each time the eval runs. See the top of the main `README.md` for the current per-step summary table.

### What the diagnostic metrics tell us

The four non-correctness metrics exist to localize *why* a question fails:

- Low `context_recall` + low `answer_correctness` → retrieval missed the document. Fix by adding the right retrieval strategy in the next step.
- High `context_recall` + low `answer_correctness` → retrieval found it; generation failed. Fix by tightening the prompt, reranker, or compression budget.
- Low `context_precision` → retrieval is noisy. Fix by adding reranking (Step 08) or domain slicing (Step 09).
- Low `faithfulness` → model is hallucinating despite having context. Fix by stricter system prompt or smaller / cleaner context.

This decomposition is what makes the step-by-step progression meaningful — each step targets one of these failure modes.

---

## Key Learning Questions — Answered

1. **When is Graph RAG better than pure vector?** Steps 05→06 showed +26pp lift (26%→85%) on entity-chain and dependency questions. Graph wins on named-entity multi-hop (who reports to whom, what depends on what). Vector wins on semantic similarity when no entity chain is needed. Not worth it for pure semantic questions.

2. **Highest ROI chunking strategy for a heterogeneous corpus?** Format-aware chunking at step 04 went from 26%→52% (+26pp) — the single largest lift. The key: treat CSVs as aggregate chunks (one chunk = entire sheet's summary), Markdown as section-split, text as paragraph-split. Format detection before chunking is the ROI driver.

3. **Agent-to-agent contracts?** `SynthesisResult`, `GraphResult`, `CSVResult` dataclasses with typed fields (answer, confidence, sources, latency_ms, status). Orchestrator only imports the contracts module, not the agent internals. Swap any agent by implementing the same dataclass output.

4. **What does context engineering actually buy?** Steps 09→10: 93%→85% — it *hurt* here. CrossEncoder reranking helps precision, but extractive compression is lossy. Net: reranking alone is worth it; compression requires tuning per query type. Lesson: measure before you compress.

5. **Evaluation system that catches regressions?** Golden 27-question suite with `required_facts` + `disqualifiers` per question catches regressions reliably. The compression regression at step 10 was instantly visible. Key: disqualifiers (wrong answers that "pass" on a reading but fail on a specific fact) are as important as required_facts.

6. **Right granularity for observability?** Per-agent traces with latency_ms + input_summary + output_summary were sufficient. Token-level tracing (Arize Phoenix) added signal for debugging but wasn't needed for pass-rate work. Verdict: per-agent traces for eval; per-token for cost optimization.

7. **When does multi-agent hurt?** Step 08 (single agentic) regressed from 89%→85% vs step 07. Tool-calling introduces LLM discretion over *when* to call the CSV tool — the LLM sometimes skips it. Lesson: for deterministic structured queries, unconditional tool invocation beats agent discretion.

8. **Contradictory information?** The multi-agent Critic (step 09) helps surface contradictions. Synthesis rule "For two things with the same name, name BOTH and state the outcome of EACH" was the key prompt pattern for disambiguation questions. Q18 (two Project Phoenix programs) remained the hardest — requires seeing *both* passages simultaneously.

9. **What does VSA buy?** Step 11 recovered 4pp from step 10 (85%→89%) using domain-specific system prompts. The keyword router added zero LLM calls for routing. VSA benefit at this scale: per-slice system prompts and compress_ratio tuning without touching orchestrator logic. Downside: keyword router can mis-route edge-case queries.

10. **When to stop retrieving?** The confidence scoring in step 10 + early cache hit is the practical answer. For retrieval depth: graph BFS with max_depth=3 + vector top-20 was sufficient for the 15-question golden set. Stopping rule: when the top-1 CrossEncoder score > 0.85, the answer is likely in that chunk — stop expanding.

---

## Project Conventions

- Each step lives in `step_XX_name/`
- Every step has: `README.md`, `implementation/`, `tests/`, `results/` (evaluation outputs)
- No step is "done" until: (a) implementation works, (b) tests pass, (c) evaluation scores logged, (d) tradeoffs documented in README
- Commit format: `step_XX: <what changed and why>`
- Code style: Python, typed, no magic globals, every function has a clear contract (input/output types)

---

## Current Status

All 10 numbered steps and 3 utility folders are implemented.

| Step | Adds | Implementation key |
|---|---|---|
| Step 01 Baseline RAG | Paragraph chunks + MiniLM + top-5 cosine | `BaselineRAG` |
| Step 02 Chunking | Section-aware Markdown / per-row CSV / contextual headers | `Step02RAG` |
| Step 03 Tools | Pandas CSV tool for aggregates | `Step03ToolsRAG` |
| Step 04 Hybrid Retrieval | BM25 fused with dense via RRF | `Step04HybridRAG` |
| Step 05 Knowledge Graph | Entity / relationship graph + multi-hop | `Step05RAG` |
| Step 06 Graph RAG | Alias resolution + BFS blast radius | `Step06RAG` |
| Step 07 Multi-Agent | QueryAnalyst → specialists → Critic → Synthesis | `Step07RAG` |
| Step 08 Context Engineering | Rerank → dedup → compress → XML budget | `Step08RAG` |
| Step 09 VSA | Domain slice router (Finance / HR / Eng) | `Step09RAG` |
| Step 10 Production | Cache + retry + confidence + health | `Step10RAG` |

| Utility | Purpose |
|---|---|
| `dataset/` | 50-file synthetic Vertexia corpus |
| `observability/` | JSONL trace store + Arize Phoenix integration |
| `evaluation/` | RAGAS-based eval runner + 15-question golden set + LLM judge |

Current per-step pass rates live in each step's `results/eval_results.json` and in the summary table at the top of the main `README.md`. The eval runner is `evaluation/run_eval.py --all`.
