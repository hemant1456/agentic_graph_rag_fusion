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

```
Level 0: Data Foundation      → What are we operating over? Rich, realistic, trappy data.
Level 1: Baseline RAG         → The floor. Dumb chunking + cosine similarity. Measure it.
Level 2: Observability        → Instrument everything before complexity grows.
Level 3: Evaluation           → Golden dataset + metrics. We can't improve what we can't measure.
Level 4: Parsing & Chunking   → Document intelligence. Format-aware, metadata-rich.
Level 5: Knowledge Graph      → Extract entities + relationships. Graph construction pipeline.
Level 6: Graph RAG            → Traverse the graph. When does it beat vector? When does it lose?
Level 7: RAG Fusion           → Combine all retrieval strategies. Reciprocal rank fusion.
Level 8: Single Agentic RAG   → ReAct agent. Tools. Decide what to retrieve and how.
Level 9: Multi-Agent System   → Orchestrator + specialized subagents. Contracts between them.
Level 10: Context Engineering → Compression, reranking, context window management.
Level 11: VSA Refactor        → Restructure the system around vertical slices.
Level 12: Production Grade    → Reliability, cost control, latency budgets, always-working.
```

---

## Step Breakdown

### STEP 00 — Foundation Dataset
**Goal**: Create a rich, realistic synthetic company corpus that will stress-test every retrieval strategy we build.

**Output**: `step_00_dataset/company_data/` — 50+ documents across 7+ formats, engineered with deliberate "traps" (temporal ambiguity, entity collision, multi-hop chains, contradictory facts, implicit relationships).

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

## Evaluation Scorecard (Tracked Per Step)

| Metric | Step 01 | Step 04 | Step 06 | Step 07 | Step 08 | Step 09 | Step 12 |
|---|---|---|---|---|---|---|---|
| Simple Lookup Accuracy | — | — | — | — | — | — | — |
| Multi-hop Success Rate | — | — | — | — | — | — | — |
| Temporal Query Accuracy | — | — | — | — | — | — | — |
| Faithfulness Score | — | — | — | — | — | — | — |
| Context Precision | — | — | — | — | — | — | — |
| Avg Latency (s) | — | — | — | — | — | — | — |
| Cost per Query ($) | — | — | — | — | — | — | — |

---

## Key Learning Questions (to answer by end of project)

1. When is Graph RAG *actually* better than pure vector retrieval, and when is it not worth the complexity?
2. What chunking strategy has the highest ROI for a heterogeneous corpus (mixed formats, mixed lengths)?
3. How do you design agent-to-agent contracts so that a subagent can be swapped without touching the orchestrator?
4. What does "context engineering" actually buy you in evaluation scores vs. just throwing more context at the model?
5. How do you build an evaluation system that catches regressions *before* they reach production?
6. What's the right granularity for observability in a multi-agent RAG system — per-agent? per-tool-call? per-token?
7. When does multi-agent architecture hurt more than it helps (coordination overhead, error propagation)?
8. How do you handle contradictory information in the corpus gracefully?
9. What does VSA buy you at this scale vs. a well-organized monolith?
10. How do you know when to stop retrieving?

---

## Project Conventions

- Each step lives in `step_XX_name/`
- Every step has: `README.md`, `implementation/`, `tests/`, `results/` (evaluation outputs)
- No step is "done" until: (a) implementation works, (b) tests pass, (c) evaluation scores logged, (d) tradeoffs documented in README
- Commit format: `step_XX: <what changed and why>`
- Code style: Python, typed, no magic globals, every function has a clear contract (input/output types)

---

## Current Status

| Step | Status | Notes |
|---|---|---|
| Step 00 | IN PROGRESS | Creating company dataset |
| Step 01–12 | NOT STARTED | — |
