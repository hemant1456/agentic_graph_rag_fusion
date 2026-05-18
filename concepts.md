# RAG to Production — A Concept Journey

> This document follows the original 12-step design plan. The project has since been consolidated to
> **7 numbered pipeline steps** (see [README.md](README.md) for the current structure):
>
> - Original Step 02 (Observability) and Step 03 (Evaluation) are now utility folders (`observability/`, `evaluation/`).
> - Original Steps 01+04 (Baseline + Format-aware Chunking) merged into today's **Step 01**.
> - Original Step 07 (RAG Fusion / BM25) became today's **Step 03**.
> - Original Step 08 (Single Agentic RAG) was dropped — multi-agent (Step 05) absorbs it.
> - Original Steps 05+06 (Knowledge Graph + Graph RAG) merged into today's **Step 04**.
> - Original Steps 10+11 (Context Engineering + VSA) merged into today's **Step 06**.
>
> Chapter numbering below still tracks the original 12 — the narrative still teaches every concept,
> just spread across a less-condensed scaffold than the current code. When you read "Step 09 Multi-Agent"
> here, that's today's `step_05_multi_agent/`. The translation table in [MASTER_PLAN.md](MASTER_PLAN.md)
> is the canonical mapping.

> Each section introduces only the concepts needed for that step — building on what came before.
> Read it in order and you'll understand not just *what* each technique does, but *why* we needed it
> and what it cost us.

---

## Prologue — The Problem Worth Solving

Imagine you join Vertexia Inc. on day one. You want to know:

- "Who does the Head of Engineering report to?"
- "What was our Q3 revenue across all products?"
- "If the NexusFlow API goes down, which services break?"

All of that information exists — buried across 48 files in 7 departments. CSV spreadsheets, Markdown wikis,
plain-text org charts, JSON configs. No single file has the full picture. Some answers require reading two
files and connecting a relationship that's never written down explicitly.

**This is the real problem.** Not "can an LLM answer questions?" — it can. The problem is getting the
*right information* to the LLM at the *right time* in a *trustworthy form*.

That's what this journey is about.

---

## Chapter 1 — The Data Foundation (Step 00)

Before we build anything, we need to understand what we're operating over.

### What is a Corpus?

A **corpus** is the collection of documents your system will search. Ours has a deliberate design:

```
company_data/
├── finance/    → CSV spreadsheets (revenue, budgets, vendor contracts)
├── hr/         → Employee directory CSV + Markdown org docs
├── engineering/ → Architecture Markdown + API dependency CSV
├── sales/      → Deal pipeline CSVs + customer lists
├── product/    → Roadmap Markdown + on-call schedules
├── legal/      → Contract TXT files
└── executive/  → Strategy Markdown
```

Seven formats, seven query styles. CSVs need aggregation (sum, count, filter). Markdown needs section
understanding. TXT files need full-text search. Each demands a different approach.

### Why We Engineered "Traps"

A corpus that's too clean teaches you nothing. We deliberately planted:

| Trap | Example | Why it's hard |
|------|---------|---------------|
| **Entity collision** | Two programs both named "Project Phoenix" | Disambiguation required |
| **Multi-hop chain** | "Who manages TechCorp?" → find CSM → find manager | No single document has the answer |
| **Temporal ambiguity** | Q3 vs Q4 revenue spans across files | Requires joining CSV rows by date |
| **Alias mismatch** | User asks "analytics dashboard" → actual name is "InsightLens" | Synonymy problem |
| **Aggregate trap** | "Total ARR across all customers" → must sum 47 rows | LLM can't add reliably from partial passages |

These traps ensure that every technique we add has a *measurable reason to exist*.

### The Golden Question Set

We created **27 questions** with exact expected answers. Each question has:
- `required_facts`: strings that MUST appear in the answer
- `disqualifiers`: wrong answers that sound plausible but are factually incorrect

This is our ruler. Every technique is measured against these 27 questions. If a new technique doesn't
move the score, it doesn't ship.

---

## Chapter 2 — The Naive Approach (Step 01)

With data in hand, the natural first instinct: throw it at an LLM.

But LLMs have a **context window** — a limit on how much text they can read at once. You can't paste
all 48 files into every query. You need to *select* the relevant parts first.

This is the core insight behind **RAG**.

### What is RAG?

**Retrieval-Augmented Generation** — fetch relevant documents, then generate an answer.

```
                    ┌─────────────────────────────────────────┐
                    │            RAG Pipeline                 │
                    └─────────────────────────────────────────┘

  Question: "Who is the Head of Engineering?"
       │
       ▼
  ┌─────────────┐     similarity     ┌─────────────────┐
  │  Embedding  │  ──────────────►  │  Vector Search  │
  │  Model      │                   │  (ChromaDB)     │
  └─────────────┘                   └────────┬────────┘
       │                                     │
       │  encode question                    │  top-5 matching chunks
       │  as vector                          │
       ▼                                     ▼
  [0.3, -0.7, 0.1, ...]          chunk_1: "Priya Sharma joined..."
                                  chunk_2: "Engineering team structure..."
                                  chunk_3: "Head of Engineering reports..."
                                       │
                                       ▼
                                ┌─────────────┐
                                │     LLM     │  ◄── answer the question
                                │  (Gemini)   │      using only these chunks
                                └─────────────┘
                                       │
                                       ▼
                            "The Head of Engineering is
                             Priya Sharma, who reports to..."
```

### Embeddings — The Core Concept

An **embedding** is a list of numbers (a vector) that represents the *meaning* of text. The key property:

```
Texts with similar meaning → vectors that point in similar directions
Texts with different meaning → vectors that point in different directions


   "Head of Engineering"  ──────────────────────►  [0.3, 0.8, -0.2, ...]
   "Engineering lead"     ──────────────────────►  [0.3, 0.7, -0.1, ...]  ← very close
   "Q3 revenue figures"   ──────────────────────►  [-0.6, 0.1,  0.9, ...]  ← far away
```

**Cosine similarity** measures how closely two vectors point in the same direction (1.0 = identical
direction, 0.0 = perpendicular, -1.0 = opposite). The search returns the chunks whose embeddings are
closest to the question's embedding.

### Chunking — Breaking Documents Into Pieces

You can't embed an entire file as one unit — an embedding is a single compressed representation and
loses detail for long texts. So we split documents into **chunks** first.

Naive approach (what Step 01 does): fixed-size windows.

```
Document: "Priya Sharma is Head of Engineering. She reports to CEO David Chen.
            The engineering team has 45 engineers across 3 sub-teams..."

Chunk 1 (0–200 chars):  "Priya Sharma is Head of Engineering. She reports to CEO David"
Chunk 2 (200–400 chars): "Chen. The engineering team has 45 engineers across 3 sub-teams"
```

Notice the problem: the sentence "She reports to CEO David Chen" gets **split across chunk 1 and 2**.
Either chunk alone is incomplete. This is the chunking problem — and it's why Step 01 achieves only 26%.

### Why Naive RAG Fails

| Question type | Why vector search fails |
|--------------|------------------------|
| Aggregate ("total ARR") | Returns 5 rows; LLM computes from those 5, misses the other 42 |
| Multi-hop ("who manages TechCorp's account?") | No single chunk contains both the customer → CSM and CSM → manager relationship |
| Alias ("analytics dashboard") | Embedding of "analytics dashboard" ≠ embedding of "InsightLens" |
| Exact numerical ("Q3 revenue was $X") | LLM picks closest number it sees; may be a different quarter |

**Step 01 result: 26% (7/27).** This is the floor. Everything from here is about fixing these failure modes.

---

## Chapter 3 — Seeing What's Happening (Step 02)

Before fixing failures, we need to *see* them. This is observability.

### What is a Trace?

A **trace** records what happened during a single query — every step, its inputs, outputs, and timing.

```
Query: "What was Q3 revenue?"
│
├── [t=0ms]    Embedding model   input: "What was Q3 revenue?"
│                                output: [0.3, -0.1, ...]  latency: 120ms
│
├── [t=120ms]  Vector search     input: embedding vector, k=5
│                                output: [chunk_7, chunk_12, chunk_2, ...]  latency: 8ms
│
├── [t=128ms]  LLM call          input: question + 5 chunks (2,400 tokens)
│                                output: "Q3 revenue was $3.2M"  latency: 1,800ms
│                                [WRONG — actual answer is $4.12M]
│
└── [t=1928ms] Total latency: 1.93s
```

Without traces you know the answer is wrong. With traces you can see *why*: the retrieved chunks were
from Q2, not Q3 — the cosine similarity matched "revenue" broadly but missed the temporal filter.

**Arize Phoenix** provides a visual trace explorer. JSONL traces give you a queryable audit log. Both
are instrumented before adding complexity, so every technique we add is immediately observable.

---

## Chapter 4 — Measuring What Matters (Step 03)

Knowing an answer is wrong isn't enough. We need to know *how* wrong, and *in what way*.

### LLM-as-Judge

We can't hand-grade every answer for every eval run. Instead, we use an LLM to grade the answers
using specific criteria. This is called **LLM-as-judge**.

The five RAGAS-style metrics we use:

```
┌─────────────────────────────────────────────────────────────────┐
│                    5 Evaluation Dimensions                      │
├────────────────────┬────────────────────────────────────────────┤
│ Faithfulness       │ Does the answer contain ONLY facts from    │
│                    │ the retrieved context? (no hallucination)  │
├────────────────────┼────────────────────────────────────────────┤
│ Context Precision  │ Of the retrieved chunks, what fraction     │
│                    │ were actually needed to answer?            │
├────────────────────┼────────────────────────────────────────────┤
│ Context Recall     │ Did retrieval find ALL the chunks needed?  │
├────────────────────┼────────────────────────────────────────────┤
│ Answer Relevance   │ Does the answer actually address what      │
│                    │ was asked? (not off-topic)                 │
├────────────────────┼────────────────────────────────────────────┤
│ Answer Correctness │ Factually, is the answer right?           │
│  (our primary)     │ Checked against required_facts             │
└────────────────────┴────────────────────────────────────────────┘
```

Our primary grader uses **deterministic fact-checking** (required_facts + disqualifiers) rather than
pure LLM judgment — cheaper, faster, reproducible.

### Why You Need Eval Before You Build

Without an eval suite, "does this technique help?" is a guess. With one, it's a measurement.

The eval suite is what tells us that adding CrossEncoder reranking at Step 10 *hurts* Q18 even while
helping Q05. Without the suite, we'd ship step 10 thinking it was an improvement.

---

## Chapter 5 — The Format Problem (Step 04)

Step 01 used fixed-size chunking for everything. The first real insight: **format dictates how to chunk**.

### Four Different Document Types, Four Different Needs

```
FORMAT          NAIVE SPLIT         PROBLEM              RIGHT SPLIT
──────────────────────────────────────────────────────────────────────
CSV             Every N chars       Row split mid-cell   Aggregate chunk
                                    Numbers lose context (entire sheet → summary)

Markdown        Every N chars       Header + body        Section split
                                    separated            (H2 = one chunk)

Plain text      Every N chars       Sentence fragments   Paragraph split
                                                         (double-newline boundary)

JSON            Every N chars       Invalid JSON         Object boundary split
```

### The Aggregate Chunk — The Key Insight for CSVs

A CSV is not a document. It's a **table**. The right way to chunk it is not to split rows — it's to
summarize the table's aggregate statistics AS a chunk.

```
revenue_by_product_2023.csv (raw):
  month,nexusflow_revenue,insightlens_revenue,pulseconnect_revenue,total_revenue
  2023-01,450000,380000,220000,1050000
  2023-02,460000,390000,225000,1075000
  ...12 rows total

Naive chunk: "2023-01,450000,380000,220000,1050000\n2023-02,460000,390000..."
  → LLM can't aggregate from partial rows

Aggregate chunk:
  "Revenue by Product 2023: 12 months of data.
   Q1 total: $3,150,000 | Q2 total: $3,420,000
   Q3 total: $4,120,000 | Q4 total: $4,890,000
   Annual total: $15,580,000
   NexusFlow peak: September 2023 ($580,000)"
  → LLM reads the answer directly
```

**Step 04 result: 52% (14/27)** — double the baseline. The biggest single lift in the project.
Format-aware chunking is the highest-ROI technique we apply.

---

## Chapter 6 — The Relationship Problem (Step 05)

Even with perfect chunking, some questions can't be answered from documents alone. Consider:

> "Who manages the TechCorp account, and what is that person's department head?"

The answer requires:
1. Find TechCorp in `customer_list.csv` → get CSM employee ID
2. Look up that employee ID in `employee_directory.csv` → get their manager
3. Look up that manager → get their department head

No single chunk contains this chain. Even the best retrieval can't retrieve a relationship that spans
three CSV files. We need a different data structure: a **knowledge graph**.

### What is a Knowledge Graph?

A knowledge graph stores **entities** (things) and **relationships** (how they connect) explicitly.

```
                    ┌──────────────────────────────────────────┐
                    │           Knowledge Graph                │
                    │                                          │
                    │  [Priya Sharma]──reports_to──►[David Chen]
                    │       │                            │      │
                    │       │                        (CEO)     │
                    │  manages_account                         │
                    │       │                                  │
                    │       ▼                                  │
                    │  [TechCorp] ──uses──► [NexusFlow]        │
                    │                           │              │
                    │                      depends_on          │
                    │                           │              │
                    │                           ▼              │
                    │                     [EventsAPI]          │
                    └──────────────────────────────────────────┘
```

Nodes are entities. Edges are typed relationships. The graph makes implicit connections explicit.

### BFS — Walking the Graph

**Breadth-First Search (BFS)** starts at one node and explores outward, hop by hop.

```
Question: "If NexusFlow fails, what else breaks?"

Step 1 — find anchor node: NexusFlow
Step 2 — follow depends_on edges (depth 1):
         NexusFlow ──depends_on──► EventsAPI
         NexusFlow ──depends_on──► AuthService
Step 3 — follow from those (depth 2):
         EventsAPI ──depends_on──► DataPipeline
         AuthService ──depends_on──► UserDB

Answer: EventsAPI, AuthService (direct), DataPipeline, UserDB (indirect)
```

BFS gives us **blast-radius analysis** and **dependency chains** — questions that pure vector search
fundamentally cannot answer because the answer requires traversing relationships, not matching text.

**Step 05 result: 78% (21/27)** — +26pp from chunking. Graph unlocks the multi-hop questions.

---

## Chapter 7 — Walking the Graph Smarter (Step 06)

The knowledge graph exists. But there's a practical problem: users don't ask "NexusFlow" — they ask
"the analytics dashboard" or "that reporting tool."

### Alias Resolution

**Alias resolution** maps user terms to canonical graph node names:

```
User says:           Maps to:
"analytics dashboard"  ──► InsightLens
"reporting tool"       ──► InsightLens
"nexus"                ──► NexusFlow
"pulse"                ──► PulseConnect
```

Without this, BFS starts from the wrong node and returns empty results — the question appears to fail
because retrieval found nothing, not because the answer doesn't exist.

### Combining Graph + Vector

Graph answers "how things relate." Vector answers "what things are." The best answers need both.

```
Question: "Who is on call this week, and what are the SLOs for their services?"

Graph traversal:  find on-call engineer → find their assigned services
Vector retrieval: find SLO definition passages for those services
                         │
                         ▼
LLM combines: "Alice Chen is on call. Her services are NexusFlow (99.9% monthly)
               and InsightLens (99.5% monthly)."
```

**Step 06 result: 85% (23/27)** — alias resolution recovers 2 more questions.

---

## Chapter 8 — The Best of All Retrieval Worlds (Step 07)

We now have vector search and graph traversal. But there's a third retrieval strategy we haven't used:
**keyword search**. And a fourth: **direct structured queries** over raw data.

### BM25 — Keyword Search

**BM25** (Best Match 25) is the algorithm behind classic search engines. Unlike embeddings, it matches
exact words and rare terms. It's fast, precise for technical terms, and doesn't need a GPU.

```
Query: "closed-won deals Q3 2023"

BM25 scores high:  chunk containing "closed-won" and "Q3 2023" exactly
Embeddings score high:  chunk about "successful deals third quarter"

They find DIFFERENT things. BM25 wins on exact terminology.
Embeddings win on paraphrase and meaning.
```

### Reciprocal Rank Fusion (RRF)

**RRF** merges ranked lists from multiple retrieval strategies into one. The intuition: if a chunk ranks
#2 in BM25 AND #3 in vector search, it's probably very relevant. If it only appears in one list, it's
less certain.

```
BM25 results:           Vector results:
  1. chunk_A  ◄─────────────────── chunk_A (rank 1)
  2. chunk_B             chunk_C (rank 2)
  3. chunk_D  ◄─────────────────── chunk_D (rank 3)
  4. chunk_C
                   │
                   ▼  RRF formula: score = Σ 1/(rank + 60)
                   │
  Fused ranking:
    chunk_A: 1/(1+60) + 1/(1+60) = 0.0328  ← appears in BOTH → high score
    chunk_D: 1/(3+60) + 1/(3+60) = 0.0317  ← appears in BOTH
    chunk_B: 1/(2+60) + 0         = 0.0161  ← only BM25
    chunk_C: 0        + 1/(2+60)  = 0.0161  ← only vector
```

The chunk that ranks well in multiple systems wins. Single-strategy outliers are penalized.

### The Structured Query Tool — Bypassing Retrieval for Aggregates

For aggregate questions ("total ARR", "Q3 revenue", "planned headcount"), retrieval is fundamentally
wrong: you're looking for a computed number that doesn't exist as text anywhere in the corpus.

The fix: **bypass retrieval entirely and run the Pandas query directly**.

```
Question: "What is the total ARR across all customers?"

Option A — Retrieval:
  Vector search finds 5 customer rows (out of 47)
  LLM sums those 5 → gives wrong answer
  (It literally cannot see the other 42 rows)

Option B — Structured Query Tool:
  detect_intent("total ARR") → "total_arr"
  pd.read_csv("customer_list.csv")["arr_usd"].sum() → $8,450,000
  inject "[AUTHORITATIVE: use $8,450,000]" into context
  LLM reads the authoritative label → correct answer
```

The tool always runs first, unconditionally. If no intent is matched, it returns empty — no harm done.
If intent is matched, the answer is pre-computed and injected before any LLM sees it.

**Step 07 result: 89% (24/27)** — the ceiling of what pure retrieval can achieve with our data.

---

## Chapter 9 — Adding a Brain (Step 08)

Steps 01–07 follow a fixed pipeline: retrieve → generate. The pipeline doesn't *think* — it runs the
same steps for every question. An agentic system changes this: the LLM decides what to do next.

### The ReAct Pattern

**ReAct** (Reason + Act) is the core pattern for agentic systems:

```
┌─────────────────────────────────────────────────────────┐
│                    ReAct Loop                           │
└─────────────────────────────────────────────────────────┘

Question: "Who does the CSM for TechCorp report to?"
    │
    ▼
┌───────────────┐
│    REASON     │  "I need to find TechCorp's CSM first."
└───────┬───────┘
        │
        ▼
┌───────────────┐
│      ACT      │  call graph_tool("TechCorp")
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   OBSERVE     │  result: "CSM is Alice Chen (E042)"
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    REASON     │  "Now I need to find who Alice Chen reports to."
└───────┬───────┘
        │
        ▼
┌───────────────┐
│      ACT      │  call graph_tool("E042 manager")
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   OBSERVE     │  result: "Reports to Bob Singh, Head of Sales"
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   SYNTHESIZE  │  "The CSM for TechCorp is Alice Chen,
└───────────────┘   who reports to Bob Singh, Head of Sales."
```

### The Problem With Agent Discretion

Agentic RAG gives the LLM *choice* over which tools to call. For some questions, this is powerful.
For others, it introduces variance.

The specific failure we observed: for aggregate questions, the agent *sometimes* skips the CSV tool
("I'll just answer from context"). The pipeline approach in Step 07 always runs the CSV tool regardless.

```
Step 07: detect_intent() → always runs → deterministic
Step 08: LLM decides "should I call csv_tool?" → sometimes skips it → non-deterministic
```

**Step 08 result: 85% (23/27)** — regression from 89%. The lesson: agent discretion is powerful but
you need guardrails. Unconditional tool invocation beats optional tool invocation for known-answer-exists queries.

### LLM Gateway — Multi-Provider Routing

Rather than hardcoding one LLM provider, Step 08 introduces a local **LLM Gateway** that routes to
Gemini, NVIDIA NIM, Groq, or Cerebras — all free tier. This gives:
- Provider fallback (if one is rate-limited, try the next)
- Consistent API surface (one client, any provider)
- Cost and latency visibility per provider

---

## Chapter 10 — A Team of Specialists (Step 09)

A single agent that does everything is hard to debug, tune, and improve. The alternative: break the
work into roles.

### The Multi-Agent Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Orchestrator                  │
                    │  (coordinates all agents, holds state)  │
                    └───────┬─────────────────────────────────┘
                            │  dispatches to
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
    ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
    │ QueryAnalyst │ │  Retrieval  │ │GraphNavigator│
    │              │ │ Specialist  │ │              │
    │ Classifies   │ │             │ │ BFS traversal│
    │ intent:      │ │ BM25+dense  │ │ from entity  │
    │ simple/multi │ │ RRF, k=20   │ │ anchor nodes │
    │ /aggregate   │ └─────────────┘ └──────────────┘
    └──────────────┘
             │              │              │
             └──────────────┼──────────────┘
                            │  all contexts merged
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
    ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
    │ Structured   │ │  Synthesis  │ │    Critic    │
    │ Data Agent   │ │             │ │              │
    │              │ │ Writes the  │ │ Reviews for  │
    │ Pandas CSV   │ │ final answer│ │ contradictions│
    │ tool always  │ │ from all    │ │ & precision  │
    │ unconditional│ │ contexts    │ └──────────────┘
    └──────────────┘ └─────────────┘
```

### Agent Contracts

Each agent has a typed input and output. The Orchestrator only depends on the **contract**, not the
implementation:

```python
# Any retrieval agent must return this shape
@dataclass
class RetrievalResult:
    chunks: list[str]
    sources: list[str]
    latency_ms: float
    strategy: str    # "bm25", "dense", "rrf"
    status: str      # "ok" | "error"
```

Swap the retrieval implementation → orchestrator doesn't change. This is why contracts come before
implementation.

### The Critic Agent

The **Critic** reads the Synthesis agent's draft answer and checks:
- Does it actually answer the question asked?
- Are there contradictions with the source context?
- Does it name exact field values (not paraphrases)?

The Critic doesn't rewrite the answer — it raises a flag. If the Critic rejects the draft, Synthesis
tries again with the Critic's feedback. This is a simple **self-reflection loop**.

**Step 09 result: 93% (25/27)** — best result in the project. The Critic + precise synthesis rules
push past what any single-agent system achieves.

---

## Chapter 11 — Less is More (Step 10)

At step 09 we retrieve up to 20 chunks and send them all to the LLM. That's ~45,000 characters of
context per query. Problems:
- LLMs have a "lost in the middle" effect — they pay less attention to chunks buried in the middle
- Near-duplicate chunks waste tokens
- Irrelevant sentences dilute the signal for the relevant ones

**Context engineering** is the discipline of curating *what* reaches the LLM, not just *how much*.

### The Four-Stage Pipeline

```
RAW RETRIEVAL (20 chunks, ~45,000 chars)
       │
       ▼ STAGE 1: RERANK
   CrossEncoder scores each (question, chunk) pair.
   A CrossEncoder reads BOTH texts together — unlike embeddings
   which encode them separately. Much more precise.
   Keeps top 8 by cross-encoder score.
   → ~18,000 chars
       │
       ▼ STAGE 2: DEDUPLICATE
   6-gram Jaccard similarity between all remaining chunk pairs.
   "6-gram" = overlapping sequences of 6 words.
   If two chunks share >72% of their 6-grams, they're near-duplicates.
   Drop the lower-ranked one.
   → ~12,000 chars
       │
       ▼ STAGE 3: COMPRESS (extractive)
   Score each sentence by keyword overlap with the question.
   Keep top 60% of sentences per chunk.
   CSV data and graph context are NEVER compressed (they're already precise).
   → ~7,000 chars
       │
       ▼ STAGE 4: FORMAT
   Structured XML with source attribution and CrossEncoder scores:
   <csv_data>...</csv_data>
   <graph_context>...</graph_context>
   <passage rank="1" src="hr/employee_directory.csv" score="0.94">...</passage>
   Enforces a hard 24,000-char budget.
   → final context to LLM
```

### The Tradeoff — Why 85% < 93%

Extractive compression is a **lossy operation**. For most questions, the sentences that get removed
are genuinely irrelevant. For two questions (Q18, Q22), the *key* sentences happen to have low
keyword overlap with the query — so they score in the bottom 40% and get removed.

```
Q18: "Two programs named Project Phoenix — what's the difference?"
Query keywords: "project", "phoenix", "difference"
                                    │
                                    ▼
CrossEncoder elevates: "Project Phoenix (2022) and Project Phoenix (2024) are two separate..."
Compressor scores this sentence: overlap with "project phoenix difference" → high → KEPT ✓

But the sentence that says "The 2022 program was cancelled, the 2024 program is ongoing" →
it contains "programme" and "2022" but not "phoenix" or "difference" → lower score → REMOVED ✗
```

This is the documented compression tradeoff. Step 10 is not a regression in principle — it's a
tradeoff between token efficiency and recall completeness.

**Step 10 result: 85% (23/27)** — the compression wins on most questions, loses on 2 edge cases.

---

## Chapter 12 — The Right Slice for the Job (Step 11)

Step 09 uses one system prompt for all questions. But Finance questions need different reasoning
than Engineering questions. A single prompt makes compromises that hurt everyone.

**Vertical Slice Architecture (VSA)** organizes the system by *domain* rather than by *layer*.

### What is VSA?

Traditional layered architecture:
```
Layer:   Retrieval → Reranking → Compression → Synthesis
         (every question goes through the same settings)
```

VSA:
```
Domain:  Finance Slice  → (its own retrieval settings, system prompt, compress_ratio)
         HR Slice       → (its own settings)
         Engineering Slice → (its own settings, force_graph=True)
         General Slice  → (default settings)
```

Each "slice" is a self-contained vertical cut through all the layers — retrieval tuning,
compression ratio, and synthesis prompt — all owned by one slice.

### The Keyword Router — Zero LLM Calls

Routing doesn't need an LLM. We use keyword scoring:

```
Question: "What is the total vendor spend in 2023?"
Keywords matched:
  "vendor" → finance slice: +1
  "spend"  → finance slice: +1
  "2023"   → (neutral)
  "total"  → finance slice: +1

Finance slice score: 3/7 words × 4.0 = 1.7 (capped at 1.0) → route to Finance ✓
```

If no slice scores above threshold → General slice handles it.

### Slice-Specific Tuning

```
┌───────────────┬──────────────┬────────────────┬───────────────────────────────────┐
│ Slice         │ compress_ratio│ force_graph    │ System Prompt Focus               │
├───────────────┼──────────────┼────────────────┼───────────────────────────────────┤
│ Finance       │ 0.70         │ False          │ Exact dollar figures, date format  │
│ HR            │ 0.75         │ False          │ Org hierarchy, departure types     │
│ Engineering   │ 0.80         │ True           │ Dependency chains, SLO values      │
│ General       │ 0.65         │ False          │ Default, cross-domain questions    │
└───────────────┴──────────────┴────────────────┴───────────────────────────────────┘
```

Engineering slice has `force_graph=True` — for any engineering question, the graph navigator always
runs regardless of query classification. Dependency questions can't be answered without it.

**Step 11 result: 89% (24/27)** — recovers Finance/HR losses from step 10; Q18 remains hard.

---

## Chapter 13 — Ready for the Real World (Step 12)

A system that works in an eval script is not a production system. Production means:
- It works reliably even when external APIs fail
- It responds in acceptable latency even under load
- It doesn't charge you $50/day re-answering the same 10 questions
- When it fails, it fails gracefully and tells you *why*

### Semantic Cache

Caching by exact string matching is useless — users rephrase the same question constantly.
**Semantic caching** stores the embedding of each question with its answer. For a new question:

```
New question: "What was Vertexia's revenue in Q3?"
Stored question: "What is the total Q3 2023 revenue?"
Embedding similarity: 0.94 → cache HIT → return stored answer immediately
(no retrieval, no LLM call, ~5ms instead of ~2000ms)
```

Threshold: if similarity > 0.92, it's the same question. Below that, run the full pipeline.

### Retry and Backoff

LLM APIs fail. Rate limits hit. Network hiccups happen. **Exponential backoff** retries with
increasing wait times:

```
Attempt 1: call API → 429 Rate Limited
  wait 1 second
Attempt 2: call API → 429 Rate Limited
  wait 2 seconds
Attempt 3: call API → 429 Rate Limited
  wait 4 seconds
Attempt 4: call API → success ✓
```

Wait time doubles each attempt, capped at some maximum. This avoids hammering a struggling API
while still eventually succeeding.

### Confidence Scoring

Not all answers are equally reliable. A confidence score gives the system (and the user) a signal:

```
High confidence (>0.85): top CrossEncoder score is high, all required facts present
Medium confidence (0.60–0.85): answer found but context was partial
Low confidence (<0.60): LLM answered but retrieval was weak → flag for review
```

### Health Monitor

A background process that:
- Runs a synthetic query every N minutes
- Checks that the answer is correct (against a known expected answer)
- Alerts if latency exceeds budget or answer quality degrades
- This is "always-on testing" for your production system

**Step 12 result: 89% (24/27) + production reliability** — same accuracy as step 11 but now with
the operational properties that make it shippable.

---

## Epilogue — What Makes Production RAG Hard

Looking back at the full journey, the pattern is clear. Every technique we added existed to fix
a specific, measurable failure:

```
PROBLEM                          FIX                           LIFT
────────────────────────────────────────────────────────────────────
Fixed-size chunks break           Format-aware chunking          +26pp
sentences and lose CSV structure

Vector can't traverse             Knowledge graph + BFS          +26pp
relationships

Embeddings miss exact terms       BM25 + RRF fusion              +7pp

Aggregates need all rows,         Deterministic CSV tool         (part of above)
not just retrieved ones

One prompt can't handle           Domain slices + per-slice      +4pp (from step 10 dip)
every question type               prompts

Compression removes key           Higher compress_ratio for      (partial recovery)
sentences for edge cases          edge-case-prone slices

External APIs fail                Retry/backoff + fallback        reliability

Same questions re-computed        Semantic cache                  latency
```

**The most important lesson**: the techniques that gave the biggest lifts weren't the flashiest ones.
Format-aware chunking (+26pp) outperformed agentic reasoning by a wide margin. A deterministic
Pandas query outperformed LLM-based aggregation every time.

**Production RAG is mostly about knowing when NOT to use an LLM.**

Use the LLM for what only an LLM can do — language understanding, synthesis, judgment.
For everything else (arithmetic, exact lookup, relationship traversal) — use the right tool for the job.

```
Final architecture — what each layer is responsible for:

Query arrives
    │
    ▼  [Semantic Cache] — have we seen this exact question? serve cached answer
    │
    ▼  [VSA Router] — which domain slice handles this? (zero LLM calls)
    │
    ▼  [Query Analyst] — what type of question is this? what do we need to retrieve?
    │
    ├──► [Structured Query Tool] — always runs; Pandas for aggregate questions
    │
    ├──► [BM25 + Dense Retrieval] — RRF merge, k=20 candidates
    │
    ├──► [Graph Navigator] — BFS from entity anchors; forced on for engineering questions
    │
    ▼  [Context Engineering] — rerank → dedup → compress → XML format
    │
    ▼  [Synthesis] — write the answer from curated context only
    │
    ▼  [Critic] — verify precision, flag contradictions
    │
    ▼  [Confidence Score] — how reliable is this answer?
    │
    ▼  [Health Monitor] — is the system performing within SLOs?
    │
    ▼  answer → user
```

Every layer has a specific job. No layer does another layer's job. That's what production looks like.
