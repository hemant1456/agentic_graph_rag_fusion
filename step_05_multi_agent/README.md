# Step 07 — Multi-Agent Orchestration

## What it adds
Replaces the single-prompt pipeline with five specialised agents coordinated by an orchestrator: a QueryAnalyst classifies the question and emits sub-questions, three specialists (RetrievalSpecialist, GraphNavigator, StructuredData) gather evidence in parallel, a Synthesis agent writes the answer, and a Critic agent reviews and rewrites it. Newly handles Tier 6 cross-document questions (Q14-Q15) that require comparing facts spread across multiple files.

## Design
- **Class:** `Step05RAG` in `step_05_multi_agent/implementation/pipeline.py`
- **Inherits from:** composes `Step03HybridRAG` for retrieval and reuses the Step 05 knowledge graph
- **Key components:**
  - `step_05_multi_agent/implementation/orchestrator.py` — top-level `run(question, retriever, graph)` that sequences the agents
  - `step_05_multi_agent/implementation/agents/query_analyst.py` — query classification and sub-question generation
  - `step_05_multi_agent/implementation/agents/retrieval_specialist.py` — wraps hybrid retrieval
  - `step_05_multi_agent/implementation/agents/graph_navigator.py` — alias-resolved graph traversal
  - `step_05_multi_agent/implementation/agents/structured_data.py` — pandas CSV tool calls
  - `step_05_multi_agent/implementation/agents/synthesis.py` and `critic.py` — answer drafting and review
  - `step_05_multi_agent/implementation/agents/contracts.py` — typed dataclasses for inter-agent messages

## How it works
The orchestrator first calls QueryAnalyst, which classifies the query (lookup, aggregate, multi-hop, cross-document) and decomposes compound queries into sub-questions. The specialists then run: RetrievalSpecialist returns hybrid chunks, GraphNavigator walks the graph from extracted entities, and StructuredData runs CSV aggregates when the intent matches. Synthesis takes the three evidence streams and the query type and writes a first-draft answer. Critic reviews the draft against the same evidence and either approves it or rewrites it. The final answer plus per-agent traces are returned.

## Run
```bash
uv run python evaluation/run_eval.py --step step_05_multi_agent
```

## Results
See `step_05_multi_agent/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Single-prompt RAG conflates classification, retrieval choice, synthesis, and verification into one opaque step. Q14 ("does the NexusFlow availability target meet the Phoenix Corp SLA?") needs the model to pull facts from two distinct documents and compare them; Q15 needs it to filter offboarding records by year and reason and report each row. Splitting the work across agents with typed contracts makes each decision inspectable, enables parallel evidence gathering, and lets the critic catch hallucinations that a single pass would emit.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 10 PASS · 2 PARTIAL · 2 FAIL out of 14 questions (71% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.807 | 0.736 | 1.000 | 0.259 | 0.650 |

### Per-question detail

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q03** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q04** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q05** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q06** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q07** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q08** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q09** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q10** | PARTIAL | 0.50 | `step_04_knowledge_graph` | Regression: was solvable at step 4; now only PARTIAL. The answer is partially correct but lacks the direct manager's title and the CSM's office location is not explicitly supported by the retrieved contexts. |
| **Q11** | PARTIAL | 0.60 | `step_04_knowledge_graph` | Regression: was solvable at step 4; now only PARTIAL. The answer correctly identifies InsightLens but hallucinates DataCraft/PulseConnect endpoints not present in the provided context, while missing the specific A… |
| **Q12** | FAIL | 0.20 | `step_04_knowledge_graph` | Regression: was solvable at step 4; now FAILED. The model hallucinated names not present in the provided context and failed to retrieve the specific employee list required to answer the question. |
| **Q13** | FAIL | 0.00 | `step_05_multi_agent` | Should PASS at this tier but FAILED — diagnose. The model failed to identify the correct direct reports and instead calculated ARR for customers assigned directly to Maya Sharma, while the retrieved contexts lacked the necessar… |
| **Q14** | PASS | 1.00 | `step_05_multi_agent` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |

> Each question's text + reference answer lives in `step_01_baseline_rag/evaluation/golden_questions.py`. The full per-question JSON (including the judge's reasoning) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
