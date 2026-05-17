# Step 07 — Multi-Agent Orchestration

## What it adds
Replaces the single-prompt pipeline with five specialised agents coordinated by an orchestrator: a QueryAnalyst classifies the question and emits sub-questions, three specialists (RetrievalSpecialist, GraphNavigator, StructuredData) gather evidence in parallel, a Synthesis agent writes the answer, and a Critic agent reviews and rewrites it. Newly handles Tier 6 cross-document questions (Q14-Q15) that require comparing facts spread across multiple files.

## Design
- **Class:** `Step07RAG` in `step_07_multi_agent/implementation/pipeline.py`
- **Inherits from:** composes `Step04HybridRAG` for retrieval and reuses the Step 05 knowledge graph
- **Key components:**
  - `step_07_multi_agent/implementation/orchestrator.py` — top-level `run(question, retriever, graph)` that sequences the agents
  - `step_07_multi_agent/implementation/agents/query_analyst.py` — query classification and sub-question generation
  - `step_07_multi_agent/implementation/agents/retrieval_specialist.py` — wraps hybrid retrieval
  - `step_07_multi_agent/implementation/agents/graph_navigator.py` — alias-resolved graph traversal
  - `step_07_multi_agent/implementation/agents/structured_data.py` — pandas CSV tool calls
  - `step_07_multi_agent/implementation/agents/synthesis.py` and `critic.py` — answer drafting and review
  - `step_07_multi_agent/implementation/agents/contracts.py` — typed dataclasses for inter-agent messages

## How it works
The orchestrator first calls QueryAnalyst, which classifies the query (lookup, aggregate, multi-hop, cross-document) and decomposes compound queries into sub-questions. The specialists then run: RetrievalSpecialist returns hybrid chunks, GraphNavigator walks the graph from extracted entities, and StructuredData runs CSV aggregates when the intent matches. Synthesis takes the three evidence streams and the query type and writes a first-draft answer. Critic reviews the draft against the same evidence and either approves it or rewrites it. The final answer plus per-agent traces are returned.

## Run
```bash
uv run python evaluation/run_eval.py --step step_07_multi_agent
```

## Results
See `step_07_multi_agent/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Single-prompt RAG conflates classification, retrieval choice, synthesis, and verification into one opaque step. Q14 ("does the NexusFlow availability target meet the Phoenix Corp SLA?") needs the model to pull facts from two distinct documents and compare them; Q15 needs it to filter offboarding records by year and reason and report each row. Splitting the work across agents with typed contracts makes each decision inspectable, enables parallel evidence gathering, and lets the critic catch hallucinations that a single pass would emit.
