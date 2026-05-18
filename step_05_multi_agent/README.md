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

### Per-question summary

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

### Per-question detail (question · reference · system answer · judge)

Each entry is collapsed; click to expand the full debug view.

<details>
<summary><b>Q01</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; What is Vertexia's customer data retention policy for hot storage and cold storage?

**Reference answer (gold):**

> Vertexia stores customer data in hot storage for 90 days, then archives to AWS S3 Glacier for 1 year of cold storage.

**System answer at this step:**

> Vertexia's customer data retention policy is 90 days in hot storage and 1 year in cold storage (S3 Glacier).

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer matches the reference exactly in content and is fully supported by multiple retrieved contexts; most retrieved contexts are relevant but a few (e.g., vendor matrix, founding story) are not directly needed.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.70 | 1.00 |

</details>

<details>
<summary><b>Q02</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; Who co-founded Vertexia and in what year was the company founded?

**Reference answer (gold):**

> Vertexia was co-founded by Arjun Mehta and Diana Volkov in March 2019.

**System answer at this step:**

> Vertexia was co-founded by Arjun Mehta and Diana Volkov in March 2019.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer perfectly matches the reference and is fully supported by the contexts, but only half of the retrieved contexts are relevant to the question.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |

</details>

<details>
<summary><b>Q03</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; In Vertexia's on-call runbook, what is the first action and the escalation owner for the PulseConnect webhook delivery failure alert?

**Reference answer (gold):**

> For the PulseConnect webhook_delivery_failure_rate > 5% alert, the first action is to check the SendGrid quota dashboard and the Twilio API status page (and inspect the outbound retry queue depth in the PulseConnect admin UI). The escalation owner is Raj Patel.

**System answer at this step:**

> The first action for the PulseConnect webhook_delivery_failure_rate > 5% alert is to check the external_sendgrid quota dashboard and the external_twilio API status page, and to inspect the outbound retry queue depth in the PulseConnect admin UI. The escalation owner for this alert is Raj Patel.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer matches the reference exactly in content and is fully supported by context [3]; minor naming variation ('external_sendgrid' vs 'SendGrid') is format-tolerant; only context [3] is relevant, others are unrelated.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q04</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total ARR across all Vertexia customers combined?

**Reference answer (gold):**

> Summing the arr_usd column across all 20 rows in customer_list.csv yields a total of $11,000,000 across all Vertexia customers combined.

**System answer at this step:**

> The total ARR across all Vertexia customers combined is $11,000,000.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer is factually correct but not supported by any retrieved context containing the customer_list.csv data or ARR sum, making it hallucinated.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |

</details>

<details>
<summary><b>Q05</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; How many active Vertexia employees are based in the Berlin office?

**Reference answer (gold):**

> There are 5 active Vertexia employees based in the Berlin office.

**System answer at this step:**

> There are 5 active Vertexia employees based in the Berlin office.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer is factually correct and matches the reference, but the retrieved contexts do not mention the Berlin office or the number of employees there, making the answer ungrounded in the provided context.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |

</details>

<details>
<summary><b>Q06</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total planned headcount across all departments in Vertexia's 2023 budget allocation?

**Reference answer (gold):**

> Summing the headcount column across all 9 department rows in budget_allocation_2023.csv yields a total planned headcount of 181 for 2023. (This is the budgeted plan, distinct from the company's ~500 actual employees referenced in prose.)

**System answer at this step:**

> The total planned headcount across all departments in Vertexia's 2023 budget allocation is 181.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer directly matches the reference and is supported by the retrieved contexts.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.90 | 1.00 |

</details>

<details>
<summary><b>Q07</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; According to Vertexia's vendor contracts summary, what is the annual contract value for Snowflake and when does that contract renew?

**Reference answer (gold):**

> Per vendor_contracts_summary.csv, Vertexia's annual contract value for Snowflake is $120,000 and the contract renews on June 30, 2024.

**System answer at this step:**

> The annual contract value for Snowflake is $120,000, and the contract renews on 2024-06-30.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the retrieved context, though only one of the eleven provided context chunks was relevant to the query.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.09 | 1.00 |

</details>

<details>
<summary><b>Q08</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Which NexusFlow API endpoint was deprecated in v2.1, and what endpoint replaced it?

**Reference answer (gold):**

> The GET /v2/events/batch endpoint was deprecated in NexusFlow API v2.1 (and removed in v2.2). It was replaced by GET /v2/events/stream.

**System answer at this step:**

> The endpoint `GET /v2/events/batch` was deprecated in v2.1, and it was replaced by `GET /v2/events/stream`.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the provided context, though only one of the eleven retrieved chunks was relevant to the query.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q09</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Per Vertexia's Q4 2023 on-call schedule, who was the on-call engineer for the Data Platform Team during the week starting 2023-10-16?

**Reference answer (gold):**

> Per on_call_schedule_q4_2023.csv, Priya Kapoor (employee_id E039) was the on-call engineer for the Data Platform Team during the week of 2023-10-16 to 2023-10-22.

**System answer at this step:**

> Per the On Call Schedule Q4 2023 in the retrieved context, the on-call engineer for the Data Platform Team for the week starting 2023-10-16 was Priya Kapoor.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer correctly identifies the on-call engineer using the provided context, though the retrieved context set contained significant irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.09 | 1.00 |

</details>

<details>
<summary><b>Q10</b> — PARTIAL · correctness 0.50 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Who is the CSM assigned to the customer 'Summit Pharma', what office location is that CSM based in, and who is that CSM's direct manager?

**Reference answer (gold):**

> The CSM assigned to Summit Pharma is Anjali Patel (per csm_account_history.csv). Anjali Patel is based in San Francisco and her direct manager is Maya Sharma (VP Customer Success), per employee_directory.csv.

**System answer at this step:**

> The CSM assigned to Summit Pharma is Anjali Patel. She is based in the San Francisco office, and her direct manager is Maya Sharma.

**Diagnosis** &nbsp; Regression: was solvable at step 4; now only PARTIAL. The answer is partially correct but lacks the direct manager's title and the CSM's office location is not explicitly supported by the retrieved contexts.

**Judge reasoning** &nbsp; The answer is partially correct but lacks the direct manager's title and the CSM's office location is not explicitly supported by the retrieved contexts.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.50 | 1.00 | 1.00 | 0.20 | 0.60 |

</details>

<details>
<summary><b>Q11</b> — PARTIAL · correctness 0.60 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Based on Vertexia's API dependency graph, which Vertexia services directly depend on NexusFlow's APIs? Name each service AND the specific NexusFlow endpoint it consumes.

**Reference answer (gold):**

> Per api_dependencies.csv, three Vertexia services directly depend on NexusFlow's APIs: InsightLens (via events_api and connectors_api), PulseConnect (via pipeline_status_api), and DataCraft (via ingest_api).

**System answer at this step:**

> Based on the provided API dependency graph and documentation, the following Vertexia services directly depend on NexusFlow APIs:
> 
> *   **InsightLens** consumes the `events/stream` endpoint.
> *   **DataCraft** consumes the `ingest` endpoint (via the `ingest_api`).
> *   **PulseConnect** consumes the `pipelines` endpoint (via the `pipeline_status_api`).
> *   **DataCraft Integration** consumes the `events/stream` endpoint.

**Diagnosis** &nbsp; Regression: was solvable at step 4; now only PARTIAL. The answer correctly identifies InsightLens but hallucinates DataCraft/PulseConnect endpoints not present in the provided context, while missing the specific A…

**Judge reasoning** &nbsp; The answer correctly identifies InsightLens but hallucinates DataCraft/PulseConnect endpoints not present in the provided context, while missing the specific API names required by the reference.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.60 | 0.80 | 1.00 | 0.40 | 0.50 |

</details>

<details>
<summary><b>Q12</b> — FAIL · correctness 0.20 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Which active Vertexia employees located in the Bangalore office report — directly or indirectly, through any number of management hops — to the CTO Sarah Chen? List every such employee.

**Reference answer (gold):**

> Five active Bangalore-based employees report (directly or indirectly) to CTO Sarah Chen: Priya Nair (Lead, Data Platform Team), Kenji Ito (Senior Engineer, Data Platform), Lin Wei (Senior Engineer, NexusFlow), Priya Kapoor (Data Engineer, Data Platform), and Omar Faruk (Engineer, Security).

**System answer at this step:**

> The active Vertexia employees located in the Bangalore office who report to Sarah Chen are Priya Nair, Kenji Ito, Adrian Blake, and Priya Kapoor.

**Diagnosis** &nbsp; Regression: was solvable at step 4; now FAILED. The model hallucinated names not present in the provided context and failed to retrieve the specific employee list required to answer the question.

**Judge reasoning** &nbsp; The model hallucinated names not present in the provided context and failed to retrieve the specific employee list required to answer the question.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.20 | 0.50 | 1.00 | 0.10 | 0.00 |

</details>

<details>
<summary><b>Q13</b> — FAIL · correctness 0.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; What is the combined ARR (sum of arr_usd in customer_list.csv) of all customers whose assigned CSM is an active direct report of Maya Sharma per the employee directory?

**Reference answer (gold):**

> Maya Sharma has two active direct reports in employee_directory.csv: Sam Rivera and Anjali Patel (Preet Kaur also reported to Maya but departed in 2023). The customers in customer_list.csv assigned to either of those two CSMs combine to a total ARR of $3,708,000 (Anjali Patel's 5 accounts sum to $2,400,000; Sam Rivera's 5 accounts sum to $1,308,000).

**System answer at this step:**

> The combined ARR of customers whose CSM is an active direct report of Maya Sharma is $1,116,000.

**Diagnosis** &nbsp; Should PASS at this tier but FAILED — diagnose. The model failed to identify the correct direct reports and instead calculated ARR for customers assigned directly to Maya Sharma, while the retrieved contexts lacked the necessar…

**Judge reasoning** &nbsp; The model failed to identify the correct direct reports and instead calculated ARR for customers assigned directly to Maya Sharma, while the retrieved contexts lacked the necessary employee directory data to identify the correct CSMs.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |

</details>

<details>
<summary><b>Q14</b> — PASS · correctness 1.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; Consider Vertexia's active Customer Success Managers (role contains 'Customer Success Manager') who started at the company in 2022 per the employee directory. What is the combined ARR in customer_list.csv of the customers assigned to those CSMs?

**Reference answer (gold):**

> The only active Customer Success Manager in employee_directory.csv who started at Vertexia in 2022 is Anjali Patel (start_date 2022-03-15). The five customers in customer_list.csv assigned to Anjali Patel (Summit Pharma, OmegaLogistics, Redwood Analytics, Ironclad Security, and Northgate Bank) have a combined ARR of $2,400,000.

**System answer at this step:**

> The active Customer Success Managers who started in 2022 are Anjali Patel (start_date: 2022-03-15). 
> 
> The customers assigned to Anjali Patel and their respective ARR are:
> * Summit Pharma: $300,000
> * OmegaLogistics: $156,000
> * Redwood Analytics: $720,000
> * Ironclad Security: $384,000
> * Northgate Bank: $840,000
> 
> The combined ARR for these customers is $2,400,000.

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The answer is factually accurate and fully supported by the provided context, though the retrieved set included several irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.45 | 1.00 |

</details>

> The full per-question JSON (retrieved contexts, latency, etc.) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
