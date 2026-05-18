# Step 05 — Knowledge Graph

## What it adds
Builds an entity knowledge graph from the structured CSVs (employees, accounts, API dependencies, departures) and uses multi-hop traversal to answer relational questions. Edges include `reports_to`, `manages_account`, `depends_on`, and `uses`. Newly handles Tier 5 questions (Q11-Q13) that require joining facts across multiple CSV rows or files.

## Design
- **Class:** `Step04RAG` in `step_04_knowledge_graph/implementation/pipeline.py`
- **Inherits from:** `Step03HybridRAG` (extends hybrid retrieval with graph context)
- **Key components:**
  - `step_04_knowledge_graph/implementation/builder.py` — reads HR, sales, engineering, and finance CSVs and emits `networkx.DiGraph` nodes/edges
  - `step_04_knowledge_graph/implementation/graph_store.py` — `load_or_build()` caches the graph to `step_04_knowledge_graph/results/graph.json`
  - `step_04_knowledge_graph/implementation/query.py` — `get_graph_context()` extracts a per-question subgraph and renders it as text

## How it works
On `build()`, the pipeline calls `Step03HybridRAG.build()` and then loads (or rebuilds) the graph from `dataset/company_data/`. Nodes are people, products, customers, and vendors; edges encode reports-to, manages-account, depends-on, and uses relations. At query time, hybrid retrieval runs first and produces the top-k chunks. The chunk texts are passed to `get_graph_context()` as seeds: it extracts named entities, walks the graph one or two hops out, and emits a textual block like `Aisha Johnson reports_to Tomás García; Tomás García reports_to Sarah Chen`. The CSV-tool result, the graph context, and the vector context are concatenated and sent to the LLM.

## Run
```bash
uv run python evaluation/run_eval.py --step step_04_knowledge_graph
```

## Results
See `step_04_knowledge_graph/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Hybrid retrieval can find the right CSV rows but it cannot follow a relation across them. Q11 ("CSM for Phoenix Corp, and their manager") needs two joins: `customer → CSM` then `CSM → manager`. Q12 ("if NexusFlow goes down, what is affected") needs a BFS over the dependency graph. Q13 ("two-hop reporting chain for Aisha Johnson") is a pure graph walk. Encoding relations as first-class edges makes these answers deterministic instead of relying on the LLM to chain implicit references.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 10 PASS · 3 PARTIAL · 1 FAIL out of 14 questions (71% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.807 | 0.986 | 0.900 | 0.286 | 0.757 |

### Per-question summary

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q03** | PARTIAL | 0.50 | `step_01_baseline_rag` | Regression: was solvable at step 1; now only PARTIAL. The retrieved context lacks the specific runbook instructions for the alert, leading to a hallucination of the first action, though the owner was correctly ide… |
| **Q04** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q05** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q06** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q07** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q08** | PASS | 0.80 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q09** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q10** | PASS | 1.00 | `step_04_knowledge_graph` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q11** | PARTIAL | 0.60 | `step_04_knowledge_graph` | Should PASS at this tier but only PARTIAL. The actual answer correctly identifies all three services but adds a non-existent 'DataCraft (Integration)' and misidentifies endpoints; it is mostly grounded but includes minor h… |
| **Q12** | PARTIAL | 0.40 | `step_04_knowledge_graph` | Should PASS at this tier but only PARTIAL. The answer correctly identifies Priya Nair but misses four other employees; all claims are grounded in context, though the retrieved context contains enough information to infer t… |
| **Q13** | FAIL | 0.00 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |
| **Q14** | PASS | 1.00 | `step_05_multi_agent` | Unexpected PASS — question targets step 5's capability, but retrieved context happened to contain enough signal. |

### Per-question detail (question · reference · system answer · judge)

Each entry is collapsed; click to expand the full debug view.

<details>
<summary><b>Q01</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; What is Vertexia's customer data retention policy for hot storage and cold storage?

**Reference answer (gold):**

> Vertexia stores customer data in hot storage for 90 days, then archives to AWS S3 Glacier for 1 year of cold storage.

**System answer at this step:**

> Vertexia's customer data retention policy is as follows:
> 
> *   **Hot Storage:** 90 days
> *   **Cold Storage:** 1 year (S3 Glacier)

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the provided context, though the retrieval system returned a large amount of irrelevant information.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q02</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; Who co-founded Vertexia and in what year was the company founded?

**Reference answer (gold):**

> Vertexia was co-founded by Arjun Mehta and Diana Volkov in March 2019.

**System answer at this step:**

> Vertexia Inc. was co-founded by Arjun Mehta and Diana Volkov in March 2019.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the provided documents, though the retrieved context set contained significant irrelevant information.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.20 | 1.00 |

</details>

<details>
<summary><b>Q03</b> — PARTIAL · correctness 0.50 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; In Vertexia's on-call runbook, what is the first action and the escalation owner for the PulseConnect webhook delivery failure alert?

**Reference answer (gold):**

> For the PulseConnect webhook_delivery_failure_rate > 5% alert, the first action is to check the SendGrid quota dashboard and the Twilio API status page (and inspect the outbound retry queue depth in the PulseConnect admin UI). The escalation owner is Raj Patel.

**System answer at this step:**

> In Vertexia's on-call runbook, the details for the **PulseConnect webhook_delivery_failure_rate > 5%** alert are as follows:
> 
> *   **First action:** Check external_sendgrid quota dashboard and the external_twilio API status page. Inspect outbound retry queue depth in PulseConnect admin UI.
> *   **Owner:** Raj Patel
> 
> (Note: The term "sendgrid" in this query refers to the product/service external_sendgrid in Vertexia's system. The term "twilio" in this query refers to the product/service external_twilio in Vertexia's system.)

**Diagnosis** &nbsp; Regression: was solvable at step 1; now only PARTIAL. The retrieved context lacks the specific runbook instructions for the alert, leading to a hallucination of the first action, though the owner was correctly ide…

**Judge reasoning** &nbsp; The retrieved context lacks the specific runbook instructions for the alert, leading to a hallucination of the first action, though the owner was correctly identified via the vendor contract.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |

</details>

<details>
<summary><b>Q04</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total ARR across all Vertexia customers combined?

**Reference answer (gold):**

> Summing the arr_usd column across all 20 rows in customer_list.csv yields a total of $11,000,000 across all Vertexia customers combined.

**System answer at this step:**

> Based on the provided context, the total ARR across all 20 customers is $11,000,000.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer correctly identifies the total ARR provided in the first retrieved context, though most of the other retrieved documents were irrelevant to the specific query.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q05</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; How many active Vertexia employees are based in the Berlin office?

**Reference answer (gold):**

> There are 5 active Vertexia employees based in the Berlin office.

**System answer at this step:**

> Based on the provided "Active Employees by Office Location" CSV data, there are 5 active employees based in the Berlin office.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and grounded in the provided CSV context, though the retrieval system returned a large amount of irrelevant noise.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q06</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total planned headcount across all departments in Vertexia's 2023 budget allocation?

**Reference answer (gold):**

> Summing the headcount column across all 9 department rows in budget_allocation_2023.csv yields a total planned headcount of 181 for 2023. (This is the budgeted plan, distinct from the company's ~500 actual employees referenced in prose.)

**System answer at this step:**

> The total planned headcount across all 9 departments in Vertexia's 2023 budget allocation is 181.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the provided context, though the retrieval included several redundant and irrelevant chunks.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |

</details>

<details>
<summary><b>Q07</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; According to Vertexia's vendor contracts summary, what is the annual contract value for Snowflake and when does that contract renew?

**Reference answer (gold):**

> Per vendor_contracts_summary.csv, Vertexia's annual contract value for Snowflake is $120,000 and the contract renews on June 30, 2024.

**System answer at this step:**

> According to the Vertexia vendor contracts summary, the annual contract value for Snowflake is $120,000, and the contract renews on June 30, 2024.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the context, though the retrieval process returned many irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q08</b> — PASS · correctness 0.80 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Which NexusFlow API endpoint was deprecated in v2.1, and what endpoint replaced it?

**Reference answer (gold):**

> The GET /v2/events/batch endpoint was deprecated in NexusFlow API v2.1 (and removed in v2.2). It was replaced by GET /v2/events/stream.

**System answer at this step:**

> The term "pulsar" in this query refers to the product/service external_pulsar in Vertexia's system.
> 
> According to the NexusFlow API Changelog, the `GET /v2/events/batch` endpoint was deprecated in v2.1 and replaced by `GET /v2/events/stream`.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer correctly identifies the deprecated and replacement endpoints but includes irrelevant information about 'pulsar' that was not requested.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.80 | 1.00 | 0.70 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q09</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Per Vertexia's Q4 2023 on-call schedule, who was the on-call engineer for the Data Platform Team during the week starting 2023-10-16?

**Reference answer (gold):**

> Per on_call_schedule_q4_2023.csv, Priya Kapoor (employee_id E039) was the on-call engineer for the Data Platform Team during the week of 2023-10-16 to 2023-10-22.

**System answer at this step:**

> The term "pulsar" in this query refers to the product/service external_pulsar in Vertexia's system.
> 
> Per the On Call Schedule Q4 2023, the on-call engineer for the Data Platform Team during the week starting 2023-10-16 was Priya Kapoor (E039).

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer correctly identifies the engineer, but the provided context did not actually contain the on-call schedule, making the answer's claim of source grounding technically hallucinated despite the factual accuracy.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 0.80 | 0.20 | 0.00 |

</details>

<details>
<summary><b>Q10</b> — PASS · correctness 1.00 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Who is the CSM assigned to the customer 'Summit Pharma', what office location is that CSM based in, and who is that CSM's direct manager?

**Reference answer (gold):**

> The CSM assigned to Summit Pharma is Anjali Patel (per csm_account_history.csv). Anjali Patel is based in San Francisco and her direct manager is Maya Sharma (VP Customer Success), per employee_directory.csv.

**System answer at this step:**

> The term "pulsar" in this query refers to the product/service external_pulsar in Vertexia's system.
> 
> Based on the provided documents, the CSM assigned to the customer 'Summit Pharma' is **Anjali Patel**. She is based in the **San Francisco** office, and her direct manager is **Maya Sharma**.

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The answer correctly identifies the CSM, location, and manager, though it includes an unnecessary and irrelevant introductory sentence about 'pulsar'.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 0.80 | 0.60 | 1.00 |

</details>

<details>
<summary><b>Q11</b> — PARTIAL · correctness 0.60 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Based on Vertexia's API dependency graph, which Vertexia services directly depend on NexusFlow's APIs? Name each service AND the specific NexusFlow endpoint it consumes.

**Reference answer (gold):**

> Per api_dependencies.csv, three Vertexia services directly depend on NexusFlow's APIs: InsightLens (via events_api and connectors_api), PulseConnect (via pipeline_status_api), and DataCraft (via ingest_api).

**System answer at this step:**

> Based on the provided documents, the following Vertexia services directly depend on NexusFlow's APIs:
> 
> *   **InsightLens:** Consumes the `/events/stream` endpoint (as identified in the `NexusFlow Events API` spec) and the `events_api` (as identified in the `InsightLens v1.0 PRD` and `Data Platform Team — On-Call Runbook`).
> *   **DataCraft:** Consumes the `/events/stream` endpoint (as identified in the `NexusFlow Events API` spec).
> *   **PulseConnect:** Consumes the `pipeline_status_api` (as identified in the `[KNOWLEDGE GRAPH CONTEXT]` and `Blast radius from PulseConnect` analysis).
> *   **DataCraft (Integration):** Consumes the `ingest_api` (as identified in the `[KNOWLEDGE GRAPH CONTEXT]` and `Blast radius from DataCraft` analysis).
> 
> *Note: While the `[KNOWLEDGE GRAPH CONTEXT]` mentions InsightLens depends on NexusFlow via `connectors_api`, the `NexusFlow Events API` specification specifically lists InsightLens as a consumer of the `/events/stream` endpoint.*

**Diagnosis** &nbsp; Should PASS at this tier but only PARTIAL. The actual answer correctly identifies all three services but adds a non-existent 'DataCraft (Integration)' and misidentifies endpoints; it is mostly grounded but includes minor h…

**Judge reasoning** &nbsp; The actual answer correctly identifies all three services but adds a non-existent 'DataCraft (Integration)' and misidentifies endpoints; it is mostly grounded but includes minor hallucinations not in the contexts.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.60 | 0.80 | 1.00 | 0.60 | 1.00 |

</details>

<details>
<summary><b>Q12</b> — PARTIAL · correctness 0.40 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Which active Vertexia employees located in the Bangalore office report — directly or indirectly, through any number of management hops — to the CTO Sarah Chen? List every such employee.

**Reference answer (gold):**

> Five active Bangalore-based employees report (directly or indirectly) to CTO Sarah Chen: Priya Nair (Lead, Data Platform Team), Kenji Ito (Senior Engineer, Data Platform), Lin Wei (Senior Engineer, NexusFlow), Priya Kapoor (Data Engineer, Data Platform), and Omar Faruk (Engineer, Security).

**System answer at this step:**

> Based on the provided documents, the active employee located in the Bangalore office who reports to CTO Sarah Chen is:
> 
> *   **Priya Nair** (Lead, Data Platform Team)
> 
> **Reasoning:**
> *   According to the Knowledge Graph, Priya Nair is located in Bangalore.
> *   Her manager is Marcus Webb (Lead, Platform Engineering).
> *   Marcus Webb’s manager is Sarah Chen (CTO).
> *   Therefore, Priya Nair reports indirectly to Sarah Chen.
> 
> The provided context does not list any other active employees in the Bangalore office. While the "Active Employees by Office Location" CSV indicates there are 5 active employees in Bangalore, the provided Knowledge Graph only identifies Priya Nair as being located there.

**Diagnosis** &nbsp; Should PASS at this tier but only PARTIAL. The answer correctly identifies Priya Nair but misses four other employees; all claims are grounded in context, though the retrieved context contains enough information to infer t…

**Judge reasoning** &nbsp; The answer correctly identifies Priya Nair but misses four other employees; all claims are grounded in context, though the retrieved context contains enough information to infer the full list with proper traversal.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.40 | 1.00 | 1.00 | 0.80 | 0.60 |

</details>

<details>
<summary><b>Q13</b> — FAIL · correctness 0.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; What is the combined ARR (sum of arr_usd in customer_list.csv) of all customers whose assigned CSM is an active direct report of Maya Sharma per the employee directory?

**Reference answer (gold):**

> Maya Sharma has two active direct reports in employee_directory.csv: Sam Rivera and Anjali Patel (Preet Kaur also reported to Maya but departed in 2023). The customers in customer_list.csv assigned to either of those two CSMs combine to a total ARR of $3,708,000 (Anjali Patel's 5 accounts sum to $2,400,000; Sam Rivera's 5 accounts sum to $1,308,000).

**System answer at this step:**

> Based on the provided context, Maya Sharma (E019) is the VP of Customer Success. Her direct reports are listed as Preet Kaur, Sam Rivera, and Anjali Patel.
> 
> The provided context does not contain information regarding the assigned customers or the ARR for Preet Kaur, Sam Rivera, or Anjali Patel. Therefore, I cannot calculate a combined ARR for their customers.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 5.

**Judge reasoning** &nbsp; The actual answer correctly states it lacks information but fails to extract available data from the context; key customer and CSM data for Sam Rivera and Anjali Patel are missing in retrieved contexts despite being inferable from Maya Sharma's direct reports and customer assignments.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.30 | 0.20 | 0.00 |

</details>

<details>
<summary><b>Q14</b> — PASS · correctness 1.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; Consider Vertexia's active Customer Success Managers (role contains 'Customer Success Manager') who started at the company in 2022 per the employee directory. What is the combined ARR in customer_list.csv of the customers assigned to those CSMs?

**Reference answer (gold):**

> The only active Customer Success Manager in employee_directory.csv who started at Vertexia in 2022 is Anjali Patel (start_date 2022-03-15). The five customers in customer_list.csv assigned to Anjali Patel (Summit Pharma, OmegaLogistics, Redwood Analytics, Ironclad Security, and Northgate Bank) have a combined ARR of $2,400,000.

**System answer at this step:**

> To determine the combined ARR of customers assigned to active Customer Success Managers (CSMs) who started in 2022, we first identify the relevant CSMs from the provided employee directory:
> 
> 1.  **Anjali Patel (E037):**
>     *   Role: Customer Success Manager
>     *   Status: Active
>     *   Start Date: 2022-03-15 (Source 4)
>     *   Assigned Customers (from Knowledge Graph):
>         *   Summit Pharma: $300,000
>         *   OmegaLogistics: $156,000
>         *   Redwood Analytics: $720,000
>         *   Ironclad Security: $384,000
>         *   Northgate Bank: $840,000
>     *   **Total ARR for Anjali Patel: $2,400,000**
> 
> 2.  **Sam Rivera (E036):**
>     *   Role: Customer Success Manager
>     *   Status: Active
>     *   Start Date: 2023-04-01 (Source 2)
>     *   *Note: Does not meet the 2022 start date criteria.*
> 
> 3.  **Preet Kaur (E030):**
>     *   Role: Customer Success Manager (departed)
>     *   Status: Departed (Source 10)
>     *   *Note: Does not meet the "active" status criteria.*
> 
> 4.  **Maya Sharma (E019):**
>     *   Role: VP Customer Success
>     *   *Note: Does not meet the role criteria of "Customer Success Manager".*
> 
> Based on the criteria, only **Anjali Patel** is an active Customer Success Manager who started in 2022. The combined ARR of her assigned customers is **$2,400,000**.

**Diagnosis** &nbsp; Unexpected PASS — question targets step 5's capability, but retrieved context happened to contain enough signal.

**Judge reasoning** &nbsp; The actual answer correctly identifies Anjali Patel as the only qualifying CSM and computes the correct combined ARR; all claims are grounded in the context; the retrieved contexts contain all necessary facts but include significant irrelevant information.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.20 | 1.00 |

</details>

> The full per-question JSON (retrieved contexts, latency, etc.) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
