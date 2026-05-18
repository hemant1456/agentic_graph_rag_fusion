# Step 06 — Context Engineering + VSA

## What it adds
Combines two layers in a single step. First, a keyword-scored Vertical Slice Architecture (VSA) router dispatches each question to a domain slice (Finance / HR / Engineering / General) that owns its own system prompt, retrieval-query augmentation, rerank top-k, and compression ratio. Second, every slice runs the same four-stage context-engineering stack: a CrossEncoder reranker reorders a wide candidate set, a Jaccard-based deduplicator drops near-identical chunks, an extractive compressor keeps only the sentences most relevant to the question, and an XML formatter packs the result into a token budget.

## Design
- **Class:** `Step06RAG` in `step_06_context_engineering/implementation/pipeline.py`
- **Inherits from:** composes `Step03HybridRAG` (k=20 wide candidates) and reuses the Step 05 agents
- **Key components:**
  - `step_06_context_engineering/implementation/router.py` — `dispatch()` scores slices by keyword overlap and returns the winning slice plus confidence
  - `step_06_context_engineering/implementation/slices/base.py` — shared `SliceConfig` contract + `run_with_config()` that executes the CE + synthesis pipeline with slice-specific overrides
  - `step_06_context_engineering/implementation/slices/finance_slice.py` — finance-tuned prompt, exact-number formatting rules, CSV-forced
  - `step_06_context_engineering/implementation/slices/hr_slice.py` — HR-tuned prompt, graph-forced, employee/org query augmentation
  - `step_06_context_engineering/implementation/slices/engineering_slice.py` — engineering-tuned prompt, graph-forced, product-name keyword expansion
  - `step_06_context_engineering/implementation/slices/general_slice.py` — fallback for low-confidence routes
  - `step_06_context_engineering/implementation/reranker.py` — CrossEncoder rerank to top `rerank_k` (default 8)
  - `step_06_context_engineering/implementation/deduplicator.py` — Jaccard near-duplicate removal
  - `step_06_context_engineering/implementation/compressor.py` — sentence-level extractive compression to `compress_ratio` (default 0.60)
  - `step_06_context_engineering/implementation/formatter.py` — XML envelope with explicit token budget
  - `step_06_context_engineering/implementation/context_engineer.py` — `engineer_context()` orchestrates the four stages and emits `ce_metrics`

## How it works
At query time, `router.dispatch()` scores the question against each slice's keyword lexicon and returns the highest-scoring slice along with a confidence in `[0, 1]`. The chosen slice runs the context-engineered pipeline using its own configuration: which system prompt, which query augmentation, which `rerank_k`, which `compress_ratio`. QueryAnalyst runs first to produce sub-questions. RetrievalSpecialist fetches 20 chunks for the (possibly augmented) main question and 10 more for each of up to four sub-questions. GraphNavigator and StructuredData run unconditionally so exact graph and CSV facts are preserved verbatim. The raw chunks then enter `engineer_context()`: the reranker reorders them by question relevance, the deduplicator removes near-duplicates, the compressor keeps the top sentences from each survivor, and the formatter wraps everything (CSV data, graph context, compressed chunks) inside an XML envelope sized to the token budget. The engineered XML is the only context sent to Synthesis using the slice's system prompt; Critic then reviews. The returned `Step06Result` exposes the slice name, router confidence, and engineering metrics.

## Run
```bash
uv run python evaluation/run_eval.py --step step_06_context_engineering
```

## Results
See `step_06_context_engineering/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Step 05 sends up to 20 chunks plus graph and CSV blocks straight to the LLM with one generic system prompt. Many chunks overlap, many contain only one or two relevant sentences, and one prompt cannot be optimal for every domain — finance answers must be exact numbers, HR answers must respect reporting chains, engineering answers must use canonical product names. CrossEncoder reranking promotes the chunks the bi-encoder underweighted; dedup cuts redundancy; compression trims filler sentences; XML formatting gives the model a clearly delimited budget. The VSA router then routes each question to a domain-specific prompt and configuration so each slice can be tuned independently. Pure-keyword routing keeps the dispatch cost near zero, and the exact graph and CSV outputs bypass compression so deterministic facts are never lossy.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 11 PASS · 1 PARTIAL · 2 FAIL out of 14 questions (79% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.814 | 0.700 | 1.000 | 0.345 | 0.657 |

### Per-question summary

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q03** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q04** | FAIL | 0.00 | `step_02_tools` | Regression: was solvable at step 2; now FAILED. The answer provides a specific figure not found in the provided context, indicating a hallucination or reliance on external data not present in the retrieved d… |
| **Q05** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q06** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q07** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q08** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q09** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q10** | PASS | 1.00 | `step_04_knowledge_graph` | Continues to PASS from an earlier tier — capability still works. |
| **Q11** | PASS | 1.00 | `step_04_knowledge_graph` | Continues to PASS from an earlier tier — capability still works. |
| **Q12** | PARTIAL | 0.40 | `step_04_knowledge_graph` | Regression: was solvable at step 4; now only PARTIAL. The actual answer correctly identifies one employee (Priya Nair) with a valid reporting chain to Sarah Chen, but misses four others listed in the reference; al… |
| **Q13** | FAIL | 0.00 | `step_05_multi_agent` | Regression: was solvable at step 5; now FAILED. The answer provides an incorrect calculation and hallucinates a value not supported by the provided context, which itself lacked the necessary data to perform… |
| **Q14** | PASS | 1.00 | `step_05_multi_agent` | Continues to PASS from an earlier tier — capability still works. |

### Per-question detail (question · reference · system answer · judge)

Each entry is collapsed; click to expand the full debug view.

<details>
<summary><b>Q01</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; What is Vertexia's customer data retention policy for hot storage and cold storage?

**Reference answer (gold):**

> Vertexia stores customer data in hot storage for 90 days, then archives to AWS S3 Glacier for 1 year of cold storage.

**System answer at this step:**

> Vertexia's customer data retention policy is 90 days for hot storage and 1 year for cold storage (S3 Glacier).

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the retrieved contexts, though some retrieved chunks were irrelevant to the specific question.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.67 | 1.00 |

</details>

<details>
<summary><b>Q02</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; Who co-founded Vertexia and in what year was the company founded?

**Reference answer (gold):**

> Vertexia was co-founded by Arjun Mehta and Diana Volkov in March 2019.

**System answer at this step:**

> Vertexia was co-founded by Arjun Mehta and Diana Volkov in 2019.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is factually accurate and fully supported by the provided contexts, though only two of the six retrieved chunks were relevant to the query.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.33 | 1.00 |

</details>

<details>
<summary><b>Q03</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; In Vertexia's on-call runbook, what is the first action and the escalation owner for the PulseConnect webhook delivery failure alert?

**Reference answer (gold):**

> For the PulseConnect webhook_delivery_failure_rate > 5% alert, the first action is to check the SendGrid quota dashboard and the Twilio API status page (and inspect the outbound retry queue depth in the PulseConnect admin UI). The escalation owner is Raj Patel.

**System answer at this step:**

> For the PulseConnect webhook delivery failure alert, the first action is to check the SendGrid quota dashboard and the external_twilio API status page, and to inspect the outbound retry queue depth in the PulseConnect admin UI. The escalation owner is Raj Patel.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer accurately extracts the required information from the relevant context, though the retrieved set included several irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.20 | 1.00 |

</details>

<details>
<summary><b>Q04</b> — FAIL · correctness 0.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total ARR across all Vertexia customers combined?

**Reference answer (gold):**

> Summing the arr_usd column across all 20 rows in customer_list.csv yields a total of $11,000,000 across all Vertexia customers combined.

**System answer at this step:**

> $11,000,000. This is the total ARR across all 20 customers as shown in the customer_list.csv data.

**Diagnosis** &nbsp; Regression: was solvable at step 2; now FAILED. The answer provides a specific figure not found in the provided context, indicating a hallucination or reliance on external data not present in the retrieved d…

**Judge reasoning** &nbsp; The answer provides a specific figure not found in the provided context, indicating a hallucination or reliance on external data not present in the retrieved documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |

</details>

<details>
<summary><b>Q05</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; How many active Vertexia employees are based in the Berlin office?

**Reference answer (gold):**

> There are 5 active Vertexia employees based in the Berlin office.

**System answer at this step:**

> There are 5 active Vertexia employees based in the Berlin office.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer matches the reference but is a hallucination because the provided context contains no information about a Berlin office or its employee count.

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

**Judge reasoning** &nbsp; The answer is accurate and grounded, though the retrieved context included irrelevant documents like the Q3 finance report.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.60 | 1.00 |

</details>

<details>
<summary><b>Q07</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; According to Vertexia's vendor contracts summary, what is the annual contract value for Snowflake and when does that contract renew?

**Reference answer (gold):**

> Per vendor_contracts_summary.csv, Vertexia's annual contract value for Snowflake is $120,000 and the contract renews on June 30, 2024.

**System answer at this step:**

> The annual contract value for Snowflake is $120,000, and the contract renews on June 30, 2024.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is perfectly accurate and grounded in the provided context, though the retrieval step included several irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.17 | 1.00 |

</details>

<details>
<summary><b>Q08</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Which NexusFlow API endpoint was deprecated in v2.1, and what endpoint replaced it?

**Reference answer (gold):**

> The GET /v2/events/batch endpoint was deprecated in NexusFlow API v2.1 (and removed in v2.2). It was replaced by GET /v2/events/stream.

**System answer at this step:**

> The NexusFlow API endpoint deprecated in v2.1 is `GET /v2/events/batch`. It was replaced by `GET /v2/events/stream`.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer correctly identifies the deprecated and replacement endpoints as per the reference, all claims are grounded in context [6], and it directly answers the question; however, only one of the six retrieved contexts ([6]) is relevant, lowering context precision.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |

</details>

<details>
<summary><b>Q09</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Per Vertexia's Q4 2023 on-call schedule, who was the on-call engineer for the Data Platform Team during the week starting 2023-10-16?

**Reference answer (gold):**

> Per on_call_schedule_q4_2023.csv, Priya Kapoor (employee_id E039) was the on-call engineer for the Data Platform Team during the week of 2023-10-16 to 2023-10-22.

**System answer at this step:**

> **Priya Kapoor** — According to Vertexia's Q4 2023 on-call schedule, the on-call engineer for the Data Platform Team during the week starting 2023-10-16 was Priya Kapoor (employee ID E039).

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer is factually correct and on-topic, but the retrieved contexts do not contain the specific on-call schedule data or mention Priya Kapoor, making the answer ungrounded in the provided context.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | 1.00 | 0.17 | 0.00 |

</details>

<details>
<summary><b>Q10</b> — PASS · correctness 1.00 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Who is the CSM assigned to the customer 'Summit Pharma', what office location is that CSM based in, and who is that CSM's direct manager?

**Reference answer (gold):**

> The CSM assigned to Summit Pharma is Anjali Patel (per csm_account_history.csv). Anjali Patel is based in San Francisco and her direct manager is Maya Sharma (VP Customer Success), per employee_directory.csv.

**System answer at this step:**

> Anjali Patel is the CSM assigned to Summit Pharma. She is based in the San Francisco office. Her direct manager is Maya Sharma (VP Customer Success).

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer correctly identifies Anjali Patel as the CSM for Summit Pharma, her San Francisco location, and her manager Maya Sharma; all claims are supported by retrieved contexts, though some retrieved contexts (e.g., engineering doc, Crestwood Pharma row) are irrelevant.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.60 | 1.00 |

</details>

<details>
<summary><b>Q11</b> — PASS · correctness 1.00 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Based on Vertexia's API dependency graph, which Vertexia services directly depend on NexusFlow's APIs? Name each service AND the specific NexusFlow endpoint it consumes.

**Reference answer (gold):**

> Per api_dependencies.csv, three Vertexia services directly depend on NexusFlow's APIs: InsightLens (via events_api and connectors_api), PulseConnect (via pipeline_status_api), and DataCraft (via ingest_api).

**System answer at this step:**

> Based on the API dependency graph, the following Vertexia services directly depend on NexusFlow's APIs:
> 
> 1. **InsightLens** — consumes the `events_api` (critical) and the `connectors_api` (optional).
> 2. **PulseConnect** — consumes the `pipeline_status_api` (optional).
> 3. **DataCraft** — consumes the `ingest_api` (critical).

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer correctly identifies all services and endpoints as in the reference, with minor unsupported details ('critical'/'optional') not in contexts; all key facts are present in retrieved contexts, but some retrieved chunks are irrelevant.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 0.80 | 1.00 | 0.60 | 1.00 |

</details>

<details>
<summary><b>Q12</b> — PARTIAL · correctness 0.40 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Which active Vertexia employees located in the Bangalore office report — directly or indirectly, through any number of management hops — to the CTO Sarah Chen? List every such employee.

**Reference answer (gold):**

> Five active Bangalore-based employees report (directly or indirectly) to CTO Sarah Chen: Priya Nair (Lead, Data Platform Team), Kenji Ito (Senior Engineer, Data Platform), Lin Wei (Senior Engineer, NexusFlow), Priya Kapoor (Data Engineer, Data Platform), and Omar Faruk (Engineer, Security).

**System answer at this step:**

> The following active employee in the Bangalore office reports to CTO Sarah Chen:
> 
> *   **Priya Nair** (Lead, Data Platform Team): Reports to Marcus Webb (Lead, Platform Engineering), who reports to Sarah Chen (CTO).

**Diagnosis** &nbsp; Regression: was solvable at step 4; now only PARTIAL. The actual answer correctly identifies one employee (Priya Nair) with a valid reporting chain to Sarah Chen, but misses four others listed in the reference; al…

**Judge reasoning** &nbsp; The actual answer correctly identifies one employee (Priya Nair) with a valid reporting chain to Sarah Chen, but misses four others listed in the reference; all claims are supported by context, but retrieved contexts lack most employee details needed for full recall.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.40 | 1.00 | 1.00 | 0.33 | 0.20 |

</details>

<details>
<summary><b>Q13</b> — FAIL · correctness 0.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; What is the combined ARR (sum of arr_usd in customer_list.csv) of all customers whose assigned CSM is an active direct report of Maya Sharma per the employee directory?

**Reference answer (gold):**

> Maya Sharma has two active direct reports in employee_directory.csv: Sam Rivera and Anjali Patel (Preet Kaur also reported to Maya but departed in 2023). The customers in customer_list.csv assigned to either of those two CSMs combine to a total ARR of $3,708,000 (Anjali Patel's 5 accounts sum to $2,400,000; Sam Rivera's 5 accounts sum to $1,308,000).

**System answer at this step:**

> The combined ARR of customers assigned to active direct reports of Maya Sharma is $2,160,000.
> 
> **Supporting Context:**
> *   **Maya Sharma's direct reports:** Preet Kaur (departed), Sam Rivera (active), and Anjali Patel (active).
> *   **Active direct reports:** Sam Rivera and Anjali Patel.
> *   **Customers assigned to Sam Rivera:** Coastal Bancorp ($216,000), NovaTech Systems ($264,000), BlueRidge Energy ($480,000), Pacific Dynamics ($48,000), and Crestwood Pharma ($300,000).
> *   **Customers assigned to Anjali Patel:** Summit Pharma ($300,000), OmegaLogistics ($156,000), Redwood Analytics ($720,000), Ironclad Security ($384,000), and Northgate Bank ($840,000).
> *   **Calculation:** $216,000 + $264,000 + $480,000 + $48,000 + $300,000 + $300,000 + $156,000 + $720,000 + $384,000 + $840,000 = $3,708,000.
> 
> *(Correction: Based on the provided customer records, the sum of ARR for customers assigned to active direct reports Sam Rivera and Anjali Patel is $3,708,000.)*

**Diagnosis** &nbsp; Regression: was solvable at step 5; now FAILED. The answer provides an incorrect calculation and hallucinates a value not supported by the provided context, which itself lacked the necessary data to perform…

**Judge reasoning** &nbsp; The answer provides an incorrect calculation and hallucinates a value not supported by the provided context, which itself lacked the necessary data to perform the full calculation.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 1.00 | 0.16 | 0.00 |

</details>

<details>
<summary><b>Q14</b> — PASS · correctness 1.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; Consider Vertexia's active Customer Success Managers (role contains 'Customer Success Manager') who started at the company in 2022 per the employee directory. What is the combined ARR in customer_list.csv of the customers assigned to those CSMs?

**Reference answer (gold):**

> The only active Customer Success Manager in employee_directory.csv who started at Vertexia in 2022 is Anjali Patel (start_date 2022-03-15). The five customers in customer_list.csv assigned to Anjali Patel (Summit Pharma, OmegaLogistics, Redwood Analytics, Ironclad Security, and Northgate Bank) have a combined ARR of $2,400,000.

**System answer at this step:**

> Anjali Patel is the only active Customer Success Manager who started in 2022.
> 
> *   **Anjali Patel (E037):** Started 2022-03-15; status: active.
> *   **Assigned Customers & ARR:**
>     *   Summit Pharma: $300,000
>     *   OmegaLogistics: $156,000
>     *   Redwood Analytics: $720,000
>     *   Ironclad Security: $384,000
>     *   Northgate Bank: $840,000
> 
> **Combined ARR:** $2,400,000

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer correctly identifies the CSM and calculates the total ARR, though the retrieved contexts included irrelevant rows for other employees.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |

</details>

> The full per-question JSON (retrieved contexts, latency, etc.) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
