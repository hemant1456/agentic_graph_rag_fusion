# Step 03 — CSV Tools

## What it adds
Introduces a structured-data tool: a pandas-backed CSV query function that runs alongside dense retrieval. When the question matches an aggregate intent (total ARR, Q3 revenue, headcount by city), the tool computes the exact answer from the source CSV and prepends the result to the LLM context. Newly handles Tier 3 questions (Q05-Q07) where vector retrieval can find the relevant rows but cannot sum them.

## Design
- **Class:** `Step02ToolsRAG` in `step_02_tools/implementation/pipeline.py`
- **Inherits from:** composes `BaselineRAG`'s ChromaDB collection (`vertexia_smart`), retrieval, and generation
- **Key components:**
  - `step_02_tools/implementation/csv_tool.py` — `detect_intent()` regex/keyword router and `run_query()` pandas executor
  - Step 02's `vertexia_smart` ChromaDB collection (no re-ingest)

## How it works
At query time, the pipeline runs the Step 02 dense retrieval to produce a vector context. In parallel, `detect_intent(question)` pattern-matches the question against a small registry of aggregate intents (total ARR, quarterly revenue sum, employee count by location, etc.). If an intent matches, `run_query(intent)` loads the relevant CSV from `dataset/company_data/` with pandas, computes the aggregate, and renders it as a labelled text block. The CSV result is prepended to the vector context, then both are sent to the LLM. If no intent matches, the pipeline degrades to pure dense retrieval.

## Run
```bash
uv run python evaluation/run_eval.py --step step_02_tools
```

## Results
See `step_02_tools/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Dense retrieval can locate the customer-list CSV chunks for Q05 but cannot sum 20 ARR values. The LLM can sometimes add small numbers it sees in context, but it routinely truncates the CSV, hallucinates rows, or drops precision. A deterministic pandas call is both correct and traceable. This step makes the structured side of the corpus first-class instead of treating CSVs as if they were prose.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 6 PASS · 2 PARTIAL · 6 FAIL out of 14 questions (43% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.514 | 0.986 | 0.700 | 0.243 | 0.529 |

### Per-question summary

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q03** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q04** | PARTIAL | 0.50 | `step_02_tools` | Should PASS at this tier but only PARTIAL. The actual answer correctly reports both the $11M CSV total and the $16.5M finance report figure, acknowledging the discrepancy without inventing explanations, but the reference a… |
| **Q05** | PASS | 1.00 | `step_02_tools` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q06** | PASS | 1.00 | `step_02_tools` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q07** | PASS | 1.00 | `step_03_hybrid_retrieval` | Unexpected PASS — question targets step 3's capability, but retrieved context happened to contain enough signal. |
| **Q08** | FAIL | 0.00 | `step_03_hybrid_retrieval` | Expected FAIL — required capability arrives at step 3. |
| **Q09** | FAIL | 0.00 | `step_03_hybrid_retrieval` | Expected FAIL — required capability arrives at step 3. |
| **Q10** | FAIL | 0.30 | `step_04_knowledge_graph` | Expected FAIL — required capability arrives at step 4. |
| **Q11** | PARTIAL | 0.40 | `step_04_knowledge_graph` | Expected — capability arrives at step 4. PARTIAL means retrieval brought some related context. The answer failed to identify PulseConnect and DataCraft as services, incorrectly attributed DataCraft as a consumer of… |
| **Q12** | FAIL | 0.00 | `step_04_knowledge_graph` | Expected FAIL — required capability arrives at step 4. |
| **Q13** | FAIL | 0.00 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |
| **Q14** | FAIL | 0.00 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |

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

**Judge reasoning** &nbsp; The actual answer matches the reference exactly and is fully supported by multiple retrieved contexts; however, only about half of the retrieved contexts are relevant to the question.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |

</details>

<details>
<summary><b>Q02</b> — PASS · correctness 1.00 · tier owner: <code>step_01_baseline_rag</code></summary>

**Question** &nbsp; Who co-founded Vertexia and in what year was the company founded?

**Reference answer (gold):**

> Vertexia was co-founded by Arjun Mehta and Diana Volkov in March 2019.

**System answer at this step:**

> Vertexia was co-founded by Arjun Mehta and Diana Volkov. The company was founded in March 2019.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer matches the reference exactly and is fully supported by the retrieved contexts, though only half of the retrieved contexts were relevant to the question.

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

> For the **PulseConnect webhook_delivery_failure_rate > 5%** alert, the details are as follows:
> 
> *   **First action:** Check SendGrid quota dashboard and the Twilio API status page. Inspect outbound retry queue depth in PulseConnect admin UI.
> *   **Owner:** Raj Patel

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer matches the reference exactly in content and is fully supported by the retrieved contexts; most retrieved contexts are relevant but some are not directly related to the PulseConnect alert.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.70 | 1.00 |

</details>

<details>
<summary><b>Q04</b> — PARTIAL · correctness 0.50 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total ARR across all Vertexia customers combined?

**Reference answer (gold):**

> Summing the arr_usd column across all 20 rows in customer_list.csv yields a total of $11,000,000 across all Vertexia customers combined.

**System answer at this step:**

> Based on the provided documents, there are conflicting figures regarding the total ARR:
> 
> *   The **Q3 2023 Finance Report** and the **Series C Announcement** state that Vertexia Inc.'s ARR was **$16.5 million** at the close of Q3 2023.
> *   The **structured CSV query** (customer_list.csv) reports a total ARR of **$11,000,000** across all 20 customers.
> 
> The provided context does not explain the discrepancy between these two figures.

**Diagnosis** &nbsp; Should PASS at this tier but only PARTIAL. The actual answer correctly reports both the $11M CSV total and the $16.5M finance report figure, acknowledging the discrepancy without inventing explanations, but the reference a…

**Judge reasoning** &nbsp; The actual answer correctly reports both the $11M CSV total and the $16.5M finance report figure, acknowledging the discrepancy without inventing explanations, but the reference answer only expects the $11M sum, making it partially correct; all claims are grounded in retrieved contexts; the answer d

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.50 | 1.00 | 1.00 | 0.20 | 1.00 |

</details>

<details>
<summary><b>Q05</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; How many active Vertexia employees are based in the Berlin office?

**Reference answer (gold):**

> There are 5 active Vertexia employees based in the Berlin office.

**System answer at this step:**

> Based on the provided context, there are 5 active employees based in the Berlin office.

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The actual answer correctly states 5 active employees in Berlin, is fully supported by context [1], directly answers the question, all reference facts are in retrieved contexts, but only 1 of 11 contexts is relevant.

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

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the provided structured query context, though the retrieval set contained significant irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.30 | 1.00 |

</details>

<details>
<summary><b>Q07</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; According to Vertexia's vendor contracts summary, what is the annual contract value for Snowflake and when does that contract renew?

**Reference answer (gold):**

> Per vendor_contracts_summary.csv, Vertexia's annual contract value for Snowflake is $120,000 and the contract renews on June 30, 2024.

**System answer at this step:**

> According to the Vertexia vendor contracts summary, the annual contract value for Snowflake is $120,000, and the contract renews on June 30, 2024.

**Diagnosis** &nbsp; Unexpected PASS — question targets step 3's capability, but retrieved context happened to contain enough signal.

**Judge reasoning** &nbsp; The answer is factually accurate and fully supported by the retrieved context, though only one of the ten provided chunks was relevant to the query.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q08</b> — FAIL · correctness 0.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Which NexusFlow API endpoint was deprecated in v2.1, and what endpoint replaced it?

**Reference answer (gold):**

> The GET /v2/events/batch endpoint was deprecated in NexusFlow API v2.1 (and removed in v2.2). It was replaced by GET /v2/events/stream.

**System answer at this step:**

> The provided documents do not contain information regarding the deprecation of a specific NexusFlow API endpoint in version 2.1, nor do they mention a replacement endpoint for such a change. The available documentation only covers API changes for v1.0 and v1.1.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 3.

**Judge reasoning** &nbsp; The model correctly identified that the information was missing from the provided context, but the context itself failed to contain the necessary information to answer the question.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.30 | 0.00 | 0.00 |

</details>

<details>
<summary><b>Q09</b> — FAIL · correctness 0.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Per Vertexia's Q4 2023 on-call schedule, who was the on-call engineer for the Data Platform Team during the week starting 2023-10-16?

**Reference answer (gold):**

> Per on_call_schedule_q4_2023.csv, Priya Kapoor (employee_id E039) was the on-call engineer for the Data Platform Team during the week of 2023-10-16 to 2023-10-22.

**System answer at this step:**

> The provided documents do not contain information regarding the specific on-call schedule for the Data Platform Team for the week starting 2023-10-16. Therefore, I cannot answer your question.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 3.

**Judge reasoning** &nbsp; The model correctly identified that the provided context lacked the requested information, but the retrieval system failed to provide the necessary document containing the on-call schedule.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.30 | 0.00 | 0.00 |

</details>

<details>
<summary><b>Q10</b> — FAIL · correctness 0.30 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Who is the CSM assigned to the customer 'Summit Pharma', what office location is that CSM based in, and who is that CSM's direct manager?

**Reference answer (gold):**

> The CSM assigned to Summit Pharma is Anjali Patel (per csm_account_history.csv). Anjali Patel is based in San Francisco and her direct manager is Maya Sharma (VP Customer Success), per employee_directory.csv.

**System answer at this step:**

> Based on the provided documents, the CSM assigned to Summit Pharma is Anjali Patel.
> 
> The provided context does not contain information regarding Anjali Patel's office location or her direct manager.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 4.

**Judge reasoning** &nbsp; The model correctly identified the CSM but failed to retrieve the office location and manager information present in the provided context, leading to an incomplete answer.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.30 | 1.00 | 0.30 | 0.20 | 0.00 |

</details>

<details>
<summary><b>Q11</b> — PARTIAL · correctness 0.40 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Based on Vertexia's API dependency graph, which Vertexia services directly depend on NexusFlow's APIs? Name each service AND the specific NexusFlow endpoint it consumes.

**Reference answer (gold):**

> Per api_dependencies.csv, three Vertexia services directly depend on NexusFlow's APIs: InsightLens (via events_api and connectors_api), PulseConnect (via pipeline_status_api), and DataCraft (via ingest_api).

**System answer at this step:**

> Based on the provided documents, the following services directly depend on NexusFlow's APIs:
> 
> *   **InsightLens:**
>     *   Consumes the **`events_api`** (specifically the `/events/stream` endpoint, as noted in the `NexusFlow Events API` spec and the `InsightLens` PRD).
>     *   Consumes the **`connectors_api`** (as noted in the `api_dependencies.csv` file).
> *   **DataCraft Integration:**
>     *   Consumes the **`events_api`** (specifically the `/events/stream` endpoint, as noted in the `NexusFlow Events API` spec).

**Diagnosis** &nbsp; Expected — capability arrives at step 4. PARTIAL means retrieval brought some related context. The answer failed to identify PulseConnect and DataCraft as services, incorrectly attributed DataCraft as a consumer of…

**Judge reasoning** &nbsp; The answer failed to identify PulseConnect and DataCraft as services, incorrectly attributed DataCraft as a consumer of the events_api, and the provided context lacked the specific dependency mapping required to fully answer the prompt.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.40 | 0.80 | 1.00 | 0.40 | 0.40 |

</details>

<details>
<summary><b>Q12</b> — FAIL · correctness 0.00 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Which active Vertexia employees located in the Bangalore office report — directly or indirectly, through any number of management hops — to the CTO Sarah Chen? List every such employee.

**Reference answer (gold):**

> Five active Bangalore-based employees report (directly or indirectly) to CTO Sarah Chen: Priya Nair (Lead, Data Platform Team), Kenji Ito (Senior Engineer, Data Platform), Lin Wei (Senior Engineer, NexusFlow), Priya Kapoor (Data Engineer, Data Platform), and Omar Faruk (Engineer, Security).

**System answer at this step:**

> Based on the provided documents, there is not enough information to list the specific active employees in the Bangalore office who report to CTO Sarah Chen.
> 
> While the documents state that Priya Nair leads the Bangalore office (Source 8) and that Sarah Chen oversees the new organizational structure including Platform Engineering and Product Engineering (Source 6), the provided context does not contain a comprehensive list of individual employees or their specific reporting lines to the CTO.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 4.

**Judge reasoning** &nbsp; The model correctly identified that the provided context lacked the necessary information to answer the question, though the context recall is 0.0 because the required data was not present in the retrieved chunks.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.30 | 0.10 | 0.00 |

</details>

<details>
<summary><b>Q13</b> — FAIL · correctness 0.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; What is the combined ARR (sum of arr_usd in customer_list.csv) of all customers whose assigned CSM is an active direct report of Maya Sharma per the employee directory?

**Reference answer (gold):**

> Maya Sharma has two active direct reports in employee_directory.csv: Sam Rivera and Anjali Patel (Preet Kaur also reported to Maya but departed in 2023). The customers in customer_list.csv assigned to either of those two CSMs combine to a total ARR of $3,708,000 (Anjali Patel's 5 accounts sum to $2,400,000; Sam Rivera's 5 accounts sum to $1,308,000).

**System answer at this step:**

> The provided context does not contain an employee directory or information regarding the reporting structure of employees at Vertexia Inc. Therefore, I cannot determine which CSMs are direct reports of Maya Sharma, and I cannot calculate the combined ARR for those customers.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 5.

**Judge reasoning** &nbsp; The model correctly identified that the employee directory was missing from the context, but the context recall is 0 because the necessary information to answer the question was not provided to the model.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.30 | 0.20 | 0.00 |

</details>

<details>
<summary><b>Q14</b> — FAIL · correctness 0.00 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; Consider Vertexia's active Customer Success Managers (role contains 'Customer Success Manager') who started at the company in 2022 per the employee directory. What is the combined ARR in customer_list.csv of the customers assigned to those CSMs?

**Reference answer (gold):**

> The only active Customer Success Manager in employee_directory.csv who started at Vertexia in 2022 is Anjali Patel (start_date 2022-03-15). The five customers in customer_list.csv assigned to Anjali Patel (Summit Pharma, OmegaLogistics, Redwood Analytics, Ironclad Security, and Northgate Bank) have a combined ARR of $2,400,000.

**System answer at this step:**

> The provided context does not contain an employee directory or information regarding the start dates of Customer Success Managers at Vertexia. Therefore, I cannot identify which CSMs started in 2022 or calculate the combined ARR for their assigned customers.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 5.

**Judge reasoning** &nbsp; The model failed to identify the employee directory in the provided context, leading to an incorrect refusal despite the necessary information being present in the retrieved chunks.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.30 | 0.10 | 0.00 |

</details>

> The full per-question JSON (retrieved contexts, latency, etc.) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
