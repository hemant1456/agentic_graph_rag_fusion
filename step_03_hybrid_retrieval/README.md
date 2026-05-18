# Step 04 — Hybrid Retrieval (BM25 + Dense)

## What it adds
Adds a BM25 lexical retriever alongside the dense vector retriever and fuses the two ranked lists with Reciprocal Rank Fusion (RRF). Dense retrieval covers semantic paraphrase; BM25 anchors on exact tokens — version strings, finding IDs, vendor names. Newly handles Tier 4 questions (Q08-Q10) where the answer hinges on a rare keyword.

## Design
- **Class:** `Step03HybridRAG` in `step_03_hybrid_retrieval/implementation/pipeline.py`
- **Inherits from:** does not subclass, but reuses Step 02's ChromaDB collection and Step 03's CSV tool
- **Key components:**
  - `step_03_hybrid_retrieval/implementation/bm25_retriever.py` — `BM25Index` built over the same chunks stored in ChromaDB
  - `_rrf_fuse()` in `step_03_hybrid_retrieval/implementation/pipeline.py` — reciprocal rank fusion with `k_rrf=60`
  - Step 03's CSV tool (`detect_intent` / `run_query`)

## How it works
On `build()`, the pipeline loads the `vertexia_smart` ChromaDB collection and constructs a BM25 index over the same chunk texts. At query time it runs dense cosine search and BM25 in parallel, each returning roughly `2k` candidates. The two ranked lists are merged with RRF: each chunk earns `1 / (60 + rank)` from each list it appears in, scores are summed, and the top-`k` chunks are kept. The fused chunks plus any CSV tool result are formatted into the LLM context. Step 04 also becomes the retrieval substrate that Steps 05-10 build on.

## Run
```bash
uv run python evaluation/run_eval.py --step step_03_hybrid_retrieval
```

## Results
See `step_03_hybrid_retrieval/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Dense embeddings normalise away the distinctive tokens that some questions depend on. Q08 ("which NexusFlow endpoint was deprecated in v2.1") needs the chunk that literally contains the string `v2.1`; cosine similarity treats `v2.0`, `v2.1`, and `v2.2` as near-equivalent. Q09 ("audit finding M-2") and Q10 ("Snowflake contract") have the same shape. BM25 ranks those rare tokens highly and RRF lets them rise without throwing away the semantic recall that dense retrieval still provides.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 9 PASS · 2 PARTIAL · 3 FAIL out of 14 questions (64% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.721 | 0.914 | 0.921 | 0.256 | 0.757 |

### Per-question summary

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q03** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q04** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q05** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q06** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q07** | PASS | 1.00 | `step_03_hybrid_retrieval` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q08** | PASS | 1.00 | `step_03_hybrid_retrieval` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q09** | PASS | 1.00 | `step_03_hybrid_retrieval` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q10** | PARTIAL | 0.50 | `step_04_knowledge_graph` | Expected — capability arrives at step 4. PARTIAL means retrieval brought some related context. The answer correctly identifies Anjali Patel as the CSM for Summit Pharma but honestly states that location and manager… |
| **Q11** | PARTIAL | 0.40 | `step_04_knowledge_graph` | Expected — capability arrives at step 4. PARTIAL means retrieval brought some related context. The actual answer correctly identifies InsightLens and mentions DataCraft Integration but incorrectly maps endpoints an… |
| **Q12** | FAIL | 0.00 | `step_04_knowledge_graph` | Expected FAIL — required capability arrives at step 4. |
| **Q13** | FAIL | 0.00 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |
| **Q14** | FAIL | 0.20 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |

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

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the retrieved documents, though several retrieved chunks were irrelevant to the specific question asked.

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

> Vertexia was co-founded by Arjun Mehta and Diana Volkov in March 2019.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The actual answer matches the reference answer and is supported by the retrieved contexts.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.20 | 1.00 |

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

**Judge reasoning** &nbsp; The answer is perfectly accurate and grounded, though the retrieval process returned many irrelevant documents.

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

> The total ARR across all 20 customers is $11,000,000.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer correctly identifies the total ARR from the provided structured context, though most of the retrieved documents were irrelevant to the specific query.

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

> Based on the provided "Active Employees by Office Location" CSV query, there are 5 active employees based in the Berlin office.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and grounded in the provided context, though the retrieval process returned many irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.09 | 1.00 |

</details>

<details>
<summary><b>Q06</b> — PASS · correctness 1.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total planned headcount across all departments in Vertexia's 2023 budget allocation?

**Reference answer (gold):**

> Summing the headcount column across all 9 department rows in budget_allocation_2023.csv yields a total planned headcount of 181 for 2023. (This is the budgeted plan, distinct from the company's ~500 actual employees referenced in prose.)

**System answer at this step:**

> The total planned headcount across all 9 departments in Vertexia's 2023 budget allocation is 181.

**Diagnosis** &nbsp; Continues to PASS from an earlier tier — capability still works.

**Judge reasoning** &nbsp; The answer is accurate and fully supported, though the retrieved context set contained significant redundant noise.

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

> According to the Vendor Contracts Summary, the annual contract value for Snowflake is $120,000 USD, and the contract renewal date is June 30, 2024.

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the retrieved context, though the system retrieved many irrelevant documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q08</b> — PASS · correctness 1.00 · tier owner: <code>step_03_hybrid_retrieval</code></summary>

**Question** &nbsp; Which NexusFlow API endpoint was deprecated in v2.1, and what endpoint replaced it?

**Reference answer (gold):**

> The GET /v2/events/batch endpoint was deprecated in NexusFlow API v2.1 (and removed in v2.2). It was replaced by GET /v2/events/stream.

**System answer at this step:**

> The `GET /v2/events/batch` endpoint was deprecated in v2.1. It was replaced by `GET /v2/events/stream`.

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The answer is accurate and fully supported by the context, though the retrieved set contained significant irrelevant documents.

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

> Per the Q4 2023 on-call schedule, the on-call engineer for the Data Platform Team during the week starting 2023-10-16 was Priya Kapoor.

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The actual answer correctly identifies Priya Kapoor as the on-call engineer for the specified week, fully matching the reference; all claims are grounded in the retrieved context (specifically context [8]); the answer directly addresses the question; only one of the ten retrieved contexts (on_call_s

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |

</details>

<details>
<summary><b>Q10</b> — PARTIAL · correctness 0.50 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Who is the CSM assigned to the customer 'Summit Pharma', what office location is that CSM based in, and who is that CSM's direct manager?

**Reference answer (gold):**

> The CSM assigned to Summit Pharma is Anjali Patel (per csm_account_history.csv). Anjali Patel is based in San Francisco and her direct manager is Maya Sharma (VP Customer Success), per employee_directory.csv.

**System answer at this step:**

> Based on the provided documents, the CSM assigned to Summit Pharma is Anjali Patel.
> 
> The provided context does not contain information regarding the office location of Anjali Patel or the identity of her direct manager.

**Diagnosis** &nbsp; Expected — capability arrives at step 4. PARTIAL means retrieval brought some related context. The answer correctly identifies Anjali Patel as the CSM for Summit Pharma but honestly states that location and manager…

**Judge reasoning** &nbsp; The answer correctly identifies Anjali Patel as the CSM for Summit Pharma but honestly states that location and manager info are missing; all claims are grounded, but the retrieved contexts lack the employee directory with manager and location details.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.50 | 1.00 | 1.00 | 0.40 | 0.60 |

</details>

<details>
<summary><b>Q11</b> — PARTIAL · correctness 0.40 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Based on Vertexia's API dependency graph, which Vertexia services directly depend on NexusFlow's APIs? Name each service AND the specific NexusFlow endpoint it consumes.

**Reference answer (gold):**

> Per api_dependencies.csv, three Vertexia services directly depend on NexusFlow's APIs: InsightLens (via events_api and connectors_api), PulseConnect (via pipeline_status_api), and DataCraft (via ingest_api).

**System answer at this step:**

> Based on the provided documents, the following services directly depend on NexusFlow's APIs:
> 
> *   **InsightLens:** Consumes the `/events/stream` endpoint (as noted in the `NexusFlow Events API` specification and the InsightLens PRD).
> *   **DataCraft Integration:** Consumes the `/events/stream` endpoint (as noted in the `NexusFlow Events API` specification).
> 
> Additionally, the `Data Platform Team — On-Call Runbook` notes that **InsightLens** depends on the `events-api` for its analytics ingestion path, and the `NexusFlow System Architecture` document mentions that NexusFlow delivers data to **internal services like InsightLens**.

**Diagnosis** &nbsp; Expected — capability arrives at step 4. PARTIAL means retrieval brought some related context. The actual answer correctly identifies InsightLens and mentions DataCraft Integration but incorrectly maps endpoints an…

**Judge reasoning** &nbsp; The actual answer correctly identifies InsightLens and mentions DataCraft Integration but incorrectly maps endpoints and misses PulseConnect; all claims are grounded in context, but key facts from api_dependencies.csv are missing in retrieval.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.40 | 1.00 | 1.00 | 0.80 | 0.60 |

</details>

<details>
<summary><b>Q12</b> — FAIL · correctness 0.00 · tier owner: <code>step_04_knowledge_graph</code></summary>

**Question** &nbsp; Which active Vertexia employees located in the Bangalore office report — directly or indirectly, through any number of management hops — to the CTO Sarah Chen? List every such employee.

**Reference answer (gold):**

> Five active Bangalore-based employees report (directly or indirectly) to CTO Sarah Chen: Priya Nair (Lead, Data Platform Team), Kenji Ito (Senior Engineer, Data Platform), Lin Wei (Senior Engineer, NexusFlow), Priya Kapoor (Data Engineer, Data Platform), and Omar Faruk (Engineer, Security).

**System answer at this step:**

> Based on the provided documents, there is no information available that lists the specific names of active employees located in the Bangalore office, nor is there a comprehensive directory linking individual employees to their specific office locations and reporting lines.
> 
> While the documents identify the leadership structure (Sarah Chen as CTO, with Marcus Webb and Tomás García reporting to her, and their respective sub-teams including Priya Nair, Daniel Osei, Yuki Tanaka, Aisha Johnson, and Raj Patel), the provided context does not specify which of these individuals or any other employees are based in the Bangalore office.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 4.

**Judge reasoning** &nbsp; The actual answer correctly states that the documents lack specific information about Bangalore-based employees reporting to Sarah Chen, which aligns with the retrieved contexts; however, the reference answer contains details not present in any retrieved context, making correctness and context recal

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

> Based on the provided documents, the employee directory (Source 6) identifies Maya Sharma as the VP of Customer Success. The directory does not list any employees who report to Maya Sharma (i.e., employees with a `manager_id` corresponding to Maya Sharma's `employee_id`, E019).
> 
> Because there are no active direct reports of Maya Sharma listed in the provided context, the combined ARR of customers assigned to such individuals is $0.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 5.

**Judge reasoning** &nbsp; The actual answer incorrectly claims Maya Sharma has no direct reports due to missing employee_directory data, but the reference confirms Sam Rivera and Anjali Patel are her direct reports, which were not retrieved or used.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.80 | 0.10 | 0.00 |

</details>

<details>
<summary><b>Q14</b> — FAIL · correctness 0.20 · tier owner: <code>step_05_multi_agent</code></summary>

**Question** &nbsp; Consider Vertexia's active Customer Success Managers (role contains 'Customer Success Manager') who started at the company in 2022 per the employee directory. What is the combined ARR in customer_list.csv of the customers assigned to those CSMs?

**Reference answer (gold):**

> The only active Customer Success Manager in employee_directory.csv who started at Vertexia in 2022 is Anjali Patel (start_date 2022-03-15). The five customers in customer_list.csv assigned to Anjali Patel (Summit Pharma, OmegaLogistics, Redwood Analytics, Ironclad Security, and Northgate Bank) have a combined ARR of $2,400,000.

**System answer at this step:**

> Based on the provided documents, here is the step-by-step breakdown:
> 
> 1.  **Identify active Customer Success Managers who started in 2022:**
>     *   **Anjali Patel (E037):** Role is "Customer Success Manager," status is "active," and start date is 2022-03-15.
>     *   **Maya Sharma:** While she is listed as a CSM in the customer list, she does not appear in the provided employee directory rows to verify her start date.
>     *   **Sam Rivera (E036):** Started in 2023.
>     *   **Preet Kaur (E030):** Status is "departed."
> 
> 2.  **Calculate the ARR for customers assigned to Anjali Patel:**
>     *   According to Source 7, Anjali Patel is the CSM for **Redwood Analytics**, which has an ARR of **$720,000**.
> 
> The provided context does not contain enough information to confirm the start dates for Maya Sharma, so I cannot determine if she meets the criteria of starting in 2022. Therefore, based strictly on the available data, the combined ARR for customers assigned to the identified CSM (Anjali Patel) is **$720,000**.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 5.

**Judge reasoning** &nbsp; The model failed to retrieve or identify all customers assigned to Anjali Patel from the provided context, leading to an incorrect ARR calculation.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.20 | 0.80 | 0.80 | 0.40 | 0.40 |

</details>

> The full per-question JSON (retrieved contexts, latency, etc.) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
