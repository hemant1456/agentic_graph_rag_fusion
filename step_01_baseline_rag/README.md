# Step 01 — Baseline RAG

## What it adds
The minimal RAG loop: corpus is split into paragraph chunks, embedded with HuggingFace MiniLM, and stored in ChromaDB. A question is embedded, top-5 cosine matches are retrieved, and a single LLM call generates the answer. This step is the floor that every later step builds on, and it already handles the simplest factoid lookups (Tier 1: Q01-Q02).

## Design
- **Class:** `BaselineRAG` in `step_01_baseline_rag/implementation/pipeline.py`
- **Inherits from:** none (root of the chain)
- **Key components:**
  - `step_01_baseline_rag/implementation/ingest.py` — paragraph chunker (~1000 chars, 200 overlap) and ChromaDB writer
  - `step_01_baseline_rag/implementation/retrieve.py` — top-k cosine retrieval and context formatter
  - `step_01_baseline_rag/implementation/generate.py` — prompt assembly and LLM call via `llm_gatewayV2`
  - Shared `chroma_db/` collection `vertexia_baseline`

## How it works
Ingestion walks `dataset/company_data/`, reads each file as plain text, splits on paragraphs with a sliding window (1000 chars, 200 overlap), embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`, and writes to ChromaDB. At query time, the question is embedded with the same model, ChromaDB returns the top-5 nearest chunks by cosine distance, they are concatenated into a context block, and a single LLM call (Groq llama-3.3-70b with Gemini fallback) produces the answer. There is no reranking, no query rewriting, no metadata filtering.

## Run
```bash
uv run python evaluation/run_eval.py --step step_01_baseline_rag
```

## Results
See `step_01_baseline_rag/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Every later step is justified by a failure mode of this one. Naive paragraph chunking shreds structured markdown (vendor matrices, runbook alerts), so Tier 2 questions like Q03 and Q04 fail. Dense embeddings alone miss keyword-exact tokens (Tier 4), cannot aggregate CSVs (Tier 3), and cannot traverse relations across files (Tier 5). The baseline gives us a concrete starting score against which each upgrade is measured.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 4 PASS · 1 PARTIAL · 9 FAIL out of 14 questions (29% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.350 | 0.986 | 0.621 | 0.264 | 0.343 |

### Per-question summary

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q03** | PASS | 1.00 | `step_01_baseline_rag` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q04** | FAIL | 0.00 | `step_02_tools` | Expected FAIL — required capability arrives at step 2. |
| **Q05** | FAIL | 0.00 | `step_02_tools` | Expected FAIL — required capability arrives at step 2. |
| **Q06** | FAIL | 0.20 | `step_02_tools` | Expected FAIL — required capability arrives at step 2. |
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

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

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

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

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

**Diagnosis** &nbsp; Pass-tier hits as designed — the step's new capability surfaces the required fact(s).

**Judge reasoning** &nbsp; The actual answer matches the reference exactly in content and is fully supported by the retrieved contexts; most retrieved contexts are relevant but some are not directly related to the PulseConnect alert.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.00 | 0.70 | 1.00 |

</details>

<details>
<summary><b>Q04</b> — FAIL · correctness 0.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total ARR across all Vertexia customers combined?

**Reference answer (gold):**

> Summing the arr_usd column across all 20 rows in customer_list.csv yields a total of $11,000,000 across all Vertexia customers combined.

**System answer at this step:**

> As of the end of Q3 2023, the total ARR for Vertexia Inc. is $16.5M.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 2.

**Judge reasoning** &nbsp; The actual answer reports $16.5M ARR from retrieved contexts, but the reference answer specifies a different value ($11M) derived from summing a CSV file not present in the contexts, making the answer incorrect despite being faithful to the provided documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.80 | 0.30 | 0.00 |

</details>

<details>
<summary><b>Q05</b> — FAIL · correctness 0.00 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; How many active Vertexia employees are based in the Berlin office?

**Reference answer (gold):**

> There are 5 active Vertexia employees based in the Berlin office.

**System answer at this step:**

> The provided documents do not contain information regarding a Berlin office. The context states that Vertexia employees are based in San Francisco, Austin, and Bangalore.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 2.

**Judge reasoning** &nbsp; The actual answer correctly states that Berlin office information is absent from the contexts, so it is faithful, but the reference answer asserts a fact (5 employees) not supported by any context, making correctness 0.0 and recall 0.0 since the required fact is missing in retrieved documents.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.30 | 0.10 | 0.00 |

</details>

<details>
<summary><b>Q06</b> — FAIL · correctness 0.20 · tier owner: <code>step_02_tools</code></summary>

**Question** &nbsp; What is the total planned headcount across all departments in Vertexia's 2023 budget allocation?

**Reference answer (gold):**

> Summing the headcount column across all 9 department rows in budget_allocation_2023.csv yields a total planned headcount of 181 for 2023. (This is the budgeted plan, distinct from the company's ~500 actual employees referenced in prose.)

**System answer at this step:**

> Based on the provided budget allocation documents, the total planned headcount for the departments listed is as follows:
> 
> *   **Platform Engineering:** 42
> *   **Revenue:** 38
> *   **People & Culture:** 5
> *   **Executive:** 3
> 
> The total planned headcount for these specific departments is 88. Please note that the provided context does not contain a comprehensive list of all departments at Vertexia Inc., so this figure may not represent the total planned headcount for the entire company.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 2.

**Judge reasoning** &nbsp; The model correctly identified partial data from the retrieved chunks but failed to retrieve or aggregate the full budget allocation table, leading to an incomplete and incorrect total.

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.20 | 1.00 | 0.80 | 0.50 | 0.40 |

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

> Based on the provided documents, there is insufficient information to list every employee in the Bangalore office who reports to CTO Sarah Chen.
> 
> While the documents establish the following:
> *   **Sarah Chen** is the CTO (appointed April 10, 2023) and oversees the organizational structure, including Platform Engineering (led by Marcus Webb) and Product Engineering (led by Tomás García).
> *   **Priya Nair** is listed as the lead for the Bangalore office in the Q3 2023 Org Chart.
> 
> However, the provided context does not contain a full employee roster, nor does it provide the reporting lines or organizational hierarchy for the Bangalore office beyond identifying Priya Nair as the lead. Therefore, it is not possible to determine which specific employees in Bangalore report to Sarah Chen.

**Diagnosis** &nbsp; Expected FAIL — required capability arrives at step 4.

**Judge reasoning** &nbsp; The model correctly identified that the provided context lacked the necessary information to answer the question, though the reference answer implies the information should have been present in the retrieved documents.

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
