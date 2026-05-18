"""Golden question set — tier-locked across 5 capability tiers (over 7 pipeline steps).

Each question is designed to be PROVABLY UNANSWERABLE at every step prior to the
one that introduces its required capability. The corpus was audited (2026-05-18)
to confirm that no prose document leaks the answer fact in a way that dense
retrieval would surface at baseline.

  Tier 1 (Q01-Q03):  Simple retrieval + format-aware chunks  — step_01_baseline_rag
  Tier 2 (Q04-Q06):  Exact CSV arithmetic                    — step_02_tools
  Tier 3 (Q07-Q09):  BM25 keyword-exact / sparse CSV rows    — step_03_hybrid_retrieval
  Tier 4 (Q10-Q12):  Multi-hop joins via foreign keys        — step_04_knowledge_graph
  Tier 5 (Q13-Q14):  Orchestrated graph + structured + agg   — step_05_multi_agent

14 questions total (no Tier 6 — could not design a robust tier-6-locked question
without leaking into earlier tiers; the spec prefers fewer-locked over more-leaky).
"""
from dataclasses import dataclass


@dataclass
class GoldenQuestion:
    id: str
    type: str
    question: str
    required_facts: list[str]
    partial_facts: list[str]
    disqualifiers: list[str]
    explanation: str
    expected_outcome: str
    fixed_by_step: str
    reference_answer: str = ""  # Natural-language gold answer for RAGAS judge


GOLDEN_QUESTIONS: list[GoldenQuestion] = [

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1 — Simple retrieval + Format-aware Chunking (step_01_baseline_rag)
    # Single fact, present in one doc, dense retrieval at k=10 finds the chunk.
    # ─────────────────────────────────────────────────────────────────────────

    GoldenQuestion(
        id="Q01",
        type="simple_lookup",
        question="What is Vertexia's customer data retention policy for hot storage and cold storage?",
        required_facts=["90 day", "glacier"],
        partial_facts=["retention", "cold", "storage"],
        disqualifiers=[],
        explanation=(
            "Stated identically in onboarding_handbook.txt, "
            "data_processing_agreement_template.txt, and phoenix_corp_msa.txt §5. "
            "Dense retrieval finds it trivially at k=10."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
        reference_answer=(
            "Vertexia stores customer data in hot storage for 90 days, "
            "then archives to AWS S3 Glacier for 1 year of cold storage."
        ),
    ),

    GoldenQuestion(
        id="Q02",
        type="simple_lookup",
        question="Who co-founded Vertexia and in what year was the company founded?",
        required_facts=["Arjun Mehta", "Diana Volkov", "2019"],
        partial_facts=["founder", "co-founder"],
        disqualifiers=[],
        explanation=(
            "founding_story.txt + series_b_investor_update.txt both name "
            "Arjun Mehta and Diana Volkov as co-founders in March 2019."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
        reference_answer="Vertexia was co-founded by Arjun Mehta and Diana Volkov in March 2019.",
    ),

    GoldenQuestion(
        id="Q03",
        type="chunking_dependent",
        question=(
            "In Vertexia's on-call runbook, what is the first action and the "
            "escalation owner for the PulseConnect webhook delivery failure alert?"
        ),
        required_facts=["SendGrid", "Twilio", "Raj Patel"],
        partial_facts=["webhook", "PulseConnect", "on-call"],
        disqualifiers=["Felix Wagner", "Aisha Johnson", "Kenji Ito", "Daniel Osei"],
        explanation=(
            "oncall_runbook_top_alerts.md '## PulseConnect "
            "webhook_delivery_failure_rate > 5%' section. Format-aware section "
            "chunking with contextual headers retrieves the exact section, not the "
            "neighbouring InsightLens / Stripe / RDS alert sections."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
        reference_answer=(
            "For the PulseConnect webhook_delivery_failure_rate > 5% alert, the "
            "first action is to check the SendGrid quota dashboard and the Twilio "
            "API status page (and inspect the outbound retry queue depth in the "
            "PulseConnect admin UI). The escalation owner is Raj Patel."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2 — Exact CSV Arithmetic (step_02_tools)
    # Each answer is computable ONLY by summing/counting a CSV column. Audited:
    # the exact result number does NOT appear in any prose doc.
    # ─────────────────────────────────────────────────────────────────────────

    GoldenQuestion(
        id="Q04",
        type="csv_aggregate",
        question="What is the total ARR across all Vertexia customers combined?",
        required_facts=["11,000,000"],
        partial_facts=["ARR", "total", "customer"],
        # Distractor: prose says $16.5M ARR (a company-level metric, NOT the sum
        # of customer_list.csv arr_usd column). Baseline reliably reports $16.5M.
        disqualifiers=["16,500,000", "16.5M", "16.5 million", "$5.2M"],
        explanation=(
            "customer_list.csv has 20 rows summing to exactly $11,000,000. "
            "GREP AUDIT: 11,000,000 / $11M does NOT appear in any prose doc. "
            "Prose reports a different metric ($16.5M total ARR) which is a "
            "company-level ARR distinct from the sum of customer_list rows. "
            "Baseline currently FAILs by returning $16.5M (confirmed in eval)."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_02_tools",
        reference_answer=(
            "Summing the arr_usd column across all 20 rows in customer_list.csv "
            "yields a total of $11,000,000 across all Vertexia customers combined."
        ),
    ),

    GoldenQuestion(
        id="Q05",
        type="csv_aggregate",
        question="How many active Vertexia employees are based in the Berlin office?",
        required_facts=["5"],
        partial_facts=["Berlin", "employee", "location"],
        disqualifiers=[],
        explanation=(
            "employee_directory.csv: filter status=active and location=Berlin → "
            "5 employees (Felix Wagner, Ravi Krishnan, Emma Fischer, Noah Zimmermann, "
            "Aleksander Nowak). "
            "GREP AUDIT: prose only mentions Berlin as a planned 2024 expansion hub "
            "(company_strategy_2024.txt) — no headcount number stated. Baseline "
            "currently FAILs and explicitly says it has no Berlin info (confirmed)."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_02_tools",
        reference_answer="There are 5 active Vertexia employees based in the Berlin office.",
    ),

    GoldenQuestion(
        id="Q06",
        type="csv_aggregate",
        question=(
            "What is the total planned headcount across all departments in "
            "Vertexia's 2023 budget allocation?"
        ),
        required_facts=["181"],
        partial_facts=["headcount", "department", "budget", "2023"],
        # Prose says "~500 employees" — a company-wide actual figure that is
        # different from the sum of the budget_allocation_2023.csv headcount
        # column. The judge must accept 181 specifically.
        disqualifiers=["500", "approximately 500", "~500"],
        explanation=(
            "budget_allocation_2023.csv has 9 department rows whose headcount column "
            "sums to exactly 181 (the 2023 PLANNED headcount per department budget). "
            "GREP AUDIT: '181' does NOT appear anywhere in the corpus. Prose talks "
            "about '~500 employees' (a different, company-wide actual figure). "
            "Matches csv_tool intent `total_headcount`."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_02_tools",
        reference_answer=(
            "Summing the headcount column across all 9 department rows in "
            "budget_allocation_2023.csv yields a total planned headcount of 181 "
            "for 2023. (This is the budgeted plan, distinct from the company's "
            "~500 actual employees referenced in prose.)"
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 3 — BM25 Keyword-Exact / Sparse CSV Rows (step_03_hybrid_retrieval)
    # Each question contains a rare exact token (version string, date, vendor
    # name, ID). The answer chunk is text-sparse so dense retrieval misses it.
    # ─────────────────────────────────────────────────────────────────────────

    GoldenQuestion(
        id="Q07",
        type="keyword_exact",
        question=(
            "According to Vertexia's vendor contracts summary, what is the annual "
            "contract value for Snowflake and when does that contract renew?"
        ),
        required_facts=["120,000", "2024-06-30"],
        partial_facts=["Snowflake", "contract", "renewal"],
        disqualifiers=[],
        explanation=(
            "vendor_contracts_summary.csv: 'Snowflake,Data Warehouse,120000,"
            "2024-06-30,Priya Nair,Usage-based + commitment'. "
            "GREP AUDIT: the number 120,000 in association with Snowflake appears "
            "ONLY in this CSV row. Snowflake is mentioned in many prose docs "
            "(DPA, vendor matrix, sales playbook, architecture) but never with "
            "contract value or expiry date. The CSV row text is sparse and "
            "dense-embedding-poor; BM25 on 'Snowflake' + 'contract' will surface "
            "the CSV row chunk. Baseline currently FAILs (confirmed)."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_hybrid_retrieval",
        reference_answer=(
            "Per vendor_contracts_summary.csv, Vertexia's annual contract value for "
            "Snowflake is $120,000 and the contract renews on June 30, 2024."
        ),
    ),

    GoldenQuestion(
        id="Q08",
        type="keyword_exact",
        question=(
            "Which NexusFlow API endpoint was deprecated in v2.1, and what "
            "endpoint replaced it?"
        ),
        required_facts=["events/batch", "events/stream"],
        partial_facts=["v2.1", "deprecated", "NexusFlow"],
        disqualifiers=[],
        explanation=(
            "nexusflow_api_changelog.md ## v2.1: 'GET /v2/events/batch — Deprecated "
            "in v2.1, removed in v2.2. Replaced by GET /v2/events/stream.' "
            "GREP AUDIT: the events/batch → events/stream pair is ONLY in this "
            "changelog doc. v2.1 is heavily semantically associated with the "
            "outage / postmortem (different doc), so dense retrieval pulls "
            "postmortem instead of changelog. BM25 on 'v2.1' + 'deprecated' "
            "surfaces the right chunk. Baseline currently FAILs (confirmed)."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_hybrid_retrieval",
        reference_answer=(
            "The GET /v2/events/batch endpoint was deprecated in NexusFlow API v2.1 "
            "(and removed in v2.2). It was replaced by GET /v2/events/stream."
        ),
    ),

    GoldenQuestion(
        id="Q09",
        type="keyword_exact",
        question=(
            "Per Vertexia's Q4 2023 on-call schedule, who was the on-call engineer "
            "for the Data Platform Team during the week starting 2023-10-16?"
        ),
        required_facts=["Priya Kapoor"],
        partial_facts=["on-call", "Data Platform", "2023-10-16"],
        disqualifiers=["Kenji Ito", "Priya Nair", "James O'Brien", "Lin Wei"],
        explanation=(
            "on_call_schedule_q4_2023.csv row: '2023-10-16,2023-10-22,Data Platform "
            "Team,Priya Kapoor,E039'. "
            "GREP AUDIT: Priya Kapoor appears ONLY in employee_directory.csv and "
            "the two on-call CSVs. She is not named in any prose. The date "
            "2023-10-16 is a rare exact token; BM25 on the date matches the row "
            "directly. Dense retrieval on 'on-call Data Platform October 2023' "
            "would surface the runbook docs (which name Kenji Ito / Priya Nair) "
            "rather than the specific schedule row. Baseline likely picks a "
            "wrong/no engineer."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_hybrid_retrieval",
        reference_answer=(
            "Per on_call_schedule_q4_2023.csv, Priya Kapoor (employee_id E039) was "
            "the on-call engineer for the Data Platform Team during the week of "
            "2023-10-16 to 2023-10-22."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 4 — Multi-hop Foreign-Key Joins (step_04_knowledge_graph)
    # Each question requires chaining lookups across 2+ CSVs by ID. Audited:
    # no prose doc names the full resolved chain (the org charts only document
    # the upper tree, not the leaf employees used here).
    # ─────────────────────────────────────────────────────────────────────────

    GoldenQuestion(
        id="Q10",
        type="multi_hop",
        question=(
            "Who is the CSM assigned to the customer 'Summit Pharma', what office "
            "location is that CSM based in, and who is that CSM's direct manager?"
        ),
        required_facts=["Anjali Patel", "San Francisco", "Maya Sharma"],
        partial_facts=["Summit Pharma", "CSM", "manager", "location"],
        disqualifiers=[],
        explanation=(
            "Three-CSV-hop chain (all keyed by IDs, no prose shortcut): "
            "(1) csm_account_history.csv: Summit Pharma → Anjali Patel (E037). "
            "(2) employee_directory.csv row E037: location=San Francisco, "
            "manager_id=E019. "
            "(3) employee_directory.csv row E019: name=Maya Sharma. "
            "GREP AUDIT: neither 'Summit Pharma' nor 'Anjali Patel' appears in "
            "any prose doc — they exist only in CSV rows. With CSV-per-row "
            "chunking, baseline retrieves Summit Pharma's row (which yields "
            "'Anjali Patel') but cannot resolve E037 → her directory row → E019 "
            "→ Maya Sharma without graph traversal. Even if Maya Sharma is in "
            "prose, the bridging chain is CSV-only."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_knowledge_graph",
        reference_answer=(
            "The CSM assigned to Summit Pharma is Anjali Patel (per "
            "csm_account_history.csv). Anjali Patel is based in San Francisco and "
            "her direct manager is Maya Sharma (VP Customer Success), per "
            "employee_directory.csv."
        ),
    ),

    GoldenQuestion(
        id="Q11",
        type="multi_hop",
        question=(
            "Based on Vertexia's API dependency graph, which Vertexia services "
            "directly depend on NexusFlow's APIs? Name each service AND the "
            "specific NexusFlow endpoint it consumes."
        ),
        required_facts=[
            "InsightLens", "PulseConnect", "DataCraft",
            "events_api", "pipeline_status_api", "ingest_api",
        ],
        partial_facts=["dependency", "depends", "connectors_api"],
        disqualifiers=[],
        explanation=(
            "api_dependencies.csv: three services consume NexusFlow APIs — "
            "InsightLens (events_api + connectors_api), PulseConnect "
            "(pipeline_status_api), and DataCraft (ingest_api). "
            "GREP AUDIT: the InsightLens→NexusFlow dependency is mentioned in "
            "prose (postmortem, architecture); the DataCraft→NexusFlow critical "
            "dependency is ONLY in api_dependencies.csv (other prose describes "
            "DataCraft loosely without naming the ingest_api dependency). "
            "Baseline currently returns PARTIAL — gets InsightLens but misses "
            "DataCraft (confirmed in eval)."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_knowledge_graph",
        reference_answer=(
            "Per api_dependencies.csv, three Vertexia services directly depend on "
            "NexusFlow's APIs: InsightLens (via events_api and connectors_api), "
            "PulseConnect (via pipeline_status_api), and DataCraft (via ingest_api)."
        ),
    ),

    GoldenQuestion(
        id="Q12",
        type="multi_hop",
        question=(
            "Which active Vertexia employees located in the Bangalore office "
            "report — directly or indirectly, through any number of management "
            "hops — to the CTO Sarah Chen? List every such employee."
        ),
        required_facts=["Priya Nair", "Kenji Ito", "Lin Wei", "Priya Kapoor", "Omar Faruk"],
        partial_facts=["Bangalore", "Sarah Chen", "reports"],
        disqualifiers=[],
        explanation=(
            "Pure employee_directory.csv reverse-BFS from E003 (Sarah Chen), "
            "filtered by location=Bangalore and status=active. The five "
            "transitive reports are Priya Nair (E010, two hops via Marcus Webb "
            "E009), Kenji Ito (E017, via Priya Nair), Lin Wei (E021, via Yuki "
            "Tanaka via Tomás García), Priya Kapoor (E039, via Priya Nair), "
            "and Omar Faruk (E032, via Daniel Osei). "
            "GREP AUDIT: Priya Nair and Kenji Ito are in prose (org chart, "
            "post-mortem), but Lin Wei, Priya Kapoor, and Omar Faruk appear ONLY "
            "in CSV rows. Baseline retrieving the org chart prose can name at "
            "most 2 of the 5 — it cannot enumerate the CSV-only employees "
            "without traversing the directory."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_knowledge_graph",
        reference_answer=(
            "Five active Bangalore-based employees report (directly or indirectly) "
            "to CTO Sarah Chen: Priya Nair (Lead, Data Platform Team), Kenji Ito "
            "(Senior Engineer, Data Platform), Lin Wei (Senior Engineer, "
            "NexusFlow), Priya Kapoor (Data Engineer, Data Platform), and Omar "
            "Faruk (Engineer, Security)."
        ),
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 5 — Orchestrated Graph + Structured + Aggregate (step_05_multi_agent)
    # Requires the Graph specialist to filter entities AND the Structured/CSV
    # specialist to aggregate over that filtered set — one tool cannot solve it.
    # ─────────────────────────────────────────────────────────────────────────

    GoldenQuestion(
        id="Q13",
        type="cross_document",
        question=(
            "What is the combined ARR (sum of arr_usd in customer_list.csv) of "
            "all customers whose assigned CSM is an active direct report of "
            "Maya Sharma per the employee directory?"
        ),
        required_facts=["3,708,000"],
        partial_facts=["combined ARR", "Maya Sharma", "direct report", "CSM"],
        disqualifiers=[],
        explanation=(
            "Three-step orchestrated query: "
            "(1) GRAPH: employee_directory.csv → find active employees with "
            "manager_id = E019 (Maya Sharma) → {Sam Rivera (E036), Anjali Patel "
            "(E037)}. Preet Kaur (E030) is departed, so excluded. "
            "(2) STRUCTURED FILTER: customer_list.csv → rows where csm ∈ {Sam "
            "Rivera, Anjali Patel} → 10 customers. "
            "(3) AGGREGATE: sum arr_usd across those 10 rows → $3,708,000 "
            "(Anjali: 300+156+720+384+840 = $2.4M; Sam: 216+264+480+48+300 = $1.308M). "
            "GREP AUDIT: $3,708,000 does NOT appear anywhere in the corpus. "
            "Tier 4 graph alone yields the CSM set but cannot sum ARR; Tier 2 "
            "CSV tool alone has no intent for this composite query. Only "
            "orchestration (graph specialist + structured specialist) can solve "
            "it."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_multi_agent",
        reference_answer=(
            "Maya Sharma has two active direct reports in employee_directory.csv: "
            "Sam Rivera and Anjali Patel (Preet Kaur also reported to Maya but "
            "departed in 2023). The customers in customer_list.csv assigned to "
            "either of those two CSMs combine to a total ARR of $3,708,000 "
            "(Anjali Patel's 5 accounts sum to $2,400,000; Sam Rivera's 5 "
            "accounts sum to $1,308,000)."
        ),
    ),

    GoldenQuestion(
        id="Q14",
        type="cross_document",
        question=(
            "Consider Vertexia's active Customer Success Managers (role contains "
            "'Customer Success Manager') who started at the company in 2022 per "
            "the employee directory. What is the combined ARR in customer_list.csv "
            "of the customers assigned to those CSMs?"
        ),
        required_facts=["2,400,000", "Anjali Patel"],
        partial_facts=["Customer Success Manager", "2022", "combined ARR"],
        # GREP AUDIT: $2,400,000 is also Phoenix Corp's individual ARR (heavily
        # in prose docs). A baseline retrieving phoenix_corp prose would return
        # "$2.4M (Phoenix Corp)" — same number, wrong provenance. Disqualify it.
        disqualifiers=["Phoenix Corp", "Phoenix Corporation"],
        explanation=(
            "(1) GRAPH/STRUCTURED on directory: filter status=active, role "
            "contains 'Customer Success Manager', start_date in 2022 → exactly "
            "one person: Anjali Patel (E037, started 2022-03-15). Preet Kaur also "
            "started in 2022 but is departed_2023-06. Sam Rivera started 2023. "
            "(2) STRUCTURED FILTER on customer_list: csm = 'Anjali Patel' → 5 "
            "customers (Summit Pharma 300k, OmegaLogistics 156k, Redwood "
            "Analytics 720k, Ironclad Security 384k, Northgate Bank 840k). "
            "(3) AGGREGATE: sum → $2,400,000. "
            "GREP AUDIT: $2,400,000 also happens to be Phoenix Corp's individual "
            "ARR (heavily in prose), so the question must be answered with the "
            "Anjali-Patel-customer-set provenance, not by retrieving Phoenix Corp "
            "prose. The required_facts include 'Anjali Patel' to enforce that. "
            "Requires composing a multi-condition CSV filter on directory plus a "
            "structured CSV aggregate — beyond any single specialist."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_multi_agent",
        reference_answer=(
            "The only active Customer Success Manager in employee_directory.csv "
            "who started at Vertexia in 2022 is Anjali Patel (start_date "
            "2022-03-15). The five customers in customer_list.csv assigned to "
            "Anjali Patel (Summit Pharma, OmegaLogistics, Redwood Analytics, "
            "Ironclad Security, and Northgate Bank) have a combined ARR of "
            "$2,400,000."
        ),
    ),
]
