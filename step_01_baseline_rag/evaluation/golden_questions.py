from dataclasses import dataclass


@dataclass
class GoldenQuestion:
    id: str
    type: str
    question: str
    required_facts: list[str]        # ALL must be present for PASS
    partial_facts: list[str]         # any present = PARTIAL (if not already PASS/FAIL)
    disqualifiers: list[str]         # any present → force FAIL (wrong answer caught)
    explanation: str
    expected_outcome: str            # expected grade at Step 01 baseline
    fixed_by_step: str               # first step that achieves PASS


GOLDEN_QUESTIONS: list[GoldenQuestion] = [

    # ── TIER 1: Simple Retrieval (Q01–Q07) ─────────────────────────────────────
    # Answerable by vector search alone — single document, single fact.
    # Should PASS at Step 01 (baseline RAG).

    GoldenQuestion(
        id="Q01",
        type="simple_lookup",
        question="What is Vertexia's customer data retention policy for hot storage and cold storage?",
        required_facts=["90 day", "glacier"],
        partial_facts=["retention", "cold", "storage"],
        disqualifiers=[],
        explanation=(
            "Answer is in onboarding_handbook.txt and data_processing_agreement_template.txt. "
            "Semantic search for 'data retention policy' retrieves both immediately."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
    ),

    GoldenQuestion(
        id="Q02",
        type="simple_lookup",
        question="Who co-founded Vertexia and in what year was the company founded?",
        required_facts=["Arjun Mehta", "Diana Volkov", "2019"],
        partial_facts=["founder", "co-founder"],
        disqualifiers=[],
        explanation=(
            "Both founders and the founding year appear in founding_story.txt and "
            "series announcements. Any query about founders retrieves these."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
    ),

    GoldenQuestion(
        id="Q03",
        type="simple_lookup",
        question="What was Sarah Chen's job title before she was promoted to CTO in April 2023?",
        required_facts=["VP Engineering"],
        partial_facts=["Sarah Chen", "engineering", "promoted"],
        disqualifiers=["CTO before", "always CTO"],
        explanation=(
            "promotion_announcements_2023.txt and org_chart_q1_2023.txt both record "
            "Sarah Chen as VP Engineering prior to the April 2023 restructuring."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
    ),

    GoldenQuestion(
        id="Q04",
        type="simple_lookup",
        question="What are the three main products Vertexia sells?",
        required_facts=["NexusFlow", "InsightLens", "PulseConnect"],
        partial_facts=["product", "platform", "analytics"],
        disqualifiers=[],
        explanation=(
            "All three products appear together in founding_story.txt, nexusflow_architecture.md, "
            "and multiple other documents. Simple semantic retrieval finds them."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
    ),

    GoldenQuestion(
        id="Q05",
        type="simple_lookup",
        question="What message queue technology does Vertexia's data pipeline use after the DataCraft integration was completed?",
        required_facts=["Pulsar"],
        partial_facts=["message queue", "migration", "DataCraft"],
        disqualifiers=["now uses kafka", "still uses kafka", "uses apache kafka as its message"],
        explanation=(
            "datacraft_migration_complete.txt and nexusflow_architecture.md both state Pulsar "
            "as the message queue after migration. Disqualifiers catch 'uses Kafka' answers; "
            "mentioning Kafka as the old system in a correct migration answer is fine."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
    ),

    GoldenQuestion(
        id="Q06",
        type="simple_lookup",
        question="What uptime SLA does the Phoenix Corp Master Service Agreement require for Vertexia's data pipeline services?",
        required_facts=["99.99"],
        partial_facts=["SLA", "uptime", "availability", "Phoenix"],
        disqualifiers=["99.9%"],
        explanation=(
            "phoenix_corp_msa.txt states 99.99% uptime requirement explicitly. "
            "99.9% is the NexusFlow architecture target — a common wrong answer."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
    ),

    GoldenQuestion(
        id="Q07",
        type="simple_lookup",
        question="How many engineers joined Vertexia through the DataCraft acquisition?",
        required_facts=["12"],
        partial_facts=["DataCraft", "acquisition", "engineer"],
        disqualifiers=[],
        explanation=(
            "datacraft_employee_integration.txt and founding_story.txt both state 12 engineers "
            "joined via the acquisition. Direct lookup from a single document."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
    ),

    # ── TIER 1.5: Format-aware Chunking (Q08–Q09) ─────────────────────────────
    # These questions live in long structured documents where each section answers
    # a different specific question. Step 01's naive paragraph chunker merges 4-5
    # sections per chunk, diluting the embedding signal so the LLM gets mixed info
    # about multiple vendors/alerts and confuses the per-section fields. Step 02's
    # section-aware chunker with a contextual header (FILE | DOC | SECTION) returns
    # the exact section needed.
    # Should be PARTIAL at Step 01 and PASS at Step 02 (Format-aware chunking).

    GoldenQuestion(
        id="Q08",
        type="chunking_dependent",
        question="In Vertexia's on-call runbook, what is the first action and the escalation owner for the PulseConnect webhook delivery failure alert?",
        required_facts=["SendGrid", "Twilio", "Raj Patel"],
        partial_facts=["webhook", "PulseConnect", "on-call"],
        disqualifiers=["Felix Wagner", "Aisha Johnson", "Kenji Ito"],
        explanation=(
            "oncall_runbook_top_alerts.md '## PulseConnect webhook_delivery_failure_rate > 5%' "
            "section: first action = check SendGrid quota and Twilio API, owner = Raj Patel. "
            "Step 01's chunker swallows multiple alert sections into one chunk, so the LLM "
            "sees a mixed bag of owners and may attach the wrong one to PulseConnect. "
            "Step 02 chunks each alert as its own section with a contextual header — dense "
            "retrieval returns exactly the PulseConnect section."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_02_chunking",
    ),

    GoldenQuestion(
        id="Q09",
        type="chunking_dependent",
        question="According to Vertexia's vendor data processing matrix, what sub-processors does Datadog use, and what is its data retention period for traces and logs?",
        required_facts=["us-east-1", "eu-west-1", "ap-southeast-1", "15 months"],
        partial_facts=["Datadog", "sub-processor", "retention"],
        disqualifiers=["us-central1", "90 days", "18 months", "24 months"],
        explanation=(
            "vendor_data_processing_matrix.md '## Datadog' section: sub-processors = AWS "
            "us-east-1, eu-west-1, ap-southeast-1; retention = 15 months for traces and logs. "
            "Step 01's chunker merges multiple vendor sections, so the LLM sees Snowflake's "
            "regions, SendGrid's 90-day retention, and Stripe's GCP us-central1 in the same "
            "chunk and frequently swaps fields across vendors. Step 02's per-vendor section "
            "chunks with prepended 'DOC: Vendor Data Processing Matrix | SECTION: Datadog' "
            "context give dense retrieval a clean target."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_02_chunking",
    ),

    # ── TIER 2: CSV Computation (Q10–Q14) ──────────────────────────────────────
    # These questions require exact arithmetic over structured CSV data.
    # Vector retrieval fails — it surfaces at most k rows, never the full table.
    # Should FAIL at Steps 01–02 and PASS at Step 03 (CSV Tool Calling).

    GoldenQuestion(
        id="Q10",
        type="csv_aggregate",
        question="What is the total ARR across all Vertexia customers combined?",
        required_facts=["11,000,000"],
        partial_facts=["ARR", "annual recurring revenue", "customer"],
        disqualifiers=["8,450", "4,120"],
        explanation=(
            "customer_list.csv has 20 customers totalling $11,000,000 ARR. "
            "Vector retrieval surfaces at most 5 rows — the LLM sums those and gets a wrong total. "
            "Requires a Pandas tool that reads all 20 rows."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_tools",
    ),

    GoldenQuestion(
        id="Q11",
        type="csv_aggregate",
        question="What was the total revenue across all products combined in Q3 2023 (July, August, and September)?",
        required_facts=["4,120,000"],
        partial_facts=["Q3", "revenue", "2023"],
        disqualifiers=["4.2M", "4,200,000"],
        explanation=(
            "revenue_by_product_2023.csv rows for 2023-07, 2023-08, 2023-09 sum to $4,120,000. "
            "The all-hands notes say '~$4.2M' (rounded bookings figure) — the disqualifier. "
            "Requires a tool that filters by month and sums."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_tools",
    ),

    GoldenQuestion(
        id="Q12",
        type="csv_aggregate",
        question="How many active Vertexia employees are based in Berlin?",
        required_facts=["5"],
        partial_facts=["Berlin", "employee", "location"],
        disqualifiers=[],
        explanation=(
            "employee_directory.csv: filtering status=active and location=Berlin gives 5 employees. "
            "Vector retrieval might surface a few CSV rows but cannot reliably group and count all of them."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_tools",
    ),

    GoldenQuestion(
        id="Q13",
        type="csv_aggregate",
        question="What is the total planned headcount across all departments in Vertexia's 2023 budget?",
        required_facts=["181"],
        partial_facts=["headcount", "budget", "department"],
        disqualifiers=[],
        explanation=(
            "budget_allocation_2023.csv has 9 departments summing to 181 planned headcount. "
            "No single retrieved chunk contains the cross-department total."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_tools",
    ),

    GoldenQuestion(
        id="Q14",
        type="csv_aggregate",
        question="What was the total ARR from all Closed-Won deals in Q3 2023?",
        required_facts=["1,692,000"],
        partial_facts=["closed-won", "Q3", "ARR", "deal"],
        disqualifiers=[],
        explanation=(
            "deal_pipeline_q3_2023.csv: 8 Closed-Won deals summing to $1,692,000 ARR. "
            "Vector retrieval returns a few rows — partial sum gives wrong total."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_tools",
    ),

    # ── TIER 3: BM25 / Keyword-Exact Retrieval (Q15–Q20) ───────────────────────
    # These questions contain technical identifiers, version strings, or proper
    # nouns that dense embedding search misses or confuses with similar docs.
    # BM25 keyword search surfaces the right document by exact term matching.
    # Should FAIL at Steps 01–03 and PASS at Step 04 (Hybrid BM25 + Dense).

    GoldenQuestion(
        id="Q15",
        type="keyword_exact",
        question="What specific action items were documented in the NexusFlow v2.1 postmortem and who owns each one?",
        required_facts=["max_connections", "config validation", "canary"],
        partial_facts=["postmortem", "action", "v2.1", "NexusFlow"],
        disqualifiers=[],
        explanation=(
            "nexusflow_v21_postmortem.txt: five action items including Priya Nair's config "
            "validation (AI-1) and Yuki Tanaka's automated canary testing (AI-2). "
            "The exact string 'v2.1' and 'max_connections' are technical identifiers — "
            "dense retrieval on 'NexusFlow issues actions' misses this document; "
            "BM25 on 'v2.1 postmortem action items' finds it directly."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_hybrid_retrieval",
    ),

    GoldenQuestion(
        id="Q16",
        type="keyword_exact",
        question="Who was the on-call engineer for the NexusFlow team during the week of August 14, 2023?",
        required_facts=["Yuki Tanaka"],
        partial_facts=["on-call", "NexusFlow", "August"],
        disqualifiers=["Lin Wei", "Sophie Laurent"],
        explanation=(
            "on_call_schedule_aug2023.csv: week_start 2023-08-14, team=NexusFlow Team → Yuki Tanaka. "
            "Lin Wei was NexusFlow on-call the previous week; Sophie Laurent the following week. "
            "Dense retrieval on 'NexusFlow on-call engineer' may surface the postmortem; "
            "BM25 finds the schedule CSV directly via exact term matching."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_hybrid_retrieval",
    ),

    GoldenQuestion(
        id="Q17",
        type="keyword_exact",
        question="What is Vertexia's annual spend on AWS and when does that contract renew?",
        required_facts=["480,000", "December 31, 2024"],
        partial_facts=["AWS", "cloud", "infrastructure", "vendor"],
        disqualifiers=[],
        explanation=(
            "vendor_contracts_summary.csv: AWS row — annual_value_usd=480000, renewal_date=2024-12-31. "
            "Dense search for 'cloud infrastructure annual cost' returns generic strategy docs. "
            "BM25 on 'AWS' finds the exact vendor row in the contracts CSV."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_hybrid_retrieval",
    ),

    GoldenQuestion(
        id="Q18",
        type="keyword_exact",
        question="What is Vertexia's annual spend on Snowflake and when does the contract expire?",
        required_facts=["120,000", "June 30, 2024"],
        partial_facts=["Snowflake", "data warehouse", "contract"],
        disqualifiers=[],
        explanation=(
            "vendor_contracts_summary.csv: Snowflake row — annual_value_usd=120000, "
            "renewal_date=2024-06-30. Dense search may return general docs; "
            "BM25 finds 'Snowflake' in the vendor CSV directly."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_hybrid_retrieval",
    ),

    GoldenQuestion(
        id="Q19",
        type="keyword_exact",
        question="Which NexusFlow API endpoint was deprecated in v2.1, and what endpoint replaced it?",
        required_facts=["events/batch", "events/stream"],
        partial_facts=["v2.1", "deprecated", "NexusFlow", "v2.2"],
        disqualifiers=[],
        explanation=(
            "nexusflow_api_changelog.md: 'GET /v2/events/batch — Deprecated in v2.1, removed in v2.2. "
            "Replaced by GET /v2/events/stream.' Both the version tag 'v2.1' and the endpoint names "
            "appear in the same chunk. Dense search for 'NexusFlow deprecated endpoint' returns "
            "generic architecture docs; BM25 on 'v2.1' scores the changelog chunk directly."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_hybrid_retrieval",
    ),

    GoldenQuestion(
        id="Q20",
        type="keyword_exact",
        question="Who is the remediation owner for security audit finding M-2, and what was the target remediation date?",
        required_facts=["Daniel Osei", "October 31, 2023"],
        partial_facts=["M-2", "TLS", "security audit"],
        disqualifiers=[],
        explanation=(
            "security_audit_2023.txt: 'FINDING M-2 (Medium): TLS 1.1 fallback. "
            "Remediation owner: Daniel Osei. Target: October 31, 2023.' "
            "The finding ID 'M-2' is a rare token that BM25 scores at rank #1 for the exact "
            "security-findings chunk. Dense search for 'security audit remediation owner' "
            "returns the audit header (names Daniel Osei as Sponsor but lacks the date)."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_hybrid_retrieval",
    ),

    # ── TIER 4: Knowledge Graph / Multi-hop (Q21–Q26) ──────────────────────────
    # These questions require traversing typed relationships across multiple CSV
    # files — an org hierarchy join, a dependency graph BFS, or a schedule+org
    # chain. No single retrieval or tool call can answer them; a graph is needed.
    # Should FAIL at Steps 01–04 and PASS at Step 05 (Knowledge Graph RAG).

    GoldenQuestion(
        id="Q21",
        type="multi_hop",
        question="Who is the CSM managing the Phoenix Corp account, and who is that person's direct manager?",
        required_facts=["Maya Sharma", "Lisa Torres"],
        partial_facts=["CSM", "Phoenix Corp", "account"],
        disqualifiers=[],
        explanation=(
            "csm_account_history.csv: Phoenix Corp → Maya Sharma (CSM). "
            "employee_directory.csv: Maya Sharma (E019) → manager E006 → Lisa Torres (CRO). "
            "Two-hop chain across two CSV files — requires graph traversal."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_knowledge_graph",
    ),

    GoldenQuestion(
        id="Q22",
        type="multi_hop",
        question="Which services have a critical dependency on NexusFlow's events_api endpoint?",
        required_facts=["InsightLens", "DataCraft"],
        partial_facts=["events_api", "dependency", "NexusFlow"],
        disqualifiers=[],
        explanation=(
            "api_dependencies.csv: InsightLens and DataCraft both list NexusFlow events_api "
            "as a critical dependency. Requires filtering the dependency graph by endpoint and criticality."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_knowledge_graph",
    ),

    GoldenQuestion(
        id="Q23",
        type="multi_hop",
        question="Who was the on-call engineer for the Data Platform team during the week of August 14, 2023?",
        required_facts=["Kenji Ito"],
        partial_facts=["on-call", "Data Platform", "August"],
        disqualifiers=["Priya Nair", "James O'Brien", "Lin Wei"],
        explanation=(
            "on_call_schedule_aug2023.csv: week_start 2023-08-14, team=Data Platform Team → Kenji Ito. "
            "Disqualifiers are engineers on neighbouring weeks. "
            "Unlike the NexusFlow on-call (Q16), this requires first reading the postmortem "
            "to determine which team was responsible, then matching against the schedule — a two-hop join."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_knowledge_graph",
    ),

    GoldenQuestion(
        id="Q24",
        type="multi_hop",
        question="If NexusFlow goes down entirely, which services are directly or indirectly affected? List all of them.",
        required_facts=["InsightLens", "PulseConnect", "DataCraft"],
        partial_facts=["blast radius", "dependency", "downstream"],
        disqualifiers=[],
        explanation=(
            "api_dependencies.csv BFS from NexusFlow: "
            "direct: InsightLens (events_api critical), PulseConnect (pipeline_status_api), DataCraft (ingest_api). "
            "Indirect: PulseConnect also depends on InsightLens (metrics_api). "
            "Requires full graph traversal — no single chunk answers this."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_knowledge_graph",
    ),

    GoldenQuestion(
        id="Q25",
        type="multi_hop",
        question="Who does Aisha Johnson report to, and who does that person report to? Give the full two-hop reporting chain.",
        required_facts=["Tomás García", "Sarah Chen"],
        partial_facts=["Aisha Johnson", "reports to", "manager"],
        disqualifiers=[],
        explanation=(
            "employee_directory.csv: Aisha Johnson (E014) → manager E012 → Tomás García → "
            "manager E003 → Sarah Chen (CTO). "
            "Two-hop chain through the org hierarchy — requires graph traversal."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_knowledge_graph",
    ),

    GoldenQuestion(
        id="Q26",
        type="multi_hop",
        question="What external and internal services does PulseConnect depend on according to the API dependency data?",
        required_facts=["InsightLens", "NexusFlow", "sendgrid", "twilio"],
        partial_facts=["PulseConnect", "depends", "dependency"],
        disqualifiers=[],
        explanation=(
            "api_dependencies.csv: PulseConnect depends on InsightLens (metrics_api), "
            "NexusFlow (pipeline_status_api), external_sendgrid (email), external_twilio (SMS). "
            "Requires reading all rows where consuming_service=PulseConnect."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_knowledge_graph",
    ),

    # ── TIER 5: Complex Multi-step Reasoning (Q27–Q31) ─────────────────────────
    # These questions require disambiguation across documents with the same name,
    # cross-document comparison, multi-source synthesis, or cross-quarter joins.
    # Should FAIL at Steps 01–06 and PASS at Step 07 (Multi-Agent RAG).

    GoldenQuestion(
        id="Q27",
        type="disambiguation",
        question="There are two different things at Vertexia referred to as 'Project Phoenix'. What is each one and what was the outcome of each?",
        required_facts=["migration", "signed", "Phoenix Corp"],
        partial_facts=["Project Phoenix", "two", "engineering", "sales"],
        disqualifiers=[],
        explanation=(
            "Two distinct things: (1) Internal Python 2→3 migration of NexusFlow — completed June 2022. "
            "(2) Phoenix Corp enterprise deal — $2.4M ARR contract signed June 2022. "
            "Both happen to share the name. Requires reading both documents and distinguishing them."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07_multi_agent",
    ),

    GoldenQuestion(
        id="Q28",
        type="cross_document",
        question="Does Vertexia's documented NexusFlow availability target meet the uptime requirement in the Phoenix Corp contract? What is the gap if any?",
        required_facts=["99.9", "99.99"],
        partial_facts=["availability", "SLA", "gap", "NexusFlow"],
        disqualifiers=["no gap", "currently meets", "fully meets", "does meet the"],
        explanation=(
            "nexusflow_architecture.md: NexusFlow targets 99.9% availability. "
            "phoenix_corp_msa.txt: Phoenix Corp requires 99.99%. "
            "Gap is 0.09 percentage points — the current target does NOT meet the SLA. "
            "Requires reading two separate documents and comparing numbers."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07_multi_agent",
    ),

    GoldenQuestion(
        id="Q29",
        type="cross_document",
        question="Was InsightLens impacted by the August 2023 NexusFlow outage? If so, explain why via the dependency chain, and name the on-call engineer for each affected service.",
        required_facts=["InsightLens", "events_api", "Kenji Ito", "Yuki Tanaka"],
        partial_facts=["dependency", "outage", "on-call", "critical"],
        disqualifiers=[],
        explanation=(
            "InsightLens critically depends on NexusFlow events_api (api_dependencies.csv). "
            "Data Platform on-call Aug 14: Kenji Ito. NexusFlow on-call Aug 14: Yuki Tanaka. "
            "Requires: dependency graph + on-call schedule + postmortem date — three separate sources."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07_multi_agent",
    ),

    GoldenQuestion(
        id="Q30",
        type="cross_document",
        question="What was the combined ARR from all Closed-Won deals across both Q3 and Q4 2023 (the full second half of the year)?",
        required_facts=["3,456,000"],
        partial_facts=["H2", "closed-won", "ARR", "Q3", "Q4"],
        disqualifiers=[],
        explanation=(
            "Q3 Closed-Won: $1,692,000 (deal_pipeline_q3_2023.csv) + "
            "Q4 Closed-Won: $1,764,000 (deal_pipeline_q4_2023.csv) = $3,456,000. "
            "Requires reading two separate CSV files, summing each with a tool, then adding the totals."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07_multi_agent",
    ),

    GoldenQuestion(
        id="Q31",
        type="cross_document",
        question="Which employees left Vertexia voluntarily in 2023? For each person, state their department, last day, and the stated reason for departure.",
        required_facts=["Adrian Blake", "FinDataCo", "Preet Kaur", "relocated"],
        partial_facts=["voluntary", "departure", "offboarding", "2023"],
        disqualifiers=[],
        explanation=(
            "offboarding_records_2023.csv: Adrian Blake (Platform Engineering, 2023-08-31, "
            "joined competitor FinDataCo) and Preet Kaur (Revenue, 2023-06-30, relocated internationally). "
            "Diana Volkov left in 2021 — she is a distractor. "
            "Requires surfacing a low-traffic HR CSV that generic retrieval rarely ranks highly."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07_multi_agent",
    ),
]
