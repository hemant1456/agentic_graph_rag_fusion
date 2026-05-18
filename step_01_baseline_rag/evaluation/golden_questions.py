"""Golden question set — 15 questions across 5 tiers (over 7 pipeline steps).

Each tier showcases a capability that the corresponding step adds:
  Tier 1 (Q01-Q04):  Simple retrieval + format-aware chunks — step_01_baseline_rag
  Tier 2 (Q05-Q07):  CSV aggregates                         — step_02_tools
  Tier 3 (Q08-Q10):  BM25 keyword-exact                     — step_03_hybrid_retrieval
  Tier 4 (Q11-Q13):  Graph multi-hop + alias resolution     — step_04_knowledge_graph
  Tier 5 (Q14-Q15):  Cross-document multi-agent             — step_05_multi_agent

Reduced from 31 → 15 for faster eval iteration on free-tier judge API.
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

    # ── TIER 1: Simple retrieval + Format-aware Chunking ──────────────────────

    GoldenQuestion(
        id="Q01",
        type="simple_lookup",
        question="What is Vertexia's customer data retention policy for hot storage and cold storage?",
        required_facts=["90 day", "glacier"],
        partial_facts=["retention", "cold", "storage"],
        disqualifiers=[],
        explanation="onboarding_handbook.txt + data_processing_agreement_template.txt both state the policy.",
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
        reference_answer="Vertexia stores customer data in hot storage for 90 days, then archives to AWS S3 Glacier for long-term cold storage.",
    ),

    GoldenQuestion(
        id="Q02",
        type="simple_lookup",
        question="Who co-founded Vertexia and in what year was the company founded?",
        required_facts=["Arjun Mehta", "Diana Volkov", "2019"],
        partial_facts=["founder", "co-founder"],
        disqualifiers=[],
        explanation="founding_story.txt names both founders and the year.",
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
        reference_answer="Vertexia was co-founded by Arjun Mehta and Diana Volkov in 2019.",
    ),

    # ── TIER 1 (cont.): Format-aware Chunking ─────────────────────────────────

    GoldenQuestion(
        id="Q03",
        type="chunking_dependent",
        question="In Vertexia's on-call runbook, what is the first action and the escalation owner for the PulseConnect webhook delivery failure alert?",
        required_facts=["SendGrid", "Twilio", "Raj Patel"],
        partial_facts=["webhook", "PulseConnect", "on-call"],
        disqualifiers=["Felix Wagner", "Aisha Johnson", "Kenji Ito"],
        explanation=(
            "oncall_runbook_top_alerts.md '## PulseConnect webhook_delivery_failure_rate > 5%' "
            "section. The merged Step 01 baseline now uses format-aware section chunking with "
            "contextual headers, so the exact section is retrieved."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
        reference_answer="For the PulseConnect webhook delivery failure alert, the first action is to check the SendGrid quota dashboard and the Twilio API status page. The escalation owner is Raj Patel.",
    ),

    GoldenQuestion(
        id="Q04",
        type="chunking_dependent",
        question="According to Vertexia's vendor data processing matrix, what sub-processors does Datadog use, and what is its data retention period for traces and logs?",
        required_facts=["us-east-1", "eu-west-1", "ap-southeast-1", "15 months"],
        partial_facts=["Datadog", "sub-processor", "retention"],
        disqualifiers=["us-central1", "90 days", "18 months", "24 months"],
        explanation=(
            "vendor_data_processing_matrix.md '## Datadog' section. The merged Step 01 baseline "
            "now uses per-vendor section chunks, so Datadog fields are no longer mixed with "
            "Snowflake / SendGrid / Stripe."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_01_baseline_rag",
        reference_answer="Datadog uses AWS sub-processors in us-east-1, eu-west-1, and ap-southeast-1. Its data retention period for traces and logs is 15 months.",
    ),

    # ── TIER 2: CSV Computation ───────────────────────────────────────────────

    GoldenQuestion(
        id="Q05",
        type="csv_aggregate",
        question="What is the total ARR across all Vertexia customers combined?",
        required_facts=["11,000,000"],
        partial_facts=["ARR", "annual recurring revenue", "customer"],
        disqualifiers=["8,450", "4,120"],
        explanation="customer_list.csv has 20 rows summing to $11,000,000 ARR. Needs a Pandas tool.",
        expected_outcome="FAIL",
        fixed_by_step="step_02_tools",
        reference_answer="The total ARR across all 20 Vertexia customers combined is $11,000,000.",
    ),

    GoldenQuestion(
        id="Q06",
        type="csv_aggregate",
        question="What was the total revenue across all products combined in Q3 2023 (July, August, and September)?",
        required_facts=["4,120,000"],
        partial_facts=["Q3", "revenue", "2023"],
        disqualifiers=["4.2M", "4,200,000"],
        explanation="revenue_by_product_2023.csv months 07/08/09 sum to $4,120,000.",
        expected_outcome="FAIL",
        fixed_by_step="step_02_tools",
        reference_answer="The total revenue across all products combined in Q3 2023 (July, August, and September) was $4,120,000.",
    ),

    GoldenQuestion(
        id="Q07",
        type="csv_aggregate",
        question="How many active Vertexia employees are based in Berlin?",
        required_facts=["5"],
        partial_facts=["Berlin", "employee", "location"],
        disqualifiers=[],
        explanation="employee_directory.csv filter status=active and location=Berlin → 5.",
        expected_outcome="FAIL",
        fixed_by_step="step_02_tools",
        reference_answer="There are 5 active Vertexia employees based in Berlin.",
    ),

    # ── TIER 3: BM25 / Keyword-Exact ──────────────────────────────────────────

    GoldenQuestion(
        id="Q08",
        type="keyword_exact",
        question="Which NexusFlow API endpoint was deprecated in v2.1, and what endpoint replaced it?",
        required_facts=["events/batch", "events/stream"],
        partial_facts=["v2.1", "deprecated", "NexusFlow"],
        disqualifiers=[],
        explanation=(
            "nexusflow_api_changelog.md: 'GET /v2/events/batch — Deprecated in v2.1' "
            "replaced by 'GET /v2/events/stream'. BM25 on 'v2.1' scores the right chunk."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_hybrid_retrieval",
        reference_answer="The GET /v2/events/batch endpoint was deprecated in NexusFlow API v2.1 and replaced by GET /v2/events/stream.",
    ),

    GoldenQuestion(
        id="Q09",
        type="keyword_exact",
        question="Who is the remediation owner for security audit finding M-2, and what was the target remediation date?",
        required_facts=["Daniel Osei", "October 31, 2023"],
        partial_facts=["M-2", "TLS", "security audit"],
        disqualifiers=[],
        explanation=(
            "security_audit_2023.txt: FINDING M-2 — Daniel Osei, target 2023-10-31. "
            "The finding ID 'M-2' is a unique BM25 token."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_03_hybrid_retrieval",
        reference_answer="The remediation owner for security audit finding M-2 is Daniel Osei, with a target remediation date of October 31, 2023.",
    ),

    GoldenQuestion(
        id="Q10",
        type="keyword_exact",
        question="What is Vertexia's annual spend on Snowflake and when does the contract expire?",
        required_facts=["120,000", "June 30, 2024"],
        partial_facts=["Snowflake", "data warehouse", "contract"],
        disqualifiers=[],
        explanation="vendor_contracts_summary.csv: Snowflake row, annual=120000, renewal=2024-06-30.",
        expected_outcome="FAIL",
        fixed_by_step="step_03_hybrid_retrieval",
        reference_answer="Vertexia's annual spend on Snowflake is $120,000, and the contract expires on June 30, 2024.",
    ),

    # ── TIER 4: Graph Multi-hop ───────────────────────────────────────────────

    GoldenQuestion(
        id="Q11",
        type="multi_hop",
        question="Who is the CSM managing the Phoenix Corp account, and who is that person's direct manager?",
        required_facts=["Maya Sharma", "Lisa Torres"],
        partial_facts=["CSM", "Phoenix Corp", "account"],
        disqualifiers=[],
        explanation=(
            "csm_account_history.csv → Maya Sharma. employee_directory.csv → manager = Lisa Torres. "
            "Two-hop join across two CSVs."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_04_knowledge_graph",
        reference_answer="Maya Sharma is the CSM managing the Phoenix Corp account, and her direct manager is Lisa Torres, the Chief Revenue Officer.",
    ),

    GoldenQuestion(
        id="Q12",
        type="multi_hop",
        question="If NexusFlow goes down entirely, which services are directly or indirectly affected? List all of them.",
        required_facts=["InsightLens", "PulseConnect", "DataCraft"],
        partial_facts=["blast radius", "dependency", "downstream"],
        disqualifiers=[],
        explanation="api_dependencies.csv BFS from NexusFlow — direct + indirect downstream services.",
        expected_outcome="FAIL",
        fixed_by_step="step_04_knowledge_graph",
        reference_answer="If NexusFlow goes down, the directly and indirectly affected services are InsightLens, PulseConnect, and DataCraft, all of which depend on NexusFlow's APIs.",
    ),

    GoldenQuestion(
        id="Q13",
        type="multi_hop",
        question="Who does Aisha Johnson report to, and who does that person report to? Give the full two-hop reporting chain.",
        required_facts=["Tomás García", "Sarah Chen"],
        partial_facts=["Aisha Johnson", "reports to", "manager"],
        disqualifiers=[],
        explanation="employee_directory.csv: Aisha Johnson → Tomás García → Sarah Chen (CTO). Two-hop org traversal.",
        expected_outcome="FAIL",
        fixed_by_step="step_04_knowledge_graph",
        reference_answer="Aisha Johnson reports to Tomas Garcia, who in turn reports to Sarah Chen, the CTO.",
    ),

    # ── TIER 5: Cross-document / Multi-step ───────────────────────────────────

    GoldenQuestion(
        id="Q14",
        type="cross_document",
        question="Does Vertexia's documented NexusFlow availability target meet the uptime requirement in the Phoenix Corp contract? What is the gap if any?",
        required_facts=["99.9", "99.99"],
        partial_facts=["availability", "SLA", "gap", "NexusFlow"],
        disqualifiers=["no gap", "currently meets", "fully meets", "does meet the"],
        explanation=(
            "nexusflow_architecture.md says 99.9% target. phoenix_corp_msa.txt requires 99.99%. "
            "Gap is 0.09 pp — current target does NOT meet the SLA. Requires comparing two documents."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_multi_agent",
        reference_answer="NexusFlow's documented availability target is 99.9% while the Phoenix Corp contract requires 99.99% uptime. There is a 0.09 percentage point gap, so the current target does not meet the SLA requirement.",
    ),

    GoldenQuestion(
        id="Q15",
        type="cross_document",
        question="Which employees left Vertexia voluntarily in 2023? For each person, state their department, last day, and the stated reason for departure.",
        required_facts=["Adrian Blake", "FinDataCo", "Preet Kaur", "relocated"],
        partial_facts=["voluntary", "departure", "offboarding", "2023"],
        disqualifiers=[],
        explanation=(
            "offboarding_records_2023.csv: Adrian Blake (Platform Eng, 2023-08-31, joined FinDataCo) "
            "and Preet Kaur (Revenue, 2023-06-30, relocated). Diana Volkov is a 2021 distractor."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05_multi_agent",
        reference_answer="Two employees left Vertexia voluntarily in 2023: Adrian Blake (Platform Engineering, last day 2023-08-31, joined competitor FinDataCo) and Preet Kaur (Revenue, last day 2023-06-30, relocated internationally).",
    ),
]
