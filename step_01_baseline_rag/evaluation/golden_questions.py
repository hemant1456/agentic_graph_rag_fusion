"""Golden question set — 15 questions across 6 tiers.

Each tier showcases a capability that the corresponding step adds:
  Tier 1 (Q01-Q02):  Simple retrieval     — step_01_baseline_rag
  Tier 2 (Q03-Q04):  Format-aware chunks  — step_02_chunking
  Tier 3 (Q05-Q07):  CSV aggregates       — step_03_tools
  Tier 4 (Q08-Q10):  BM25 keyword-exact   — step_04_hybrid_retrieval
  Tier 5 (Q11-Q13):  Graph multi-hop      — step_05_knowledge_graph
  Tier 6 (Q14-Q15):  Cross-document       — step_07_multi_agent

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


GOLDEN_QUESTIONS: list[GoldenQuestion] = [

    # ── TIER 1: Simple Retrieval ───────────────────────────────────────────────

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
    ),

    # ── TIER 2: Format-aware Chunking ─────────────────────────────────────────

    GoldenQuestion(
        id="Q03",
        type="chunking_dependent",
        question="In Vertexia's on-call runbook, what is the first action and the escalation owner for the PulseConnect webhook delivery failure alert?",
        required_facts=["SendGrid", "Twilio", "Raj Patel"],
        partial_facts=["webhook", "PulseConnect", "on-call"],
        disqualifiers=["Felix Wagner", "Aisha Johnson", "Kenji Ito"],
        explanation=(
            "oncall_runbook_top_alerts.md '## PulseConnect webhook_delivery_failure_rate > 5%' "
            "section. Step 01's chunker merges multiple alert sections; Step 02's section-aware "
            "chunker with contextual headers returns the exact section."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_02_chunking",
    ),

    GoldenQuestion(
        id="Q04",
        type="chunking_dependent",
        question="According to Vertexia's vendor data processing matrix, what sub-processors does Datadog use, and what is its data retention period for traces and logs?",
        required_facts=["us-east-1", "eu-west-1", "ap-southeast-1", "15 months"],
        partial_facts=["Datadog", "sub-processor", "retention"],
        disqualifiers=["us-central1", "90 days", "18 months", "24 months"],
        explanation=(
            "vendor_data_processing_matrix.md '## Datadog' section. Step 01 mixes Snowflake / "
            "SendGrid / Stripe fields into the same chunk; Step 02's per-vendor chunks isolate them."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_02_chunking",
    ),

    # ── TIER 3: CSV Computation ───────────────────────────────────────────────

    GoldenQuestion(
        id="Q05",
        type="csv_aggregate",
        question="What is the total ARR across all Vertexia customers combined?",
        required_facts=["11,000,000"],
        partial_facts=["ARR", "annual recurring revenue", "customer"],
        disqualifiers=["8,450", "4,120"],
        explanation="customer_list.csv has 20 rows summing to $11,000,000 ARR. Needs a Pandas tool.",
        expected_outcome="FAIL",
        fixed_by_step="step_03_tools",
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
        fixed_by_step="step_03_tools",
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
        fixed_by_step="step_03_tools",
    ),

    # ── TIER 4: BM25 / Keyword-Exact ──────────────────────────────────────────

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
        fixed_by_step="step_04_hybrid_retrieval",
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
        fixed_by_step="step_04_hybrid_retrieval",
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
        fixed_by_step="step_04_hybrid_retrieval",
    ),

    # ── TIER 5: Graph Multi-hop ───────────────────────────────────────────────

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
        fixed_by_step="step_05_knowledge_graph",
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
        fixed_by_step="step_05_knowledge_graph",
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
        fixed_by_step="step_05_knowledge_graph",
    ),

    # ── TIER 6: Cross-document / Multi-step ───────────────────────────────────

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
        fixed_by_step="step_07_multi_agent",
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
        fixed_by_step="step_07_multi_agent",
    ),
]
