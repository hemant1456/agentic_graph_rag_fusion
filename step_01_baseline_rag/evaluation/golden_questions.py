"""
Golden question set — designed to expose retrieval failure modes progressively.

Target baseline score: ~25–35% (2-3 PASS, 3-4 PARTIAL, 3-4 FAIL).
Each non-anchor question targets a specific capability gap that a later step fixes.

disqualifiers: if any of these strings appear in the answer, grade → FAIL
regardless of required_facts matches. Used for questions where the model can
produce a plausible-sounding but factually wrong answer.
"""

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
    expected_outcome: str
    fixed_by_step: str


GOLDEN_QUESTIONS: list[GoldenQuestion] = [
    # ── ANCHORS (baseline must pass — prove the system works at all) ──────────

    GoldenQuestion(
        id="Q01",
        type="simple_lookup",
        question="What is Vertexia's customer data retention policy?",
        required_facts=["90 day", "glacier"],
        partial_facts=["retention", "storage", "cold"],
        disqualifiers=[],
        explanation="Single-document answer in onboarding_handbook.txt. Baseline anchor.",
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),
    GoldenQuestion(
        id="Q02",
        type="simple_aggregation",
        question="How many employees joined Vertexia through the DataCraft acquisition?",
        required_facts=["12"],
        partial_facts=["DataCraft", "acquisition"],
        disqualifiers=[],
        explanation="Explicit number in datacraft_employee_integration.txt. Second anchor.",
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),

    # ── SHOULD FAIL BASELINE ──────────────────────────────────────────────────

    GoldenQuestion(
        id="Q03",
        type="csv_arithmetic",
        question=(
            "What is the uptime gap in percentage points between the SLA in our "
            "highest-ARR customer contract and the availability target in our primary "
            "data pipeline's architecture document?"
        ),
        required_facts=["0.09"],   # 99.99% − 99.9% = 0.09 pp
        partial_facts=["99.99", "99.9", "gap", "uptime"],
        disqualifiers=[],
        explanation=(
            "3-hop + arithmetic: (1) highest-ARR customer = Phoenix Corp ($2.4M, from CSV); "
            "(2) their SLA = 99.99% (legal/phoenix_corp_msa.txt); "
            "(3) pipeline target = 99.9% (engineering/nexusflow_architecture.md); "
            "(4) gap = 0.09 percentage points. "
            "Even if retrieval finds both documents, model must compute 99.99 − 99.9 = 0.09. "
            "Most LLMs will mention both numbers but skip the arithmetic step."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_08 (agent with calculation tool)",
    ),
    GoldenQuestion(
        id="Q04",
        type="multi_hop_implicit",
        question=(
            "If the analytics dashboard stopped ingesting data tonight, "
            "which on-call engineer would be responsible for the fix "
            "based on the August 14 2023 rotation schedule?"
        ),
        required_facts=["Kenji", "Ito"],
        partial_facts=["on-call", "data platform", "schedule"],
        disqualifiers=["Priya Nair", "James O'Brien", "Lin Wei"],
        explanation=(
            "4-hop chain: 'analytics dashboard' = InsightLens → "
            "api_dependencies.csv: InsightLens depends on NexusFlow events_api (critical) → "
            "data_platform_runbook.md: events_api owned by Data Platform Team → "
            "on_call_schedule_aug2023.csv: week of Aug 14 = Kenji Ito. "
            "Baseline retrieves the schedule CSV but the model anchors to 'tonight' "
            "and reads the last/most recent row instead of week-of-Aug-14."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_08 (agent: structured date-range query on CSV)",
    ),
    GoldenQuestion(
        id="Q07",
        type="csv_aggregation",
        question="What is the total annual recurring revenue from all customers currently using NexusFlow?",
        required_facts=["3.6"],   # $3.612M: Phoenix $2.4M + Apex $240k + TechVenture $420k + Meridian $360k + Pinnacle $192k
        partial_facts=["phoenix", "nexusflow", "arr"],
        disqualifiers=[],
        explanation=(
            "Requires filtering customer_list.csv WHERE products contains 'NexusFlow', "
            "then summing ARR: Phoenix $2.4M + Apex $240k + TechVenture $420k + "
            "Meridian $360k + Pinnacle $192k = $3.612M. "
            "Top-5 retrieval gets at most 2-3 rows; model can't aggregate all 5."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query tool for CSV)",
    ),
    GoldenQuestion(
        id="Q08",
        type="multi_format_multi_hop",
        question=(
            "Which products were directly or indirectly affected by the August 2023 outage, "
            "and what was their combined revenue in the month the outage occurred?"
        ),
        required_facts=["InsightLens", "1.02"],  # NexusFlow $520k + InsightLens $500k = $1.02M in August
        partial_facts=["NexusFlow", "outage", "revenue"],
        disqualifiers=["PulseConnect"],  # PulseConnect was NOT affected
        explanation=(
            "Chain: postmortem → NexusFlow outage → "
            "api_dependencies.csv → InsightLens critical dep on events_api → "
            "revenue_by_product_2023.csv → August: NexusFlow $520k + InsightLens $500k = $1.02M. "
            "Baseline will name NexusFlow (retrieved via postmortem) but miss InsightLens "
            "and definitely won't compute the revenue sum."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (graph + structured query fusion)",
    ),

    # ── SHOULD BE PARTIAL AT BASELINE ────────────────────────────────────────

    GoldenQuestion(
        id="Q05",
        type="temporal_inference",
        question=(
            "What was the title of the person who is currently Vertexia's CTO, "
            "at the time the DataCraft acquisition was completed?"
        ),
        required_facts=["VP Engineering"],
        partial_facts=["Sarah Chen", "engineering", "2022"],
        disqualifiers=[],
        explanation=(
            "Requires: (1) who is current CTO → Sarah Chen; "
            "(2) when did DataCraft close → January 2022; "
            "(3) Sarah Chen's title in January 2022 → VP Engineering (became CTO April 2023). "
            "No date or person name in query. Model likely retrieves CTO-related docs "
            "and may answer 'CTO' (wrong temporal state) or correctly reason to VP Engineering."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_05 (graph: temporal edges on Person nodes)",
    ),
    GoldenQuestion(
        id="Q06",
        type="stale_reference",
        question=(
            "According to our customer records, who is the customer success manager "
            "for Apex Financial, and are they currently employed at Vertexia?"
        ),
        required_facts=["Preet Kaur", "departed"],
        partial_facts=["Apex", "customer success", "CSM"],
        disqualifiers=[],
        explanation=(
            "customer_list.csv names Preet Kaur as CSM. "
            "hr/offboarding_records_2023.csv shows she departed June 30 2023. "
            "These are in separate CSVs with no explicit link. "
            "Baseline will name Preet Kaur (PARTIAL) but almost certainly won't "
            "cross-reference the offboarding CSV to confirm she has departed."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_05 (graph: Person node with employment_status edge)",
    ),
    GoldenQuestion(
        id="Q09",
        type="disambiguation_no_name",
        question=(
            "Vertexia has two separate efforts that share the same internal name. "
            "What are they, and what was the outcome of each?"
        ),
        required_facts=["Python", "Phoenix Corp", "completed", "signed"],
        partial_facts=["migration", "enterprise", "2022"],
        disqualifiers=[],
        explanation=(
            "The shared name is 'Phoenix' — never stated in the query. "
            "Engineering 'Project Phoenix' = Python 2→3 migration (completed June 2022). "
            "Sales 'Phoenix' = Phoenix Corp enterprise contract (signed June 2022). "
            "Without 'Phoenix' in the query, cosine search has no anchor. "
            "Model must retrieve and connect two semantically unrelated documents."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_08 (agent: entity disambiguation + multi-query strategy)",
    ),
    GoldenQuestion(
        id="Q10",
        type="sla_breach_inference",
        question=(
            "Based on the August 2023 incident report, did Vertexia meet its "
            "standard 99.9% uptime commitment to customers that month? "
            "Show your calculation."
        ),
        required_facts=["did not", "99.4"],   # 4h23min / 31-day month = 0.589% downtime = 99.411% uptime
        partial_facts=["4 hour", "outage", "99.9", "uptime"],
        disqualifiers=["yes"],
        explanation=(
            "Requires: (1) postmortem → 4h 23min outage duration; "
            "(2) nexusflow_architecture.md → 99.9% standard SLA; "
            "(3) calculate: 263 min downtime / 44640 min in August = 0.589% downtime "
            "= 99.411% uptime < 99.9% → MISSED. "
            "Baseline retrieves the postmortem and knows there was an outage, but "
            "won't retrieve the architecture SLA target AND perform the arithmetic."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_10 (context engineering: chain-of-thought + calculation)",
    ),
]
