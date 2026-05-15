"""
Golden question set for Step 01 evaluation.

10 questions designed to test different retrieval failure modes.
Each question has:
  - type: the failure mode category being tested
  - question: the query string
  - required_facts: strings that MUST appear in a correct answer (case-insensitive)
  - explanation: why this question is hard for naive RAG
  - expected_outcome: our hypothesis for baseline RAG performance
"""

from dataclasses import dataclass


@dataclass
class GoldenQuestion:
    id: str
    type: str
    question: str
    required_facts: list[str]      # ALL must be present for PASS
    partial_facts: list[str]       # subset presence = PARTIAL
    explanation: str
    expected_outcome: str          # PASS / PARTIAL / FAIL hypothesis


GOLDEN_QUESTIONS: list[GoldenQuestion] = [
    GoldenQuestion(
        id="Q01",
        type="simple_lookup",
        question="What is Vertexia's data retention policy for customer data?",
        required_facts=["90 day", "cold storage", "glacier"],
        partial_facts=["retention", "storage"],
        explanation="Answer lives verbatim in onboarding_handbook.txt. Easy for vector search.",
        expected_outcome="PASS",
    ),
    GoldenQuestion(
        id="Q02",
        type="comparative",
        question="How did Q3 2023 NexusFlow revenue compare to Q2 2023?",
        required_facts=["1.6", "1.4"],   # Q3 NexusFlow $1.6M, Q2 $1.4M
        partial_facts=["revenue", "Q3", "Q2"],
        explanation="Answer requires reading the revenue CSV and comparing rows. CSV rows embed poorly.",
        expected_outcome="PARTIAL",
    ),
    GoldenQuestion(
        id="Q03",
        type="multi_hop",
        question="Who was the on-call engineer for the data platform during the August 2023 outage?",
        required_facts=["Kenji", "Ito"],
        partial_facts=["on-call", "data platform", "outage"],
        explanation=(
            "3-hop chain: postmortem → data platform team ownership → on-call schedule CSV. "
            "No single document contains the answer. Vector search will retrieve the postmortem "
            "but miss the on-call schedule."
        ),
        expected_outcome="FAIL",
    ),
    GoldenQuestion(
        id="Q04",
        type="temporal",
        question="What was Sarah Chen's title in January 2023?",
        required_facts=["VP Engineering"],
        partial_facts=["Sarah Chen", "engineering"],
        explanation=(
            "Sarah Chen was VP Engineering until April 2023, then CTO. "
            "Vector search may retrieve both eras of documents and confuse them."
        ),
        expected_outcome="PARTIAL",
    ),
    GoldenQuestion(
        id="Q05",
        type="disambiguation",
        question="What is Project Phoenix at Vertexia?",
        required_facts=["Python", "migration", "Phoenix Corp"],
        partial_facts=["Phoenix"],
        explanation=(
            "Two completely different things share the name 'Phoenix': "
            "an internal Python 2→3 migration (engineering) and an enterprise customer deal (sales). "
            "A correct answer must acknowledge both. Naive RAG will likely return one or the other."
        ),
        expected_outcome="PARTIAL",
    ),
    GoldenQuestion(
        id="Q06",
        type="implicit_link",
        question="Does the Phoenix Corp SLA requirement exceed NexusFlow's documented availability target?",
        required_facts=["99.99", "99.9", "yes"],
        partial_facts=["SLA", "uptime", "availability"],
        explanation=(
            "Requires connecting two documents from different departments: "
            "legal/phoenix_corp_msa.txt (99.99% SLA) and engineering/nexusflow_architecture.md (99.9% target). "
            "No single document contains both numbers. Classic cross-document reasoning."
        ),
        expected_outcome="FAIL",
    ),
    GoldenQuestion(
        id="Q07",
        type="contradictory_data",
        question="What was Vertexia's total revenue in Q3 2023?",
        required_facts=["4.12", "4.2"],   # must acknowledge BOTH figures
        partial_facts=["revenue", "Q3", "million"],
        explanation=(
            "Two correct but different answers exist: $4.12M (GAAP, finance report) "
            "and $4.2M (bookings, sales/all-hands). Naive RAG returns whichever it retrieves first "
            "without noting the discrepancy."
        ),
        expected_outcome="PARTIAL",
    ),
    GoldenQuestion(
        id="Q08",
        type="aggregation",
        question="How many employees joined Vertexia through the DataCraft acquisition?",
        required_facts=["12"],
        partial_facts=["DataCraft", "acquisition", "employees"],
        explanation="The number 12 appears explicitly in datacraft_employee_integration.txt. Should be retrievable.",
        expected_outcome="PASS",
    ),
    GoldenQuestion(
        id="Q09",
        type="stale_reference",
        question="Who is the VP of Customer Success at Vertexia?",
        required_facts=["Maya Sharma"],
        partial_facts=["customer success", "VP"],
        explanation=(
            "Current answer is Maya Sharma. But old documents may surface Preet Kaur "
            "(departed CSM). Tests whether naive RAG retrieves stale info. "
            "Org chart Q3 2023 has the correct current answer."
        ),
        expected_outcome="PASS",
    ),
    GoldenQuestion(
        id="Q10",
        type="cross_format",
        question="Was InsightLens affected by the August 2023 NexusFlow outage? Explain why or why not.",
        required_facts=["yes", "events_api"],
        partial_facts=["InsightLens", "NexusFlow", "dependency"],
        explanation=(
            "InsightLens depends on NexusFlow's events_api (critical dependency). "
            "This fact ONLY exists in engineering/api_dependencies.csv — a single CSV row. "
            "No prose document states this link. Tests whether CSV row chunks surface via vector search."
        ),
        expected_outcome="FAIL",
    ),
]
