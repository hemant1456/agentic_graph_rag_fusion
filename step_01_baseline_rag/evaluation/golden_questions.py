"""
Golden question set — designed to expose retrieval failure modes progressively.

Target baseline score: ~30–40% PASS (6–9 PASS, 7–9 PARTIAL, 5–8 FAIL) out of 22 questions.

Question categories:
  ANCHOR       — baseline must pass; prove the system works at all
  HARD_FAIL    — structural retrieval limits; require full CSV table scan or cross-CSV join
                 with no semantic keyword overlap → vector top-k cannot surface all needed rows
  PARTIAL      — some facts retrievable, but full answer requires multi-hop or arithmetic step

disqualifiers: if any of these strings appear in the answer, grade → FAIL
regardless of required_facts matches. Used to catch plausible but factually wrong answers.
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

    # ── ANCHORS (baseline must PASS) ─────────────────────────────────────────

    GoldenQuestion(
        id="Q01",
        type="simple_lookup",
        question="What is Vertexia's customer data retention policy for hot storage and cold storage?",
        required_facts=["90 day", "glacier"],
        partial_facts=["retention", "storage", "cold"],
        disqualifiers=[],
        explanation=(
            "Single-document answer in data_processing_agreement_template.txt or "
            "onboarding_handbook.txt. Semantic search for 'data retention policy' retrieves "
            "both documents immediately. Baseline anchor."
        ),
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),

    GoldenQuestion(
        id="Q02",
        type="simple_aggregation",
        question="How many engineers joined Vertexia through the DataCraft acquisition?",
        required_facts=["12"],
        partial_facts=["DataCraft", "acquisition", "engineer"],
        disqualifiers=[],
        explanation=(
            "Explicit number in founding_story.txt and datacraft_employee_integration.txt. "
            "Multiple documents contain '12 engineers'. Baseline anchor."
        ),
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),

    GoldenQuestion(
        id="Q03",
        type="simple_lookup",
        question="Who is the CEO of Vertexia and who co-founded the company with them?",
        required_facts=["Arjun Mehta", "Diana Volkov"],
        partial_facts=["CEO", "co-founder", "founder"],
        disqualifiers=[],
        explanation=(
            "Both names in founding_story.txt (CEO narrates it), employee_directory.csv, "
            "and series announcements. Any query about CEO or founders retrieves these. Anchor."
        ),
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),

    GoldenQuestion(
        id="Q04",
        type="simple_lookup",
        question="In what year was Vertexia Inc. founded?",
        required_facts=["2019"],
        partial_facts=["founded", "started", "San Francisco"],
        disqualifiers=[],
        explanation=(
            "founding_story.txt opens with 'March 2019'. Trivially retrieved. Anchor."
        ),
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),

    GoldenQuestion(
        id="Q05",
        type="simple_lookup",
        question="What message queue technology does NexusFlow use for its core pipeline?",
        required_facts=["Pulsar"],
        partial_facts=["message queue", "pipeline", "kafka"],
        disqualifiers=[],
        explanation=(
            "founding_story.txt explicitly mentions the Kafka → Apache Pulsar transition. "
            "nexusflow_architecture.md also references Pulsar. Easy retrieval. Anchor."
        ),
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),

    GoldenQuestion(
        id="Q06",
        type="simple_lookup",
        question="What is the NexusFlow platform's monthly availability SLO target as defined in its architecture document?",
        required_facts=["99.9"],
        partial_facts=["SLO", "availability", "uptime", "three nines"],
        disqualifiers=[],
        explanation=(
            "nexusflow_architecture.md states '99.9% (three nines) measured monthly'. "
            "Direct semantic match on 'NexusFlow architecture availability SLO'. Anchor."
        ),
        expected_outcome="PASS",
        fixed_by_step="baseline",
    ),

    # ── HARD FAIL — structural retrieval limits ───────────────────────────────
    # These require seeing every row of a CSV (N > top-k=5) or joining two CSVs
    # that share no semantic keywords with the query.

    GoldenQuestion(
        id="Q07",
        type="csv_full_aggregation",
        question="What is the total annual recurring revenue across all of Vertexia's current customers?",
        required_facts=["11"],   # 20 customers summing to $11,000,000 exactly
        partial_facts=["phoenix corp", "arr", "customer"],
        disqualifiers=[],
        explanation=(
            "customer_list.csv has 20 rows. top-k=5 retrieval returns at most 5 rows. "
            "The model will see Phoenix Corp ($2.4M) and QuantumBank ($3.2M) prominently and sum a partial set. "
            "Full sum requires reading all 20 rows: $11,000,000. "
            "Model will likely state $7M–$8M range or refuse to total."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured CSV query tool: SUM arr_usd)",
    ),

    GoldenQuestion(
        id="Q08",
        type="csv_full_aggregation",
        question="What is Vertexia's total annual spend across all vendor contracts?",
        required_facts=["956"],   # $956,400 across 15 vendors
        partial_facts=["AWS", "vendor", "annual", "contract"],
        disqualifiers=[],
        explanation=(
            "vendor_contracts_summary.csv has 15 rows. AWS ($480K) dominates top-5 retrieval. "
            "Full sum: 480K+120K+36K+48K+18K+24K+12K+8.4K+9.6K+14.4K+36K+48K+60K+24K+18K = $956,400. "
            "Model retrieves 5 of 15 rows (AWS + a few others) and computes a partial wrong sum. "
            "Without seeing all 15 rows the model cannot produce $956,400."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured CSV query tool: SUM annual_value_usd)",
    ),

    GoldenQuestion(
        id="Q09",
        type="csv_full_scan_filter",
        question="Which Vertexia employees are based in Berlin? List all of them.",
        required_facts=["Emma Fischer", "Noah Zimmermann", "Aleksander Nowak"],   # all 5: Felix Wagner, Ravi Krishnan, Emma Fischer, Noah Zimmermann, Aleksander Nowak
        partial_facts=["Felix Wagner", "Berlin", "DataCraft"],
        disqualifiers=[],
        explanation=(
            "employee_directory.csv has 48 rows. Only 5 have location=Berlin: "
            "Felix Wagner (E016), Ravi Krishnan (E026), Emma Fischer (E027), Noah Zimmermann (E028), "
            "Aleksander Nowak (E047). "
            "top-k retrieval on 'Berlin employees' will likely return the DataCraft Integration rows "
            "but Felix Wagner is prominent (team lead) and may appear in prose too. "
            "Emma Fischer, Noah Zimmermann, and Aleksander Nowak appear only in the CSV. "
            "PASS requires naming all three less-prominent engineers."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured CSV query: WHERE location='Berlin')",
    ),

    GoldenQuestion(
        id="Q10",
        type="csv_arithmetic_aggregation",
        question=(
            "Which Vertexia department has the highest annual budget per employee headcount, "
            "and what is that figure?"
        ),
        required_facts=["Platform Engineering", "195"],   # $8.2M / 42 HC = $195,238/head
        partial_facts=["budget", "headcount", "department"],
        disqualifiers=["166,666", "166666"],   # model sees 5/9 rows, declares Executive ($167K) as winner — wrong
        explanation=(
            "budget_allocation_2023.csv has 9 department rows. Finding the highest ratio "
            "requires dividing budget by headcount for all 9 rows and comparing. "
            "Platform Engineering: $8,200,000 / 42 = $195,238/head is the highest. "
            "top-k retrieval returns ~5 rows (Executive, Finance, Product, Legal, Revenue) "
            "and the model declares Executive at $166,666/head as winner — a confident wrong answer. "
            "Disqualifiers catch the wrong $166,666 figure."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: computed column + ORDER BY)",
    ),

    GoldenQuestion(
        id="Q11",
        type="csv_full_aggregation",
        question=(
            "How many deals did Vertexia close as Closed-Won in Q3 2023, "
            "and what was their combined ARR?"
        ),
        required_facts=["1.69"],   # 8 deals, $1,692,000
        partial_facts=["closed-won", "q3", "deals"],
        disqualifiers=[],
        explanation=(
            "deal_pipeline_q3_2023.csv has 8 Closed-Won rows (D100–D107). "
            "Total ARR: 240K+180K+420K+96K+360K+144K+192K+60K = $1,692,000. "
            "top-k=5 retrieves 5 of 8 rows; model gets an incomplete sum. "
            "Two Closed-Lost deals in the same CSV also risk confusing the model."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: WHERE stage='Closed-Won' AND SUM arr)",
    ),

    GoldenQuestion(
        id="Q12",
        type="csv_arithmetic_cross_doc",
        question=(
            "What percentage of Vertexia's total annual recurring revenue "
            "comes from enterprise-segment customers?"
        ),
        required_facts=["65"],   # $7.16M / $11M = 65.1% (Phoenix Corp $2.4M + QuantumBank $3.2M + Redwood Analytics $720K + Northgate Bank $840K)
        partial_facts=["Phoenix Corp", "enterprise", "percent"],
        disqualifiers=[],
        explanation=(
            "Requires two full-table passes: (1) filter customer_list.csv WHERE segment='enterprise' "
            "→ 4 enterprise customers: Phoenix Corp ($2.4M), QuantumBank ($3.2M), "
            "Redwood Analytics ($720K), Northgate Bank ($840K) = $7,160,000; "
            "(2) SUM all 20 ARRs = $11,000,000; (3) divide = 65.1%. "
            "top-k retrieval surfaces Phoenix Corp and QuantumBank prominently but cannot scan all 20 customers "
            "for the denominator, and may miss Redwood Analytics or Northgate Bank as enterprise."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: GROUP BY segment + total SUM)",
    ),

    GoldenQuestion(
        id="Q13",
        type="cross_csv_multi_hop",
        question=(
            "The engineer who departed Vertexia to join FinDataCo — "
            "what is the name of their direct manager?"
        ),
        required_facts=["Priya Nair"],
        partial_facts=["Adrian Blake", "manager", "FinDataCo", "E010"],
        disqualifiers=[],
        explanation=(
            "3-hop cross-CSV with zero keyword overlap on the final hop: "
            "(1) offboarding_records_2023.csv: 'FinDataCo' → Adrian Blake (E029); "
            "(2) employee_directory.csv row E029: manager_id = E010; "
            "(3) employee_directory.csv row E010: Priya Nair. "
            "The query contains 'FinDataCo' which retrieves offboarding CSV → Adrian Blake. "
            "But linking E029 → E010 → Priya Nair requires two lookups in employee_directory.csv "
            "that have no keyword connection to 'FinDataCo'. "
            "Model will say 'Adrian Blake departed to join FinDataCo' but cannot name the manager."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05 (graph: Person→reports_to edge traversal)",
    ),

    GoldenQuestion(
        id="Q14",
        type="cross_csv_multi_hop",
        question=(
            "Who is listed as the owner of our Snowflake data warehouse contract, "
            "and who is their direct manager?"
        ),
        required_facts=["Priya Nair", "Marcus Webb"],
        partial_facts=["Snowflake", "contract", "owner"],
        disqualifiers=[],
        explanation=(
            "3-hop cross-CSV: "
            "(1) vendor_contracts_summary.csv: Snowflake owner = Priya Nair; "
            "(2) employee_directory.csv: Priya Nair (E010), manager_id = E009; "
            "(3) employee_directory.csv row for E009 = Marcus Webb. "
            "Query 'Snowflake contract owner manager' retrieves vendor CSV (Priya Nair) "
            "but has no keywords to pull Marcus Webb's employee row. "
            "The join requires matching E009 ID across two unrelated CSV rows."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_05 (graph: Person→reports_to edge) + step_07 (vendor lookup)",
    ),

    GoldenQuestion(
        id="Q15",
        type="csv_arithmetic_full_scan",
        question=(
            "What percentage of Vertexia's total 2023 annual budget is allocated "
            "to the two engineering departments combined (Platform Engineering and Product Engineering)?"
        ),
        required_facts=["59"],   # ($8.2M + $9.8M) / $30.2M = 59.6%
        partial_facts=["Platform Engineering", "Product Engineering", "budget"],
        disqualifiers=[],
        explanation=(
            "budget_allocation_2023.csv: 9 department rows. "
            "Platform Engineering $8.2M + Product Engineering $9.8M = $18M. "
            "Total across all 9 departments = $30.2M. "
            "Percentage = 18 / 30.2 = 59.6%. "
            "top-k=5 retrieval will surface Platform Eng and Product Eng rows (largest budgets). "
            "But computing the denominator ($30.2M total) requires all 9 rows. "
            "Without the total, the model cannot compute the percentage and will say "
            "'I cannot determine the total company budget to calculate the percentage.' "
            "Model correctly identifies numerator (~$18M) but cannot get denominator."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: SUM annual_budget_usd for denominator)",
    ),

    GoldenQuestion(
        id="Q16",
        type="csv_date_filter_aggregation",
        question=(
            "What is the total combined ARR of all customers who signed contracts "
            "in the second half of 2023 (July through December)?"
        ),
        required_facts=["3.12"],   # 11 customers: 420K+180K+360K+96K+144K+192K+60K+480K+48K+840K+300K = $3,120,000
        partial_facts=["TechVenture", "2023", "contract"],
        disqualifiers=[],
        explanation=(
            "customer_list.csv: 11 customers have contract_start >= 2023-07-01: "
            "TechVenture ($420K), GreenLeaf ($180K), Meridian ($360K), Harbor ($96K), "
            "Cascade ($144K), Pinnacle ($192K), Stellarpath ($60K), BlueRidge Energy ($480K), "
            "Pacific Dynamics ($48K), Northgate Bank ($840K), Crestwood Pharma ($300K) = $3,120,000. "
            "Requires date-range filter on the contract_start column + SUM across all 11 rows. "
            "Cosine search on 'H2 2023 contracts' retrieves 2-3 rows at best; "
            "model cannot see all 11 to produce the correct sum."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: WHERE contract_start >= '2023-07-01' AND SUM)",
    ),

    # ── SHOULD BE PARTIAL AT BASELINE ────────────────────────────────────────

    GoldenQuestion(
        id="Q17",
        type="stale_reference",
        question=(
            "According to our customer records, who is the customer success manager "
            "for Apex Financial, and based on all available HR records, are they "
            "still at Vertexia and under what circumstances did they leave?"
        ),
        required_facts=["Preet Kaur", "departed", "voluntary"],
        partial_facts=["Apex", "customer success", "CSM"],
        disqualifiers=[],
        explanation=(
            "customer_list.csv names Preet Kaur as CSM for Apex Financial. "
            "offboarding_records_2023.csv has departure_type='voluntary' for her. "
            "csm_account_history.csv shows the transition but does NOT have departure_type. "
            "Vector search for 'CSM Apex Financial HR records' retrieves csm_account_history "
            "and customer_list — but NOT offboarding_records (no semantic overlap). "
            "Model will say 'Preet Kaur departed' (PARTIAL) but cannot confirm 'voluntary' "
            "without retrieving offboarding_records."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_05 (graph: Person node with employment_status edge)",
    ),

    GoldenQuestion(
        id="Q18",
        type="disambiguation_no_name",
        question=(
            "Vertexia has two separate internal efforts that share the same name. "
            "What are they, and what was the outcome of each?"
        ),
        required_facts=["Python", "Phoenix Corp", "completed", "signed"],
        partial_facts=["migration", "enterprise", "2022", "Project Phoenix"],
        disqualifiers=[],
        explanation=(
            "The shared name is 'Phoenix' — never stated in the query. "
            "Engineering 'Project Phoenix' = Python 2→3 migration (completed June 2022). "
            "Sales 'Phoenix' = Phoenix Corp enterprise contract (signed June 2022). "
            "founding_story.txt mentions both, but the model needs to identify the "
            "disambiguation and state BOTH outcomes precisely. "
            "'signed' specifically (for the contract) is the hardest fact to surface."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_08 (agent: entity disambiguation + multi-query strategy)",
    ),

    GoldenQuestion(
        id="Q19",
        type="sla_breach_inference",
        question=(
            "Based on the August 2023 incident report, did Vertexia meet its "
            "standard 99.9% uptime commitment to customers that month? "
            "Show your calculation."
        ),
        required_facts=["did not", "99.4", "263"],   # 263 min downtime / 44640 min in August = 99.411%
        partial_facts=["4 hour", "outage", "99.9", "uptime"],
        disqualifiers=["yes"],
        explanation=(
            "Requires: (1) postmortem → 4h 23min outage duration; "
            "(2) nexusflow_architecture.md → 99.9% SLO; "
            "(3) 263 min / 44640 min in August = 0.589% downtime = 99.411% uptime < 99.9% → MISSED. "
            "Model often retrieves the postmortem and knows there was an outage, but "
            "must also retrieve the architecture doc for the SLO target AND do the arithmetic. "
            "'99.4' must appear as the calculated uptime figure (99.411% rounds to 99.4%)."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_10 (context engineering: chain-of-thought + calculation prompt)",
    ),

    GoldenQuestion(
        id="Q20",
        type="multi_format_multi_hop",
        question=(
            "Which products were directly or indirectly affected by the August 2023 NexusFlow outage, "
            "and what was their combined revenue in the month the outage occurred?"
        ),
        required_facts=["InsightLens", "1.02"],   # NexusFlow $520K + InsightLens $500K = $1.02M in Aug
        partial_facts=["NexusFlow", "outage", "revenue"],
        disqualifiers=["PulseConnect"],   # PulseConnect was NOT critically affected
        explanation=(
            "Chain: postmortem → NexusFlow outage → "
            "api_dependencies.csv → InsightLens critical dep on events_api → "
            "revenue_by_product_2023.csv → August: NexusFlow $520K + InsightLens $500K = $1.02M. "
            "Baseline names NexusFlow (retrieved via postmortem) but misses InsightLens "
            "(requires api_dependencies lookup) and won't compute the revenue sum."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_05 (graph: product dependency edges) + step_07 (revenue lookup)",
    ),

    GoldenQuestion(
        id="Q21",
        type="multi_hop_implicit",
        question=(
            "If the analytics dashboard had an ingestion failure on August 29, 2023, "
            "which on-call Data Platform engineer would be responsible for the fix?"
        ),
        required_facts=["Priya Nair"],
        partial_facts=["on-call", "data platform", "august"],
        disqualifiers=["Kenji Ito", "Lin Wei", "James O'Brien", "Yuki Tanaka"],
        explanation=(
            "4-hop chain: 'analytics dashboard' = InsightLens → events_api → Data Platform Team → "
            "on_call_schedule_aug2023.csv: week of Aug 28–Sep 3 = Priya Nair (E010). "
            "founding_story.txt prominently names Kenji Ito for 'the August 2023 outage' → "
            "the model anchors to Kenji Ito for ANY August 2023 analytics question. "
            "Aug 29 is week Aug 28-Sep 3; the correct answer is Priya Nair. "
            "This question exploits the model's retrieval bias toward the famous outage narrative "
            "(Kenji Ito) instead of the correct schedule row for Aug 29."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_08 (agent: structured date-range query on CSV)",
    ),

    GoldenQuestion(
        id="Q22",
        type="blast_radius_multi_hop",
        question=(
            "If Vertexia's primary message queue infrastructure went down completely, "
            "trace the full blast radius: which internal products would be affected "
            "and which specific enterprise customers would lose service?"
        ),
        required_facts=["InsightLens", "events_api", "QuantumBank"],
        partial_facts=["Pulsar", "NexusFlow", "pipeline", "Phoenix Corp", "enterprise"],
        disqualifiers=[],
        explanation=(
            "Chain: Pulsar down → NexusFlow (critical dep on external_pulsar in api_dependencies.csv) → "
            "NexusFlow events_api fails → InsightLens dashboard ingestion fails (critical dep) → "
            "enterprise customers on NexusFlow or InsightLens lose service → "
            "includes QuantumBank ($3.2M ARR, NexusFlow+InsightLens+PulseConnect) and Phoenix Corp. "
            "'events_api' requires api_dependencies.csv. "
            "'QuantumBank' requires customer_list.csv (only enterprise customers data source). "
            "Baseline retrieves architecture/postmortem prose docs; neither api_dependencies.csv "
            "nor customer_list.csv is pulled by cosine similarity on this query. "
            "Model names NexusFlow and InsightLens generically but cannot name QuantumBank "
            "without retrieving customer_list.csv."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_05 (graph: product→dependency→customer edges)",
    ),
]
