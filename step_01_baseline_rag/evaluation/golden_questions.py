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
            "Multiple documents contain '12 employees / 12 engineers'. Baseline anchor."
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

    GoldenQuestion(
        id="Q07",
        type="csv_full_aggregation",
        question="What is the total annual recurring revenue across all of Vertexia's current customers?",
        required_facts=["11,000"],   # $11,000,000 exact across 20 customers
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "customer_list.csv has 20 rows summing to exactly $11,000,000. "
            "top-k=5 returns ~5 rows; model computes a partial sum around $4–5M and states "
            "that confidently. Required fact '11,000' matches '$11,000,000' but not any "
            "partial sum. partial_facts=[] so a wrong calculation does not earn PARTIAL."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured CSV query: SUM arr_usd)",
    ),

    GoldenQuestion(
        id="Q08",
        type="csv_full_aggregation",
        question="What is Vertexia's total annual spend across all vendor contracts?",
        required_facts=["956,400"],   # exact total across 15 vendors
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "vendor_contracts_summary.csv has 15 rows totalling $956,400. "
            "AWS ($480K) dominates top-5 retrieval. Model sees 5 of 15 rows and computes "
            "a partial sum around $130–200K. '956,400' will not appear in that answer. "
            "partial_facts=[] so any wrong sum is FAIL."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured CSV query: SUM annual_value_usd)",
    ),

    GoldenQuestion(
        id="Q09",
        type="csv_full_scan_filter",
        question="How many Vertexia employees are currently based in the Berlin office?",
        required_facts=["5"],   # Felix Wagner, Ravi Krishnan, Emma Fischer, Noah Zimmermann, Aleksander Nowak
        partial_facts=[],
        disqualifiers=["2", "3", "4"],   # common partial-scan wrong answers
        explanation=(
            "employee_directory.csv has 47 rows; exactly 5 have location=Berlin: "
            "Felix Wagner (E016), Ravi Krishnan (E026), Emma Fischer (E027), "
            "Noah Zimmermann (E028), Aleksander Nowak (E047). "
            "top-k=5 retrieval from org_chart + employee_directory typically surfaces "
            "1–2 Berlin rows → model says '2' or '3'. Disqualifiers catch those wrong counts. "
            "partial_facts=[] so naming some but not all Berlin employees does not earn PARTIAL. "
            "Changed from 'list all' to count because the scoring system can't check exhaustiveness."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured CSV query: WHERE location='Berlin' COUNT)",
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
        disqualifiers=["166,666", "166666"],   # model sees 5/9 rows, names Executive as winner — wrong
        explanation=(
            "budget_allocation_2023.csv has 9 department rows. Highest ratio: "
            "Platform Engineering $8,200,000 / 42 = $195,238/head. "
            "top-k=5 returns Executive, Finance, Product, Legal, DataCraft rows; "
            "model declares Executive at $166,666/head as winner — disqualifiers catch this. "
            "partial_facts kept because the model demonstrates correct methodology (dividing "
            "budget by headcount) even when it uses incomplete data."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: computed column budget/headcount + ORDER BY DESC)",
    ),

    GoldenQuestion(
        id="Q11",
        type="csv_full_aggregation",
        question=(
            "How many deals did Vertexia close as Closed-Won in Q3 2023, "
            "and what was their combined ARR?"
        ),
        required_facts=["1,692"],   # 8 deals, $1,692,000 (comma format to match LLM output)
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "deal_pipeline_q3_2023.csv has 8 Closed-Won deals (D100–D107): "
            "240K+180K+420K+96K+360K+144K+192K+60K = $1,692,000. "
            "top-k=5 retrieves 3 of 8 Closed-Won rows; model computes an incomplete sum. "
            "'1,692' matches '$1,692,000' but not any partial sum. "
            "partial_facts=[] so any wrong sum is FAIL, not PARTIAL."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: WHERE stage='Closed-Won' AND SUM arr_usd)",
    ),

    GoldenQuestion(
        id="Q12",
        type="csv_arithmetic_cross_doc",
        question=(
            "What percentage of Vertexia's total annual recurring revenue "
            "comes from enterprise-segment customers?"
        ),
        required_facts=["65"],   # $7,160,000 / $11,000,000 = 65.1%
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "Requires two full-table passes over customer_list.csv: "
            "(1) filter WHERE segment='enterprise' → 4 customers: Phoenix Corp ($2.4M), "
            "QuantumBank ($3.2M), Redwood Analytics ($720K), Northgate Bank ($840K) = $7,160,000; "
            "(2) SUM all 20 ARRs = $11,000,000; (3) divide = 65.1%. "
            "top-k=5 retrieves prose strategy docs that mention enterprise broadly but "
            "never the full customer table. Model cannot calculate without all 20 rows. "
            "partial_facts=[] so a generic mention of 'enterprise' does not earn PARTIAL."
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
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "3-hop cross-CSV: "
            "(1) offboarding_records_2023.csv: 'FinDataCo' → Adrian Blake (E029); "
            "(2) employee_directory.csv row E029: manager_id = E010; "
            "(3) employee_directory.csv row E010: Priya Nair. "
            "Query retrieves offboarding CSV → Adrian Blake is findable. "
            "But linking E029 → E010 → 'Priya Nair' requires two directory lookups "
            "with no keyword bridge from 'FinDataCo'. "
            "partial_facts=[] because Adrian Blake (the intermediate entity) is not "
            "the answer — giving PARTIAL credit for finding the departing employee "
            "would misrepresent the question's difficulty."
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
        required_facts=["Marcus Webb"],   # the manager — that's the answer being tested
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "3-hop cross-CSV: "
            "(1) vendor_contracts_summary.csv: Snowflake owner = Priya Nair (E010); "
            "(2) employee_directory.csv row E010: manager_id = E009; "
            "(3) employee_directory.csv row E009: Marcus Webb. "
            "Query 'Snowflake contract owner manager' retrieves vendor CSV (Priya Nair). "
            "But linking E010 → E009 → Marcus Webb requires two lookups with no keyword "
            "connection to the Snowflake query. "
            "required_facts=[Marcus Webb] only — the question is asking for the manager. "
            "partial_facts=[] to avoid Priya Nair (the contract owner, not the answer) "
            "accidentally triggering PARTIAL."
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
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "budget_allocation_2023.csv: 9 department rows. "
            "Platform Engineering $8.2M + Product Engineering $9.8M = $18M (numerator). "
            "Total across all 9 departments = $30.2M (denominator). "
            "Percentage = 18 / 30.2 = 59.6%. "
            "top-k=5 surfaces Platform Eng and Product Eng rows (largest budgets) so the "
            "numerator ($18M) is computable, but the denominator requires all 9 rows. "
            "Without the total, the model says it cannot compute the percentage. "
            "partial_facts=[] so correctly naming the engineering departments does not "
            "earn PARTIAL — the question is about the percentage, not the departments."
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
        required_facts=["3,120"],   # 11 customers, $3,120,000 (comma format to match LLM output)
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "customer_list.csv: 11 customers have contract_start >= 2023-07-01: "
            "TechVenture ($420K), GreenLeaf ($180K), Meridian ($360K), Harbor ($96K), "
            "Cascade ($144K), Pinnacle ($192K), Stellarpath ($60K), BlueRidge Energy ($480K), "
            "Pacific Dynamics ($48K), Northgate Bank ($840K), Crestwood Pharma ($300K) = $3,120,000. "
            "Requires date-range filter on contract_start column + SUM across 11 rows. "
            "top-k=5 on 'H2 2023 contracts ARR' retrieves 2–3 rows at best. "
            "'3,120' matches '$3,120,000'; no partial sum would hit this string. "
            "partial_facts=[] so finding one H2 customer does not earn PARTIAL."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: WHERE contract_start >= '2023-07-01' AND SUM arr_usd)",
    ),

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
            "csm_account_history.csv shows the account transition but says only "
            "'CSM departure (Preet Kaur relocated internationally)' — no 'voluntary' field. "
            "Vector search for 'CSM Apex Financial HR records' retrieves csm_account_history "
            "and customer_list, which gives 'Preet Kaur' + 'departed' but NOT 'voluntary'. "
            "offboarding_records has no semantic overlap with this query string. "
            "PARTIAL: model identifies the person and departure; cannot confirm circumstances."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_05 (graph: Person node with employment_status + departure_type)",
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
            "founding_story.txt mentions both, but the model must identify the "
            "ambiguity and state BOTH outcomes precisely. "
            "'signed' (for the Phoenix Corp deal outcome) is the fact the baseline "
            "most consistently misses — it describes the partnership's status rather "
            "than its strategic rationale, which is what the model tends to focus on."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_08 (agent: entity disambiguation + multi-query strategy)",
    ),

    GoldenQuestion(
        id="Q19",
        type="csv_full_aggregation",
        question=(
            "What is Vertexia's total planned headcount across all departments "
            "as outlined in the 2023 budget allocation?"
        ),
        required_facts=["181"],   # sum of all 9 department headcounts: 42+58+12+14+38+6+3+5+3 = 181
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "budget_allocation_2023.csv has 9 department rows. "
            "Total headcount: Platform Eng 42 + Product Eng 58 + DataCraft 12 + "
            "Product 14 + Revenue 38 + Finance 6 + Legal 3 + People & Culture 5 + Executive 3 = 181. "
            "top-k=5 retrieves the 5 largest-budget departments (~164 headcount partial sum). "
            "The org_chart files mention ~430–480 total employees (a different metric) "
            "which the model may cite instead. Either way, '181' will not appear. "
            "partial_facts=[] so naming some departments does not earn PARTIAL."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: SUM headcount FROM budget_allocation_2023)",
    ),

    GoldenQuestion(
        id="Q20",
        type="csv_arithmetic_full_scan",
        question=(
            "What was Vertexia's total combined revenue across all product lines "
            "in Q3 2023 (July, August, and September combined)?"
        ),
        required_facts=["4,120"],   # $1,310K + $1,360K + $1,450K = $4,120,000
        partial_facts=[],
        disqualifiers=[],
        explanation=(
            "revenue_by_product_2023.csv has one row per month. "
            "Q3 2023 total revenue: July $1,310,000 + August $1,360,000 + September $1,450,000 = $4,120,000. "
            "top-k=5 from 11 rows returns a mix of months — model sees 3–4 months total, "
            "not the 3 consecutive Q3 months needed. Partial sums will be wrong. "
            "'4,120' matches '$4,120,000'; no partial quarterly sum will contain this string. "
            "partial_facts=[] so retrieving any revenue row does not earn PARTIAL."
        ),
        expected_outcome="FAIL",
        fixed_by_step="step_07 (structured query: WHERE month BETWEEN '2023-07' AND '2023-09' SUM total_revenue)",
    ),

    GoldenQuestion(
        id="Q21",
        type="multi_hop_implicit",
        question=(
            "If the analytics dashboard had an ingestion failure on August 29, 2023, "
            "which on-call Data Platform engineer would be responsible for the fix?"
        ),
        required_facts=["Priya Nair", "InsightLens"],
        partial_facts=["on-call", "data platform"],
        disqualifiers=["Kenji Ito", "Lin Wei", "James O'Brien", "Yuki Tanaka"],
        explanation=(
            "4-hop chain: 'analytics dashboard' → InsightLens (product name for the analytics UI) "
            "→ InsightLens depends on events_api (api_dependencies.csv) → events_api owned by "
            "Data Platform Team → on_call_schedule_aug2023.csv: week Aug 28–Sep 3 = Priya Nair (E010). "
            "PASS requires: (1) identifying 'analytics dashboard' = InsightLens by name, "
            "AND (2) naming Priya Nair as the on-call engineer. "
            "Baseline shortcut: model retrieves on_call_schedule directly from 'on-call August 29' "
            "semantic match → finds Priya Nair correctly but never identifies InsightLens by name. "
            "PARTIAL: model names Priya Nair but not InsightLens. "
            "Disqualifiers catch the common confusion with Kenji Ito (Aug 2023 outage incident lead "
            "who is prominent in founding_story and postmortem)."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_08 (agent: resolve 'analytics dashboard' to InsightLens via product catalog)",
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
        partial_facts=["Pulsar", "NexusFlow", "pipeline", "Phoenix Corp"],
        disqualifiers=[],
        explanation=(
            "Chain: Pulsar down → NexusFlow (critical dep on external_pulsar) → "
            "NexusFlow events_api fails → InsightLens (critical dep on events_api) fails → "
            "enterprise customers on NexusFlow or InsightLens lose service → "
            "QuantumBank ($3.2M ARR, NexusFlow+InsightLens+PulseConnect) is one such customer. "
            "'events_api' is in api_dependencies.csv; 'InsightLens' link requires that CSV. "
            "'QuantumBank' is only in customer_list.csv — not in any prose blast-radius docs. "
            "Baseline retrieves architecture/postmortem prose: gets NexusFlow + Phoenix Corp "
            "(mentioned in postmortem SLA context) but not api_dependencies.csv or customer_list.csv. "
            "PARTIAL: model gets InsightLens + events_api (architecture docs) + Phoenix Corp "
            "(postmortem) but not QuantumBank (customer_list.csv only)."
        ),
        expected_outcome="PARTIAL",
        fixed_by_step="step_05 (graph: product→dependency→customer edges)",
    ),

    # ── Q23–Q27: target steps 09–11 (step 12 adds operational hardening, not accuracy) ──

    GoldenQuestion(
        id="Q23",
        type="csm_multi_source_join",
        question=(
            "The CSM listed for Apex Financial in Vertexia's customer records is no longer "
            "with the company. According to all available records, who currently manages this "
            "account, when did they take over, and what is the account's current renewal risk?"
        ),
        required_facts=["Sam Rivera", "2023-07-01", "high"],
        partial_facts=["Apex Financial", "CSM", "Preet Kaur", "transition"],
        disqualifiers=[],
        explanation=(
            "3-source join: customer_list.csv (Preet Kaur as CSM) + "
            "csm_account_history.csv (Sam Rivera took over 2023-07-01) + "
            "customer_health_scores.csv (renewal_risk=high for Apex Financial). "
            "Step 08 retrieves customer_list.csv (shows Preet Kaur) but its single-turn "
            "CSV query lacks the sub-question strategy to also pull csm_account_history "
            "and health_scores in the same pass. "
            "Step 09 QueryAnalyst generates: 'current CSM for Apex Financial?', "
            "'account transition date?', 'renewal risk?' — three targeted sub-queries "
            "that surface all three files. required_facts=[Sam Rivera, 2023-07-01, high]."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_09 (multi-agent: QueryAnalyst sub-questions across 3 CSVs)",
    ),

    GoldenQuestion(
        id="Q24",
        type="technical_precision",
        question=(
            "Per the NexusFlow V2.1 post-mortem, what was the exact configuration key "
            "that was misconfigured, the incorrect value that was deployed, and the "
            "intended correct value?"
        ),
        required_facts=["max_connections", "10", "1000"],
        partial_facts=["rate limiter", "configuration", "ingestion", "outage"],
        disqualifiers=[],
        explanation=(
            "nexusflow_v21_postmortem.txt ROOT CAUSE section: "
            "rate_limiter.yaml had max_connections: 10 instead of max_connections: 1000. "
            "Steps 09 retrieves many near-duplicate postmortem chunks (root cause, timeline, "
            "lessons learned); the LLM paraphrases as 'rate limiter set too low' without "
            "citing the exact key 'max_connections' or the exact values 10 / 1000. "
            "Step 10 CrossEncoder re-ranks and selects precisely the ROOT CAUSE passage "
            "that contains the config key and both values; extractive compression keeps "
            "that sentence intact → LLM has no choice but to cite the exact values."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_10 (context engineering: CrossEncoder elevates exact ROOT CAUSE section)",
    ),

    GoldenQuestion(
        id="Q25",
        type="finance_exact_computation",
        question=(
            "What is the average annual contract value across all Vertexia mid-market "
            "customers? State the exact dollar figure."
        ),
        required_facts=["303,273"],
        partial_facts=["mid-market", "average", "303"],
        disqualifiers=["276,000", "330,000"],
        explanation=(
            "customer_list.csv: 11 mid-market customers, total ARR = $3,336,000, "
            "average = $3,336,000 / 11 = $303,272.73 → $303,273. "
            "Steps 09/10 with a generic synthesis prompt produce 'approximately $303K' "
            "or round to '$303,000' — both miss the required '303,273' string. "
            "Finance slice (step 11) uses force_csv=True + a system prompt that mandates "
            "comma-formatted exact numbers → LLM outputs '$303,273' → PASS. "
            "Disqualifiers catch wrong segment averages: $276K (enterprise) or $330K (SMB)."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_11 (Finance slice: force_csv + exact-number system prompt)",
    ),

    GoldenQuestion(
        id="Q26",
        type="hr_graph_skip_level",
        question=(
            "For each employee who departed Vertexia in 2023, who was their "
            "skip-level manager (their manager's manager)?"
        ),
        required_facts=["Marcus Webb", "Lisa Torres"],
        partial_facts=["Adrian Blake", "Preet Kaur", "departed", "skip"],
        disqualifiers=[],
        explanation=(
            "2023 departures: Adrian Blake (E029) and Preet Kaur (E030). "
            "Skip-level chain: Adrian Blake → Priya Nair (E010) → Marcus Webb (E009). "
            "Skip-level chain: Preet Kaur → Maya Sharma (E019) → Lisa Torres (E006). "
            "QueryAnalyst's graph heuristic triggers on keywords like 'reports to', "
            "'manager', 'org chart' — but 'skip-level' is not in that vocabulary, "
            "so steps 09/10 never invoke GraphNavigator → miss Marcus Webb and Lisa Torres. "
            "HR slice (step 11) sets force_graph=True, bypassing QueryAnalyst — "
            "GraphNavigator always runs and resolves the 2-hop chain for each departure."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_11 (HR slice: force_graph overrides QueryAnalyst heuristic)",
    ),

    GoldenQuestion(
        id="Q27",
        type="engineering_rfc_precision",
        question=(
            "According to RFC-001, what is the author's exact job title and the name "
            "of their direct manager?"
        ),
        required_facts=["Lead, Data Platform Team", "Marcus Webb"],
        partial_facts=["Priya Nair", "RFC", "event schema", "Platform Engineering"],
        disqualifiers=[],
        explanation=(
            "rfc_001_event_schema.md: author = Priya Nair. "
            "employee_directory.csv E010: role = 'Lead, Data Platform Team', manager_id = E009. "
            "employee_directory.csv E009: Marcus Webb. "
            "QueryAnalyst may not trigger graph for an RFC authorship question "
            "(no 'reports to' / 'manager' / 'org chart' keywords in the question) → "
            "steps 09/10 retrieve the RFC document (Priya Nair ✓) but skip GraphNavigator "
            "→ miss the exact role title and Marcus Webb. "
            "Engineering slice (step 11) sets force_graph=True → GraphNavigator resolves "
            "Priya Nair's node → returns exact title 'Lead, Data Platform Team' + manager Marcus Webb."
        ),
        expected_outcome="PASS",
        fixed_by_step="step_11 (Engineering slice: force_graph + precision system prompt)",
    ),
]
