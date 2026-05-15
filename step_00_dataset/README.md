# Step 00 — Foundation Dataset

## Goal

Create a rich, realistic synthetic corpus for **Vertexia Inc.**, a fictional B2B SaaS company. This corpus must be complex enough to expose the failure modes of every retrieval strategy we will build in Steps 01–12.

**The dataset is the experiment.** If the data is too clean, we will build a system that "works" but teaches us nothing. The data must have the same messiness, ambiguity, and cross-document complexity as real company data.

---

## The Company: Vertexia Inc.

**Type**: B2B SaaS  
**Size**: ~500 employees (as of end 2023)  
**Founded**: 2019, San Francisco  
**Stage**: Series C ($45M, October 2023)  
**Headquarters**: San Francisco + remote engineering hubs in Austin and Bangalore  

### Products

| Product | Codename | Description | GA Date |
|---|---|---|---|
| NexusFlow | FLOW | Customer data pipeline & ETL platform | March 2020 |
| InsightLens | LENS | Business analytics & BI dashboard | September 2021 |
| PulseConnect | PULSE | CRM & customer engagement platform | April 2022 |

### Key Events Timeline (Built Into Documents)

| Date | Event | Why It's Tricky |
|---|---|---|
| 2019-03 | Company founded by Arjun Mehta & Diana Volkov | Founding context scattered across docs |
| 2021-07 | Series B ($18M) | Financial docs reference this |
| 2022-01 | Acquisition of DataCraft (small ETL startup) | DataCraft docs exist separately, people have dual history |
| 2022-06 | "Project Phoenix" enterprise deal with Phoenix Corp | "Phoenix" is also an internal engineering migration — **collision** |
| 2023-04 | Q1 restructuring: VP Eng → CTO promotion for Sarah Chen | Org docs split before/after — **temporal ambiguity** |
| 2023-08-14 | NexusFlow major outage (v2.1 deployment bug) | Postmortem references multiple teams — **multi-hop chain** |
| 2023-10 | Series C ($45M, led by Sequoia) | Multiple docs reference this |
| 2024-01 | New OKR cycle | Forward-looking planning docs |

### Org Structure (Post-Restructuring, Q2 2023 onward)

```
CEO: Arjun Mehta
CTO: Sarah Chen (formerly VP Engineering)
    ├── Platform Engineering (Lead: Marcus Webb)
    │   ├── Infrastructure Team (Marcus Webb, direct)
    │   ├── Data Platform Team (Lead: Priya Nair)
    │   └── Security Team (Lead: Daniel Osei)
    ├── Product Engineering (Lead: Tomás García)
    │   ├── NexusFlow Team (Lead: Yuki Tanaka)
    │   ├── InsightLens Team (Lead: Aisha Johnson)
    │   └── PulseConnect Team (Lead: Raj Patel)
    └── DataCraft Integration Team (Lead: Felix Wagner, DataCraft founder)

CPO: Nadia Kim
    ├── Product Management
    └── Design

CFO: Robert Okafor
    ├── Finance
    └── Legal (VP Legal: Carmen Reyes)

CRO: Lisa Torres
    ├── Sales (VP Sales: Ben Carter)
    ├── Customer Success (VP CS: Maya Sharma)
    └── Marketing

VP People & Culture: Zara Ahmed
```

---

## The "Traps" — Why This Data Will Break Naive RAG

### Trap 1: Temporal Ambiguity (Entity state changes over time)
- Sarah Chen is "VP Engineering" in all pre-April 2023 documents
- Sarah Chen is "CTO" in all post-April 2023 documents
- No single document says "Sarah Chen WAS VP Engineering and IS NOW CTO"
- **Naive RAG will return documents from both eras and give inconsistent answers**
- **Multi-hop/Graph needed**: Person → role change event → current role

### Trap 2: Name Collision — "Phoenix"
- Engineering: "Project Phoenix" = internal migration of NexusFlow from Python 2 to Python 3 (completed June 2022)
- Sales: "Phoenix deal" or "Phoenix Corp contract" = $2.4M ARR enterprise customer signed June 2022
- Both happen to be the same month — *purely coincidental in-universe*
- **Naive RAG returns both** when asked about "Phoenix"
- **Requires disambiguation**: context around "Phoenix" determines which one

### Trap 3: Multi-hop Reasoning Chain (Outage Attribution)
- Question: "Who was the on-call engineer responsible during the August 2023 outage?"
- Postmortem says: "The v2.1 deployment caused a cascade in the data ingestion service"
- Separate runbook says: "Data ingestion service is owned by the Data Platform Team"
- Separate HR schedule says: "On-call rotation for Data Platform week of Aug 14: Kenji Ito"
- **No single document answers this question**
- **Vector RAG retrieves postmortem but misses the on-call schedule**

### Trap 4: Contradictory Numbers (Two Sources, One Truth)
- Finance report: "Q3 2023 revenue: $4.12M"
- Sales dashboard export: "Q3 2023 closed-won: $4.2M"
- **Both are technically correct** (finance uses GAAP recognition; sales uses booking date)
- **Naive RAG doesn't know which to trust** — will confidently return one or the other

### Trap 5: Acquisition Context Bleed
- DataCraft was acquired Jan 2022. Felix Wagner joined as DataCraft Integration Team lead.
- Pre-acquisition DataCraft docs exist (product specs, architecture) that reference their old tech stack (Kafka-based, vs. Vertexia's Pulsar-based)
- **Question: "What message queue does the data pipeline use?"** → DataCraft docs say Kafka, Vertexia docs say Pulsar
- After integration (mid-2023), they migrated DataCraft to Pulsar
- The answer is: "Pulsar (migrated from Kafka in Q2 2023)" — but scattered across 3 docs

### Trap 6: Implicit Relationship (Contract → SLA → Engineering Requirement)
- Phoenix Corp contract (Legal): "99.99% uptime SLA for data pipeline services"
- NexusFlow architecture doc (Engineering): "Target availability: 99.9%" (written before Phoenix deal)
- **Question: "What uptime does the Phoenix Corp contract require?"** 
- **Then: "Does the current NexusFlow architecture meet that SLA?"**
- These are in completely separate documents with no explicit link
- **Graph needed**: Contract → Customer (Phoenix Corp) → Product (NexusFlow) → Architecture spec

### Trap 7: Dense Numerical Tables vs. Prose
- Q3 2023 revenue breakdown is in *both* `finance/q3_2023_revenue_report.csv` (precise) and `executive/q3_allhands_notes.txt` (rounded, prose)
- "What was NexusFlow's Q3 2023 contribution to total revenue?" — CSV gives exact, prose gives approximate
- **Chunking**: The CSV row with the answer is a single short row — may get lost in vector search

### Trap 8: Renamed Departments
- Before April 2023: "Engineering" department with VP Engineering Sarah Chen
- After April 2023: Split into "Platform Engineering" and "Product Engineering"
- HR records updated; old job descriptions not updated
- **Question: "Who is in the Engineering department?"** — old docs give one answer, new org chart gives different answer

### Trap 9: Stale Job Descriptions
- 3 people left the company in 2023 (documented in offboarding records)
- Their names still appear in some internal wikis and architecture docs
- **Question: "Who can I contact about X system?"** — the doc says Person A, but they left in August 2023

### Trap 10: Cross-product Dependency Hidden in a Table
- InsightLens depends on NexusFlow's `events_api` endpoint (one line in a CSV of API dependencies)
- **Question: "Was InsightLens affected by the August 2023 outage?"** — Yes, because of this dependency
- But no prose document ever says "InsightLens was affected by the outage"
- **The link exists only in structured data**

---

## Document Inventory

### Format Distribution
| Format | Count | Why Included |
|---|---|---|
| `.txt` | 12 | Meeting notes, emails, wikis (unstructured, variable length) |
| `.md` | 8 | Architecture docs, runbooks, READMEs (semi-structured) |
| `.csv` | 10 | Employee data, sales pipeline, revenue, bug tracker, API deps |
| `.json` | 4 | API specs, system config, OKR tracking |
| `.py` | 2 | Code files with docstrings (engineering context) |

**Total**: ~36 core documents + generated variations = 50+ artifacts

---

### Engineering Documents (`engineering/`)

| File | Format | Content | Traps |
|---|---|---|---|
| `nexusflow_architecture.md` | MD | System design, components, tech stack (Pulsar, not Kafka) | Trap 5, Trap 6 |
| `nexusflow_v21_postmortem.txt` | TXT | Aug 14 2023 outage root cause analysis, timeline, action items | Trap 3, core document |
| `data_platform_runbook.md` | MD | On-call procedures, service ownership, escalation | Trap 3 (ownership info) |
| `project_phoenix_migration.md` | MD | Python 2→3 migration plan, timeline, completion note | Trap 2 |
| `api_dependencies.csv` | CSV | Service → dependency mapping, including InsightLens→NexusFlow | Trap 10 |
| `on_call_schedule_aug2023.csv` | CSV | Week-by-week on-call rotation Aug 2023 | Trap 3 |
| `datacraft_original_architecture.md` | MD | DataCraft's Kafka-based architecture (pre-acquisition) | Trap 5 |
| `datacraft_migration_complete.txt` | TXT | Migration completion notice, Pulsar cutover June 2023 | Trap 5 |
| `security_audit_2023.txt` | TXT | Annual security review findings, recommendations | Background |
| `engineering_rfcs/rfc_001_event_schema.md` | MD | Event schema standardization RFC | Background, cross-product |

### Product Documents (`product/`)

| File | Format | Content | Traps |
|---|---|---|---|
| `nexusflow_prd_v2.md` | MD | Product requirements for NexusFlow 2.0 | Trap 6 (availability req) |
| `insightlens_prd_v1.md` | MD | InsightLens product requirements | Trap 10 |
| `pulseconnect_roadmap_2024.txt` | TXT | PulseConnect planned features, owners | Forward-looking |
| `user_research_q3_2023.txt` | TXT | Customer interview synthesis, pain points | Background |
| `product_okrs_2023.json` | JSON | OKR tree for product team 2023 | Trap 7 (targets vs actuals) |

### HR Documents (`hr/`)

| File | Format | Content | Traps |
|---|---|---|---|
| `employee_directory.csv` | CSV | Full employee list: id, name, dept, role, manager, start, status | Trap 1, 8, 9 |
| `org_chart_q1_2023.txt` | TXT | Org chart snapshot before restructuring | Trap 1, Trap 8 |
| `org_chart_q3_2023.txt` | TXT | Org chart snapshot after restructuring | Trap 1, Trap 8 |
| `onboarding_handbook.txt` | TXT | General onboarding guide, policies, tools | Background |
| `offboarding_records_2023.csv` | CSV | People who left in 2023: name, last day, reason | Trap 9 |
| `promotion_announcements_2023.txt` | TXT | All promotions, includes Sarah Chen CTO announcement | Trap 1 (source of truth) |
| `datacraft_employee_integration.txt` | TXT | How DataCraft employees were onboarded post-acquisition | Trap 5 |

### Finance Documents (`finance/`)

| File | Format | Content | Traps |
|---|---|---|---|
| `revenue_by_product_2023.csv` | CSV | Monthly revenue per product, exact GAAP figures | Trap 4, Trap 7 |
| `q3_2023_finance_report.txt` | TXT | Quarterly finance narrative: $4.12M revenue | Trap 4 |
| `budget_allocation_2023.csv` | CSV | Budget per department, headcount costs | Background |
| `vendor_contracts_summary.csv` | CSV | Vendor, contract value, renewal date, owner | Background |
| `series_b_investor_update.txt` | TXT | July 2021 investor update letter | Historical context |
| `series_c_announcement.txt` | TXT | Oct 2023 Series C announcement | Background |

### Sales Documents (`sales/`)

| File | Format | Content | Traps |
|---|---|---|---|
| `deal_pipeline_q3_2023.csv` | CSV | Open and closed deals, ARR, stage | Trap 4 |
| `phoenix_corp_deal_summary.txt` | TXT | Phoenix Corp enterprise deal memo, $2.4M ARR | Trap 2, Trap 6 |
| `customer_list.csv` | CSV | Customer name, industry, ARR, product, CSM | Background |
| `sales_playbook_2023.txt` | TXT | Sales methodology, ICP, objection handling | Background |
| `q3_closed_won_report.txt` | TXT | "Q3 closed-won: $4.2M" (booking-date accounting) | Trap 4 |

### Legal Documents (`legal/`)

| File | Format | Content | Traps |
|---|---|---|---|
| `phoenix_corp_msa.txt` | TXT | Master Service Agreement with Phoenix Corp, incl. 99.99% SLA | Trap 6 |
| `data_processing_agreement_template.txt` | TXT | GDPR/CCPA DPA template | Background |
| `datacraft_acquisition_summary.txt` | TXT | Acquisition terms, IP transfer, employee transition | Trap 5 |
| `ip_policy.txt` | TXT | Company IP policy, assignment, open-source use | Background |

### Executive Documents (`executive/`)

| File | Format | Content | Traps |
|---|---|---|---|
| `q3_2023_allhands_notes.txt` | TXT | All-hands meeting notes: "~$4.2M revenue in Q3" (rounded) | Trap 4, Trap 7 |
| `board_meeting_april_2023.txt` | TXT | Board meeting covering restructuring, Sarah Chen CTO decision | Trap 1 |
| `company_strategy_2024.txt` | TXT | 2024 strategic plan, priorities, targets | Forward-looking |
| `founding_story.txt` | TXT | Company history, Arjun + Diana founding, early pivots | Background |

---

## Generation Approach

Documents are generated with a Python script (`implementation/generate_dataset.py`) using Claude API.

**Generation principles**:
1. Each document is internally consistent with the company timeline
2. Documents reference each other implicitly (never hyperlinked, just named)
3. All names, numbers, and dates are consistent across documents *except where a Trap is designed*
4. Documents use realistic corporate language, jargon, abbreviations
5. Length variation: short memos (~200 words) to long reports (~2000 words)
6. CSVs have realistic data volume: employee directory has all ~500 employees (abbreviated), deals have 50+ rows

---

## The 10 Golden Questions (Baseline Test Set)

These are the questions we will use to measure every system we build. The correct answer for each is documented below.

```
Q1 [Simple Lookup]:    "What is Vertexia's data retention policy for customer data?"
    Expected: 90 days hot, 1 year cold storage (from onboarding handbook)
    
Q2 [Comparative]:      "How did Q3 2023 NexusFlow revenue compare to Q2 2023?"
    Expected: NexusFlow Q2 $1.4M, Q3 $1.6M — 14.3% QoQ growth (from revenue CSV)
    
Q3 [Multi-hop]:        "Who was the on-call engineer for the data platform during the August 2023 outage?"
    Expected: Kenji Ito (postmortem → service ownership → on-call schedule)
    
Q4 [Temporal]:         "What was Sarah Chen's title in January 2023?"
    Expected: VP Engineering (became CTO in April 2023)
    
Q5 [Entity Disambiguation]: "What is Project Phoenix?"
    Expected: Two things — (a) internal Python 2→3 migration (eng), (b) Phoenix Corp enterprise deal (sales). Must distinguish.
    
Q6 [Implicit Link]:    "Does the Phoenix Corp SLA requirement exceed NexusFlow's documented availability target?"
    Expected: Yes — Phoenix Corp requires 99.99%, NexusFlow architecture doc targets 99.9%
    
Q7 [Contradictory Data]: "What was Q3 2023 total revenue?"
    Expected: $4.12M per GAAP (finance), $4.2M per bookings (sales). Both correct, different accounting.
    
Q8 [Aggregation]:      "How many employees joined Vertexia through the DataCraft acquisition?"
    Expected: 12 employees (from datacraft_employee_integration.txt + employee_directory.csv)
    
Q9 [Stale Reference]:  "Who should I contact about the InsightLens data pipeline integration?"
    Expected: The doc says Marcus Webb, but Marcus is still at the company (not stale); distinguish from trap where person left
    
Q10 [Cross-format]:    "Was InsightLens affected by the August 2023 NexusFlow outage? Explain why."
    Expected: Yes, because InsightLens depends on NexusFlow's events_api (only in api_dependencies.csv)
```

---

## Done Criteria

Step 00 is complete when:

- [ ] All documents in the inventory exist with realistic content
- [ ] Every Trap is verifiable (we can demonstrate each one exists in the data)
- [ ] The 10 golden questions have documented correct answers
- [ ] A human can browse the `company_data/` folder and understand the company well enough to answer the questions manually
- [ ] Document metadata catalog (`results/document_catalog.csv`) lists: filename, format, department, date_range, word_count, traps_encoded
- [ ] Generation script is committed and reproducible

---

## Next Step

Once Step 00 is complete: → Step 01 (Baseline Vector RAG) uses this exact corpus. The first real test: how many of the 10 golden questions can naive RAG answer correctly?
