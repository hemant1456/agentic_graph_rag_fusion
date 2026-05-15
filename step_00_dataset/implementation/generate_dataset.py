"""
Dataset generation script for Vertexia Inc. synthetic corpus.

Generates all company documents using Gemini as primary LLM (free tier),
falling back to Anthropic Claude if Gemini fails.

Usage:
    uv run python step_00_dataset/implementation/generate_dataset.py

Requirements (in .env):
    GOOGLE_API_KEY   — primary (Gemini, free)
    ANTHROPIC_API_KEY — fallback (Claude)
"""

import anthropic
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

# ── Shared company context ─────────────────────────────────────────────────
# This object is injected into every document prompt to ensure consistency.

COMPANY_CONTEXT = """
COMPANY FACTS (use these exact names, dates, and numbers — never deviate):

Company: Vertexia Inc.
Founded: March 2019, San Francisco
Stage: Series C ($45M, October 2023, led by Sequoia)
Employees: ~500 as of end 2023
HQ: San Francisco + Austin + Bangalore

PRODUCTS:
- NexusFlow (internal codename FLOW): Customer data pipeline & ETL. GA: March 2020.
- InsightLens (internal codename LENS): Business analytics & BI. GA: September 2021.
- PulseConnect (internal codename PULSE): CRM & customer engagement. GA: April 2022.

KEY PEOPLE (use these names exactly, no variations):
- Arjun Mehta — CEO (co-founder)
- Diana Volkov — Co-founder, left to start another company in late 2021
- Sarah Chen — CTO (as of April 2023; was VP Engineering from founding through March 2023)
- Marcus Webb — Lead, Platform Engineering (reports to Sarah Chen)
- Priya Nair — Lead, Data Platform Team (reports to Marcus Webb)
- Daniel Osei — Lead, Security Team (reports to Marcus Webb)
- Tomás García — Lead, Product Engineering (reports to Sarah Chen)
- Yuki Tanaka — Lead, NexusFlow Team (reports to Tomás García)
- Aisha Johnson — Lead, InsightLens Team (reports to Tomás García)
- Raj Patel — Lead, PulseConnect Team (reports to Tomás García)
- Felix Wagner — DataCraft founder, now leads DataCraft Integration Team
- Kenji Ito — Senior Engineer, Data Platform Team (on-call Aug 14 2023)
- Nadia Kim — CPO
- Robert Okafor — CFO
- Carmen Reyes — VP Legal
- Lisa Torres — CRO
- Ben Carter — VP Sales
- Maya Sharma — VP Customer Success
- Zara Ahmed — VP People & Culture

KEY EVENTS (dates are EXACT):
- 2022-01-15: DataCraft acquisition closed ($8M, 12 employees join)
- 2022-06-01: Phoenix Corp MSA signed ($2.4M ARR), 99.99% uptime SLA
- 2022-06-30: Engineering "Project Phoenix" (Python 2→3 migration) completed
- 2023-04-10: Sarah Chen promoted from VP Engineering to CTO; Engineering split into Platform Engineering and Product Engineering
- 2023-08-14: NexusFlow v2.1 major outage, duration 4h 23min, root cause: misconfigured rate limiter in data ingestion service
- 2023-10-05: Series C announced ($45M)
- 2021-07-01: Series B ($18M)

FINANCIALS (exact):
- Q1 2023 total revenue: $3.8M (NexusFlow $1.2M, InsightLens $1.4M, PulseConnect $1.2M)
- Q2 2023 total revenue: $4.0M (NexusFlow $1.4M, InsightLens $1.4M, PulseConnect $1.2M)
- Q3 2023 total revenue: $4.12M GAAP / $4.2M bookings (NexusFlow $1.6M, InsightLens $1.5M, PulseConnect $1.02M) — GAAP vs bookings difference is intentional
- ARR at end Q3 2023: $16.5M

TECHNOLOGY STACK:
- Message queue: Apache Pulsar (migrated from Kafka after DataCraft acquisition completed Q2 2023)
- DataCraft original stack used Apache Kafka (pre-migration)
- Cloud: AWS primarily, some GCP for ML workloads
- Languages: Python (backend), TypeScript (frontend), Go (data pipeline hot path)

DATA POLICIES:
- Customer data retention: 90 days hot storage, 1 year cold storage (S3 Glacier)
- Logs retention: 30 days
- Backup frequency: daily snapshots, 30-day retention

CRITICAL TRAP FACTS (embed these naturally in the right documents):
- "Project Phoenix" in engineering = Python 2→3 migration (completed June 2022)
- "Phoenix" or "Phoenix Corp" in sales/legal = enterprise customer (signed June 2022, $2.4M ARR)
- Sarah Chen's title depends on date: VP Engineering (≤ March 2023), CTO (≥ April 2023)
- InsightLens depends on NexusFlow events_api (so it WAS affected by Aug 2023 outage)
- The on-call engineer for Data Platform on Aug 14 2023 was Kenji Ito
- Q3 revenue is $4.12M GAAP (finance uses this) vs $4.2M bookings (sales uses this)
- 12 employees joined from DataCraft acquisition
- NexusFlow architecture doc targets 99.9% uptime; Phoenix Corp SLA requires 99.99%
"""


@dataclass
class DocumentSpec:
    path: str          # relative to company_data/
    format: str        # txt, md, csv, json
    description: str   # what to generate
    extra_context: str = ""  # additional instructions for this specific doc


# ── Document specifications ─────────────────────────────────────────────────

DOCUMENTS: list[DocumentSpec] = [
    # ── ENGINEERING ──────────────────────────────────────────────────────────
    DocumentSpec(
        path="engineering/nexusflow_architecture.md",
        format="md",
        description="NexusFlow system architecture document. Cover: system overview, components (ingestion service, transformation engine, delivery service), tech stack (Python, Go for hot path, Apache Pulsar for message queue, PostgreSQL metadata store, AWS infrastructure), scalability design, availability target of 99.9% (three nines — NOT four nines). Written by Priya Nair, dated February 2023. Professional engineering doc style with sections, diagrams described in text, numbered components. About 800 words.",
        extra_context="IMPORTANT: The documented availability target is 99.9% (three nines). Do NOT write 99.99%. This creates a deliberate gap with the Phoenix Corp SLA requirement."
    ),
    DocumentSpec(
        path="engineering/nexusflow_v21_postmortem.txt",
        format="txt",
        description="Post-mortem report for the NexusFlow v2.1 outage on August 14 2023. Timeline: deployment at 09:15 PST, first alerts 09:23 PST, incident declared 09:31 PST, service restored 13:38 PST (4h 23min). Root cause: a misconfigured rate limiter in the data ingestion service that was introduced in the v2.1 release. The rate limiter incorrectly set max_connections=10 instead of max_connections=1000. The data ingestion service is owned by the Data Platform Team. Do NOT mention who the on-call engineer was — that is in a separate runbook/schedule. Action items include: deployment checklist for config validation, automated canary testing, and blameless culture reminder. About 600 words. Written by Yuki Tanaka.",
        extra_context="Do NOT name the on-call engineer in this document. Leave a reference to 'the on-call engineer was paged' without naming them."
    ),
    DocumentSpec(
        path="engineering/data_platform_runbook.md",
        format="md",
        description="On-call runbook for the Data Platform Team. Sections: service ownership (data ingestion service, transformation engine, events API — all owned by Data Platform Team under Priya Nair), escalation procedures, common incident types, recovery procedures, on-call rotation policy (weekly rotation, schedule maintained in on_call_schedule_aug2023.csv). Professional runbook format. About 500 words.",
        extra_context="Mention that the on-call schedule is in a separate CSV file. This creates the multi-hop chain."
    ),
    DocumentSpec(
        path="engineering/project_phoenix_migration.md",
        format="md",
        description="Engineering project document for 'Project Phoenix' — the internal initiative to migrate NexusFlow codebase from Python 2 to Python 3. Background: Python 2 EOL in 2020 necessitated migration. Project kicked off January 2022, completed June 30 2022. Led by Yuki Tanaka. Challenges: 187,000 lines of Python code, 43 external libraries to update, 6-month timeline. Results: migration complete, all tests passing, 8% performance improvement post-migration. Status: COMPLETED. About 500 words.",
        extra_context="This is ENGINEERING's 'Project Phoenix'. Do NOT reference sales or Phoenix Corp. The name collision with the enterprise deal is the trap."
    ),
    DocumentSpec(
        path="engineering/datacraft_original_architecture.md",
        format="md",
        description="DataCraft's original system architecture document (written before acquisition, ~late 2021). Describes DataCraft's ETL pipeline: Kafka-based event streaming, Spark for batch processing, custom Python ETL framework called DataWeave. Written in DataCraft's company voice (smaller, scrappier startup). Author: Felix Wagner. Note at top: 'ARCHIVED — See datacraft_migration_complete.txt for current state'. About 400 words.",
        extra_context="Use Apache Kafka as the message queue. This is the pre-migration architecture that conflicts with Vertexia's Pulsar stack."
    ),
    DocumentSpec(
        path="engineering/datacraft_migration_complete.txt",
        format="txt",
        description="Migration completion notice sent by Felix Wagner to the engineering team in June 2023. Subject: DataCraft Integration Migration Complete. Announces that the DataCraft data pipeline has been fully migrated from Kafka to Apache Pulsar as part of the Vertexia platform standardization. Migration took 18 months. All DataCraft customers have been migrated. The old DataCraft architecture doc is now archived. Key note: the DataWeave framework has been deprecated and replaced with NexusFlow's transformation engine. Brief, email-style format. About 250 words.",
    ),
    DocumentSpec(
        path="engineering/security_audit_2023.txt",
        format="txt",
        description="Annual security audit report for Vertexia Inc., conducted in September 2023. Covers: authentication systems (using Okta SSO), authorization model (RBAC), data encryption (at-rest AES-256, in-transit TLS 1.3), penetration test results (3 medium findings, 0 critical), SOC 2 Type II compliance status (audit in progress), recommendations. Audited by external firm 'SecureAudit Partners'. Professional security report format. About 700 words.",
    ),
    DocumentSpec(
        path="engineering/rfc_001_event_schema.md",
        format="md",
        description="RFC-001: Standardized Event Schema for Cross-Product Analytics. Author: Priya Nair. Problem: NexusFlow, InsightLens, and PulseConnect each have slightly different event schemas making cross-product analytics impossible. Proposal: a unified event schema with mandatory fields (event_id, timestamp, product_id, user_id, event_type, properties) and optional extension fields. Decision: ACCEPTED, to be implemented in Q4 2023. This RFC establishes that InsightLens consumes NexusFlow's events_api. About 500 words.",
        extra_context="Explicitly mention that InsightLens will consume NexusFlow's events_api for cross-product analytics. This is background context for the InsightLens-NexusFlow dependency."
    ),

    # ── PRODUCT ───────────────────────────────────────────────────────────────
    DocumentSpec(
        path="product/nexusflow_prd_v2.md",
        format="md",
        description="Product Requirements Document for NexusFlow 2.0. Author: Nadia Kim. Contains: problem statement, target users (data engineers at mid-market companies), key features (real-time streaming, 200+ connectors, visual pipeline builder), success metrics, non-functional requirements. The availability requirement section states: 'Platform must achieve 99.9% uptime per our standard SLA. Enterprise customers with custom SLAs may require higher — see Legal for specific contract terms.' About 700 words.",
        extra_context="State 99.9% as the standard availability requirement, and note that enterprise contracts may have custom SLAs. This is key for Trap 6."
    ),
    DocumentSpec(
        path="product/insightlens_prd_v1.md",
        format="md",
        description="Product Requirements Document for InsightLens v1.0. Author: Nadia Kim. Contains: vision (make every business decision data-driven), user personas, core features (drag-and-drop dashboard builder, 50+ chart types, scheduled reports, data connectors). Mentions that InsightLens connects to NexusFlow's data pipeline as a primary data source via the events_api. Success metrics include DAU, dashboard creation rate, time-to-insight. About 600 words.",
    ),
    DocumentSpec(
        path="product/pulseconnect_roadmap_2024.txt",
        format="txt",
        description="PulseConnect 2024 Product Roadmap. Author: Raj Patel (product lead). Q1: AI-powered lead scoring, email sequence automation. Q2: Native Slack integration, customer health scoring. Q3: Revenue intelligence module. Q4: Mobile app v2. Each item has owner, rough sizing (S/M/L/XL), and dependencies. Conversational planning document style. About 500 words.",
    ),
    DocumentSpec(
        path="product/user_research_q3_2023.txt",
        format="txt",
        description="User Research Synthesis: Q3 2023 Customer Interviews. Conducted by Nadia Kim's team. 12 customer interviews across NexusFlow, InsightLens, PulseConnect. Top pain points: (1) NexusFlow: setup complexity, (2) InsightLens: slow dashboard loads with large datasets, (3) PulseConnect: missing mobile app. Key themes, representative quotes, recommendations. About 600 words.",
    ),

    # ── HR ────────────────────────────────────────────────────────────────────
    DocumentSpec(
        path="hr/org_chart_q1_2023.txt",
        format="txt",
        description="Vertexia Inc. Org Chart — Q1 2023 (snapshot as of January 2023). Text-format org chart. Structure: CEO Arjun Mehta → CTO level does NOT exist yet → VP Engineering Sarah Chen → single Engineering department (no Platform/Product split yet). Include all key people. Note at top: 'Org Chart — Q1 2023'. About 300 words.",
        extra_context="Sarah Chen's title is VP Engineering here. No CTO role. Engineering is one department. This is the pre-restructuring org chart."
    ),
    DocumentSpec(
        path="hr/org_chart_q3_2023.txt",
        format="txt",
        description="Vertexia Inc. Org Chart — Q3 2023 (snapshot as of July 2023, post-restructuring). Text-format org chart. Structure reflects the April 2023 restructuring: CEO → CTO Sarah Chen → Platform Engineering (Marcus Webb) and Product Engineering (Tomás García) as separate orgs + DataCraft Integration Team (Felix Wagner). Also include CPO Nadia Kim, CFO Robert Okafor (with VP Legal Carmen Reyes), CRO Lisa Torres (with VP Sales Ben Carter, VP CS Maya Sharma), VP People Zara Ahmed. About 400 words.",
        extra_context="Sarah Chen is now CTO. Engineering is now split. This is post-restructuring."
    ),
    DocumentSpec(
        path="hr/onboarding_handbook.txt",
        format="txt",
        description="Vertexia Inc. Employee Onboarding Handbook. Sections: Welcome & company values (Move Fast, Build Trust, Own It), first week checklist, tools setup (Slack, Notion, GitHub, Okta SSO, 1Password), data handling policies (customer data retention: 90 days hot, 1 year cold storage on S3 Glacier; logs: 30 days), expense policy, PTO policy (unlimited with 2-week minimum recommendation), security requirements. Warm, professional tone. About 800 words.",
        extra_context="Include the exact data retention policy: 90 days hot storage, 1 year cold storage (S3 Glacier). This is the answer to Golden Question 1."
    ),
    DocumentSpec(
        path="hr/promotion_announcements_2023.txt",
        format="txt",
        description="A collection of promotion announcements sent company-wide in 2023 (via email/Slack). Include: (1) April 10 2023: Sarah Chen promoted to CTO; Engineering restructured into Platform Engineering and Product Engineering; Tomás García elevated to lead Product Engineering; (2) June 2023: Aisha Johnson to Senior Manager; (3) August 2023: Raj Patel to Staff PM; (4) other minor promotions. Email announcement format for each. About 500 words.",
        extra_context="The April 10 2023 announcement must clearly say Sarah Chen is promoted FROM VP Engineering TO CTO. This is the source-of-truth document for Trap 1."
    ),
    DocumentSpec(
        path="hr/datacraft_employee_integration.txt",
        format="txt",
        description="Internal memo from Zara Ahmed (VP People) about the DataCraft acquisition employee integration. Dated February 2022. Lists the 12 DataCraft employees who joined, their roles, their Vertexia org placement. Felix Wagner becomes DataCraft Integration Team lead, reports to Sarah Chen (VP Engineering at the time). Covers: benefits enrollment deadline, laptop replacement, Okta account setup. Professional HR memo. About 400 words.",
        extra_context="State that exactly 12 employees joined from DataCraft. This is the answer to Golden Question 8."
    ),

    # ── EXECUTIVE ─────────────────────────────────────────────────────────────
    DocumentSpec(
        path="executive/q3_2023_allhands_notes.txt",
        format="txt",
        description="All-hands meeting notes from October 2023 (after Q3 close). Presenter: Arjun Mehta. Agenda: Q3 results, Series C announcement, Q4 priorities, team shout-outs. Revenue section: Arjun says 'We hit approximately $4.2M in revenue for Q3' (using bookings/rounded figure). Also covers headcount growth, product milestones. Notes are written by an EA, include audience questions. About 600 words.",
        extra_context="Use '$4.2M' — the bookings/rounded number. Finance report uses $4.12M GAAP. This is Trap 4 and Trap 7."
    ),
    DocumentSpec(
        path="executive/board_meeting_april_2023.txt",
        format="txt",
        description="Board meeting minutes from April 3 2023. Agenda: Q1 2023 results review, org restructuring proposal, Sarah Chen CTO appointment, Series C preparation. The board unanimously approves the restructuring. Minutes note: 'The board approved the appointment of Sarah Chen as Chief Technology Officer, effective April 10 2023. The Engineering organization will be restructured into Platform Engineering and Product Engineering.' Formal board minutes format. About 500 words.",
    ),
    DocumentSpec(
        path="executive/company_strategy_2024.txt",
        format="txt",
        description="Vertexia 2024 Strategic Plan. Author: Arjun Mehta + leadership team. Priorities: (1) Expand enterprise segment — target 20 new enterprise logos, (2) International expansion — UK and Germany, (3) AI-powered features across all products, (4) Platform unification — complete DataCraft integration. OKRs for each priority. Tone: ambitious, forward-looking. About 700 words.",
    ),
    DocumentSpec(
        path="executive/founding_story.txt",
        format="txt",
        description="The Vertexia founding story. Written by Arjun Mehta for the company Notion wiki. How Arjun and Diana Volkov met at Stanford, the problem they saw (data pipelines were too hard for non-engineers), early pivots, first customers, Diana's departure in late 2021 to pursue a separate startup (amicable, she remains a small shareholder), key milestones. Personal, narrative tone. About 600 words.",
    ),

    # ── SALES ─────────────────────────────────────────────────────────────────
    DocumentSpec(
        path="sales/phoenix_corp_deal_summary.txt",
        format="txt",
        description="Deal summary memo for the Phoenix Corp enterprise deal. Written by Ben Carter (VP Sales). Phoenix Corp is a Fortune 500 financial services company. Signed MSA on June 1 2022. ARR: $2.4M (NexusFlow enterprise tier + InsightLens). 3-year contract. Key requirements: 99.99% uptime SLA (higher than standard), dedicated CSM (Maya Sharma), quarterly business reviews, on-premise deployment option in year 2. This is Vertexia's largest deal to date. Professional deal memo. About 400 words.",
        extra_context="This is SALES's 'Phoenix' — Phoenix Corp the customer. Do NOT confuse with engineering's Project Phoenix migration. Clearly state 99.99% uptime SLA requirement."
    ),
    DocumentSpec(
        path="sales/sales_playbook_2023.txt",
        format="txt",
        description="Vertexia Sales Playbook 2023. Sections: Ideal Customer Profile (mid-market to enterprise B2B companies, 200-5000 employees, data-driven culture), discovery questions, product positioning (NexusFlow vs Fivetran, InsightLens vs Tableau/Looker), common objections and responses, deal stages (Prospect/Qualify/Demo/Eval/Negotiate/Close), MEDDIC framework application. Professional sales enablement doc. About 800 words.",
    ),
    DocumentSpec(
        path="sales/q3_closed_won_report.txt",
        format="txt",
        description="Q3 2023 Closed-Won Report by Ben Carter (VP Sales). Key metrics: Total bookings $4.2M (booking-date accounting), 8 new logos, 15 expansions, NRR 118%. Top deals listed. Note in the report: 'Total bookings of $4.2M represents booking-date accounting. Finance team will report GAAP recognized revenue of $4.12M for Q3.' About 400 words.",
        extra_context="Explicitly state both $4.2M bookings AND $4.12M GAAP here to partially explain the discrepancy, but also note this is in SALES docs — finance report only states $4.12M."
    ),

    # ── LEGAL ─────────────────────────────────────────────────────────────────
    DocumentSpec(
        path="legal/phoenix_corp_msa.txt",
        format="txt",
        description="Master Service Agreement between Vertexia Inc. and Phoenix Corp. Dated June 1 2022. This is a summarized/excerpted version (not full legal text). Key terms: (1) Service Availability SLA: 99.99% monthly uptime for all contracted services; (2) downtime credits: 10% fee credit per 0.01% below SLA; (3) contract term: 3 years; (4) data processing per standard DPA; (5) dedicated support tier. About 500 words. Legal document style.",
        extra_context="99.99% uptime SLA must be clearly stated. This is the key legal fact for Trap 6."
    ),
    DocumentSpec(
        path="legal/datacraft_acquisition_summary.txt",
        format="txt",
        description="Summary of the DataCraft acquisition terms. Dated January 15 2022. Acquisition price: $8M (all cash). 12 DataCraft employees join Vertexia with 18-month retention packages. IP transfer: all DataCraft patents, trademarks, and software IP transfer to Vertexia. Non-compete: Felix Wagner 3-year non-compete in ETL space (except as Vertexia employee). Legal summary/memo style. About 400 words.",
    ),
    DocumentSpec(
        path="legal/ip_policy.txt",
        format="txt",
        description="Vertexia Inc. Intellectual Property Policy. Covers: employee IP assignment (all work product belongs to Vertexia), open source contribution policy (approved OSS list, prohibited licenses), confidentiality obligations, whistleblower protections. Effective date: January 2021, last updated March 2023. About 500 words.",
    ),
    DocumentSpec(
        path="legal/data_processing_agreement_template.txt",
        format="txt",
        description="Vertexia Inc. Data Processing Agreement (DPA) Template — GDPR/CCPA compliant. Standard clauses: data controller/processor definitions, processing purposes, security measures, sub-processor list (AWS, Snowflake, Okta), data subject rights procedures, breach notification (72 hours for GDPR). Template format with [CUSTOMER NAME] placeholders. About 600 words.",
    ),
    DocumentSpec(
        path="finance/series_b_investor_update.txt",
        format="txt",
        description="Investor update letter from Arjun Mehta following Series B close, July 2021. $18M Series B led by Andreessen Horowitz. Use of proceeds: 60% engineering headcount, 30% go-to-market, 10% infrastructure. ARR at Series B close: $5.2M. Milestones: launched InsightLens, crossed 50 customers, NPS 67. Tone: optimistic, strategic. About 400 words.",
    ),
    DocumentSpec(
        path="finance/series_c_announcement.txt",
        format="txt",
        description="Internal announcement from Arjun Mehta of Series C close, October 5 2023. $45M Series C led by Sequoia Capital. ARR at Series C close: $16.5M. 3x ARR growth since Series B. Use of proceeds: international expansion, AI features, enterprise sales team expansion. Company-wide Slack announcement style, then detailed FAQ. About 500 words.",
    ),
    DocumentSpec(
        path="finance/q3_2023_finance_report.txt",
        format="txt",
        description="Q3 2023 Finance Report by Robert Okafor (CFO). Sections: Revenue (GAAP recognized revenue $4.12M — NexusFlow $1.6M, InsightLens $1.5M, PulseConnect $1.02M), gross margin 72%, operating expenses by category, cash position, headcount costs. Professional finance narrative. Note: uses GAAP revenue, not bookings. About 600 words.",
        extra_context="Revenue is $4.12M GAAP. Do NOT use $4.2M. This conflicts with the sales report and all-hands notes (which use bookings). This is Trap 4."
    ),
]


# ── CSV and JSON documents (generated directly, not via LLM) ────────────────

def generate_employee_directory(out_path: Path) -> None:
    """Generate a realistic employee directory CSV with ~50 key employees."""
    employees = [
        # id, name, department, role, manager, start_date, status, location
        ("E001", "Arjun Mehta", "Executive", "CEO", "", "2019-03-01", "active", "San Francisco"),
        ("E002", "Diana Volkov", "Executive", "Co-Founder (departed)", "", "2019-03-01", "departed_2021", "San Francisco"),
        ("E003", "Sarah Chen", "Platform Engineering", "CTO", "E001", "2019-06-15", "active", "San Francisco"),
        ("E004", "Nadia Kim", "Product", "CPO", "E001", "2020-02-01", "active", "San Francisco"),
        ("E005", "Robert Okafor", "Finance", "CFO", "E001", "2020-04-15", "active", "San Francisco"),
        ("E006", "Lisa Torres", "Revenue", "CRO", "E001", "2021-01-10", "active", "Austin"),
        ("E007", "Zara Ahmed", "People & Culture", "VP People & Culture", "E001", "2020-09-01", "active", "San Francisco"),
        ("E008", "Carmen Reyes", "Legal", "VP Legal", "E005", "2021-03-15", "active", "San Francisco"),
        ("E009", "Marcus Webb", "Platform Engineering", "Lead, Platform Engineering", "E003", "2020-01-15", "active", "San Francisco"),
        ("E010", "Priya Nair", "Platform Engineering", "Lead, Data Platform Team", "E009", "2020-03-01", "active", "Bangalore"),
        ("E011", "Daniel Osei", "Platform Engineering", "Lead, Security Team", "E009", "2021-05-10", "active", "San Francisco"),
        ("E012", "Tomás García", "Product Engineering", "Lead, Product Engineering", "E003", "2019-08-01", "active", "Austin"),
        ("E013", "Yuki Tanaka", "Product Engineering", "Lead, NexusFlow Team", "E012", "2019-10-15", "active", "San Francisco"),
        ("E014", "Aisha Johnson", "Product Engineering", "Senior Manager, InsightLens Team", "E012", "2021-02-01", "active", "Austin"),
        ("E015", "Raj Patel", "Product Engineering", "Staff PM, PulseConnect Team", "E012", "2021-07-15", "active", "San Francisco"),
        ("E016", "Felix Wagner", "DataCraft Integration", "Lead, DataCraft Integration Team", "E003", "2022-01-15", "active", "Berlin"),
        ("E017", "Kenji Ito", "Platform Engineering", "Senior Engineer, Data Platform", "E010", "2022-04-01", "active", "Bangalore"),
        ("E018", "Ben Carter", "Revenue", "VP Sales", "E006", "2021-04-01", "active", "Austin"),
        ("E019", "Maya Sharma", "Revenue", "VP Customer Success", "E006", "2021-06-15", "active", "San Francisco"),
        ("E020", "James O'Brien", "Platform Engineering", "Staff Engineer, Infrastructure", "E009", "2020-06-01", "active", "San Francisco"),
        ("E021", "Lin Wei", "Product Engineering", "Senior Engineer, NexusFlow", "E013", "2022-02-15", "active", "Bangalore"),
        ("E022", "Sophie Laurent", "Product Engineering", "Senior Engineer, InsightLens", "E014", "2022-05-01", "active", "San Francisco"),
        ("E023", "Amara Diallo", "People & Culture", "HR Business Partner", "E007", "2022-03-01", "active", "San Francisco"),
        ("E024", "Carlos Mendez", "Revenue", "Account Executive", "E018", "2022-08-15", "active", "Austin"),
        ("E025", "Fatima Al-Hassan", "Product", "Senior PM, NexusFlow", "E004", "2023-01-10", "active", "San Francisco"),
        # DataCraft employees (joined 2022-01-15)
        ("E026", "Ravi Krishnan", "DataCraft Integration", "Senior Engineer, DataCraft", "E016", "2022-01-15", "active", "Berlin"),
        ("E027", "Emma Fischer", "DataCraft Integration", "Engineer, DataCraft", "E016", "2022-01-15", "active", "Berlin"),
        ("E028", "Noah Zimmermann", "DataCraft Integration", "Data Engineer, DataCraft", "E016", "2022-01-15", "active", "Berlin"),
        # Departed employees (still appear in old docs — Trap 9)
        ("E029", "Adrian Blake", "Platform Engineering", "Senior Engineer (departed)", "E010", "2021-11-01", "departed_2023-08", "San Francisco"),
        ("E030", "Preet Kaur", "Revenue", "Customer Success Manager (departed)", "E019", "2022-01-01", "departed_2023-06", "Austin"),
        # Additional employees to reach realistic scale (abbreviated)
        ("E031", "Wei Zhang", "Finance", "Senior Financial Analyst", "E005", "2022-06-01", "active", "San Francisco"),
        ("E032", "Omar Faruk", "Platform Engineering", "Engineer, Security", "E011", "2023-02-01", "active", "Bangalore"),
        ("E033", "Isabella Romano", "Product Engineering", "Senior Engineer, PulseConnect", "E015", "2022-09-15", "active", "Austin"),
        ("E034", "David Kim", "Revenue", "Account Executive", "E018", "2023-03-01", "active", "San Francisco"),
        ("E035", "Nora Andersen", "Product", "UX Designer", "E004", "2022-11-01", "active", "San Francisco"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["employee_id", "name", "department", "role", "manager_id", "start_date", "status", "location"])
        writer.writerows(employees)


def generate_on_call_schedule(out_path: Path) -> None:
    """Generate on-call schedule for August 2023. Kenji Ito is on call week of Aug 14."""
    rows = [
        ("2023-07-31", "2023-08-06", "Data Platform Team", "Priya Nair", "E010"),
        ("2023-08-07", "2023-08-13", "Data Platform Team", "James O'Brien", "E020"),
        ("2023-08-14", "2023-08-20", "Data Platform Team", "Kenji Ito", "E017"),  # ← THE KEY ROW
        ("2023-08-21", "2023-08-27", "Data Platform Team", "Lin Wei", "E021"),
        ("2023-08-28", "2023-09-03", "Data Platform Team", "Priya Nair", "E010"),
        ("2023-07-31", "2023-08-06", "NexusFlow Team", "Yuki Tanaka", "E013"),
        ("2023-08-07", "2023-08-13", "NexusFlow Team", "Lin Wei", "E021"),
        ("2023-08-14", "2023-08-20", "NexusFlow Team", "Yuki Tanaka", "E013"),
        ("2023-08-21", "2023-08-27", "NexusFlow Team", "Sophie Laurent", "E022"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["week_start", "week_end", "team", "on_call_engineer", "employee_id"])
        writer.writerows(rows)


def generate_api_dependencies(out_path: Path) -> None:
    """
    API dependency map. InsightLens → NexusFlow events_api is the critical row for Trap 10.
    """
    rows = [
        ("InsightLens", "NexusFlow", "events_api", "v2", "critical", "Dashboard data ingestion"),
        ("InsightLens", "NexusFlow", "connectors_api", "v1", "optional", "Connector status polling"),
        ("PulseConnect", "InsightLens", "metrics_api", "v1", "optional", "Embedded analytics widget"),
        ("PulseConnect", "NexusFlow", "pipeline_status_api", "v1", "optional", "Pipeline health display"),
        ("InsightLens", "external_snowflake", "jdbc", "native", "critical", "Snowflake data connector"),
        ("NexusFlow", "external_aws_s3", "sdk", "native", "critical", "Cold storage delivery"),
        ("NexusFlow", "external_pulsar", "client", "v3.0", "critical", "Message queue"),
        ("PulseConnect", "external_sendgrid", "api_v3", "v3", "critical", "Email delivery"),
        ("PulseConnect", "external_twilio", "api", "v2", "optional", "SMS notifications"),
        ("DataCraft", "NexusFlow", "ingest_api", "v2", "critical", "DataCraft pipeline handoff"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["consuming_service", "providing_service", "api_endpoint", "version", "criticality", "purpose"])
        writer.writerows(rows)


def generate_revenue_csv(out_path: Path) -> None:
    """Monthly revenue by product for 2023 (GAAP).

    Quarterly totals:
      Q1: NexusFlow $1.2M, InsightLens $1.4M, PulseConnect $1.2M = $3.8M
      Q2: NexusFlow $1.4M, InsightLens $1.4M, PulseConnect $1.2M = $4.0M
      Q3: NexusFlow $1.6M, InsightLens $1.5M, PulseConnect $1.02M = $4.12M (GAAP)
    """
    rows = [
        # month, nexusflow, insightlens, pulseconnect, total
        # Q1: NF=1.2M, IL=1.4M, PC=1.2M → total 3.8M
        ("2023-01", 390_000, 460_000, 390_000, 1_240_000),
        ("2023-02", 400_000, 470_000, 400_000, 1_270_000),
        ("2023-03", 410_000, 470_000, 410_000, 1_290_000),
        # Q2: NF=1.4M, IL=1.4M, PC=1.2M → total 4.0M
        ("2023-04", 450_000, 460_000, 390_000, 1_300_000),
        ("2023-05", 470_000, 470_000, 400_000, 1_340_000),
        ("2023-06", 480_000, 470_000, 410_000, 1_360_000),
        # Q3: NF=1.6M, IL=1.5M, PC=1.02M → total 4.12M
        ("2023-07", 500_000, 480_000, 330_000, 1_310_000),
        ("2023-08", 520_000, 500_000, 340_000, 1_360_000),
        ("2023-09", 580_000, 520_000, 350_000, 1_450_000),
        # Q4 partial
        ("2023-10", 600_000, 540_000, 360_000, 1_500_000),
        ("2023-11", 620_000, 560_000, 380_000, 1_560_000),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "nexusflow_revenue", "insightlens_revenue", "pulseconnect_revenue", "total_revenue"])
        writer.writerows(rows)


def generate_deal_pipeline(out_path: Path) -> None:
    """Q3 2023 sales pipeline with closed/open deals."""
    companies = [
        ("Apex Financial", "Financial Services", 240_000, "Closed-Won", "NexusFlow"),
        ("GreenLeaf Retail", "Retail", 180_000, "Closed-Won", "InsightLens"),
        ("TechVenture AI", "Technology", 420_000, "Closed-Won", "NexusFlow+InsightLens"),
        ("Harbor Logistics", "Logistics", 96_000, "Closed-Won", "PulseConnect"),
        ("Meridian Health", "Healthcare", 360_000, "Closed-Won", "NexusFlow"),
        ("Cascade Media", "Media", 144_000, "Closed-Won", "InsightLens"),
        ("Pinnacle Mfg", "Manufacturing", 192_000, "Closed-Won", "NexusFlow"),
        ("Stellarpath", "SaaS", 60_000, "Closed-Won", "PulseConnect"),
        ("BlueRidge Energy", "Energy", 480_000, "Eval", "NexusFlow+InsightLens"),
        ("Northgate Bank", "Financial Services", 840_000, "Negotiate", "NexusFlow"),
        ("Crestwood Pharma", "Pharma", 300_000, "Demo", "InsightLens"),
        ("DataFlow Corp", "Technology", 120_000, "Closed-Lost", "NexusFlow"),
        ("Vanguard Retail", "Retail", 90_000, "Closed-Lost", "PulseConnect"),
    ]
    rows = [(f"D{i+100}", name, industry, arr, stage, product, "Ben Carter", "2023-Q3")
            for i, (name, industry, arr, stage, product) in enumerate(companies)]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["deal_id", "company_name", "industry", "arr", "stage", "products", "owner", "quarter"])
        writer.writerows(rows)


def generate_vendor_contracts(out_path: Path) -> None:
    rows = [
        ("AWS", "Cloud Infrastructure", 480_000, "2024-12-31", "Robert Okafor", "Annual commitment, multi-year"),
        ("Snowflake", "Data Warehouse", 120_000, "2024-06-30", "Priya Nair", "Usage-based + commitment"),
        ("Okta", "Identity & SSO", 36_000, "2024-01-31", "Daniel Osei", "Per-seat, 500 seats"),
        ("Datadog", "Monitoring", 48_000, "2024-03-31", "Marcus Webb", "Hosts + custom metrics"),
        ("Sendgrid", "Email Delivery", 18_000, "2024-12-31", "Raj Patel", "Volume-based"),
        ("Twilio", "SMS/Communications", 24_000, "2024-12-31", "Raj Patel", "Usage-based"),
        ("GitHub", "Source Control", 12_000, "2024-12-31", "Marcus Webb", "Enterprise, 200 seats"),
        ("1Password", "Password Management", 8_400, "2024-12-31", "Daniel Osei", "Teams plan"),
        ("Notion", "Wiki/Docs", 9_600, "2024-12-31", "Zara Ahmed", "Plus plan, company-wide"),
        ("Figma", "Design", 14_400, "2024-12-31", "Nadia Kim", "Professional, design team"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vendor", "category", "annual_value_usd", "renewal_date", "owner", "notes"])
        writer.writerows(rows)


def generate_customer_list(out_path: Path) -> None:
    customers = [
        ("Phoenix Corp", "Financial Services", 2_400_000, "NexusFlow+InsightLens", "Maya Sharma", "enterprise", "2022-06-01"),
        ("Apex Financial", "Financial Services", 240_000, "NexusFlow", "Preet Kaur (departed)", "mid-market", "2022-09-15"),
        ("TechVenture AI", "Technology", 420_000, "NexusFlow+InsightLens", "Maya Sharma", "mid-market", "2023-07-01"),
        ("Meridian Health", "Healthcare", 360_000, "NexusFlow", "Maya Sharma", "mid-market", "2023-08-01"),
        ("GreenLeaf Retail", "Retail", 180_000, "InsightLens", "Maya Sharma", "mid-market", "2023-07-15"),
        ("Harbor Logistics", "Logistics", 96_000, "PulseConnect", "Maya Sharma", "smb", "2023-08-01"),
        ("Cascade Media", "Media", 144_000, "InsightLens", "Maya Sharma", "smb", "2023-08-15"),
        ("Pinnacle Mfg", "Manufacturing", 192_000, "NexusFlow", "Maya Sharma", "mid-market", "2023-09-01"),
        ("Stellarpath", "SaaS", 60_000, "PulseConnect", "Maya Sharma", "smb", "2023-09-15"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["company_name", "industry", "arr_usd", "products", "csm", "segment", "contract_start"])
        writer.writerows(customers)


def generate_budget_allocation(out_path: Path) -> None:
    rows = [
        ("Platform Engineering", 8_200_000, 42, "Marcus Webb", "Engineering infrastructure and platform"),
        ("Product Engineering", 9_800_000, 58, "Tomás García", "Product development across 3 products"),
        ("DataCraft Integration", 1_400_000, 12, "Felix Wagner", "DataCraft migration and integration"),
        ("Product", 2_100_000, 14, "Nadia Kim", "Product management and design"),
        ("Revenue", 6_400_000, 38, "Lisa Torres", "Sales, CS, and marketing"),
        ("Finance", 800_000, 6, "Robert Okafor", "Finance, accounting, FP&A"),
        ("Legal", 400_000, 3, "Carmen Reyes", "Legal and compliance"),
        ("People & Culture", 600_000, 5, "Zara Ahmed", "HR, recruiting, culture"),
        ("Executive", 500_000, 3, "Arjun Mehta", "Executive team and ops"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["department", "annual_budget_usd", "headcount", "owner", "notes"])
        writer.writerows(rows)


def generate_offboarding_records(out_path: Path) -> None:
    rows = [
        ("E029", "Adrian Blake", "Platform Engineering", "2023-08-31", "voluntary", "Joined competitor FinDataCo", "E010"),
        ("E030", "Preet Kaur", "Revenue", "2023-06-30", "voluntary", "Relocated internationally", "E019"),
        ("E002", "Diana Volkov", "Executive", "2021-12-15", "voluntary_founder", "Left to found new startup (amicable)", "E001"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["employee_id", "name", "department", "last_day", "departure_type", "notes", "manager_id"])
        writer.writerows(rows)


def generate_product_okrs(out_path: Path) -> None:
    okrs = {
        "year": 2023,
        "owner": "Nadia Kim",
        "objectives": [
            {
                "id": "O1",
                "objective": "Make NexusFlow the easiest enterprise data pipeline to deploy",
                "key_results": [
                    {"id": "KR1.1", "metric": "Time-to-first-pipeline", "target": "< 30 minutes", "q3_actual": "47 minutes"},
                    {"id": "KR1.2", "metric": "Enterprise connector count", "target": "200+", "q3_actual": "187"},
                    {"id": "KR1.3", "metric": "NPS score", "target": "65+", "q3_actual": "61"}
                ]
            },
            {
                "id": "O2",
                "objective": "Establish InsightLens as a best-in-class BI tool for mid-market",
                "key_results": [
                    {"id": "KR2.1", "metric": "DAU/MAU ratio", "target": "0.45", "q3_actual": "0.38"},
                    {"id": "KR2.2", "metric": "Dashboard creation rate (per user/month)", "target": "3.0", "q3_actual": "2.4"},
                    {"id": "KR2.3", "metric": "Dashboard load time p95 (ms)", "target": "< 2000", "q3_actual": "3200"}
                ]
            },
            {
                "id": "O3",
                "objective": "Launch PulseConnect to product-led growth (PLG) motion",
                "key_results": [
                    {"id": "KR3.1", "metric": "PLG free signups Q3", "target": "500", "q3_actual": "312"},
                    {"id": "KR3.2", "metric": "Free-to-paid conversion rate", "target": "8%", "q3_actual": "5.2%"},
                    {"id": "KR3.3", "metric": "Mobile app DAU (post-launch)", "target": "N/A Q3", "q3_actual": "Not launched"}
                ]
            }
        ]
    }
    with open(out_path, "w") as f:
        json.dump(okrs, f, indent=2)


def generate_api_spec(out_path: Path) -> None:
    spec = {
        "service": "NexusFlow Events API",
        "version": "v2.0",
        "base_url": "https://api.nexusflow.vertexia.com/v2",
        "owner_team": "Data Platform Team",
        "owner": "Priya Nair",
        "availability_sla": "99.9%",
        "endpoints": [
            {
                "path": "/events",
                "method": "POST",
                "description": "Ingest a batch of events",
                "auth": "Bearer token",
                "rate_limit": "10000 req/min per tenant"
            },
            {
                "path": "/events/stream",
                "method": "GET",
                "description": "Stream events for downstream consumers (InsightLens primary consumer)",
                "auth": "Bearer token",
                "consumers": ["InsightLens", "DataCraft Integration"]
            },
            {
                "path": "/pipelines",
                "method": "GET",
                "description": "List pipeline status",
                "auth": "Bearer token"
            },
            {
                "path": "/connectors",
                "method": "GET",
                "description": "List available connectors",
                "auth": "Bearer token"
            }
        ],
        "changelog": {
            "v2.1": "Released 2023-08-14 — ROLLED BACK same day due to outage. Rate limiter config bug.",
            "v2.0": "Released 2023-06-01 — Pulsar migration complete, new streaming endpoint",
            "v1.9": "Released 2023-02-15 — Connector framework v2"
        }
    }
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)


# ── LLM generation ──────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
SYSTEM_INSTRUCTION = (
    "You are a corporate document writer creating realistic synthetic documents "
    "for an AI research project. Write only the requested document, nothing else."
)


def _build_prompt(spec: DocumentSpec) -> str:
    extra = f"\nADDITIONAL REQUIREMENTS: {spec.extra_context}" if spec.extra_context else ""
    return f"""You are writing a synthetic but realistic corporate document for a fictional company called Vertexia Inc.

COMPANY CONTEXT (follow these facts exactly):
{COMPANY_CONTEXT}

DOCUMENT TO WRITE:
File: {spec.path}
Format: {spec.format}
Instructions: {spec.description}
{extra}

Write only the document content. No preamble, no "here is the document", no meta-commentary.
Use appropriate format for the file type ({spec.format}).
Be specific with names, dates, and numbers — use the exact values from the company context.
Make it feel authentic to corporate culture — not a caricature, but real.
"""


def _try_gemini(spec: DocumentSpec) -> str:
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=google_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(spec),
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )
    return response.text


def _try_anthropic(spec: DocumentSpec) -> str:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=SYSTEM_INSTRUCTION,
        messages=[{"role": "user", "content": _build_prompt(spec)}],
    )
    return message.content[0].text


def generate_with_llm(spec: DocumentSpec, out_path: Path) -> None:
    try:
        text = _try_gemini(spec)
        provider = "gemini"
    except Exception as gemini_err:
        print(f"    Gemini failed ({gemini_err}), falling back to Anthropic...")
        text = _try_anthropic(spec)
        provider = "anthropic"
    out_path.write_text(text)
    print(f"  Generated [{provider}]: {spec.path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    base = Path(__file__).parent.parent / "company_data"

    print("=== Generating structured data files (no LLM) ===")
    structured = {
        "hr/employee_directory.csv": generate_employee_directory,
        "engineering/on_call_schedule_aug2023.csv": generate_on_call_schedule,
        "engineering/api_dependencies.csv": generate_api_dependencies,
        "finance/revenue_by_product_2023.csv": generate_revenue_csv,
        "finance/budget_allocation_2023.csv": generate_budget_allocation,
        "finance/vendor_contracts_summary.csv": generate_vendor_contracts,
        "hr/offboarding_records_2023.csv": generate_offboarding_records,
        "sales/deal_pipeline_q3_2023.csv": generate_deal_pipeline,
        "sales/customer_list.csv": generate_customer_list,
        "product/product_okrs_2023.json": generate_product_okrs,
        "engineering/nexusflow_api_spec.json": generate_api_spec,
    }
    for rel_path, fn in structured.items():
        out_path = base / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fn(out_path)
        print(f"  Created: {rel_path}")

    print("\n=== Generating LLM documents ===")
    for spec in DOCUMENTS:
        out_path = base / spec.path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"  Skipping (exists): {spec.path}")
            continue
        generate_with_llm(spec, out_path)

    print("\n=== Generating document catalog ===")
    catalog_path = Path(__file__).parent.parent / "results" / "document_catalog.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(base.rglob("*")):
        if f.is_file():
            rows.append({
                "path": str(f.relative_to(base)),
                "format": f.suffix.lstrip("."),
                "department": f.parent.name,
                "size_bytes": f.stat().st_size,
            })
    with open(catalog_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "format", "department", "size_bytes"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Catalog written: {catalog_path}")
    print(f"\nDone. {len(rows)} files generated.")


if __name__ == "__main__":
    main()
