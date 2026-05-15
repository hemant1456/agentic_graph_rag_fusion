"""
Dataset integrity tests for Step 00.

These tests verify that the generated corpus has the right structure and
that all 10 "traps" are verifiably present in the data.

Run with:
    uv run pytest step_00_dataset/tests/ -v
"""

import csv
import json
import re
from pathlib import Path

COMPANY_DATA = Path(__file__).parent.parent / "company_data"


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_text(rel_path: str) -> str:
    return (COMPANY_DATA / rel_path).read_text()


def read_csv(rel_path: str) -> list[dict]:
    with open(COMPANY_DATA / rel_path) as f:
        return list(csv.DictReader(f))


def read_json(rel_path: str) -> dict:
    with open(COMPANY_DATA / rel_path) as f:
        return json.load(f)


# ── Structural tests — all documents must exist ───────────────────────────────

REQUIRED_FILES = [
    "engineering/nexusflow_architecture.md",
    "engineering/nexusflow_v21_postmortem.txt",
    "engineering/data_platform_runbook.md",
    "engineering/project_phoenix_migration.md",
    "engineering/api_dependencies.csv",
    "engineering/on_call_schedule_aug2023.csv",
    "engineering/on_call_schedule_q4_2023.csv",
    "engineering/nexusflow_api_changelog.md",
    "engineering/datacraft_original_architecture.md",
    "engineering/datacraft_migration_complete.txt",
    "engineering/security_audit_2023.txt",
    "engineering/rfc_001_event_schema.md",
    "engineering/nexusflow_api_spec.json",
    "product/nexusflow_prd_v2.md",
    "product/insightlens_prd_v1.md",
    "product/pulseconnect_roadmap_2024.txt",
    "product/user_research_q3_2023.txt",
    "product/product_okrs_2023.json",
    "hr/employee_directory.csv",
    "hr/org_chart_q1_2023.txt",
    "hr/org_chart_q3_2023.txt",
    "hr/onboarding_handbook.txt",
    "hr/offboarding_records_2023.csv",
    "hr/promotion_announcements_2023.txt",
    "hr/datacraft_employee_integration.txt",
    "finance/revenue_by_product_2022.csv",
    "finance/revenue_by_product_2023.csv",
    "finance/q3_2023_finance_report.txt",
    "finance/budget_allocation_2023.csv",
    "finance/vendor_contracts_summary.csv",
    "finance/series_b_investor_update.txt",
    "finance/series_c_announcement.txt",
    "sales/phoenix_corp_deal_summary.txt",
    "sales/sales_playbook_2023.txt",
    "sales/q3_closed_won_report.txt",
    "sales/deal_pipeline_q3_2023.csv",
    "sales/deal_pipeline_q4_2023.csv",
    "sales/customer_list.csv",
    "sales/customer_health_scores_2023.csv",
    "sales/csm_account_history.csv",
    "legal/phoenix_corp_msa.txt",
    "legal/datacraft_acquisition_summary.txt",
    "legal/ip_policy.txt",
    "legal/data_processing_agreement_template.txt",
    "executive/q3_2023_allhands_notes.txt",
    "executive/board_meeting_april_2023.txt",
    "executive/company_strategy_2024.txt",
    "executive/founding_story.txt",
]


def test_all_required_files_exist():
    missing = [f for f in REQUIRED_FILES if not (COMPANY_DATA / f).exists()]
    assert not missing, f"Missing files: {missing}"


def test_minimum_file_count():
    all_files = list(COMPANY_DATA.rglob("*"))
    actual_files = [f for f in all_files if f.is_file()]
    assert len(actual_files) >= 48, f"Expected 48+ files, got {len(actual_files)}"


def test_format_variety():
    all_files = list(COMPANY_DATA.rglob("*"))
    suffixes = {f.suffix for f in all_files if f.is_file()}
    required_formats = {".md", ".txt", ".csv", ".json"}
    missing_formats = required_formats - suffixes
    assert not missing_formats, f"Missing file formats: {missing_formats}"


# ── Trap 1: Temporal Ambiguity (Sarah Chen's title) ──────────────────────────

def test_trap1_sarah_chen_vp_in_q1_org_chart():
    text = read_text("hr/org_chart_q1_2023.txt").lower()
    assert "sarah chen" in text
    assert "vp engineering" in text or "vp, engineering" in text or "vice president" in text


def test_trap1_sarah_chen_cto_in_q3_org_chart():
    text = read_text("hr/org_chart_q3_2023.txt").lower()
    assert "sarah chen" in text
    assert "cto" in text or "chief technology officer" in text


def test_trap1_promotion_announcement_exists():
    text = read_text("hr/promotion_announcements_2023.txt").lower()
    assert "sarah chen" in text
    assert "cto" in text or "chief technology officer" in text
    assert "april" in text or "2023-04" in text


def test_trap1_different_titles_across_era_docs():
    """Pre-restructuring doc says VP Eng; post says CTO. Both exist."""
    q1_text = read_text("hr/org_chart_q1_2023.txt").lower()
    q3_text = read_text("hr/org_chart_q3_2023.txt").lower()
    # Q1 should NOT say CTO for Sarah Chen
    # Q3 should NOT say VP Engineering for Sarah Chen
    assert "vp engineering" in q1_text or "vp, engineering" in q1_text
    assert "cto" in q3_text


# ── Trap 2: Name Collision — "Phoenix" ───────────────────────────────────────

def test_trap2_engineering_project_phoenix_exists():
    text = read_text("engineering/project_phoenix_migration.md").lower()
    assert "project phoenix" in text
    assert "python" in text  # It's a Python 2→3 migration


def test_trap2_sales_phoenix_corp_exists():
    text = read_text("sales/phoenix_corp_deal_summary.txt").lower()
    assert "phoenix corp" in text
    assert "arr" in text or "revenue" in text or "2.4" in text


def test_trap2_both_phoenixes_are_different():
    """Confirm the two "Phoenix" documents are about completely different things."""
    eng_text = read_text("engineering/project_phoenix_migration.md").lower()
    sales_text = read_text("sales/phoenix_corp_deal_summary.txt").lower()
    # Engineering Phoenix is about code migration
    assert "migration" in eng_text or "python" in eng_text
    # Sales Phoenix is about a customer deal
    assert "customer" in sales_text or "enterprise" in sales_text or "contract" in sales_text


# ── Trap 3: Multi-hop — Outage Attribution ──────────────────────────────────

def test_trap3_postmortem_does_not_name_oncall():
    """The postmortem must NOT name the on-call engineer — that's in the schedule."""
    text = read_text("engineering/nexusflow_v21_postmortem.txt").lower()
    assert "kenji" not in text, "On-call engineer name should NOT be in postmortem"
    assert "data ingestion service" in text or "ingestion" in text


def test_trap3_runbook_names_data_platform_team_as_owner():
    text = read_text("engineering/data_platform_runbook.md").lower()
    assert "data platform team" in text or "data platform" in text
    assert "priya nair" in text or "priya" in text


def test_trap3_oncall_schedule_has_kenji_aug14():
    rows = read_csv("engineering/on_call_schedule_aug2023.csv")
    aug14_row = next(
        (r for r in rows if r.get("week_start") == "2023-08-14" and "Data Platform" in r.get("team", "")),
        None
    )
    assert aug14_row is not None, "No on-call row for Aug 14 Data Platform"
    assert "Kenji Ito" in aug14_row.get("on_call_engineer", ""), \
        f"Expected Kenji Ito, got: {aug14_row.get('on_call_engineer')}"


def test_trap3_chain_is_complete():
    """All three links in the chain exist:
    1. Postmortem → data ingestion service involved
    2. Runbook → data ingestion owned by Data Platform Team
    3. On-call schedule → Kenji Ito on call Aug 14
    """
    postmortem = read_text("engineering/nexusflow_v21_postmortem.txt").lower()
    runbook = read_text("engineering/data_platform_runbook.md").lower()
    schedule = read_csv("engineering/on_call_schedule_aug2023.csv")

    assert "ingestion" in postmortem
    assert "data platform" in runbook
    kenji_row = next((r for r in schedule if "Kenji" in r.get("on_call_engineer", "")), None)
    assert kenji_row is not None


# ── Trap 4: Contradictory Revenue Numbers ────────────────────────────────────

def test_trap4_finance_report_says_4_12m():
    text = read_text("finance/q3_2023_finance_report.txt")
    assert "4.12" in text or "4,120" in text, "Finance report should state $4.12M GAAP revenue"


def test_trap4_allhands_says_4_2m():
    text = read_text("executive/q3_2023_allhands_notes.txt")
    assert "4.2" in text, "All-hands notes should state ~$4.2M (bookings)"


def test_trap4_both_numbers_exist_in_corpus():
    finance = read_text("finance/q3_2023_finance_report.txt")
    allhands = read_text("executive/q3_2023_allhands_notes.txt")
    assert "4.12" in finance or "4,120" in finance
    assert "4.2" in allhands


# ── Trap 5: DataCraft Stack Conflict (Kafka vs Pulsar) ───────────────────────

def test_trap5_datacraft_original_uses_kafka():
    text = read_text("engineering/datacraft_original_architecture.md").lower()
    assert "kafka" in text, "DataCraft original architecture must mention Kafka"


def test_trap5_vertexia_uses_pulsar():
    text = read_text("engineering/nexusflow_architecture.md").lower()
    assert "pulsar" in text, "NexusFlow architecture must mention Apache Pulsar"


def test_trap5_migration_completes_the_story():
    text = read_text("engineering/datacraft_migration_complete.txt").lower()
    assert "kafka" in text or "pulsar" in text
    assert "migrat" in text  # migration/migrated


# ── Trap 6: SLA Gap (99.9% vs 99.99%) ────────────────────────────────────────

def test_trap6_nexusflow_architecture_targets_three_nines():
    text = read_text("engineering/nexusflow_architecture.md")
    assert "99.9%" in text, "Architecture doc must state 99.9% availability (three nines)"
    assert "99.99%" not in text, "Architecture doc must NOT claim 99.99% — that's the trap"


def test_trap6_phoenix_corp_msa_requires_four_nines():
    text = read_text("legal/phoenix_corp_msa.txt")
    assert "99.99%" in text, "Phoenix Corp MSA must specify 99.99% uptime SLA"


def test_trap6_gap_is_explorable():
    """Confirm the gap: phoenix contract requires more than architecture delivers."""
    arch_text = read_text("engineering/nexusflow_architecture.md")
    legal_text = read_text("legal/phoenix_corp_msa.txt")
    assert "99.9%" in arch_text
    assert "99.99%" in legal_text
    # The gap should exist (99.9% < 99.99%)
    assert "99.99%" not in arch_text  # Architecture does NOT claim 99.99%


# ── Trap 7: Numbers in Different Formats ─────────────────────────────────────

def test_trap7_revenue_csv_has_exact_q3():
    rows = read_csv("finance/revenue_by_product_2023.csv")
    q3_rows = [r for r in rows if r.get("month", "").startswith("2023-0") and
               r.get("month", "") in ("2023-07", "2023-08", "2023-09")]
    assert len(q3_rows) == 3, "Must have 3 months of Q3 data (July, August, September)"
    total = sum(int(r.get("total_revenue", 0)) for r in q3_rows)
    # Q3 GAAP total should be ~4,120,000
    assert abs(total - 4_120_000) < 100_000, f"Q3 revenue total unexpected: {total}"


# ── Trap 8: Renamed Department ────────────────────────────────────────────────

def test_trap8_q1_has_single_engineering_dept():
    text = read_text("hr/org_chart_q1_2023.txt").lower()
    assert "engineering" in text
    # Should NOT have "platform engineering" and "product engineering" as separate orgs
    # (those come after the April 2023 restructuring)


def test_trap8_q3_has_split_engineering_dept():
    text = read_text("hr/org_chart_q3_2023.txt").lower()
    assert "platform engineering" in text
    assert "product engineering" in text


def test_trap8_employee_directory_reflects_new_structure():
    rows = read_csv("hr/employee_directory.csv")
    depts = {r["department"] for r in rows}
    assert "Platform Engineering" in depts, "Employee directory must have Platform Engineering"
    assert "Product Engineering" in depts, "Employee directory must have Product Engineering"


# ── Trap 9: Stale References (Departed Employees) ────────────────────────────

def test_trap9_offboarding_records_exist():
    rows = read_csv("hr/offboarding_records_2023.csv")
    assert len(rows) >= 2, "Should have at least 2 offboarding records for 2023"


def test_trap9_departed_employees_have_status():
    rows = read_csv("hr/employee_directory.csv")
    departed = [r for r in rows if "departed" in r.get("status", "")]
    assert len(departed) >= 2, "At least 2 employees should be marked departed"


def test_trap9_customer_list_references_departed_csm():
    """Customer list still shows Preet Kaur (departed) as CSM for Apex Financial."""
    rows = read_csv("sales/customer_list.csv")
    apex_row = next((r for r in rows if "Apex" in r.get("company_name", "")), None)
    if apex_row:
        # Check that the CSM referenced is a departed employee
        assert "Preet Kaur" in apex_row.get("csm", "") or "departed" in apex_row.get("csm", "").lower(), \
            "Apex Financial CSM should reference Preet Kaur (departed)"


# ── Trap 10: Cross-format Dependency (InsightLens → NexusFlow) ───────────────

def test_trap10_api_dependencies_has_insightlens_nexusflow_link():
    rows = read_csv("engineering/api_dependencies.csv")
    link = next(
        (r for r in rows
         if r.get("consuming_service", "").lower() == "insightlens"
         and r.get("providing_service", "").lower() == "nexusflow"
         and r.get("criticality", "").lower() == "critical"),
        None
    )
    assert link is not None, "Must have a critical InsightLens → NexusFlow dependency in api_dependencies.csv"


def test_trap10_no_prose_doc_explicitly_mentions_outage_affected_insightlens():
    """
    The InsightLens-outage connection is ONLY in the CSV.
    Prose documents should not explicitly state 'InsightLens was affected by the August outage'.
    This validates the trap: you NEED the structured data to answer the question.
    """
    postmortem = read_text("engineering/nexusflow_v21_postmortem.txt").lower()
    # Postmortem should focus on NexusFlow, not explicitly say InsightLens was affected
    # (the connection is implicit via the API dependency CSV)
    assert "nexusflow" in postmortem or "data ingestion" in postmortem


# ── Golden Questions — answerable facts exist in corpus ──────────────────────

def test_gq1_data_retention_policy_exists():
    """Q1: What is Vertexia's data retention policy?"""
    text = read_text("hr/onboarding_handbook.txt").lower()
    assert "90 day" in text or "90-day" in text or "90 days" in text
    assert "glacier" in text or "cold storage" in text or "1 year" in text


def test_gq2_nexusflow_revenue_comparison_possible():
    """Q2: NexusFlow Q2 vs Q3 2023 revenue comparison."""
    rows = read_csv("finance/revenue_by_product_2023.csv")
    q2_rows = [r for r in rows if r.get("month") in ("2023-04", "2023-05", "2023-06")]
    q3_rows = [r for r in rows if r.get("month") in ("2023-07", "2023-08", "2023-09")]
    assert len(q2_rows) == 3
    assert len(q3_rows) == 3
    q2_nexusflow = sum(int(r["nexusflow_revenue"]) for r in q2_rows)
    q3_nexusflow = sum(int(r["nexusflow_revenue"]) for r in q3_rows)
    assert q3_nexusflow > q2_nexusflow, f"Q3 ({q3_nexusflow}) should be higher than Q2 ({q2_nexusflow})"


def test_gq8_datacraft_12_employees():
    """Q8: How many employees joined from DataCraft?"""
    text = read_text("hr/datacraft_employee_integration.txt").lower()
    assert "12" in text, "DataCraft integration memo must mention 12 employees"


def test_gq10_chain_for_insightlens_outage_impact():
    """Q10: Was InsightLens affected by the August 2023 outage?"""
    # Step 1: Outage happened to NexusFlow
    postmortem = read_text("engineering/nexusflow_v21_postmortem.txt").lower()
    assert "2023" in postmortem and ("august" in postmortem or "aug" in postmortem)

    # Step 2: InsightLens depends on NexusFlow events_api (critical)
    deps = read_csv("engineering/api_dependencies.csv")
    critical_dep = next(
        (r for r in deps
         if "insightlens" in r.get("consuming_service", "").lower()
         and "nexusflow" in r.get("providing_service", "").lower()
         and "critical" in r.get("criticality", "").lower()),
        None
    )
    assert critical_dep is not None, "The critical InsightLens→NexusFlow dependency must exist"


# ── Expanded dataset integrity ────────────────────────────────────────────────

def test_customer_list_has_20_rows():
    rows = read_csv("sales/customer_list.csv")
    assert len(rows) == 20, f"customer_list.csv must have 20 rows, got {len(rows)}"


def test_customer_list_total_arr_is_11m():
    rows = read_csv("sales/customer_list.csv")
    total = sum(int(r["arr_usd"]) for r in rows)
    assert total == 11_000_000, f"Total ARR must be $11,000,000, got ${total:,}"


def test_customer_list_enterprise_segment_total():
    rows = read_csv("sales/customer_list.csv")
    enterprise_arr = sum(int(r["arr_usd"]) for r in rows if r.get("segment") == "enterprise")
    assert enterprise_arr == 7_160_000, f"Enterprise ARR must be $7,160,000, got ${enterprise_arr:,}"


def test_employee_directory_has_48_rows():
    rows = read_csv("hr/employee_directory.csv")
    assert len(rows) >= 48, f"employee_directory.csv must have 48+ rows, got {len(rows)}"


def test_employee_directory_has_5_berlin_employees():
    rows = read_csv("hr/employee_directory.csv")
    berlin = [r for r in rows if r.get("location", "").lower() == "berlin"]
    names = [r["name"] for r in berlin]
    assert len(berlin) == 5, f"Expected 5 Berlin employees, got {len(berlin)}: {names}"
    assert any("Aleksander Nowak" in n for n in names), "Aleksander Nowak must be in Berlin employees"


def test_vendor_contracts_has_15_rows():
    rows = read_csv("finance/vendor_contracts_summary.csv")
    assert len(rows) == 15, f"vendor_contracts_summary.csv must have 15 rows, got {len(rows)}"


def test_vendor_contracts_total_spend():
    rows = read_csv("finance/vendor_contracts_summary.csv")
    total = sum(float(r.get("annual_value_usd", 0)) for r in rows)
    assert abs(total - 956_400) < 1, f"Total vendor spend must be $956,400, got ${total:,.0f}"


def test_csm_account_history_preet_kaur_transition():
    """Apex Financial CSM transitioned from Preet Kaur to Sam Rivera."""
    rows = read_csv("sales/csm_account_history.csv")
    apex_rows = [r for r in rows if "Apex" in r.get("company_name", "")]
    assert len(apex_rows) == 2, f"Apex Financial must have 2 CSM history rows, got {len(apex_rows)}"
    preet_row = next((r for r in apex_rows if "Preet Kaur" in r.get("csm_name", "")), None)
    assert preet_row is not None, "Preet Kaur must be in Apex Financial CSM history"
    sam_row = next((r for r in apex_rows if "Sam Rivera" in r.get("csm_name", "")), None)
    assert sam_row is not None, "Sam Rivera must be in Apex Financial CSM history"


def test_customer_health_scores_high_risk_customers():
    rows = read_csv("sales/customer_health_scores_2023.csv")
    high_risk = [r["company_name"] for r in rows if r.get("renewal_risk") == "high"]
    assert len(high_risk) >= 3, f"Expected 3+ high-risk customers, got {len(high_risk)}: {high_risk}"
    assert "Apex Financial" in high_risk, "Apex Financial must be high renewal risk"


def test_deal_pipeline_q4_2023_closed_won():
    rows = read_csv("sales/deal_pipeline_q4_2023.csv")
    closed_won = [r for r in rows if r.get("stage") == "Closed-Won"]
    assert len(closed_won) >= 4, f"Q4 2023 must have 4+ Closed-Won deals, got {len(closed_won)}"


def test_nexusflow_api_changelog_has_breaking_changes():
    text = read_text("engineering/nexusflow_api_changelog.md")
    assert "v2.1" in text
    assert "events/stream" in text or "events/batch" in text
    assert "Breaking" in text or "breaking" in text


def test_revenue_by_product_2022_exists_and_has_12_months():
    rows = read_csv("finance/revenue_by_product_2022.csv")
    assert len(rows) == 12, f"revenue_by_product_2022.csv must have 12 months, got {len(rows)}"
    months = {r.get("month", "") for r in rows}
    assert "2022-01" in months and "2022-12" in months


def test_on_call_schedule_q4_2023_coverage():
    rows = read_csv("engineering/on_call_schedule_q4_2023.csv")
    assert len(rows) >= 20, f"Q4 2023 on-call schedule must have 20+ rows, got {len(rows)}"
    teams = {r.get("team", "") for r in rows}
    assert "Data Platform Team" in teams
    assert "NexusFlow Team" in teams
