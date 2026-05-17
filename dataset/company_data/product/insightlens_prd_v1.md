# InsightLens v1.0 — Product Requirements Document

**Author:** Nadia Kim, CPO
**Status:** Approved for Engineering
**Last updated:** May 12, 2021
**Target GA:** September 2021

---

## 1. Vision

**Make every business decision data-driven.**

Today, most operational decisions inside mid-market SaaS companies are made on gut, on stale spreadsheets, or on dashboards owned by a single overburdened analyst. The data exists — increasingly, it already flows through NexusFlow — but it isn't reaching the people who need it in a form they can use.

InsightLens is Vertexia's answer: a self-serve business analytics product that puts dashboarding, exploration, and scheduled reporting directly into the hands of operators (RevOps, Marketing, CS, Finance) without requiring SQL or a ticket to the data team.

InsightLens is the second pillar of the Vertexia platform. NexusFlow moves and shapes the data; InsightLens turns it into decisions.

## 2. User Personas

**Priya — RevOps Manager (primary).** Lives in spreadsheets. Needs weekly pipeline and forecast views, wants to slice by segment without filing a Jira ticket. Will champion the tool internally if it's faster than her current BI stack.

**Marcus — Marketing Lead (primary).** Cares about campaign attribution and funnel conversion. Wants scheduled email digests for his team every Monday.

**Elena — Customer Success Director (secondary).** Needs health-score dashboards and churn-risk views. Doesn't want to learn SQL, ever.

**Sam — Data Analyst (power user / enabler).** Builds the foundational dashboards and shared datasets that the personas above consume. Needs governance, version history, and the ability to write raw SQL when drag-and-drop isn't enough.

## 3. Core Features (v1.0 scope)

### 3.1 Drag-and-Drop Dashboard Builder
- Grid-based canvas with resizable tiles
- Add chart → pick dataset → pick dimensions/measures → done (target: under 60 seconds for first chart)
- Filter bar with cross-filtering across tiles on a single dashboard
- Light/dark themes, brandable logo for Enterprise tier

### 3.2 50+ Chart Types
Bar, stacked bar, line, area, scatter, pie/donut, funnel, sankey, cohort heatmap, geo map, KPI tile, table, pivot table, gauge, waterfall, box plot, and 35+ more. Charting layer built on a single rendering primitive to keep the matrix maintainable.

### 3.3 Scheduled Reports
- Email and Slack delivery
- PDF, PNG, and CSV attachments
- Daily / weekly / monthly / custom cron
- Per-recipient row-level filtering (v1.0 stretch; may slip to 1.1)

### 3.4 Data Connectors
v1.0 ships with:
- **NexusFlow (primary)** — see §4
- Postgres, MySQL, Snowflake, BigQuery, Redshift
- CSV upload
- Google Sheets

## 4. Integration with NexusFlow

InsightLens treats NexusFlow as a first-class, first-party source. Customers who already run NexusFlow should be able to add InsightLens and see live dashboards within minutes.

- InsightLens reads from NexusFlow via the **`events_api`** endpoint, subscribing to materialized event streams and curated tables exposed by the pipeline.
- Auth is shared via the Vertexia workspace identity, so no separate credentials are required.
- Schema discovery is automatic: any dataset published in NexusFlow appears in the InsightLens dataset picker within ~5 minutes.

**Dependency note:** This makes InsightLens operationally dependent on NexusFlow's `events_api`. Reliability work on that surface (owned by the NexusFlow team) is a shared concern. Yuki and team are aware.

## 5. Success Metrics

We will judge InsightLens v1.0 against three metrics in the 90 days post-GA:

| Metric | Target (90d post-GA) |
|---|---|
| **DAU / workspace** | ≥ 8 across pilot accounts |
| **Dashboard creation rate** | ≥ 3 net-new dashboards per workspace per week |
| **Time-to-insight** | Median < 5 min from signup to first published chart |

Secondary: scheduled-report adoption (% of workspaces with ≥1 active schedule), connector mix, and NexusFlow attach rate among existing FLOW customers (target: 40% within 6 months).

## 6. Out of Scope for v1.0

- Embedded analytics / white-label SDK
- ML-driven anomaly detection (roadmapped for v1.2)
- Mobile native app (responsive web only)
- Write-back to source systems

---

*Reviewers: Arjun Mehta, Sarah Chen, Aisha Johnson (incoming LENS team lead), Tomás García.*