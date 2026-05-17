# NexusFlow 2.0 — Product Requirements Document

**Author:** Nadia Kim, CPO
**Status:** Draft v2 — for review
**Last updated:** November 2023
**Reviewers:** Sarah Chen (CTO), Tomás García (Product Engineering), Yuki Tanaka (NexusFlow Lead), Marcus Webb (Platform Engineering), Carmen Reyes (Legal)

---

## 1. Problem Statement

NexusFlow has shipped as Vertexia's data pipeline and ETL product since GA in March 2020. Over the last 18 months, we've seen our customer profile shift: the same data engineers who originally adopted FLOW for batch ETL are now being asked by their businesses to move toward real-time or near-real-time pipelines. At the same time, our 2023 win/loss interviews surfaced two recurring themes:

1. **Connector breadth gaps.** ~40% of lost deals in H1 2023 cited "we'd need to build connectors X and Y ourselves." Our current catalog sits at 87 connectors; competitors lead with 200+.
2. **Pipeline authoring friction.** Non-platform data engineers (analytics engineers, embedded data engineers in product teams) find the YAML-first authoring model slow. They want a visual builder that still produces version-controlled artifacts.

The August 14, 2023 outage also surfaced architectural debt in the ingestion path — the rate limiter incident exposed how tightly coupled the events_api is to downstream consumers (notably InsightLens). NexusFlow 2.0 is the right moment to address this.

## 2. Target Users

**Primary persona:** Data engineers at mid-market companies (200–2,000 employees), typically on a data platform team of 2–8 people. They own ingestion, transformation, and delivery to a warehouse or lakehouse. They are comfortable with SQL, Python, and Git, but do not want to build or maintain bespoke connector code.

**Secondary persona:** Analytics engineers and embedded data practitioners in product/ops teams who consume from pipelines and increasingly want to author simple ones themselves.

**Non-goal:** Citizen-developer / no-code business user. We are not chasing the Zapier persona.

## 3. Key Features

### 3.1 Real-time streaming
- First-class streaming pipeline type, in addition to existing batch and micro-batch.
- Built on our Apache Pulsar backbone (migration completed Q2 2023).
- Sub-second end-to-end latency target for the hot path (Go ingestion → Pulsar → sink).
- Exactly-once semantics for supported sinks.

### 3.2 200+ connectors
- Expand connector catalog from 87 to 200+ by GA.
- Tiered support model: Vertexia-maintained (SLA-backed) vs. community / partner.
- Connector SDK published externally so partners (and customers) can contribute.

### 3.3 Visual pipeline builder
- Browser-based DAG editor (TypeScript frontend).
- Bidirectional sync with the YAML / Git-backed pipeline spec — visual edits produce diffs reviewable in PRs.
- Inline schema inference and column-level lineage preview.

## 4. Success Metrics

| Metric | Baseline (Q3 2023) | 12-month target post-GA |
|---|---|---|
| NexusFlow ARR | $1.6M quarterly revenue | $2.6M quarterly revenue |
| Connector catalog size | 87 | 200+ |
| Median time-to-first-pipeline (new account) | 11 days | < 3 days |
| % pipelines authored via visual builder | 0% | 35% |
| Net revenue retention (FLOW) | 112% | ≥ 125% |
| Win rate vs. top competitor in mid-market | 31% | ≥ 45% |

## 5. Non-Functional Requirements

### 5.1 Availability
Platform must achieve **99.9% uptime** per our standard SLA. Enterprise customers with custom SLAs may require higher — see Legal for specific contract terms.

(Note to engineering: any architectural decisions that would constrain our ability to offer higher availability tiers under negotiated contracts should be flagged in design review.)

### 5.2 Performance
- p95 ingestion latency, streaming pipelines: < 1s end-to-end.
- p95 batch job startup overhead: < 15s.
- Visual builder: DAG canvas must remain interactive (< 100ms input latency) up to 250 nodes.

### 5.3 Scalability
- Single tenant must support 10,000 concurrent active pipelines.
- Horizontal scale-out of ingestion workers without operator intervention up to 5x baseline within 90 seconds.

### 5.4 Security & Compliance
- All connectors must support secrets management via our standard vault integration (owners: Daniel Osei's team).
- SOC 2 Type II controls preserved; no regression in audit scope.
- PII tagging propagated through lineage.

### 5.5 Data Retention
Aligned with company policy: 90 days hot, 1 year cold (S3 Glacier). Logs 30 days. No change for 2.0.

## 6. Open Questions

1. Do we expose the visual builder to InsightLens users directly, or keep it FLOW-scoped for v2.0? (Owner: Nadia Kim, Aisha Johnson)
2. Connector SDK licensing model — to be confirmed with Carmen Reyes.
3. Migration story for existing v1.x customers on YAML-only pipelines. (Owner: Yuki Tanaka)

## 7. Out of Scope for 2.0

- ML feature store functionality.
- Reverse ETL (tracked separately under PulseConnect roadmap with Raj Patel).
- Mobile authoring UI.

---

*Next review: product council, with Arjun, Sarah, and Tomás. Please leave comments inline.*