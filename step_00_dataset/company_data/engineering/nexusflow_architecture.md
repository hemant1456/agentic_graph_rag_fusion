# NexusFlow System Architecture

**Author:** Priya Nair, Lead — Data Platform Team
**Date:** February 2023
**Status:** Living document (v3.2)
**Audience:** Engineering, SRE, Solutions Architects

---

## 1. Overview

NexusFlow (internal codename: FLOW) is Vertexia's customer data pipeline and ETL platform, GA since March 2020. It ingests events and records from customer source systems, applies user-defined transformations, and delivers normalized data to downstream destinations (warehouses, internal services like InsightLens, and customer-owned sinks).

This document describes the production architecture as of February 2023, the operational targets we design against, and the constraints engineers should be aware of when proposing changes.

### 1.1 Design tenets

1. **Throughput over latency.** FLOW is a pipeline, not a transactional system. P99 end-to-end latency targets are in the seconds, not milliseconds.
2. **At-least-once by default, exactly-once where it matters.** Idempotency keys are required on all transformation outputs.
3. **Backpressure is a first-class signal.** Every stage must be able to apply and respect backpressure.
4. **Operational simplicity.** Prefer one well-tuned component over three clever ones.

### 1.2 Availability target

The platform-level availability SLO for NexusFlow is **99.9% (three nines)** measured monthly against the `events_api` ingestion endpoint and the delivery success rate. This corresponds to a monthly error budget of ~43 minutes.

> **Note for account teams:** Some enterprise contracts negotiate availability commitments above the platform SLO. Where those exist, they must be tracked as customer-specific commitments by Customer Success and Legal — they are *not* the platform engineering target. Any proposal to raise the platform SLO requires a formal review with Marcus Webb and Sarah Chen.

---

## 2. High-Level Architecture

```
   [Customer Sources]
          │
          ▼
   ┌──────────────────┐
   │ (1) Ingestion    │  Go, behind ALB
   │     Service      │
   └────────┬─────────┘
            │  (Pulsar topic: flow.raw.<tenant>)
            ▼
   ┌──────────────────┐        ┌────────────────────┐
   │ (2) Transformation│◄──────│ Metadata Store      │
   │     Engine        │       │ (PostgreSQL/Aurora) │
   └────────┬─────────┘        └────────────────────┘
            │  (Pulsar topic: flow.norm.<tenant>)
            ▼
   ┌──────────────────┐
   │ (3) Delivery     │  Python workers
   │     Service      │
   └────────┬─────────┘
            ▼
   [Warehouses / Sinks / InsightLens]
```

Three numbered components, three Pulsar hops, one metadata store of truth. All three components are stateless at the request level; durable state lives in Apache Pulsar (messages, cursors) and PostgreSQL (configuration, schema registry, run history).

---

## 3. Components

### 3.1 Ingestion Service

- **Language:** Go (this is the hot path; we moved it off Python during late-2021 hardening).
- **Deployment:** AWS EKS, multi-AZ in `us-east-1` and `us-west-2`, fronted by an ALB with AWS Shield Standard.
- **Responsibilities:** Auth (HMAC + OAuth2), schema validation against the registry, tenant-level rate limiting, batching, and write-through to Pulsar.
- **Rate limiter:** Token-bucket per tenant, config-driven via the metadata store. Limits are pushed to instances via a config-reload sidecar. Misconfiguration here is the single largest operational risk in the ingestion path — changes go through a two-person review.

### 3.2 Transformation Engine

- **Language:** Python 3.10 (the migration from 2.7 was completed in June 2022 as part of Project Phoenix).
- **Execution model:** Per-tenant worker pools consuming from `flow.raw.<tenant>` Pulsar topics, executing user-defined DAGs of transformation steps (map, filter, enrich, join-with-reference-table).
- **State:** Reference tables and DAG definitions are cached locally and invalidated via Pulsar control topics. Source of truth is the PostgreSQL metadata store.
- **Isolation:** Noisy-neighbor protection via per-tenant consumer concurrency caps and CPU cgroups at the pod level.

### 3.3 Delivery Service

- **Language:** Python 3.10.
- **Responsibilities:** Adapter layer for sinks (Snowflake, BigQuery, Redshift, S3, webhooks, and the internal `lens.ingest` endpoint that feeds InsightLens), retries with exponential backoff, and dead-letter handling.
- **Important coupling:** InsightLens consumes its source data through NexusFlow's delivery service via the internal `events_api` route. An ingestion-side incident in FLOW will surface as data freshness degradation in LENS. This is documented here so SRE incident reviewers do not treat LENS as an independent failure domain.

---

## 4. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Message queue | Apache Pulsar | Migration from Kafka was completed Q2 2023 after the DataCraft acquisition consolidated tooling. |
| Hot-path language | Go | Ingestion service only. |
| Application language | Python 3.10 | Transformation, delivery, control plane. |
| Metadata store | PostgreSQL (Aurora) | Multi-AZ, daily snapshots, 30-day retention. |
| Object storage | S3 (Standard + Glacier) | 90-day hot, 1-year cold per data policy. |
| Orchestration | EKS | Multi-AZ; multi-region active/active for ingestion, active/passive for transformation. |
| Observability |