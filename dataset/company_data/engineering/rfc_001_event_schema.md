# RFC-001: Standardized Event Schema for Cross-Product Analytics

**Author:** Priya Nair, Lead, Data Platform Team
**Status:** ACCEPTED
**Created:** 2023-09-18
**Last Updated:** 2023-09-29
**Reviewers:** Marcus Webb, Sarah Chen, Yuki Tanaka, Aisha Johnson, Raj Patel, Felix Wagner
**Target Implementation:** Q4 2023

---

## 1. Summary

This RFC proposes a unified event schema to be adopted across NexusFlow, InsightLens, and PulseConnect. Today, each product emits events with subtly different shapes, naming conventions, and timestamp formats. As we scale toward $20M+ ARR and build cross-product analytics features promised in the Nadia Kim / Lisa Torres joint roadmap, the lack of a shared schema has become a blocker.

## 2. Background

Each product team built its own event pipeline at different points in the company's history:

- **NexusFlow** (GA March 2020) emits events through `events_api`, currently the most mature pipeline. Format is loosely based on the original ETL schema from 2020.
- **InsightLens** (GA September 2021) emits internal telemetry events with snake_case keys and millisecond UNIX timestamps.
- **PulseConnect** (GA April 2022) emits events using camelCase keys, ISO-8601 timestamps, and a separate `meta` envelope.

The Data Platform team has fielded 14 separate ad-hoc requests in 2023 alone for "join the three event streams" — typically from Maya Sharma's CS analytics team or from Ben Carter's RevOps group. Each one currently requires a bespoke transformation job.

Additionally, the post-incident review for the August 14, 2023 NexusFlow outage (4h 23min, rate limiter misconfiguration) surfaced a related problem: InsightLens dashboards went dark for the duration of the incident because **InsightLens consumes NexusFlow's `events_api` as its primary upstream source for product-usage analytics**. This dependency is not currently documented anywhere outside tribal knowledge, and the schema mismatch between the two products requires a translation layer that itself is a source of bugs.

## 3. Proposal

Adopt a single canonical event envelope across all three products.

### 3.1 Mandatory fields

| Field | Type | Notes |
|---|---|---|
| `event_id` | UUIDv7 | Globally unique, time-ordered |
| `timestamp` | RFC 3339 string | UTC, microsecond precision |
| `product_id` | enum | `nexusflow` \| `insightlens` \| `pulseconnect` |
| `user_id` | string | Tenant-scoped, hashed for PII |
| `event_type` | string | Dotted namespace, e.g. `pipeline.run.completed` |
| `properties` | object | Event-specific payload |

### 3.2 Optional extension fields

`session_id`, `org_id`, `source_ip_hash`, `trace_id`, `schema_version`, `experiment_assignments`.

Unknown fields outside the envelope will be rejected at the ingestion boundary. `schema_version` defaults to `"1.0"`.

### 3.3 Transport

Events will continue to flow through Apache Pulsar (post-Kafka migration, completed Q2 2023). Each product will publish to a product-specific topic; a new `events.unified.v1` topic will be populated by a fan-in subscription owned by the Data Platform team.

### 3.4 InsightLens-NexusFlow consumption

As part of this RFC, the existing InsightLens dependency on NexusFlow's `events_api` will be formalized. InsightLens will subscribe to `events.unified.v1` (filtered by `product_id = nexusflow`) rather than calling `events_api` directly. This decouples the two products at the transport layer while preserving the analytical dependency.

## 4. Migration Plan

- **Oct 2023:** Schema published, Pulsar topics provisioned, validation library shipped (Python + Go).
- **Nov 2023:** NexusFlow (Yuki Tanaka) and PulseConnect (Raj Patel) emit dual-write.
- **Dec 2023:** InsightLens (Aisha Johnson) cuts over to `events.unified.v1`. Legacy `events_api` direct consumption deprecated.
- **Q1 2024:** Legacy schemas removed.

## 5. Decision

**ACCEPTED** by Sarah Chen and Marcus Webb on 2023-09-29. Priya Nair owns delivery; Kenji Ito will lead the Pulsar topic provisioning work.