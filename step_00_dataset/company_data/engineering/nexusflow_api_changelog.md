# NexusFlow API Changelog

**Maintained by:** Yuki Tanaka, NexusFlow Team Lead  
**Last updated:** December 2023

This document records all breaking changes, deprecations, and new endpoints across NexusFlow API versions. Consumers of the API must review breaking changes before upgrading.

---

## v2.1 — October 2023

**Status:** Current stable release

### New endpoints
- `POST /v2/pipelines/{id}/replay` — Replay a pipeline from a specific checkpoint. Enables recovery without full re-ingestion.
- `GET /v2/pipelines/{id}/lineage` — Returns full data lineage graph for a pipeline (source → transform → sink).
- `PATCH /v2/connectors/{id}/rate_limit` — Per-connector rate limit override. Added in response to the August 2023 rate-limiter misconfiguration incident (see postmortem `nexusflow_v21_postmortem.txt`).

### Breaking changes
- `PUT /v2/pipelines/{id}/config` — `rate_limit_rps` field is now **required**. Previously optional with a system-wide default. Callers that omit this field receive HTTP 422.
- `GET /v2/events/batch` — **Deprecated in v2.1, removed in v2.2.** Replaced by `GET /v2/events/stream`. All InsightLens consumers using `events_api` must migrate before the v2.2 cutover date (Q2 2024).

### Bug fixes
- Fixed race condition in `POST /v2/ingest` when multiple concurrent sources write to the same partition key.
- `GET /v2/pipelines` pagination cursor now stable across page boundaries.

---

## v2.0 — March 2023

**Status:** Supported (security patches only after December 2023)

### Overview
v2.0 was the first major release after the DataCraft integration. The DataCraft team contributed the connector framework (v2 connector spec) and the schema registry integration. The Kafka-to-Pulsar migration also landed in this release cycle.

### New endpoints
- `GET /v2/connectors` — Lists all configured connectors with health status.
- `POST /v2/connectors/{id}/test` — Validates connector credentials and connectivity.
- `GET /v2/schema-registry/subjects` — Lists all registered Avro/JSON Schema subjects.
- `POST /v2/schema-registry/subjects/{subject}/versions` — Registers a new schema version.

### Breaking changes from v1
- **Authentication:** v1 used API keys in the `X-API-Key` header. v2.0 requires OAuth2 Bearer tokens via `/v2/auth/token`. All v1 API key integrations **must** be rotated before October 2023 (EOL for v1 auth).
- **Pagination:** v1 used `page` + `per_page`. v2.0 uses cursor-based pagination (`next_cursor`). All v1 clients with list endpoints must update.
- `POST /v1/ingest` — **Removed.** Replaced by `POST /v2/ingest`. v1 ingest endpoint was officially EOL as of March 2023.
- `GET /v1/pipelines/{id}/status` — **Removed.** Replaced by `GET /v2/pipelines/{id}` (status embedded in pipeline object).

### Migration guide
See `engineering/datacraft_migration_complete.txt` for the full DataCraft integration migration notes.

---

## v1.2 — September 2022

**Status:** End-of-Life (EOL March 2023)

### Changes
- Added `GET /v1/pipelines/{id}/metrics` — throughput, lag, error rate.
- Added `POST /v1/pipelines/{id}/pause` and `POST /v1/pipelines/{id}/resume`.
- `POST /v1/ingest` — Added optional `schema_id` field for schema validation at ingestion.
- Rate limit raised from 100 RPS to 500 RPS per API key.

---

## v1.1 — February 2022

**Status:** End-of-Life (EOL March 2023)

### Changes
- First release after Vertexia acquired DataCraft. Stability and compatibility focus.
- `GET /v1/pipelines` — Added `source_type` filter parameter.
- `POST /v1/connectors` — New endpoint for programmatic connector creation (previously config-file only).
- Fixed: `DELETE /v1/pipelines/{id}` incorrectly returned 200 instead of 204.

---

## v1.0 — March 2020

**Status:** End-of-Life (EOL March 2023)

### Initial GA release
- Core endpoints: `POST /v1/ingest`, `GET /v1/pipelines`, `GET /v1/pipelines/{id}/status`.
- Kafka-backed event bus (later migrated to Apache Pulsar in v2.0).
- API key authentication via `X-API-Key` header.
- Simple page-based pagination.

---

## Deprecation schedule

| Feature | Deprecated | Removed | Migration path |
|---------|-----------|---------|---------------|
| v1 API key auth (`X-API-Key`) | v2.0 (Mar 2023) | v2.2 (Q2 2024) | OAuth2 Bearer token |
| `GET /v2/events/batch` | v2.1 (Oct 2023) | v2.2 (Q2 2024) | `GET /v2/events/stream` |
| `rate_limit_rps` optional | v2.0 (Mar 2023) | v2.1 (Oct 2023) | Set explicit value in pipeline config |
| v1 pagination (`page`/`per_page`) | v2.0 (Mar 2023) | v2.2 (Q2 2024) | Cursor-based (`next_cursor`) |

---

## Consumer impact matrix

| Consumer | Current API version | Migration required | Owner |
|----------|--------------------|--------------------|-------|
| InsightLens dashboard | v2.0 (`events_api`) | Migrate to `events/stream` before Q2 2024 | Aisha Johnson |
| DataCraft integration pipeline | v2.1 | None | Felix Wagner |
| PulseConnect pipeline status | v1.2 (`pipeline_status_api`) | Must upgrade to v2.1 | Raj Patel |
| External customer connectors | Mixed v1.1–v2.0 | v1 auth rotation required | Customer Success |
