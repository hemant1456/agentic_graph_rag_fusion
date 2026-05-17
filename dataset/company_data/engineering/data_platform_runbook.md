# Data Platform Team — On-Call Runbook

**Owner:** Priya Nair, Lead, Data Platform Team
**Last reviewed:** August 2023
**Audience:** Data Platform on-call engineers

---

## 1. Service Ownership

The Data Platform Team owns the following production services. All three are critical-tier (Tier 0) and underpin NexusFlow as well as downstream consumers (notably InsightLens, which depends on `events-api` for its analytics ingestion path).

| Service | Description | Repo | Runtime |
|---|---|---|---|
| `data-ingestion-service` | Customer-facing ingest endpoints; writes to Apache Pulsar | `vertexia/data-ingestion` | Go |
| `transformation-engine` | Stream processing, schema enforcement, enrichment | `vertexia/xform-engine` | Go / Python |
| `events-api` | Internal events distribution API; consumed by NexusFlow and InsightLens | `vertexia/events-api` | Go |

All three services run on AWS (us-east-1 primary, us-west-2 standby). ML-adjacent batch jobs run on GCP and are out of scope for this runbook.

---

## 2. On-Call Rotation Policy

- **Rotation length:** Weekly, Monday 10:00 PT handoff.
- **Roster:** 6 engineers from the Data Platform Team. Primary + secondary each week.
- **Schedule:** Maintained in `on_call_schedule_aug2023.csv` in the team's `oncall/` directory. Always consult the current month's CSV — do **not** rely on PagerDuty UI alone, as overrides are reflected in the CSV first.
- **Handoff:** Mandatory 15-minute sync between outgoing and incoming primary. Open incidents, pending changes, and known risks must be reviewed.
- **Compensation:** Per Zara Ahmed's on-call policy (rev. May 2023).

---

## 3. Escalation Procedures

1. **Primary on-call** acknowledges page within 5 minutes.
2. If unacknowledged after 10 minutes, **secondary on-call** is paged automatically.
3. **Severity 1** (customer-impacting outage, SLA risk): page Priya Nair immediately and open an incident channel `#inc-<service>-<date>`.
4. If incident exceeds 30 minutes or impacts Phoenix Corp (99.99% SLA contractual customer), notify Marcus Webb and CC Maya Sharma (Customer Success).
5. Security-related incidents: loop in Daniel Osei (Security Team Lead) without delay.
6. For incidents impacting InsightLens via `events-api`, notify Aisha Johnson.

---

## 4. Common Incident Types

- **Ingest backpressure / rate limiter misconfiguration.** See post-mortem from the 2023-08-14 NexusFlow v2.1 outage (4h 23min, root cause: misconfigured rate limiter in `data-ingestion-service`). Always verify rate limiter config matches the documented baseline before deploying ingest changes.
- **Pulsar broker degradation.** Post-Kafka migration (completed Q2 2023), most queue issues now manifest as broker lag or bookie disk pressure.
- **Schema drift** in `transformation-engine` causing downstream consumer failures.
- **`events-api` latency spikes**, often correlated with InsightLens query load.

---

## 5. Recovery Procedures

### Data Ingestion Service
1. Check rate limiter config: `kubectl get configmap ingest-rl -o yaml`.
2. Compare against baseline in `vertexia/data-ingestion/configs/baseline.yaml`.
3. If anomalous, roll back via `make rollback ENV=prod`.
4. Drain Pulsar producer connections before restart.

### Transformation Engine
1. Inspect DLQ topic `xform.dlq`.
2. If schema validation failures dominate, pause affected pipeline and notify the producing team.
3. Replay from cold storage (S3 Glacier, 1-year retention) if reprocessing is required — note thaw lead time.

### Events API
1. Check downstream consumer health (NexusFlow consumers, InsightLens ingest).
2. Scale horizontally via HPA override; cap at 40 replicas without Marcus Webb approval.
3. If circuit breakers tripping, freeze deploys and open Sev-2.

---

## 6. References

- `on_call_schedule_aug2023.csv` — current rotation
- `postmortems/2023-08-14-nexusflow-v2.1.md`
- NexusFlow architecture doc (target 99.9% uptime; note Phoenix Corp contractual SLA is 99.99%)
- Logs: 30-day retention; hot data 90 days, cold 1 year (S3 Glacier)