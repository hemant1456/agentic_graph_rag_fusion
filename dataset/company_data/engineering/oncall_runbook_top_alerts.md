# Vertexia On-Call Runbook — Top Production Alerts

Maintained by: Yuki Tanaka, NexusFlow Team Lead
Last revised: November 2023

This runbook covers the 8 most frequent production alerts that the on-call engineer may encounter. Each section is self-contained: alert name, severity, first responder action, owner, rollback path, and runbook ID. Do not page across alerts — use the per-section escalation owner.

## NexusFlow ingest_lag_seconds > 300

**Severity:** P2
**First action:** Check Pulsar consumer offsets on producer-1 cluster via `pulsar-admin topics stats`. Confirm no zombie consumer holding subscriptions.
**Owner:** Felix Wagner
**Rollback:** Pause downstream consumers, replay from last checkpoint with `nexusflow-cli pipelines replay --from-checkpoint`. Notify InsightLens team since dashboards depend on this stream.
**Runbook ID:** RB-001

## InsightLens query_p99 > 5000ms sustained 10 minutes

**Severity:** P1
**First action:** Check ClickHouse cluster shard distribution; look for hot shard via `system.metrics`. Inspect long-running queries with `SELECT * FROM system.processes`.
**Owner:** Aisha Johnson
**Rollback:** Failover to read replica via the `dashboard-failover` script. Cordon affected shard and rebalance during low-traffic window.
**Runbook ID:** RB-002

## PulseConnect webhook_delivery_failure_rate > 5%

**Severity:** P1
**First action:** Check SendGrid quota dashboard and the Twilio API status page. Inspect outbound retry queue depth in PulseConnect admin UI.
**Owner:** Raj Patel
**Rollback:** Switch to email-only mode via feature flag `pulse_sms_off`. Notify CSM team; affected customers will see only email notifications until SMS is restored.
**Runbook ID:** RB-003

## Kafka offset_lag > 1000000 on events.unified.v1

**Severity:** P2
**First action:** Scale consumer group from 4 to 8 workers using the `events-unified-consumer` Helm chart values override. Confirm no schema-validation errors are causing dead-lettering.
**Owner:** Kenji Ito
**Rollback:** N/A — forward-only recovery. If lag does not drain within 30 minutes, escalate to Priya Nair.
**Runbook ID:** RB-004

## AWS RDS cpu_utilization > 90% sustained 15 minutes

**Severity:** P2
**First action:** Identify long-running queries via `pg_stat_activity`. Check if a recent migration is still holding ACCESS EXCLUSIVE locks. Confirm autovacuum is not stalled.
**Owner:** Priya Nair
**Rollback:** Failover to standby; rollback application via `terraform apply -target=rds_primary_swap`. Block the offending query at the connection-pool layer if identified.
**Runbook ID:** RB-005

## Okta authentication_failure_rate > 2% over 5 minutes

**Severity:** P1
**First action:** Verify Okta status page; check whether the spike is regional. Inspect WebAuthn vs. password breakdown in Okta System Log.
**Owner:** Daniel Osei
**Rollback:** Enable break-glass local auth via the emergency override script. Notify Security Team and start an incident channel within 5 minutes.
**Runbook ID:** RB-006

## Stripe webhook signature_verification_failure spike

**Severity:** P2
**First action:** Verify the webhook signing secret matches between Stripe dashboard and the billing service's secret store. Check for clock skew on billing pods.
**Owner:** Aisha Johnson
**Rollback:** Disable the failing webhook endpoint in Stripe dashboard; re-enable after secret rotation. Replay missed events from Stripe's events API.
**Runbook ID:** RB-007

## Datadog ingestion_billing_alert > 80% of monthly cap

**Severity:** P3
**First action:** Identify the high-volume custom metric via Datadog Usage Attribution. Confirm whether a recent deploy introduced a high-cardinality tag.
**Owner:** Marcus Webb
**Rollback:** Drop the offending tag via the platform `metric_filter` config; revert the offending deploy if cost overrun is severe.
**Runbook ID:** RB-008
