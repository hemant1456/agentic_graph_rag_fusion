# Vertexia Vendor Data Processing Matrix

Maintained by: Daniel Osei, Security & Compliance Lead
Last reviewed: October 2023

This matrix records data-handling terms for every third-party processor that receives Vertexia customer data. Each vendor section is self-contained and authoritative; DO NOT cross-reference terms from one section to another, as audit rights, retention, and sub-processor lists vary materially across vendors.

## Snowflake

**Data categories:** Aggregated analytics events, hashed customer IDs (no raw PII).
**Retention:** 18 months hot storage, then archived 36 months cold tier with auto-purge after 54 months total.
**Sub-processors:** AWS regions us-east-1 and eu-west-1 only. No transfer to Asia-Pacific permitted.
**SCC clause version:** EU SCC 2021/914 Module 2 (controller-to-processor) with UK Addendum.
**Audit cadence:** Annual SOC 2 Type II review delivered to Vertexia legal; ad-hoc penetration test rights with 30-day notice.
**Processor ID:** VENDOR-PROC-01

## SendGrid (Twilio)

**Data categories:** Email addresses, message content metadata (subject lines and timestamps only; bodies are not stored beyond delivery).
**Retention:** 90 days, then hard-deleted from all sub-processor storage.
**Sub-processors:** AWS us-east-1 (primary), GCP us-central1 (failover only).
**SCC clause version:** EU SCC 2021/914 Module 2 (controller-to-processor).
**Audit cadence:** SOC 2 Type II review only. No on-site audit rights; pen-test rights are explicitly excluded by contract.
**Processor ID:** VENDOR-PROC-02

## Datadog

**Data categories:** Application metrics, request traces (PII fields stripped by tracer config), log samples (10% sampling, scrubbed for emails/tokens).
**Retention:** 15 months for traces and logs, 6 months for raw metric dimensions, 25 months for rolled-up time-series.
**Sub-processors:** AWS us-east-1, eu-west-1, ap-southeast-1.
**SCC clause version:** EU SCC 2021/914 Module 1 and Module 2 (joint controllership for metric data only).
**Audit cadence:** Annual external penetration test report delivered to Vertexia; on-demand audit rights for any major Datadog incident (defined in MSA §7.4).
**Processor ID:** VENDOR-PROC-03

## Stripe

**Data categories:** Customer billing names, billing addresses, payment tokens (PCI scope is handled entirely by Stripe; raw card data never reaches Vertexia).
**Retention:** Indefinite billing legal hold for transaction records; tokenized card references are deleted 90 days after a customer's account closure.
**Sub-processors:** AWS us-east-1 (primary processing), GCP us-central1 (fraud-ML inference), TransUnion (KYC checks for B2B accounts only).
**SCC clause version:** EU SCC 2021/914 Module 2 with separate intra-group transfer agreement.
**Audit cadence:** PCI DSS Level 1 attestation reviewed annually; quarterly fraud-model bias review delivered to Vertexia compliance.
**Processor ID:** VENDOR-PROC-04

## HubSpot

**Data categories:** Lead names, business email addresses, CRM activity logs (call notes, deal stages).
**Retention:** 24 months post-disqualification, then anonymized to lead-stage statistics.
**Sub-processors:** AWS us-east-1 only (no EU sub-processor — affects cross-border eligibility).
**SCC clause version:** EU SCC 2021/914 Module 2 and Module 3 (processor-to-processor for HubSpot's own integrations).
**Audit cadence:** Annual SOC 2 Type II review only. Pen-test rights are not granted.
**Processor ID:** VENDOR-PROC-05
