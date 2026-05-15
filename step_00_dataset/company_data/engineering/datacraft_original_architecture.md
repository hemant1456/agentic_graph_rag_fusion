> **ARCHIVED — See datacraft_migration_complete.txt for current state**
> *This document reflects DataCraft's architecture prior to the Vertexia acquisition (closed 2022-01-15). Retained for historical reference.*

---

# DataCraft Platform Architecture

**Author:** Felix Wagner
**Last updated:** November 2021
**Status:** Living document (we'll fix the diagrams when we get a chance)

## Overview

DataCraft is a real-time ETL platform built for teams who outgrew Fivetran but don't want to babysit Airflow. Our north star: get a customer's first pipeline running in under 30 minutes, end-to-end.

We're 12 people. The architecture reflects that — we optimize for "two engineers can hold the whole system in their head" over theoretical purity.

## High-level shape

```
[Source connectors] → [Kafka] → [DataWeave workers] → [Sinks]
                         ↓
                    [Spark batch jobs]
                         ↓
                    [Warehouse / S3]
```

Three moving parts. That's it. Anything more and we'd need a platform team we can't afford.

## Event streaming: Apache Kafka

Kafka is the spine. Every source connector (Postgres CDC, Stripe, Segment, custom webhooks) writes to a topic. Every sink reads from a topic. DataWeave workers sit in the middle doing transforms.

We run a 3-broker MSK cluster in us-east-1, replication factor 3, with `min.insync.replicas=2`. Topics are partitioned by tenant ID so we get natural per-customer isolation and parallelism.

Why Kafka and not Kinesis / Pulsar / NATS? Honestly, because we know it. Felix and Anu spent four years on it at the last gig. The "best tool" is the one you can debug at 2am.

## Batch processing: Apache Spark

Streaming gets you 80% of customer use cases. The other 20% — historical backfills, daily aggregations, the "rebuild this whole table from scratch" button — is Spark on EMR, triggered by Airflow.

Spark jobs read directly from S3 (we tee every Kafka topic to S3 via Kafka Connect) rather than from Kafka itself. This keeps batch and streaming decoupled and means a bad Spark job can't back-pressure the live pipeline.

## DataWeave: our Python ETL framework

DataWeave is the part we actually wrote. It's a Python framework (3.9+) that lets a customer define a transform as a decorated function:

```python
@dataweave.transform(source="stripe.charges", sink="warehouse.revenue")
def normalize_charge(event):
    ...
```

The framework handles Kafka consumer groups, schema validation (via Avro + Schema Registry), retries with exponential backoff, dead-letter routing, and metric emission. Workers run as containers on ECS Fargate, autoscaled on consumer lag.

DataWeave is ~14k lines of Python. It is not glamorous. It is the reason customers stay.

## What we're not doing (yet)

- Multi-region. One region, one cluster, one Kafka.
- Exactly-once across sinks. We're at-least-once with idempotent sink keys, and we tell customers that honestly.
- A UI worth screenshotting. The CLI is the product.

— Felix