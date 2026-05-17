# Project Phoenix: NexusFlow Python 2 → Python 3 Migration

**Status:** COMPLETED ✅
**Start Date:** January 10, 2022
**Completion Date:** June 30, 2022
**Project Lead:** Yuki Tanaka (Lead, NexusFlow Team)
**Engineering Sponsor:** Sarah Chen (then VP Engineering)

---

## Background

Python 2 reached official End-of-Life on January 1, 2020. By late 2021, the NexusFlow codebase — which had been built up since pre-GA development in 2019 — was still running on Python 2.7 in several critical service boundaries, primarily the ingestion workers and the legacy connector framework inherited from the original FLOW prototype.

Continued reliance on Python 2 was creating compounding risk:

- Several upstream dependencies had dropped Python 2 support, forcing us to pin to older, increasingly insecure versions.
- New hires were frustrated with the dual-runtime developer environment.
- Security patches from CPython were no longer flowing to 2.7.
- The Data Platform team flagged Py2 as a blocker for the upcoming Pulsar evaluation work.

Project Phoenix was greenlit in December 2021 and officially kicked off January 10, 2022, with a 6-month target timeline.

## Scope

- **Codebase size:** ~187,000 lines of Python across the NexusFlow monorepo
- **External dependencies to update or replace:** 43 libraries
- **Services in scope:** ingestion-api, connector-framework, scheduler, events_api, transform-runner, and the CLI tooling (`nfctl`)
- **Out of scope:** Go hot-path services (already on Go 1.17), TypeScript frontend, InsightLens-specific consumers

## Approach

Yuki's team adopted a phased strategy rather than a big-bang cutover:

1. **Phase 1 (Jan–Feb 2022):** `futurize`-driven mechanical conversion, dependency audit, CI matrix updated to test both Py2.7 and Py3.9 in parallel.
2. **Phase 2 (Mar–Apr 2022):** Service-by-service cutover behind feature flags, beginning with the lowest-risk batch transforms and ending with ingestion-api.
3. **Phase 3 (May–Jun 2022):** Removal of Py2 compatibility shims (`six`, `__future__` imports), cleanup of conditional code paths, deprecation of the dual-runtime Docker images.

The team allocated roughly 2.5 FTEs continuously, with surge support from Priya Nair's Data Platform team during the ingestion-api cutover in April.

## Challenges

- **Unicode handling in connectors:** A long tail of customer-supplied connectors assumed `str`/`bytes` interchangeability. Resolved by introducing an explicit decode layer at the connector boundary.
- **Dependency replacements:** 6 libraries had no maintained Python 3 equivalent and required forks or rewrites. The most painful was an internal fork of the legacy SOAP adapter used by two enterprise customers.
- **Test coverage gaps:** Initial coverage on the scheduler was ~58%, requiring a parallel effort to write characterization tests before cutover. Coverage at project end: 81%.
- **Timeline pressure:** The June 30 deadline was tight; we deferred two non-blocking refactors (connector-framework type annotations, `nfctl` argparse cleanup) into Q3 backlog tickets.

## Results

- ✅ 100% of in-scope code migrated to Python 3.9
- ✅ All test suites passing on Py3-only CI as of June 28, 2022
- ✅ Py2.7 runtime fully removed from production by June 30, 2022
- ✅ **8% performance improvement** measured on the ingestion-api p50 latency post-migration, attributed primarily to the newer `asyncio` implementation and dict ordering improvements
- ✅ ~14% reduction in container image sizes after shim removal

## Post-Mortem & Follow-ups

A retro was held July 12, 2022. Key learnings logged in the engineering wiki under `retros/2022-07-phoenix`. Deferred items tracked in JIRA epic `NF-2841`. No production incidents were attributed to the migration during or after the cutover window.

---

*Document owner: Yuki Tanaka. Last updated: July 14, 2022.*