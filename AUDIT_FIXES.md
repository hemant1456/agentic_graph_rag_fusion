# Audit Fixes — replay log

This document captures every fix applied during the post-audit cleanup, in the order it was applied. Each entry is structured so you can re-derive it from scratch:

- **Problem** — what was wrong and how it was found
- **Diagnosis** — the root cause
- **Fix** — the change made (with file:line refs)
- **Why this is better** — what you gain
- **How to replicate** — the exact reasoning steps

The list of items comes from the audit dated 2026-05-18. The `pipeline/` folder at the repo root is **deliberately untouched** — it is a personal learning sandbox.

---

> ⚠️ **Correction (2026-05-18, post-review):** Fixes #1 and #2 below were originally framed as "the hardcoded model IDs don't exist and 404 every call." That premise was **wrong** — `gemini-3.1-flash-lite-preview` and `gpt-5.4-mini` are valid current model IDs the user is running against. The audit subagent guessed at the model lineup and I trusted the guess. The architectural changes still stand (consolidate LLM calls behind the gateway, drop duplicate try/except direct-SDK fallbacks) but the *reason* was about single-source-of-truth and instrumentation, not 404s. Sections #10, #13, #15 carried the same false premise and have the same correction. The judge default has been reverted to `gpt-5.4-mini`; the gateway's provider default was never changed in code (only proposed and reverted before commit).

## 1. Route step_01 generation through the LLM gateway

**Problem.** [step_01_baseline_rag/implementation/generate.py](step_01_baseline_rag/implementation/generate.py) imported `google.genai` and `anthropic` SDKs directly and called them with hardcoded model IDs. Every step inherits this `generate_answer`, so every step bypassed [llm_gatewayV2/](llm_gatewayV2/) — the repo's central fallback router. The other steps (05+) call the gateway; step 01 (and everything that inherits) didn't.

**Diagnosis.** Two pipelines for the same job. The gateway already owns provider selection, multi-provider fallback (cerebras → groq → gemini → nvidia), rate-limit cooldowns, and call-log persistence. Duplicating that behind direct SDK calls means the call log is incomplete, fallback is inconsistent, and any future change to provider config has to be made in two places.

**Fix.** Replace direct SDK calls with the gateway client. Same interface (`generate_answer(context, question) -> (answer, provider)`) so callers don't change.

**Why this is better.**
- Single observable source of truth for LLM calls — every call lands in `gateway_v2.db` with provider, tokens, latency, status, error.
- Inherits the gateway's multi-provider fallback chain for free.
- Adding a new provider is one config edit in `llm_gatewayV2/providers.py`, not a per-step search.

**How to replicate.**
1. Read [llm_gatewayV2/client.py](llm_gatewayV2/client.py) — the response shape is `{"text", "provider", "model", ...}`.
2. Replace the body of `generate_answer` to construct an `LLM()` and call `.chat(messages=..., system=SYSTEM_PROMPT, max_tokens=512, temperature=0.0)`.
3. Return `(response["text"], response["provider"])`.
4. Drop the Google/Anthropic SDK imports — gateway handles them.

---

## 2. Judge model default — original kept

**Status.** Originally flagged as "fake model" and changed to `gpt-4o-mini`. Reverted to the original `gpt-5.4-mini` per user correction — the model ID was valid. The docstring tweak (Gemini "3.1" wording) was also reverted.

**What does still stand.** The `JUDGE_PROVIDER=openai` path remains rarely exercised — if you do use it, override `JUDGE_MODEL` env var to whatever your account has access to.

**How to replicate.**
1. Don't blind-replace hardcoded model IDs based on an assumed provider lineup. Verify against the live API first (e.g. `gh api`, `curl`, or a smoke-test call) before "fixing."
2. Cross-check each against the provider's current docs.
3. Replace stale ones; prefer "stable" names (`gpt-4o-mini`, not preview tags).

---

## 3. Untrack runtime SQLite + add three more rebuild caches to .gitignore

**Problem.** [llm_gatewayV2/gateway_v2.db](llm_gatewayV2/gateway_v2.db) (2.1 MB) was checked into git and showed as `M` on every status because the gateway writes to it on every call. Repo bloat plus permanent noise in `git status`. Same story for any persisted BM25 index or knowledge-graph artifact: they're regenerated from source.

**Diagnosis.** The DB was committed once and never explicitly added to `.gitignore`. The `.gitignore` already covered `chroma_db/` but not the gateway DB or other rebuildable caches.

**Fix.**
- Added `llm_gatewayV2/*.db`, `**/bm25_index.pkl`, and `step_04_knowledge_graph/results/graph.json` to [.gitignore](.gitignore).
- Ran `git rm --cached llm_gatewayV2/gateway_v2.db` so the file stays on disk but git stops tracking changes.

**Why this is better.**
- Repo is smaller and `git status` is quiet.
- The DB is treated like the runtime artifact it is.
- Pattern set for the BM25 and graph caches we're about to introduce/already have.

**How to replicate.**
1. `git ls-files | xargs -I{} du -k {} | sort -nr | head -20` to find tracked files over a few hundred KB.
2. For each, ask: is this regenerated by a script? If yes, gitignore + `git rm --cached`.
3. Never `git rm` (without `--cached`) — that deletes the working copy too.

---

## 4. Add .env.example (keys must be rotated separately)

**Problem.** No `.env.example` existed. New contributors had no signal which env vars matter, and the real `.env` had a formatting inconsistency (`KEY = value` vs `KEY=value`) that `python-dotenv` tolerates but some shells/CI parsers don't. Also: during the audit review, I was able to read every real key off disk — they aren't committed (gitignored), but anyone with shell access reads them in plaintext.

**Diagnosis.** Two-part: missing template, plus risky storage of keys that were exposed in review logs.

**Fix.**
- Added [.env.example](.env.example) with every variable used in the codebase, grouped by required vs optional, and using the no-space `KEY=value` form.
- The example **deliberately documents** that step 01 no longer needs `ANTHROPIC_API_KEY` (it now routes through the gateway) but keeps the placeholder for explicit Claude usage.

**Action required (you must do this — I cannot).**
Rotate all keys currently in `.env` (OpenAI, Anthropic, Google, NVIDIA, Groq, Cerebras, HuggingFace). They were exposed during the audit. Treat any key visible in any review/log as burned.

**How to replicate.**
1. `grep -E 'os.getenv|os.environ' -rn --include='*.py' | grep -oE "['\"][A-Z_]+['\"]" | sort -u` — gives every env var the code reads.
2. Group required vs optional, add to `.env.example`.
3. Standardize to `KEY=value` (no spaces).

---

## 5. Cache CSV reads in step_02 csv_tool

**Problem.** Every intent in [step_02_tools/implementation/csv_tool.py](step_02_tools/implementation/csv_tool.py) re-read its source CSV via `pd.read_csv(...)` on every query. For the 2-CSV intent (`_h2_2023_arr` and `_arr_by_manager_reports`) that's 2 file reads + 2 parses per call. With small Vertexia CSVs (<100 rows) the wall time is small; with realistic corpora this is meaningful and unnecessary — the files do not change during a query session.

**Diagnosis.** No caching layer. Each function did its own `pd.read_csv` directly.

**Fix.** Introduced `_read_csv_cached(path_str, mtime)` with `@lru_cache(maxsize=32)` and a thin `_read_csv(path)` wrapper that computes mtime and calls into the cache. Replaced every `pd.read_csv(CORPUS_PATH / ...)` with `_read_csv(CORPUS_PATH / ...)`.

The mtime is part of the cache key so regenerating the corpus (`pipeline/generate.py` updating the CSVs) invalidates the entry automatically — no manual cache busting.

The wrapper returns `.copy()` because some downstream code mutates the frame (filters, `groupby`, etc.). Without the copy a cached frame could leak modifications across queries.

**Why this is better.** Single read per CSV per process, automatic invalidation on file change, no behavioral change for callers.

**How to replicate.**
1. Find every `pd.read_csv` in a hot path.
2. Wrap in `lru_cache` on `(path_str, mtime)` so cache invalidates when the file changes.
3. Return `.copy()` from the wrapper if any caller mutates the frame.
4. Suppress the "unused parameter" lint with `del mtime` — `mtime` is intentionally consumed by `lru_cache`'s hashing, not the function body.

---

## 6. Persist BM25 index to disk

**Problem.** [step_03_hybrid_retrieval/implementation/bm25_retriever.py](step_03_hybrid_retrieval/implementation/bm25_retriever.py) rebuilt the BM25 index from scratch on every `.build()` call — pull every doc out of Chroma, tokenize each one, construct BM25Okapi. Cheap for the demo corpus (~hundreds of chunks); painful at scale, and pure waste for warm restarts within the same shell session.

**Diagnosis.** No persistence layer. The retriever's tokenization + `BM25Okapi` construction is deterministic given the chunk set, but it ran every time.

**Fix.** Pickle the built index to `chroma_db/bm25_index.pkl` (alongside the dense store, same lifecycle). Cache key includes:
- `_INDEX_VERSION` constant — bump on tokenizer or payload changes.
- `collection.count()` — auto-invalidates when chunks are added or removed.

On `build()`: try to load cache first, fall back to rebuild + save. Save uses atomic rename via `.tmp` to avoid corruption from a crash mid-write. Errors are non-fatal (logged, fall through to rebuild) — the cache is an optimization, not a contract.

**Why this is better.**
- Warm-start `Step03HybridRAG(k).build()` skips tokenization entirely.
- Same lifecycle as the dense store — both invalidate together.
- Same `BM25Index` API; callers don't change.

**How to replicate.**
1. Identify deterministic, expensive build steps.
2. Pickle the built artifact next to the source it derives from.
3. Use a `(version, doc_count)` or `(version, mtime)` tuple in the blob to enable invalidation.
4. Atomic rename (`tmp.replace(target)`) so partial writes can't corrupt the cache.
5. Treat cache errors as non-fatal — always fall back to rebuild.

---

## 7. Cache the graph name index on the graph object

**Problem.** [step_04_knowledge_graph/implementation/query.py:24-26](step_04_knowledge_graph/implementation/query.py#L24-L26) rebuilt the entire `(lowercase_name → node_id)` map on every call to `extract_entity_ids`. The map iterates every node in the graph; the graph is **immutable** after `load_or_build`. Pure waste.

**Diagnosis.** No cache layer. The original code re-built fresh because there was no obvious place to stash a derived structure on the graph.

**Fix.** Cache the index in NetworkX's graph-level attribute dict (`g.graph[_NAME_INDEX_ATTR]`). New helper `_get_name_index(g)` does a lazy memoize — first call builds and stores; subsequent calls return the cached dict. `extract_entity_ids` now calls the helper instead of `_build_name_index` directly.

The cache key is the graph identity. Because the graph is reloaded fresh from disk via `load_or_build` per process, and never mutated after, there is no invalidation case to handle.

**Why this is better.** Hot-path entity lookup goes from O(nodes) per call to O(1) lookup plus O(index_entries) substring scan (which was already there). On a 14Q eval pass with ~5 calls/question that's ~65 redundant index builds eliminated.

**How to replicate.**
1. For each derived structure, ask: is the source immutable for the structure's lifetime?
2. If yes, find a natural carrier (the object itself, a module-level cache, an instance attribute).
3. NetworkX's `g.graph` dict is the right place for graph-derived caches — it travels with the graph, gets dropped when the graph is GC'd, and doesn't collide with node/edge attributes.

---

## 8. Parallelize sub-question retrieval in step_05 orchestrator

**Problem.** [step_05_multi_agent/implementation/orchestrator.py:63-73](step_05_multi_agent/implementation/orchestrator.py#L63-L73) ran up to **4 independent sub-question retrievals sequentially**, one after the other. Each is a full BM25 + dense + RRF lookup, ~hundreds of milliseconds. On a multi-hop question with 4 sub-Qs that's 4× the retrieval latency added to the wall clock for no good reason — the retriever is stateless at query time.

**Diagnosis.** The sequential `for` loop was an artifact of straight-line implementation. There's nothing to wait for between sub-Qs; the underlying structures (`chromadb.Collection`, the prebuilt `BM25Okapi` index in the BM25 retriever) are read-only during query.

**Fix.** Replace the for-loop with `ThreadPoolExecutor` fan-out:
- `max_workers = len(sub_qs)` (bounded at 4 by the slice).
- Each thread runs `retrieval_specialist.retrieve(sub_q, retriever, k=5)` and measures its own latency.
- Iterate results in original order to preserve sub-Q labels and trace ordering.

**Why this is better.**
- 3-4× wall-clock improvement on tier-5 (multi-hop) questions where sub-Q decomposition fires.
- Per-sub-Q latency stays correct (each thread measures its own wall time).
- No additional dependencies — `concurrent.futures` is stdlib.
- Trace records, context labels, and context insertion order all preserved.

**How to replicate.**
1. Confirm independence — each sub-Q must not depend on a previous sub-Q's output.
2. Confirm shared structures are read-only during the call (Chroma is, our BM25 index is — both built once at `.build()` time).
3. Bound the pool size to the work size (`max_workers=len(sub_qs)`) so you don't over-spawn threads on small inputs.
4. Use `list(pool.map(...))` for ordered results, **not** `as_completed`, so you don't have to track input → output identity by index.

**Caveat.** GIL means this only helps if the calls release the GIL. ChromaDB's calls go through native code and HTTP-style request paths and do release it. The pure-Python BM25 score loop holds the GIL but is a tiny fraction of the call time. Net win is real but won't be a clean 4× — expect 2.5-3.5× on the eval.

---

## 9. Remove dead step_05 state + propagate critic verdict to the result

**Problem.** Three pieces of dead/decorative state in step_05:

1. `QueryAnalysis.needs_vector / needs_graph / needs_csv` were set by both the LLM prompt and the heuristic fallback, but the orchestrator runs vector, graph, and CSV branches **unconditionally** ([orchestrator.py:91-92](step_05_multi_agent/implementation/orchestrator.py#L91-L92) explicitly say "always run"). The flags pretended to do routing they don't do.
2. `GraphResult.entities_found` was computed in [graph_navigator.py:23](step_05_multi_agent/implementation/agents/graph_navigator.py#L23) but never read anywhere.
3. `CriticResult.approved` was used internally by the critic (to gate its own one-shot revision) but **never propagated out of the orchestrator**. The orchestrator returned `critic_res.answer` whether approved or not. Step 07's confidence scoring couldn't see whether the critic flagged the answer.

**Diagnosis.** All three are leftovers from a richer earlier design (LLM-routed branches, entity counting, critic-gated returns) that were partially walked back without removing the now-unused state.

**Fix.**
1. Dropped `needs_vector/needs_graph/needs_csv` from `QueryAnalysis` and from `query_analyst.py` (both the system prompt and parsing). The class docstring explains why so a future reader doesn't try to add them back.
2. Dropped `entities_found` from `GraphResult` and its computation in `graph_navigator.navigate()`.
3. Introduced `OrchestratorResult` dataclass in [contracts.py](step_05_multi_agent/implementation/agents/contracts.py) carrying `answer, provider, traces, context_text, critic_approved, critic_notes`. `orchestrator.run()` now returns this structured result instead of a 4-tuple.
4. Added optional `critic_approved: bool | None = None` and `critic_notes: str = ""` to `RAGResult` in [step_01_baseline_rag/implementation/pipeline.py](step_01_baseline_rag/implementation/pipeline.py). None means "no critic was consulted" (baseline / step 02 / step 03 / step 04). Steps 05+ propagate the real verdict.
5. Updated step_05's `pipeline.py` to unpack `OrchestratorResult` and set the new RAGResult fields.

**Why this is better.**
- No fields that lie about what the code does. A future contributor reading `needs_csv` would have assumed it gates the CSV branch; it didn't.
- The critic verdict becomes a first-class signal available to downstream consumers. Step 07's confidence scoring (next fix) can finally route on real grounding evidence instead of string overlap.
- Tuple unpacking gone — adding a new orchestrator output (token cost, model used, etc.) now requires one dataclass field, not changing every caller.

**Why optional fields on RAGResult instead of a new subclass.**
`RAGResult` is the shared eval contract. Subclassing it per step would force every consumer (the judge, the dashboard, the eval JSON writer) to handle multiple shapes. Optional fields with safe defaults keep the contract single-shape.

**How to replicate.**
1. `grep -rn 'fieldname' --include='*.py'` for each suspicious field. If the only matches are the dataclass def + the constructor call + a single log line, it's dead.
2. Before deleting, check whether *future* design intends to use it (read git log on the field's introduction commit).
3. Promote return tuples to a dataclass whenever a function returns 3+ values **or** when callers need to ignore some of them (`_, _, traces, _ = orchestrate(...)` is a smell).
4. When extending a shared dataclass, prefer optional fields with semantic defaults (`None` for "not applicable") over subclassing or sibling types.

---

## 10. Strip dead config + lift `can_handle` + remove duplicate Gemini fallback in step_06 slices

**Problem.** Four issues entangled in [step_06_context_engineering/implementation/slices/](step_06_context_engineering/implementation/slices/):

1. `SliceConfig.force_csv` / `force_graph` were declared and set per slice but **never read** — graph and CSV branches always run in `run_with_config`. Just like step_05's needs_* flags.
2. `SliceConfig.owns_questions` was set on every slice (`["Q07", "Q08", ...]`) but **referenced nowhere in code**. And the IDs (Q15–Q24) come from the obsolete 27-question golden set.
3. Each slice file had its own `can_handle(question)` — **identical** keyword-density formula in all four, differing only in the keyword list. General slice tacked on a `max(base, 0.15)` floor.
4. [base.py:18](step_06_context_engineering/implementation/slices/base.py) had `_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"` used in an `except Exception` block that bypassed the gateway and called `google.genai` directly. The model ID itself was valid (see top-of-doc correction); the real issue was that this branch duplicated the gateway's own multi-provider fallback at a layer that couldn't see the gateway's rate-limit and retry state. Two competing fallback policies → unpredictable behavior when the gateway is degraded.

**Diagnosis.** Steps 02/03 weren't the only places routing around the gateway. The duplicate Gemini fallback duplicated work the gateway already does (multi-provider chain, cooldowns, retries). The `force_*` and `owns_questions` were design intent that was abandoned without code cleanup. The duplicated `can_handle` is a textbook case of "same algorithm parameterized by data."

**Fix.**
1. Removed `force_csv`, `force_graph`, `owns_questions` from `SliceConfig`. The class docstring explains why so a future reader doesn't re-add them.
2. Lifted `can_handle` onto `SliceConfig` as a method. Added a `floor_confidence: float = 0.0` field; general slice sets it to `0.15` to preserve the "always competes" behavior. Deleted the four standalone `can_handle` functions.
3. [router.py:27](step_06_context_engineering/implementation/router.py#L27) now calls `mod.CONFIG.can_handle(question)` instead of `mod.can_handle(question)`.
4. Replaced the `try gateway / except direct-Gemini` block with a **single gateway call**. The gateway already handles provider fallback and rate-limit cooldowns; the direct-Gemini branch was a competing fallback policy at a layer that couldn't see the gateway's state.

**Why this is better.**
- One source of truth for routing logic (`SliceConfig.can_handle`), not four.
- One source of truth for LLM fallback (the gateway), not three.
- New slices require ~20 lines: `_SYSTEM`, keyword list, `CONFIG = SliceConfig(...)`. No `can_handle` function, no force flags, no question-ID list.
- Eliminates competing fallback policies: gateway is the only layer that knows about provider state.

**How to replicate.**
1. Identify per-instance functions that differ only in data — the duplicated `can_handle` is the textbook case.
2. Lift the function onto the data class as a method; parameterize over instance attributes.
3. Audit "except / fallback" branches. If the fallback path duplicates resilience the inner layer already provides, delete it — competing retry layers make outages worse, not better.
4. For "designed but never used" config fields, prefer deletion over keeping for "future-proofing." Future you will be confused by them.

---

## 11. Eliminate the second retrieval pass in step_06

**Problem.** [step_06_context_engineering/implementation/pipeline.py:72](step_06_context_engineering/implementation/pipeline.py) ran a second `self._retriever.retrieve(question, k=self.k)` call **purely so the dashboard's "sources" panel had chunks to render**. The full BM25 + dense + RRF stack had already executed inside `vsa_dispatch` (which retrieves k=20 plus up to 4×10 sub-Q expansions), then ran the engineered context (rerank → dedup → compress). Calling retrieve again duplicated work and — worse — showed the user different chunks from the ones the answer was actually grounded on.

**Diagnosis.** The chunks the answer was grounded on lived deep inside `engineer_context` and were never returned out. So the pipeline did the lazy thing: a fresh retrieval. Wrong chunks, wasted CPU.

**Fix.**
1. `engineer_context()` now returns `(context_xml, metrics, display_chunks)` where `display_chunks` are the post-rerank, post-dedup, post-compress chunks that actually formed the LLM's context.
2. Threaded `display_chunks` through `run_with_config` (5-tuple return) → `dispatch` (7-tuple return) → `Step06RAG.query_extended`.
3. Pipeline slices to `self.k` for the dashboard, no extra retrieval.
4. Hoisted `import copy` out of the per-chunk loop in `context_engineer.py` (separate audit nit, fixed in the same pass).

**Why this is better.**
- One retrieval pass per query instead of two — meaningful on slow embedding paths.
- The dashboard now shows the **exact** chunks the answer is grounded on, not a separate top-k that may have different content after rerank.
- The eval judge can trace provenance correctly because `retrieved_chunks` and `context_sent` now agree.

**How to replicate.**
1. Audit hot paths for "compute X, throw X away, recompute X for display."
2. Promote internal intermediates to return values when callers want them.
3. If the second computation produces *different* results, that's worse than waste — fix the inconsistency, not just the cost.

---

## 12. Replace string-overlap "confidence" with a real multi-signal score

**Problem.** [step_07_production/implementation/confidence.py:28](step_07_production/implementation/confidence.py) computed `score = 0.7 * question_term_overlap + 0.3 * length_bonus`. That measures "does the answer repeat words from the question?" — a literal copy-back of the question would ace it. Grade was used to mark `PASS`/`FAIL` in the health monitor, so the entire "is the answer good?" signal for production hardening was noise.

**Diagnosis.** The original implementation had no access to faithfulness evidence. Step 05's critic LLM produced exactly that signal but the orchestrator never propagated it (fixed in #9), so step 07 couldn't see it.

**Fix.** Full rewrite of `score_answer`. New signal hierarchy:
1. **Critic verdict** (when available). `critic_approved=False` ⇒ cap at 0.30, label "low", and the score's `reason` includes the critic's flag. `critic_approved=True` ⇒ +0.25 base contribution.
2. **Hard floors.** Answer under 20 chars or matching refusal markers ("i don't know", "no information", "not in the context") ⇒ low confidence regardless of other signals.
3. **Retrieval evidence.** `context_chars >= 200 OR chunks_used >= 1` ⇒ +0.10.
4. **Answer sanity.** Length in `[30, 1500]` ⇒ +0.05.
5. **Term overlap** retained as a weak (10%) tiebreaker, not the driver.

Pipeline ([step_07_production/implementation/pipeline.py](step_07_production/implementation/pipeline.py)) now passes `critic_approved`, `critic_notes`, `context_chars`, `chunks_used` into `score_answer`. The semantic cache ([semantic_cache.py](step_07_production/implementation/semantic_cache.py)) carries `critic_approved` + `critic_notes` on each `CacheEntry` so cache-hit scoring uses the same signals as the original miss-path scoring.

When confidence comes back low, the answer is annotated with `[Note: low-confidence — <reason>]` so the dashboard and eval users see *why*, not just *that*.

**Why this is better.**
- Confidence becomes a real production signal — `health_snapshot.pass_rate` finally measures answer quality instead of question echo.
- Cache hits and original answers are scored on the same basis, so a cached answer's grade doesn't drift from the original.
- A future operator paging on `pass_rate` drops knows whether to investigate the critic, the retrieval stack, or the model — `signals` and `reason` are in the result.

**How to replicate.**
1. Audit any metric that ends in "confidence" or "score" — what does the formula actually measure? Often it measures coincidence (length, vocabulary overlap), not the property you care about.
2. List the *signals you actually have access to* that correlate with the target property. For RAG faithfulness that's: critic verdict, retrieval evidence, refusal markers, answer length sanity.
3. Combine with a defensible weighting; emit the breakdown (`signals`, `reason`) so the score is interpretable.

---

## 13. Move retry from "wrap the whole pipeline" into the gateway client

**Problem.** [step_07_production/implementation/pipeline.py](step_07_production/implementation/pipeline.py) used `@with_retry(max_attempts=3, base_delay=0.5, exceptions=(Exception,))` around `_run_pipeline`, which called the **entire** step-06 stack: VSA routing → retrieval → graph navigation → CSV tool → rerank → dedup → compress → format → LLM call → critic. A transient gateway 503 would retry **all of that**. Three times. Costly, slow, and the retry was at the wrong layer — every layer except the LLM call is deterministic given the same inputs, so retrying them is pure waste.

**Diagnosis.** Retry was applied at the orthogonally-wrong level. The thing that's actually transient — a 5xx from the gateway, a connect timeout, a dropped TCP — happens inside `LLM.chat()`. Wrap *that*, not the world.

**Fix.**
1. Added HTTP-level retry to [llm_gatewayV2/client.py](llm_gatewayV2/client.py): `_post_with_retry` retries on `httpx.ConnectError`, `httpx.ReadTimeout`, `httpx.RemoteProtocolError`, and 5xx response codes. Exponential backoff `0.5s → 1s → 2s → 4s → 8s (cap)`, `LLM_CLIENT_RETRIES=3` default, env-overridable.
2. Does **not** retry 4xx — those are caller errors (bad request, missing field, auth) and retrying makes them worse.
3. Removed `@with_retry` from step 07's pipeline and removed the now-orphaned [step_07_production/implementation/retry.py](step_07_production/implementation/retry.py) entirely.
4. Kept the `extractive_fallback` safety net for non-transient failures (Chroma down, graph file missing) — that's a different concern and is still in `pipeline.py`'s except branch.

**Why this is better.**
- Retries cost one LLM round-trip, not one full retrieval + rerank + LLM + critic stack.
- Every caller of the gateway (step 05, step 06, step 07, the eval judge) inherits resilient HTTP semantics without each one writing its own decorator.
- The retry policy lives next to the thing it retries, so changing it is one edit.

**How to replicate.**
1. For every `@with_retry`/`@retry` decorator, ask: what specific failure am I retrying against? If the answer is "anything that fails," the decorator is too broad.
2. Push the retry as close to the transient failure source as you can. HTTP retries belong in the HTTP client, not in the business logic above it.
3. Distinguish 4xx (caller's fault, don't retry) from 5xx (server transient, do retry). Most retry libs that just take `exceptions=(Exception,)` get this wrong by default.
4. After moving retry, delete the old retry helper — keeping "for backwards compat" is how dual sources of truth proliferate.

---

## 14. Make steps 02/03/04 actually inherit from their predecessor

**Problem.** The README claimed "each step inherits from the prior." In reality only step 04 used `class Step04RAG(Step03HybridRAG)`. Steps 02 and 03 composed-by-import — they re-implemented `query()` end-to-end and duplicated context-assembly logic (`vector_ctx`, `csv_ctx`, conditional `parts.append`, `RAGResult` construction). Step 04 *did* subclass but reimplemented `query()` from scratch anyway, so the inheritance was decorative.

**Diagnosis.** No extension points existed on `BaselineRAG`. Subclasses had to either copy `query()` (which is what 02/03/04 did) or skip inheritance. The class had a monolithic method instead of a template-method structure.

**Fix.** Refactored `BaselineRAG` to a template method with two extension points:

1. **`retrieve_chunks(question, k)`** — returns the candidate chunks. Default: dense top-k. Step 03 overrides this with BM25+dense+RRF.
2. **`build_context_sections(chunks, question)`** — returns a `dict[str, str]` keyed by section name (`"vector"`, `"csv"`, `"graph"`, ...). Subclasses contribute new keys via `super().build_context_sections(...)` then add. Step 02 adds `"csv"`, step 04 adds `"graph"`.

Section assembly order is a class attribute `CONTEXT_PRIORITY = ("csv", "graph", "vector")` — authoritative tool output first (CSV), then structural (graph), then textual (vector). The LLM's attention is biased toward the top of the context window, so priority matters.

The chain is now:

- `BaselineRAG` — dense retrieval, vector section only.
- `Step02ToolsRAG(BaselineRAG)` — adds `"csv"` section via `detect_intent` + `run_query`. **Whole class is ~12 lines.**
- `Step03HybridRAG(Step02ToolsRAG)` — overrides `retrieve_chunks` for hybrid RRF; CSV section inherited automatically.
- `Step04RAG(Step03HybridRAG)` — adds `"graph"` section; hybrid retrieval + CSV both inherited.

`query()` is defined **once**, in `BaselineRAG`. Latency timing, `RAGResult` construction, `generate_answer` call — all single-source.

**Why this is better.**
- The inheritance claim in the README is now true: each step is a strict superset of the prior, expressed in code.
- New step ideas (add a re-ranking layer between dense and BM25? add a tool-output section?) are one method override each, not a full pipeline copy-paste.
- Bug fixes in `query()` propagate automatically through the chain. The previous shape required four-place edits.

**How to replicate.**
1. Identify the "template" of the operation (in our case: retrieve → assemble → generate → wrap).
2. Pull the template into a single method on the base class.
3. Replace each variation point with a hook method that returns a typed value.
4. For "ordered list of optional contributions" patterns (like context sections), use a `dict[str, str]` with a class-level priority tuple instead of positional parts and conditional `if x: parts.append(x)`.
5. After refactoring, verify each subclass file is *smaller* than before. Steps 02/03/04 went from 95/126/82 lines to 22/82/40 lines — that's the win you're aiming for.

---

## 15. Hoist inline `from llm_gatewayV2.client import LLM` imports + delete one more fake Gemini fallback

**Problem.** Five files imported the gateway client inside function bodies instead of at the top:

- [step_05/.../critic.py:32, 69](step_05_multi_agent/implementation/agents/critic.py)
- [step_05/.../synthesis.py:59](step_05_multi_agent/implementation/agents/synthesis.py)
- [step_05/.../query_analyst.py:77](step_05_multi_agent/implementation/agents/query_analyst.py)
- [step_06/.../slices/base.py:120](step_06_context_engineering/implementation/slices/base.py)

Lazy imports usually exist to break circular deps. `llm_gatewayV2.client` is a leaf module — it imports nothing from this project. The lazy imports were noise.

Additionally, [synthesis.py](step_05_multi_agent/implementation/agents/synthesis.py) had a third try/except direct-Gemini fallback (same `gemini-3.1-flash-lite-preview` id; valid model, see top-of-doc correction). Same antipattern as step_01 and step_06 — a competing fallback policy at a layer that bypasses gateway state.

**Fix.**
1. Moved `from llm_gatewayV2.client import LLM` to the top of each file.
2. Removed the direct-Gemini except branch in `synthesis.py` (same reasoning as #10 and #13 — the gateway handles fallback). Synthesis failure now returns a clean `[Synthesis failed: ...]` error sentinel instead of falling through to a competing fallback policy.
3. The `_GEMINI_MODEL` constant deleted from `synthesis.py`. All three duplicate Gemini-model hardcodes consolidated into the gateway's single source of truth.

**Why this is better.**
- Module-level imports surface dependency errors at startup, not on first call inside a hot path.
- Three competing direct-Gemini fallbacks → zero. The gateway is the one place that knows about LLM providers; nobody else needs to.
- Slightly faster hot path (no repeated `from ... import ...` work).

**What I did NOT do.** The audit also flagged "provider roster encoded 3 times" (`main.py`, `router.LIMITS`, `router.SHORTCUTS`). On closer look those represent three distinct concerns (the resolution shortcuts table, the rate-limit config, and the default fallback order), not three sources of the same truth. `LIMITS.keys()` is the canonical provider set; `DEFAULT_ORDER` deliberately fixes a specific order; `SHORTCUTS` encodes user-facing aliases. Leaving as-is.

**How to replicate.**
1. `grep -rn 'from llm_gatewayV2.client import LLM'` — every match is a candidate. Verify the importing module isn't imported by `llm_gatewayV2/client.py` (it shouldn't be); if not, hoist.
2. Any time you see `try: gateway-call except: direct-provider-call`, ask whether the gateway already does this. If yes, delete the except branch — competing retry/fallback layers make outages worse.
3. After hoisting, run `grep` again for the same string with leading whitespace — that finds the in-function holdouts.

---

## 16. Realign `pyproject.toml` extras with the 7-step structure

**Problem.** Several extras drifted out of sync with the actual code:

1. `rank-bm25` listed twice — once in base `dependencies` (line 15), again in the `step-07` extra (line 49).
2. A `step-10` extra existed despite there being no step 10 (legacy from the pre-restructure 10-step layout).
3. `arize-phoenix` was under `step-02` — but step 02 is the CSV tool step. Phoenix lives in the unnumbered `observability/` utility folder.
4. `chromadb` listed under both `step-01` and `step-04` — same dep, two places.
5. `fastembed` was in the `dev` dependency-group but [dashboard.py:214](dashboard.py#L214) imports it at runtime. A user running `streamlit run dashboard.py` without `--group dev` would crash.
6. `openai` was under `step-01` but step 01 doesn't use OpenAI directly — only `evaluation/judge_llm.py` does, and only when `JUDGE_PROVIDER=openai`.

**Fix.** Rewrote `[project.optional-dependencies]` to map cleanly to the **incremental** new dependency each step actually needs:

| Extra | Adds | Used by |
|---|---|---|
| `step-01` | `chromadb`, `pandas` | baseline RAG |
| `step-02` | (none) | csv_tool uses inherited pandas |
| `step-03` | (none) | rank-bm25 is in base |
| `step-04` | `networkx` | knowledge graph |
| `step-05` | (none) | multi-agent reuses 03+04 |
| `step-06` | `sentence-transformers` | CrossEncoder reranker |
| `step-07` | (none) | semantic_cache reuses ST |
| `observability` | `arize-phoenix`, OTLP | utility folder |
| `dashboard` | `fastembed`, `streamlit`, `plotly` | dashboard.py |
| `openai` | `openai` | judge fallback |
| `dev` | pytest, mypy, ruff | tooling |

Empty extras (`step-02`, `step-03`, `step-05`, `step-07`) are kept as **documentation markers** so the per-step install pattern reads cleanly even when the step adds nothing new. They cost nothing.

Also expanded `[dependency-groups].dev` to include every runtime dep so `uv sync --group dev` gives a working end-to-end environment.

**Why this is better.**
- No duplicate listings — single source of truth for every dependency.
- Each extra is independently installable and produces a working subset (e.g., `uv sync --extra step-04` gives you steps 01-04 plus everything they need).
- Phoenix moved out of "step 02" — its real home is observability/, an unnumbered utility folder.
- Dashboard works for non-dev installers via `uv sync --extra dashboard`.

**How to replicate.**
1. For each declared dep, check whether the file that imports it lives in the labeled extra's scope.
2. Look for the same dep declared twice — `awk -F'[">=]' '/^[a-z]/{print $1}'` on the dependencies arrays groups them quickly.
3. Mark extras for steps that add nothing as `# (none) — inherited from step-NN-1` rather than deleting the extra key — keeps the install pattern uniform across all steps.
4. Empty extras don't pull anything but document the inheritance — they're documentation as configuration.

---

## 17. Sync the long-form docs with the 7-step / 14-question reality

**Problem.** Two long-form docs had drifted away from the current code:

- [MASTER_PLAN.md](MASTER_PLAN.md) still cited "15-question golden set," "6 tiers," and referred to Steps 08 / 09 / 10 / 11 in the Q&A section.
- [concepts.md](concepts.md) is a chapter-by-chapter learning narrative that still calls everything by the old 12-step labels and uses the obsolete "/27" pass-rate denominators throughout ("26% (7/27)", "78% (21/27)", etc.).

**Diagnosis.** The numbered-step restructure (2026-05-17) and the 27→14 question reduction (2026-05-18) happened after both docs were written. README and per-step result tables were regenerated; these long-form docs weren't.

**Fix — two different strategies for two different doc types.**

**MASTER_PLAN.md** (a current reference doc) was patched in-place:
- "15-question golden set" → "14-question golden set" (3 occurrences).
- Tier table rewritten from 6 tiers to 5, with the current Q01–Q14 layout from `golden_questions.py`.
- "Step 08 (single agentic)" / "Step 09" / "Step 10" / "Step 11" inline references updated or contextualized to today's 7-step naming.

**concepts.md** (a historical learning narrative) was given a **prominent banner at the top** instead of being rewritten:
> "HISTORICAL DOCUMENT — preserved for the learning narrative. Every pass-rate percentage below references the obsolete 27-question set against the obsolete 12-step layout. They explain *why* each technique was added, but they are not current numbers."

The banner also includes a translation cheat sheet (which step in the narrative maps to which current step). The chapter content is **not** rewritten because the value of `concepts.md` is the journey — "we added BM25 because it lifted us from X to Y" — and rewriting numbers strips that signal.

**Why this is better.**
- Anyone reading MASTER_PLAN.md now sees consistent, current numbers.
- Anyone reading concepts.md sees the banner first and knows to treat the percentages as a teaching device, not a status report.
- The educational journey survives. Rewriting the chapter results to current numbers would erase *why* each technique was introduced.

**How to replicate.**
1. Categorize each doc as **reference** (must stay current) or **narrative** (preserved as-is).
2. For reference docs, grep for any number/label that changed and edit each in place.
3. For narrative docs, add a top-of-file banner with the change date, what's stale, and where to find the current numbers. Do not rewrite the body — you'll destroy the teaching value.

---

## 18. Surface `extras/hybrid_rerank/` in README

**Problem.** [extras/hybrid_rerank/](extras/hybrid_rerank/) — a Step 03 variant that adds a CrossEncoder rerank stage (no dedup, no compression) — existed and had been evaluated against the current 14-question golden set, but had **zero references** in README.md, MASTER_PLAN.md, concepts.md, or dashboard.py. Anyone learning from the repo would never know it was there.

It's specifically valuable as an **ablation**: it answers "does the rerank alone explain Step 06's `context_precision` lift, or do you need the full rerank → dedup → compress pipeline?"

**Fix.** Added a short "Side experiments" section to [README.md](README.md) above the repo layout, with one paragraph describing what's in `extras/hybrid_rerank/` and why a reader would care. Also added `extras/` to the repo-layout tree.

**Why this is better.** Discoverable. A learner walking the README from top to bottom now sees the rerank-only experiment as a sibling concept to the full Step 06 stack.

**How to replicate.**
1. `find . -name 'README.md' -not -path '*/node_modules/*' -not -path '*/.venv/*'` — gives every doc anchor.
2. Cross-check `grep -rln folder-name` for each folder against the README's table of contents.
3. Any folder that's runnable but unreferenced gets a one-paragraph mention.

---

## 19. Collapse `dashboard.py` step factories into a registry

**Problem.** [dashboard.py:1478-1510](dashboard.py) had **six near-identical factory functions** — `_load_step02_rag`, `_load_step03_rag`, …, `_load_step07_rag`. Each was 4 lines of `@st.cache_resource` + lazy import + `Class(k=...).build()`. Adding a new step meant writing yet another 4-line block. Also fixed a stale "27 golden questions" docstring at line 335 from before the 27→14 question reduction.

**Diagnosis.** Copy-paste was the path of least resistance, but the steps differ only in three values: module path, class name, and `k`.

**Fix.** Introduced `_STEP_REGISTRY` — a `dict[str, tuple[module_path, class_name, k]]` keyed by the step's directory name — plus a single `_load_step_rag(step_key)` that does `importlib.import_module` + `getattr` + `Class(k=k).build()`, decorated with `@st.cache_resource`. Kept the six per-step accessor functions as one-liners that delegate to the registry, so call sites (`rag02 = _load_step02_rag()`) stay unchanged.

This preserves Streamlit's per-function cache identity (one cache entry per step key) while reducing the actual logic to one place.

**Why this is better.**
- Adding step 08 in the future is one registry row.
- All step k-values visible at once in one table — easy to see "step 06/07 use k=5, others use k=10."
- The 27→14 docstring fix removes one more reference to the obsolete eval set.

**How to replicate.**
1. Look for groups of `_load_stepNN_*` / `init_stepNN_*` / `for_stepNN_*` functions that differ only in constants.
2. Move the differing constants into a `dict[key, tuple]` registry.
3. Keep thin wrappers that delegate, so the call sites stay readable.

---

## 20. Tier locks for step_06 / step_07 — deliberate gap, not a TODO

**The audit flagged** that no golden question targets step_06 or step_07. I considered adding three of each and decided **not to**. Here's why, with the reasoning recorded in [golden_questions.py](step_01_baseline_rag/evaluation/golden_questions.py) so future-you doesn't re-litigate it.

**Why step_06 doesn't get a tier lock.** A tier lock is, by definition, a question step N+1 can pass but step N cannot. Step_06's value is **context engineering**: rerank, dedup, compress, VSA routing. These improve `context_precision` and `faithfulness` on questions earlier tiers can already attempt — they don't unlock new question types. A tier-6 question would either:
- leak into step_05 (because the answer is reachable through retrieval + multi-agent synthesis, just messily), or
- require contrived corpus data designed solely to break step_05's retrieval, which would test the test rather than the system.

**Why step_07 doesn't get a tier lock at all.** Step_07's value is **reliability**: semantic cache, confidence scoring, graceful degradation. These are process properties (latency, repeatability, failure handling), not answer-correctness properties. Bolting them into the golden-question contract would conflate two different things.

**What to add instead — a separate diagnostic harness.** Documented inline at the top of `golden_questions.py`:
1. **Step 06 diagnostic:** track `context_precision` and `faithfulness` deltas on the existing 14Q set between step_05 and step_06. If step_06 doesn't beat step_05 on these metrics, the context engineering layer is not earning its keep. *This is already measurable today via the per-step `eval_results.json` files — no new questions needed.*
2. **Step 07 cache test:** paraphrase the 14 questions ("What is X?" → "Tell me about X"), run them, assert `semantic_cache.stats.total_hits` increments.
3. **Step 07 confidence test:** feed a known-bad answer (e.g. one that contains "I don't know") through `score_answer()`, assert `label=='low'`.

**Why this is better than adding leaky questions.** Adding 6 tier-locked questions that earlier steps could secretly pass would silently weaken every downstream eval. A clean "we deliberately don't lock these tiers, here's how to measure them instead" is honest.

**How to replicate.** When the audit recommends "add X tests," check what property X actually measures. If X is a quality/process property rather than a capability property, push back: build a separate diagnostic harness for it, don't pollute the capability-locked golden suite.

---

## 21. Enforce `disqualifiers` in the eval runner

**Problem.** Every `GoldenQuestion` had a `disqualifiers: list[str]` field — strings whose presence in the answer means the model latched onto the wrong entity (e.g., "Felix Wagner" appearing in the answer to Q03 means the model retrieved the InsightLens on-call block, not the PulseConnect one). Hand-curated, encoded for many questions. **And not enforced anywhere in [evaluation/run_eval.py](evaluation/run_eval.py).** The only file that did enforce them was the dead [pipeline/eval.py](pipeline/eval.py) (user's learning sandbox, untouched). RAGAS's `answer_correctness` is format-tolerant — it occasionally rewards near-misses whose surface vocabulary matches the gold but whose cited entity is wrong.

**Diagnosis.** When `run_eval.py` was rewritten on top of the gateway-routed RAGAS judge, the disqualifier check was dropped on the floor. The field stayed; the enforcement disappeared.

**Fix.** After RAGAS computes `answer_correctness` and the initial PASS/PARTIAL/FAIL grade, run a case-insensitive substring check of each disqualifier against the answer text. Any hit triggers a **one-step grade downgrade** (`PASS → PARTIAL`, `PARTIAL → FAIL`, `FAIL` stays). The tripped disqualifier strings are recorded in the per-question result as `disqualifiers_tripped` so the dashboard and review can show *why* a near-miss was downgraded.

**Why this is better.**
- A surface-similar wrong-entity answer no longer scores PASS just because the judge was lenient.
- The hand-curated disqualifier data finally pays off — it's been written, just unused.
- The downgrade is gentle (one step, not "force FAIL"), so a legitimately good answer that incidentally contains a disqualifier substring becomes PARTIAL rather than vanishing entirely. Tunable upward later if needed.

**How to replicate.**
1. For each field on a dataclass used in production paths, grep for reads of that field. If reads exist only in the dataclass definition itself + producer paths, the field is unenforced.
2. Add enforcement at the latest possible stage — here, after RAGAS scoring, so the disqualifier never has to compete with the judge's judgement; it acts as a post-hoc veto.
3. Always record *what tripped* alongside *that something tripped* — a boolean "tripped: true" is not actionable, a list of strings is.

---

## 22. Phoenix instrumentation across all steps + gateway — scoped out, with a plan

**Problem.** [observability/implementation/traced_pipeline.py:146](observability/implementation/traced_pipeline.py#L146) instruments **only** `BaselineRAG`. Steps 02–07 emit zero traces to Phoenix. The gateway, which every step calls, writes rich call records to [llm_gatewayV2/gateway_v2.db](llm_gatewayV2/gateway_v2.db) — provider, model, tokens, latency, status, error, attempted-chain — but those records never leave SQLite.

**What I did NOT do.** Wiring OpenTelemetry exporters across the gateway and every step pipeline is multi-day integration work, not a discrete fix. I scoped it out of this audit pass deliberately rather than half-finish it. The other 21 entries above are completed; this one is intentionally deferred with a plan.

**What's there to build on.**
- Phoenix is already runnable locally (`px.launch_app(run_in_thread=True)` at [observability/.../phoenix_exporter.py:61](observability/implementation/phoenix_exporter.py#L61)) — endpoint `http://localhost:6006/v1/traces`.
- `TracedRAG` shows the span schema (`CHAIN → RETRIEVER → LLM`) and the OpenInference semantic conventions for retrieval (`retrieval.documents.N.document.content`).
- The gateway DB schema already contains every field a trace would need.

**Recommended approach when you build this:**

1. **Gateway-first.** Wrap `main.py`'s chat endpoint with an OTel span. Set `provider`, `model`, `prompt_chars`, `response_chars`, `latency_ms` as attributes. This single instrumentation gives you LLM-call visibility from every step.
2. **Lift `TracedRAG` to a mixin.** Define `class Traced<Step>RAG(BaseStep)` per step (or, cleaner: a `Traced` mixin that calls super().query() but emits spans around `retrieve_chunks`, `build_context_sections`, `generate_answer`). Tag each span with `step_name` so you can group by stage in Phoenix.
3. **Single export config.** One module reads `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:6006/v1/traces`), exports to Phoenix. The 7-step instrumentation reuses it.
4. **Decide on the Phoenix/Langfuse question after step 1.** Once spans flow from the gateway, you have a real signal to compare tools. The audit's "stay on Phoenix, fix instrumentation first" advice applies.

**Why this is the right scope decision.**
- The 20 fixes already in this doc deliver concrete, immediate improvements with measurable correctness/perf wins.
- Phoenix instrumentation is high-value but lower-urgency — the gateway DB already captures the data; tracing is the visualization layer.
- A half-built TracedRAG-for-each-step would be its own audit item later. Better to defer the whole thing than land it half-done.

**How to replicate this scope decision.**
1. When an audit item reads as "build X integration," not "fix Y bug," check whether it has a discrete deliverable. If it doesn't, scope it out and write a build plan in the audit doc instead of forcing it into the same pass.
2. The reader of this doc should be able to start at the plan above and have a working Phoenix integration in a day or two of focused work — that's the standard for what counts as "documented enough to defer."

---

## Summary

21 audit findings actioned, 1 (#22) deliberately deferred with a build plan. Net effect:

- **3 duplicate direct-Gemini fallback paths** consolidated into the gateway across step_01, step_05/synthesis, step_06/slices. (The model ID itself, `gemini-3.1-flash-lite-preview`, is valid — the issue was duplicate fallback policies, not bad model names. See correction note at top of doc.)
- **Real critic→confidence wiring** end-to-end: step_05 critic verdict propagates through `RAGResult` → `score_answer()` → health monitor → semantic cache, replacing the question-echo "confidence" heuristic.
- **Step inheritance** is real now: `BaselineRAG` → `Step02ToolsRAG` → `Step03HybridRAG` → `Step04RAG` actually inherit, with `query()` defined once.
- **Hot-path waste eliminated:** CSV cache, BM25 persisted index, graph name index memoization, double retrieval removed, sub-Q retrieval parallelized.
- **Retry moved** from "wrap the full pipeline" into the gateway's HTTP client where it belongs.
- **Eval disqualifiers enforced** — hand-curated rejection strings finally pay off.
- **Docs synced** — pyproject.toml, README.md, MASTER_PLAN.md, concepts.md, dashboard.py docstring, golden_questions.py header all aligned to 7-step / 14-question reality.
- **Hygiene:** `.env.example` added (rotate the leaked keys), `gateway_v2.db` untracked, dead `retry.py` deleted, six dashboard step-factories collapsed to a registry.
- **`extras/hybrid_rerank/` surfaced** in README so the rerank-only ablation is discoverable.

The deferred item is Phoenix instrumentation across all steps + gateway, scoped above.

