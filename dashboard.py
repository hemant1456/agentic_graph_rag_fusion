"""
RAG Fusion Lab — interactive experimentation dashboard.

Four tabs:
  🧪 Experiment Lab    — build custom indexes, compare vs baselines
  📊 Analysis          — deep multi-run comparison, latency, question types
  🔍 Query Explorer    — live retrieval from any index
  🗂 Chunk Browser     — browse chunks from any index

Run with:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Fusion Lab",
    layout="wide",
    page_icon="🔬",
)

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent
GRAPH_PATH  = ROOT / "step_04_knowledge_graph" / "results" / "graph.json"
# 7-step layout (renumbered 2026-05-17). All steps point to <step>/results/eval_results.json.
STEP01_EVAL = ROOT / "step_01_baseline_rag"        / "results" / "eval_results.json"
STEP02_EVAL = ROOT / "step_02_tools"               / "results" / "eval_results.json"
STEP03_EVAL = ROOT / "step_03_hybrid_retrieval"    / "results" / "eval_results.json"
STEP04_EVAL = ROOT / "step_04_knowledge_graph"     / "results" / "eval_results.json"
STEP05_EVAL = ROOT / "step_05_multi_agent"         / "results" / "eval_results.json"
STEP06_EVAL = ROOT / "step_06_context_engineering" / "results" / "eval_results.json"
STEP07_EVAL = ROOT / "step_07_production"          / "results" / "eval_results.json"
# Shared ChromaDB at project root — single collection "vertexia_smart" after step_01+02 merge.
CHROMA_DB   = ROOT / "chroma_db"
CHROMA_COLLECTION = "vertexia_smart"
CORPUS      = ROOT / "dataset" / "company_data"
EXP_DB_ROOT = ROOT / "step_01_baseline_rag" / "results" / "chroma_experiments"

# ── Constants ──────────────────────────────────────────────────────────────────

GRADE_COLOR = {"PASS": "#2e7d32", "PARTIAL": "#f57f17", "FAIL": "#c62828"}
GRADE_EMOJI = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}
GRADE_NUM   = {"PASS": 2, "PARTIAL": 1, "FAIL": 0}

CHUNK_TYPE_ICON = {
    "aggregate": "🔮",
    "row":       "📋",
    "section":   "📄",
    "prose":     "📝",
}
CHUNK_TYPE_COLOR = {
    "aggregate": "#6a1b9a",
    "row":       "#1565c0",
    "section":   "#00695c",
    "prose":     "#4e342e",
}

LOCAL_EMBED_MODELS: dict[str, str] = {
    "all-MiniLM-L6-v2 · 384d · fast":             "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5 · 384d · retrieval-tuned":  "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5 · 768d · retrieval-tuned":   "BAAI/bge-base-en-v1.5",
    "nomic-embed-text-v1.5 · 768d":                "nomic-ai/nomic-embed-text-v1.5",
}

STEP01_BASELINE_CHUNKS = 152   # fixed baseline for delta display

# ── Minimal CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stExpander"] summary {
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state bootstrap ────────────────────────────────────────────────────

st.session_state.setdefault("exp_history", [])

# ── Data loaders ───────────────────────────────────────────────────────────────


@st.cache_data
def load_eval(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def _rechunk_corpus(chunk_size: int, overlap: int, strategy: str) -> list[dict]:
    """
    Re-chunk every file in the corpus using configurable params. No API calls.
    strategy: "paragraph" | "fixed"
    """
    results: list[dict] = []

    for path in sorted(CORPUS.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        source = str(path.relative_to(CORPUS))
        dept = path.parent.name

        if suffix == ".csv":
            # CSV rows — each row is its own chunk regardless of strategy
            import csv
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    row_text = f"[{source}]\n" + " | ".join(f"{k}: {v}" for k, v in row.items())
                    results.append({
                        "text": row_text,
                        "source": source,
                        "department": dept,
                        "format": "csv",
                        "chunk_type": "row",
                        "chunk_index": idx,
                        "chars": len(row_text),
                    })
            continue

        if suffix == ".json":
            import json as _json
            try:
                data = _json.loads(path.read_text(errors="replace"))
                text = _json.dumps(data, indent=2)
            except Exception:
                text = path.read_text(errors="replace")
            fmt = "json"
        elif suffix in (".txt", ".md", ".py"):
            text = path.read_text(errors="replace")
            fmt = suffix.lstrip(".")
        else:
            continue

        if strategy == "paragraph":
            pieces = _rechunk_text_para(text, chunk_size, overlap)
        else:
            pieces = _rechunk_text_fixed(text, chunk_size, overlap)

        for i, piece in enumerate(pieces):
            results.append({
                "text": piece,
                "source": source,
                "department": dept,
                "format": fmt,
                "chunk_type": "prose",
                "chunk_index": i,
                "chars": len(piece),
            })

    return results


def _rechunk_text_para(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunker: accumulate paragraphs up to chunk_size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap:].strip() if overlap > 0 else ""
                current = (tail + "\n\n" + para).strip()
            else:
                step = max(chunk_size - overlap, 1)
                for i in range(0, len(para), step):
                    chunks.append(para[i : i + chunk_size])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _rechunk_text_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size chunker: slide a window of chunk_size with given overlap."""
    step = max(chunk_size - overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]


# ── Embedding helpers ──────────────────────────────────────────────────────────


@st.cache_resource
def _get_local_embedder(model_name: str):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name)


def _embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    model = _get_local_embedder(model_name)
    return [v.tolist() for v in model.embed(texts)]


def _embed_with_baseline(texts: list[str]) -> list[list[float]]:
    """Embed via the project's baseline HuggingFace MiniLM (no API calls).

    Step 01 now uses sentence-transformers/all-MiniLM-L6-v2 locally — the old
    Gemini embedder was removed when the project was renumbered.
    """
    from step_01_baseline_rag.implementation.ingest import _embedder
    return _embedder.embed_documents(texts)


def _embed_query_baseline(query: str) -> list[float]:
    """Embed a single query string with the project's baseline MiniLM embedder."""
    from step_01_baseline_rag.implementation.ingest import embed_query
    return embed_query(query)


# ── ChromaDB helpers ───────────────────────────────────────────────────────────


@st.cache_resource
def _get_collection(db_path_str: str, collection_name: str):
    import chromadb
    client = chromadb.PersistentClient(path=db_path_str)
    return client.get_collection(collection_name)


# ── Plotly chart helpers ───────────────────────────────────────────────────────


def _comparison_bar(runs: list[tuple[str, dict]]) -> go.Figure:
    """Grouped bar chart: PASS / PARTIAL / FAIL per run."""
    labels    = [label for label, _ in runs]
    pass_vals  = [d["grade_counts"]["PASS"]    for _, d in runs]
    part_vals  = [d["grade_counts"]["PARTIAL"] for _, d in runs]
    fail_vals  = [d["grade_counts"]["FAIL"]    for _, d in runs]

    fig = go.Figure(data=[
        go.Bar(name="PASS",    x=labels, y=pass_vals,  marker_color=GRADE_COLOR["PASS"],    opacity=0.9),
        go.Bar(name="PARTIAL", x=labels, y=part_vals,  marker_color=GRADE_COLOR["PARTIAL"], opacity=0.9),
        go.Bar(name="FAIL",    x=labels, y=fail_vals,  marker_color=GRADE_COLOR["FAIL"],    opacity=0.9),
    ])
    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="# Questions",
    )
    return fig


def _grade_heatmap(runs: list[tuple[str, dict]]) -> go.Figure:
    """Heatmap: rows = Q01..Q29, columns = run labels, colour = grade."""
    if not runs:
        return go.Figure()

    # Build Q-ID list from first run that has results
    q_ids: list[str] = []
    for _, d in runs:
        if d.get("results"):
            q_ids = [r["id"] for r in d["results"]]
            break
    if not q_ids:
        return go.Figure()

    z: list[list[int]] = []
    text_annot: list[list[str]] = []
    col_labels = [label for label, _ in runs]

    for _, d in runs:
        grade_map = {r["id"]: r.get("grade", "FAIL") for r in d.get("results", [])}
        col_z    = [GRADE_NUM.get(grade_map.get(qid, "FAIL"), 0) for qid in q_ids]
        col_text = [grade_map.get(qid, "—") for qid in q_ids]
        z.append(col_z)
        text_annot.append(col_text)

    # Transpose: z[run][question] → z_T[question][run]
    z_T    = [[z[col][row] for col in range(len(runs))] for row in range(len(q_ids))]
    text_T = [[text_annot[col][row] for col in range(len(runs))] for row in range(len(q_ids))]

    fig = go.Figure(go.Heatmap(
        z=z_T,
        x=col_labels,
        y=q_ids,
        text=text_T,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=[[0, "#c62828"], [0.5, "#f57f17"], [1.0, "#2e7d32"]],
        zmin=0,
        zmax=2,
        showscale=False,
    ))
    fig.update_layout(
        height=max(350, len(q_ids) * 22),
        margin=dict(t=20, b=20, l=60, r=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── _build_custom_index ────────────────────────────────────────────────────────


def _build_custom_index(
    chunk_size: int,
    overlap: int,
    k_eval: int,
    local_model: str | None,
    strategy: str,
    include_csv_agg: bool,
) -> dict:
    """
    Full pipeline: rechunk corpus → embed → store in ChromaDB → run 14 golden questions.

    local_model: fastembed model name, or None to use the baseline HuggingFace MiniLM.
    strategy: "paragraph" | "fixed"
    include_csv_agg: whether to include aggregate chunks from CSV parser.
    """
    import chromadb
    from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
    from step_01_baseline_rag.evaluation.run_eval import score as grade_fn
    from step_01_baseline_rag.implementation.generate import generate_answer
    from step_01_baseline_rag.implementation.pipeline import RAGResult
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context

    embed_label = local_model or "baseline_minilm"
    model_slug  = (local_model or "baseline_minilm").replace("/", "_").replace("-", "_")

    # ── 1. Rechunk ────────────────────────────────────────────────────────────
    st.write("**1 / 4** — Chunking corpus…")
    text_chunks = _rechunk_corpus(chunk_size, overlap, strategy)

    # CSV aggregate chunks (optional)
    if include_csv_agg:
        from step_01_baseline_rag.implementation.chunker import load_and_chunk as _smart_chunk
        smart = _smart_chunk(CORPUS)
        agg_chunks_smart = [c for c in smart if c.chunk_type == "aggregate"]
    else:
        agg_chunks_smart = []

    all_texts: list[str] = [c["text"] for c in text_chunks]
    all_meta: list[dict[str, Any]] = [
        {
            "source":     c["source"],
            "department": c["department"],
            "format":     c["format"],
            "chunk_type": c["chunk_type"],
            "chunk_index": c["chunk_index"],
        }
        for c in text_chunks
    ]
    for sc in agg_chunks_smart:
        all_texts.append(sc.text)
        all_meta.append(sc.to_metadata())

    st.write(
        f"  → {len(all_texts)} chunks "
        f"({len(text_chunks)} text/CSV-rows + {len(agg_chunks_smart)} agg)"
    )

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    st.write(f"**2 / 4** — Embedding with **{embed_label}**…")
    prog = st.progress(0.0, text="Embedding…")

    if local_model:
        st.write("  Loading model (first run downloads weights ~30–100 MB)…")
        embeddings = _embed_local(all_texts, local_model)
        prog.progress(1.0, text=f"Embedded {len(embeddings)} chunks locally.")
    else:
        # "API" branch retired — Gemini embedder was removed in the 7-step renumber.
        # Fall back to the project's baseline MiniLM (same model Step 01 uses).
        st.write("  Using baseline HuggingFace MiniLM (all-MiniLM-L6-v2) — no API calls.")
        embeddings = _embed_with_baseline(all_texts)
        prog.progress(1.0, text=f"Embedded {len(embeddings)} chunks with baseline MiniLM.")

    # ── 3. Store in ChromaDB ──────────────────────────────────────────────────
    st.write("**3 / 4** — Storing in ChromaDB…")
    db_path = EXP_DB_ROOT / f"cs{chunk_size}_ov{overlap}_{strategy[:3]}_{model_slug}"
    EXP_DB_ROOT.mkdir(parents=True, exist_ok=True)
    db_client = chromadb.PersistentClient(path=str(db_path))
    coll_name = "vertexia_exp"
    try:
        db_client.delete_collection(coll_name)
    except Exception:
        pass
    coll = db_client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})

    ids = [f"chunk_{i}" for i in range(len(all_texts))]
    batch = 100
    for i in range(0, len(all_texts), batch):
        coll.upsert(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],  # type: ignore[arg-type]
            documents=all_texts[i : i + batch],
            metadatas=all_meta[i : i + batch],     # type: ignore[arg-type]
        )
    st.write(f"  → Stored {coll.count()} chunks at `{db_path.relative_to(ROOT)}`")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    st.write(f"**4 / 4** — Evaluating {len(GOLDEN_QUESTIONS)} golden questions (k={k_eval})…")
    grade_counts: dict[str, int] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    all_results: list[dict[str, Any]] = []
    q_prog = st.progress(0.0, text="Running questions…")

    for qi, gq in enumerate(GOLDEN_QUESTIONS):
        t_ret = time.perf_counter()
        if local_model:
            qvec = _embed_local([gq.question], local_model)[0]
        else:
            qvec = _embed_query_baseline(gq.question)
        ret_ms = (time.perf_counter() - t_ret) * 1000

        res = coll.query(
            query_embeddings=[qvec],
            n_results=min(k_eval, coll.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs  = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        dists = (res["distances"] or [[]])[0]

        chunk_objs = [
            RetrievedChunk(
                text=d,
                source=str(m.get("source", "")),
                department=str(m.get("department", "")),
                format=str(m.get("format", "")),
                chunk_index=int(str(m.get("chunk_index") or 0)),
                distance=dist,
            )
            for d, m, dist in zip(docs, metas, dists)
        ]
        context = format_context(chunk_objs)

        t_gen = time.perf_counter()
        answer, provider = generate_answer(context, gq.question)
        gen_ms = (time.perf_counter() - t_gen) * 1000

        fake = RAGResult(
            question=gq.question,
            answer=answer,
            provider=provider,
            retrieved_chunks=chunk_objs,
            context_sent=context,
            context_chars=len(context),
            retrieval_latency_ms=ret_ms,
            generation_latency_ms=gen_ms,
        )
        scoring = grade_fn(fake, gq)
        grade_counts[scoring["grade"]] += 1
        all_results.append({
            "id":                   gq.id,
            "type":                 gq.type,
            "question":             gq.question,
            "answer":               answer,
            "grade":                scoring["grade"],
            "expected_outcome":     gq.expected_outcome,
            "required_hits":        scoring["required_hits"],
            "required_missing":     scoring["required_missing"],
            "sources_retrieved":    [c.source for c in chunk_objs],
            "source_similarities":  [round(c.similarity, 3) for c in chunk_objs],
            "context_chars":        len(context),
            "retrieval_latency_ms": round(ret_ms, 1),
            "generation_latency_ms": round(gen_ms, 1),
            "provider":             provider,
        })
        q_prog.progress((qi + 1) / len(GOLDEN_QUESTIONS), text=f"{gq.id}: {scoring['grade']}")

    total = len(GOLDEN_QUESTIONS)
    strat_short = "para" if strategy == "paragraph" else "fixed"
    model_short = (local_model or "minilm").split("/")[-1] if local_model else "minilm"
    label = f"cs={chunk_size} ov={overlap} {strat_short} {model_short}"

    return {
        "label":              label,
        "chunk_size":         chunk_size,
        "overlap":            overlap,
        "k":                  k_eval,
        "strategy":           strategy,
        "embed_model":        embed_label,
        "db_path":            str(db_path),
        "coll_name":          coll_name,
        "total_chunks":       len(all_texts),
        "total_questions":    total,
        "grade_counts":       grade_counts,
        "pass_rate":          round(grade_counts["PASS"] / total, 2),
        "pass_or_partial_rate": round((grade_counts["PASS"] + grade_counts["PARTIAL"]) / total, 2),
        "results":            all_results,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────


def _sidebar() -> None:
    with st.sidebar:
        st.title("🔬 RAG Fusion Lab")
        st.caption("Agentic Graph RAG — step-by-step")
        st.divider()

        st.markdown("**7-step pipeline**")
        steps = [
            ("✅", "Step 01", "Baseline RAG + format-aware chunking"),
            ("✅", "Step 02", "CSV tool calling (exact aggregates)"),
            ("✅", "Step 03", "BM25 hybrid retrieval (BM25 + dense RRF)"),
            ("✅", "Step 04", "Knowledge graph + Graph RAG (alias + BFS)"),
            ("✅", "Step 05", "Multi-Agent (QueryAnalyst → specialists → Critic)"),
            ("✅", "Step 06", "Context engineering + VSA (rerank → dedup → compress)"),
            ("✅", "Step 07", "Production hardening (cache + retry + confidence)"),
        ]
        for icon, step_id, desc in steps:
            color = "" if icon == "✅" else "color:#9e9e9e"
            st.markdown(
                f'<span style="{color}">{icon} **{step_id}** — {desc}</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Key insight — the deliberate hack**")
        st.markdown(
            "Step 01's **aggregate chunks** pre-compute sums, breakdowns, and "
            "date-period totals at index time. This is a deliberate hack that "
            "proves why pure vector RAG fails on tabular data. "
            "The real fix (Step 02) runs a **structured query tool** (Pandas) "
            "at query time — not a pre-baked summary. Aggregate chunks are "
            "stale, inflexible, and don't generalise to filters you didn't predict."
        )
        st.divider()
        st.markdown("**Architecture stack (top of pipeline)**")
        st.markdown(
            "- Step 06 VSA routes each query to one of 4 domain slices (Finance / HR / Engineering / General)\n"
            "- Each slice has its own system prompt, retrieval overrides, and evaluation suite\n"
            "- Step 07: production hardening — reliability, cost control, SLOs"
        )

        n_exp = len(st.session_state.get("exp_history", []))
        if n_exp:
            st.divider()
            st.caption(f"Experiments in session: **{n_exp}**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Experiment Lab
# ══════════════════════════════════════════════════════════════════════════════


def tab_experiment_lab() -> None:
    st.info(
        "**Step 01 (baseline RAG + format-aware chunking) is the fixed starting point.** "
        "Each time you click **Run**, you create a **new experiment** added to the history below. "
        "The comparison shows how your experiments stack up against the baselines from "
        "each step's eval_results.json.",
        icon="ℹ️",
    )

    left, right = st.columns([1, 2], gap="large")

    # ── LEFT: parameter panel ──────────────────────────────────────────────────
    with left:
        st.markdown("#### Parameters")

        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200, max_value=5000, value=2000, step=100,
            help="Step 01 baseline = 2000 chars.",
        )
        overlap = st.slider(
            "Overlap (characters)",
            min_value=0, max_value=600, value=200, step=50,
            help="Characters carried over between consecutive chunks.",
        )
        k_val = st.slider(
            "k (chunks to retrieve)",
            min_value=1, max_value=20, value=10,
        )
        strategy = st.radio(
            "Chunking strategy",
            ["Paragraph-aware", "Fixed-size (Step 01 style)"],
            help="Paragraph-aware accumulates paragraphs up to the size limit; fixed-size slides a window.",
        )
        strategy_key = "paragraph" if "Paragraph" in strategy else "fixed"

        include_csv_agg = st.checkbox(
            "Include CSV aggregate chunks",
            value=True,
            help=(
                "Aggregate chunks pre-compute sums/totals; "
                "disabling shows raw CSV rows only."
            ),
        )

        embed_backend = st.radio(
            "Embedding backend",
            ["🤗 Local (fastembed)", "🤗 Baseline MiniLM (Step 01)"],
            help=(
                "Both options run locally. 'Local (fastembed)' lets you pick a model; "
                "'Baseline MiniLM' uses the same all-MiniLM-L6-v2 the Step 01 index uses."
            ),
        )
        use_local = embed_backend.startswith("🤗 Local")

        local_model_name: str | None = None
        if use_local:
            model_label = st.selectbox(
                "Local model",
                list(LOCAL_EMBED_MODELS.keys()),
                index=0,
                help="Downloaded once (~30–100 MB) and cached in ~/.cache/fastembed/",
            )
            local_model_name = LOCAL_EMBED_MODELS[model_label]
            st.caption(f"`{local_model_name}` — ONNX runtime, no API calls.")
        else:
            st.caption(
                "Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace — "
                "same model the Step 01 baseline index was built with. No API calls."
            )

    # ── RIGHT: instant corpus preview ─────────────────────────────────────────
    with right:
        st.markdown("#### Corpus preview _(no API calls)_")

        with st.spinner("Re-chunking corpus…"):
            preview_chunks = _rechunk_corpus(chunk_size, overlap, strategy_key)

        sizes = [c["chars"] for c in preview_chunks]
        total_c = len(preview_chunks)
        delta_vs_step01 = total_c - STEP01_BASELINE_CHUNKS

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "Total chunks",
            total_c,
            delta=f"{delta_vs_step01:+d} vs Step 01",
        )
        mc2.metric("Avg size", f"{int(sum(sizes)/max(len(sizes),1))} ch")
        mc3.metric("Min size", f"{min(sizes) if sizes else 0} ch")
        mc4.metric("Max size", f"{max(sizes) if sizes else 0} ch")

        fig_hist = go.Figure(go.Histogram(x=sizes, nbinsx=40, marker_color="#1565c0"))
        fig_hist.update_layout(
            xaxis_title="Characters",
            yaxis_title="Count",
            height=220,
            margin=dict(t=10, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        all_sources = sorted(set(c["source"] for c in preview_chunks))
        sel_source = st.selectbox("File preview — pick a source file", all_sources, key="preview_src")
        file_chunks = [c for c in preview_chunks if c["source"] == sel_source]
        st.caption(f"{len(file_chunks)} chunks from `{sel_source}`")
        for fc in file_chunks[:10]:
            with st.expander(f"Chunk {fc['chunk_index']+1} — {fc['chars']} chars"):
                st.text(fc["text"][:600] + ("…" if fc["chars"] > 600 else ""))
        if len(file_chunks) > 10:
            st.caption(f"(showing first 10 of {len(file_chunks)} chunks)")

    # ── RUN section ────────────────────────────────────────────────────────────
    st.divider()

    est_time = "~30 s (local CPU)" if use_local else "~30–60 s (local CPU)"
    cost_note = "free (local)"

    col_run, col_info = st.columns([1, 4])
    run_btn = col_run.button("▶ Run Experiment", type="primary", use_container_width=True)
    col_info.info(
        f"chunk_size={chunk_size}  overlap={overlap}  k={k_val}  "
        f"strategy={strategy_key}  agg={'yes' if include_csv_agg else 'no'}  "
        f"embed={'fastembed' if use_local else 'baseline_minilm'}  |  "
        f"{est_time}  |  {cost_note}"
    )

    if run_btn:
        with st.status("Running experiment…", expanded=True):
            exp_result = _build_custom_index(
                chunk_size=chunk_size,
                overlap=overlap,
                k_eval=k_val,
                local_model=local_model_name,
                strategy=strategy_key,
                include_csv_agg=include_csv_agg,
            )
        st.session_state["exp_history"].append(exp_result)
        gc = exp_result["grade_counts"]
        st.success(
            f"Done — **{exp_result['label']}**: "
            f"{gc['PASS']} PASS / {gc['PARTIAL']} PARTIAL / {gc['FAIL']} FAIL  "
            f"({exp_result['pass_rate']:.0%} pass rate)"
        )

    # ── Experiment history table ───────────────────────────────────────────────
    history: list[dict] = st.session_state["exp_history"]
    if history:
        st.markdown("#### Experiment History")
        import pandas as pd

        hist_rows = []
        for exp in history:
            gc = exp["grade_counts"]
            hist_rows.append({
                "Label":      exp["label"],
                "Chunks":     exp["total_chunks"],
                "EmbedModel": exp["embed_model"],
                "Strategy":   exp.get("strategy", "—"),
                "ChunkSize":  exp["chunk_size"],
                "Overlap":    exp["overlap"],
                "k":          exp["k"],
                "PASS":       gc["PASS"],
                "PARTIAL":    gc["PARTIAL"],
                "FAIL":       gc["FAIL"],
                "PassRate":   f"{exp['pass_rate']:.0%}",
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

        # Comparison bar chart
        s1 = load_eval(STEP01_EVAL)
        s2 = load_eval(STEP02_EVAL)
        s3 = load_eval(STEP03_EVAL)
        s4 = load_eval(STEP04_EVAL)
        s5 = load_eval(STEP05_EVAL)
        s6 = load_eval(STEP06_EVAL)
        s7 = load_eval(STEP07_EVAL)
        all_runs: list[tuple[str, dict]] = []
        if s1:
            all_runs.append(("Step01 (baseline + format-aware)", s1))
        if s2 and s2.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step02 (CSV tool)", s2))
        if s3 and s3.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step03 (BM25 hybrid)", s3))
        if s4 and s4.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step04 (knowledge graph)", s4))
        if s5 and s5.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step05 (multi-agent)", s5))
        if s6 and s6.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step06 (context eng. + VSA)", s6))
        if s7 and s7.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step07 (production)", s7))
        for exp in history:
            all_runs.append((exp["label"], exp))

        st.markdown("#### Pass Rate Comparison")
        st.plotly_chart(_comparison_bar(all_runs), use_container_width=True)

        # Grade heatmap
        st.markdown("#### Grade Heatmap (Q01–Q29)")
        st.plotly_chart(_grade_heatmap(all_runs), use_container_width=True)

        # Per-experiment detail
        st.markdown("#### Experiment Detail")
        exp_labels = [exp["label"] for exp in history]
        sel_exp_label = st.selectbox("Select experiment", exp_labels, key="sel_exp_detail")
        sel_exp = next((e for e in history if e["label"] == sel_exp_label), None)
        if sel_exp:
            for r in sel_exp.get("results", []):
                grade = r.get("grade", "FAIL")
                icon = GRADE_EMOJI.get(grade, "")
                with st.expander(f"{icon} {r['id']} — {r.get('type','')}: {grade}"):
                    st.markdown(f"**Question:** {r['question']}")
                    st.markdown(f"**Answer:** {r.get('answer','')[:400]}{'…' if len(r.get('answer','')) > 400 else ''}")
                    hits = r.get("required_hits", [])
                    miss = r.get("required_missing", [])
                    if hits:
                        st.success(f"Found: {hits}")
                    if miss:
                        st.error(f"Missing: {miss}")
                    st.caption(f"Sources: {r.get('sources_retrieved', [])}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analysis
# ══════════════════════════════════════════════════════════════════════════════


def tab_analysis() -> None:
    s1 = load_eval(STEP01_EVAL)
    s2 = load_eval(STEP02_EVAL)
    s3 = load_eval(STEP03_EVAL)
    s4 = load_eval(STEP04_EVAL)
    s5 = load_eval(STEP05_EVAL)
    s6 = load_eval(STEP06_EVAL)
    s7 = load_eval(STEP07_EVAL)

    if not any([s1, s2, s3, s4, s5, s6, s7]):
        st.warning("No baseline eval results found. Run the evaluation scripts first.")
        st.code(
            "uv run python evaluation/run_eval.py --step step_01_baseline_rag\n"
            "uv run python evaluation/run_eval.py --step step_02_tools\n"
            "uv run python evaluation/run_eval.py --step step_03_hybrid_retrieval\n"
            "uv run python evaluation/run_eval.py --step step_04_knowledge_graph\n"
            "uv run python evaluation/run_eval.py --step step_05_multi_agent\n"
            "uv run python evaluation/run_eval.py --step step_06_context_engineering\n"
            "uv run python evaluation/run_eval.py --step step_07_production"
        )
        return

    # Build run registry — pass rates come from each step's eval_results.json (live).
    base_runs: list[tuple[str, dict]] = []
    step_specs: list[tuple[dict | None, str]] = [
        (s1, "Step01 (baseline + format-aware)"),
        (s2, "Step02 (CSV tool)"),
        (s3, "Step03 (BM25 hybrid)"),
        (s4, "Step04 (knowledge graph)"),
        (s5, "Step05 (multi-agent)"),
        (s6, "Step06 (context eng. + VSA)"),
        (s7, "Step07 (production)"),
    ]
    for sd, label in step_specs:
        if sd and sd.get("grade_counts", {}).get("PASS", 0) > 0:
            pct = round(sd["pass_rate"] * 100)
            base_runs.append((f"{label} ({pct}%)", sd))

    history: list[dict] = st.session_state["exp_history"]
    exp_options = [exp["label"] for exp in history]

    selected_exp_labels: list[str] = []
    if exp_options:
        selected_exp_labels = st.multiselect(
            "Add experiments to comparison (baselines always shown)",
            exp_options,
            default=exp_options[:3] if len(exp_options) <= 3 else exp_options[:3],
        )

    selected_exps = [exp for exp in history if exp["label"] in selected_exp_labels]
    all_runs: list[tuple[str, dict]] = base_runs + [(e["label"], e) for e in selected_exps]

    if not all_runs:
        st.info("No runs to display.")
        return

    # ── Grade distribution ─────────────────────────────────────────────────────
    st.subheader("Grade Distribution")
    st.plotly_chart(_comparison_bar(all_runs), use_container_width=True)

    # ── Per-question comparison table ──────────────────────────────────────────
    st.subheader("Per-question Comparison")
    import pandas as pd
    from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS

    q_ids = [q.id for q in GOLDEN_QUESTIONS]
    grade_maps: dict[str, dict[str, str]] = {}
    for label, d in all_runs:
        grade_maps[label] = {r["id"]: r.get("grade", "—") for r in d.get("results", [])}

    rows: list[dict[str, Any]] = []
    for qid in q_ids:
        gq = next((q for q in GOLDEN_QUESTIONS if q.id == qid), None)
        row: dict[str, Any] = {
            "ID":       qid,
            "Type":     gq.type if gq else "—",
            "Expected": gq.expected_outcome if gq else "—",
        }
        for label, _ in all_runs:
            g = grade_maps[label].get(qid, "—")
            row[label] = f"{GRADE_EMOJI.get(g, '')} {g}"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=500)

    # ── Question type breakdown ────────────────────────────────────────────────
    st.subheader("Question Type Breakdown")
    type_run_label, type_run_data = all_runs[0]
    if len(all_runs) > 1:
        type_run_label = st.selectbox(
            "Show type breakdown for run", [lbl for lbl, _ in all_runs], key="type_run_sel"
        )
        _, type_run_data = next(r for r in all_runs if r[0] == type_run_label)

    type_grades: dict[str, dict[str, int]] = {}
    for r in type_run_data.get("results", []):
        qt = r.get("type", "unknown")
        g  = r.get("grade", "FAIL")
        type_grades.setdefault(qt, {"PASS": 0, "PARTIAL": 0, "FAIL": 0})
        type_grades[qt][g] += 1

    if type_grades:
        types = list(type_grades.keys())
        fig_type = go.Figure(data=[
            go.Bar(name="PASS",    x=types, y=[type_grades[t]["PASS"]    for t in types], marker_color=GRADE_COLOR["PASS"]),
            go.Bar(name="PARTIAL", x=types, y=[type_grades[t]["PARTIAL"] for t in types], marker_color=GRADE_COLOR["PARTIAL"]),
            go.Bar(name="FAIL",    x=types, y=[type_grades[t]["FAIL"]    for t in types], marker_color=GRADE_COLOR["FAIL"]),
        ])
        fig_type.update_layout(barmode="stack", height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_type, use_container_width=True)

    # ── Latency scatter ────────────────────────────────────────────────────────
    st.subheader("Latency Analysis")
    latency_run_label = all_runs[0][0]
    if len(all_runs) > 1:
        latency_run_label = st.selectbox(
            "Latency data for run", [lbl for lbl, _ in all_runs], key="lat_run_sel"
        )
    _, latency_data = next(r for r in all_runs if r[0] == latency_run_label)

    lat_results = [
        r for r in latency_data.get("results", [])
        if r.get("retrieval_latency_ms", 0) > 0 or r.get("generation_latency_ms", 0) > 0
    ]
    if lat_results:
        ret_ms  = [r.get("retrieval_latency_ms", 0)  for r in lat_results]
        gen_ms  = [r.get("generation_latency_ms", 0) for r in lat_results]
        q_ids_l = [r["id"] for r in lat_results]
        grades  = [r.get("grade", "FAIL") for r in lat_results]
        colors_l = [GRADE_COLOR.get(g, "#555") for g in grades]

        fig_lat = go.Figure(go.Scatter(
            x=ret_ms, y=gen_ms,
            mode="markers+text",
            text=q_ids_l,
            textposition="top center",
            marker=dict(color=colors_l, size=9),
        ))
        fig_lat.update_layout(
            xaxis_title="Retrieval latency (ms)",
            yaxis_title="Generation latency (ms)",
            height=350,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.caption("No latency data available for this run (experiments don't capture it yet for baselines).")

    # ── Context length distribution ────────────────────────────────────────────
    st.subheader("Context Length Distribution")
    ctx_run_label = all_runs[0][0]
    if len(all_runs) > 1:
        ctx_run_label = st.selectbox(
            "Context lengths for run", [lbl for lbl, _ in all_runs], key="ctx_run_sel"
        )
    _, ctx_data = next(r for r in all_runs if r[0] == ctx_run_label)
    ctx_sizes = [r.get("context_chars", 0) for r in ctx_data.get("results", []) if r.get("context_chars", 0) > 0]
    if ctx_sizes:
        fig_ctx = go.Figure(go.Histogram(x=ctx_sizes, nbinsx=20, marker_color="#1565c0"))
        fig_ctx.update_layout(
            xaxis_title="Context size (chars)",
            yaxis_title="Count",
            height=250,
            margin=dict(t=10, b=20),
        )
        st.plotly_chart(fig_ctx, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Query Explorer
# ══════════════════════════════════════════════════════════════════════════════


def tab_query_explorer() -> None:
    st.caption(
        "Live retrieval from any stored index. "
        "The Step 01 baseline index was built with HuggingFace MiniLM "
        "(all-MiniLM-L6-v2). Experiments use whatever embedder you chose when running them."
    )

    # ── Index selector ─────────────────────────────────────────────────────────
    history: list[dict] = st.session_state["exp_history"]
    index_options: dict[str, tuple[Path, str, str]] = {}  # label → (db_path, coll_name, built_with)

    if CHROMA_DB.exists():
        index_options["Step 01 (baseline + format-aware)"] = (CHROMA_DB, CHROMA_COLLECTION, "local")
    for exp in history:
        db_p = Path(exp["db_path"])
        if db_p.exists():
            index_options[exp["label"]] = (db_p, exp["coll_name"], exp["embed_model"])

    if not index_options:
        st.warning("No indexes available. Run an evaluation first.")
        return

    sel_index_label = st.selectbox("Index", list(index_options.keys()))
    db_path, coll_name, built_with = index_options[sel_index_label]

    is_baseline_index = sel_index_label.startswith("Step 0")
    if is_baseline_index:
        st.info(
            "The Step 01 baseline index was built with **HuggingFace MiniLM "
            "(all-MiniLM-L6-v2, 384d)**. Use the same embedder for the query "
            "or you'll get nonsense matches."
        )

    # ── Embedding mode ─────────────────────────────────────────────────────────
    embed_options = ["Same as index was built with (recommended)", "Baseline MiniLM (Step 01)"] + list(LOCAL_EMBED_MODELS.keys())
    embed_sel = st.selectbox("Query embedding mode", embed_options, key="qe_embed")

    if embed_sel == "Same as index was built with (recommended)":
        query_embed_mode = built_with
        query_local_model: str | None = None
    elif embed_sel == "Baseline MiniLM (Step 01)":
        query_embed_mode = "baseline"
        query_local_model = None
    else:
        query_embed_mode = "local"
        query_local_model = LOCAL_EMBED_MODELS[embed_sel]

    # ── Question input ─────────────────────────────────────────────────────────
    s1 = load_eval(STEP01_EVAL)
    golden_map: dict[str, str] = {}
    if s1:
        for r in s1["results"]:
            golden_map[f"{r['id']} — {r['question'][:70]}"] = r["question"]

    preset = st.selectbox(
        "Start from a golden question (optional)",
        ["— custom —"] + list(golden_map.keys()),
        key="qe_preset",
    )
    default_q = golden_map.get(preset, "")
    question_raw = st.text_input(
        "Question",
        value=default_q,
        placeholder="Ask anything about Vertexia…",
        key="qe_question",
    )
    question = (question_raw or "").strip()

    qe_k = st.slider("k (chunks to retrieve)", 1, 20, 10, key="qe_k")

    # ── Retrieve ───────────────────────────────────────────────────────────────
    retrieve_btn = st.button("🔍 Retrieve", disabled=not question, key="qe_retrieve")

    if retrieve_btn and question:
        with st.spinner("Embedding query…"):
            try:
                if query_embed_mode == "local" and query_local_model:
                    qvec = _embed_local([question], query_local_model)[0]
                else:
                    # "baseline" or any unknown/legacy mode → use baseline MiniLM
                    qvec = _embed_query_baseline(question)
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                st.stop()

        try:
            coll = _get_collection(str(db_path), coll_name)
        except Exception as e:
            st.error(f"Could not open collection: {e}")
            st.stop()

        res = coll.query(
            query_embeddings=[qvec],
            n_results=min(qe_k, coll.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs_r  = (res["documents"] or [[]])[0]
        metas_r = (res["metadatas"] or [[]])[0]
        dists_r = (res["distances"] or [[]])[0]

        st.session_state["qe_last"] = {
            "question": question,
            "preset":   preset,
            "docs":     docs_r,
            "metas":    metas_r,
            "dists":    dists_r,
        }

    # ── Show results ───────────────────────────────────────────────────────────
    if "qe_last" in st.session_state:
        last = st.session_state["qe_last"]
        st.markdown(f"---\n**Retrieved for:** _{last['question']}_")

        docs_show  = last["docs"]
        metas_show = last["metas"]
        dists_show = last["dists"]

        for i, (doc, meta, dist) in enumerate(zip(docs_show, metas_show, dists_show)):
            sim = 1.0 - dist / 2.0
            ctype = str(meta.get("chunk_type", meta.get("format", "—")))
            icon  = CHUNK_TYPE_ICON.get(ctype, "📄")
            bar_w = int(sim * 30)
            bar   = "█" * bar_w + "░" * (30 - bar_w)
            src   = meta.get("source", "—")
            label = f"{icon} #{i+1}  `{src}` [{ctype}]  {bar}  {sim:.3f}"
            with st.expander(label, expanded=(i == 0)):
                st.text(doc[:1200] + ("…" if len(doc) > 1200 else ""))
                st.caption(
                    f"dept: {meta.get('department','—')}  |  "
                    f"chunk_index: {meta.get('chunk_index','—')}"
                )

        # Aggregate chunk callout
        agg_positions = [i for i, m in enumerate(metas_show) if m.get("chunk_type") == "aggregate"]
        if agg_positions:
            st.success(
                f"🔮 Aggregate chunk(s) at positions {[p+1 for p in agg_positions]} — "
                "aggregation queries can be answered!"
            )

        # Grading preview if golden question
        if last["preset"] != "— custom —" and s1:
            qid_preview = last["preset"].split(" — ")[0]
            from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
            gq = next((q for q in GOLDEN_QUESTIONS if q.id == qid_preview), None)
            if gq:
                combined = " ".join(docs_show).lower()
                hits_c   = [f for f in gq.required_facts if f.lower() in combined]
                miss_c   = [f for f in gq.required_facts if f.lower() not in combined]
                disq_c   = [d for d in gq.disqualifiers  if d.lower() in combined]
                st.markdown("**Grading preview** (based on retrieved context, before LLM generation):")
                if disq_c:
                    st.error(f"Disqualifier in context: {disq_c}")
                if hits_c:
                    st.success(f"Required facts in context: {hits_c}")
                if miss_c:
                    st.error(f"Required facts NOT in context: {miss_c}")
                if not miss_c and not disq_c:
                    st.success("All required facts present — LLM will likely PASS.")

        # Optional LLM answer
        st.markdown("---")
        st.caption("Generating an answer costs ~$0.001 in API credits.")
        if st.button("🤖 Generate Answer", key="qe_gen"):
            from step_01_baseline_rag.implementation.generate import generate_answer
            from step_01_baseline_rag.implementation.pipeline import RAGResult
            from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context

            chunk_objs = [
                RetrievedChunk(
                    text=d,
                    source=str(m.get("source", "")),
                    department=str(m.get("department", "")),
                    format=str(m.get("format", "")),
                    chunk_index=int(str(m.get("chunk_index") or 0)),
                    distance=dist,
                )
                for d, m, dist in zip(docs_show, metas_show, dists_show)
            ]
            context = format_context(chunk_objs)
            with st.spinner("Generating…"):
                answer, provider = generate_answer(context, last["question"])

            st.markdown(f"**Answer** _(via {provider})_:")
            st.write(answer)

            # Grade if golden
            if last["preset"] != "— custom —" and s1:
                qid_g = last["preset"].split(" — ")[0]
                from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
                from step_01_baseline_rag.evaluation.run_eval import score as grade_fn
                gq_g = next((q for q in GOLDEN_QUESTIONS if q.id == qid_g), None)
                if gq_g:
                    fake = RAGResult(
                        question=last["question"],
                        answer=answer,
                        provider=provider,
                        retrieved_chunks=chunk_objs,
                        context_sent=context,
                        context_chars=len(context),
                        retrieval_latency_ms=0,
                        generation_latency_ms=0,
                    )
                    sc   = grade_fn(fake, gq_g)
                    grd  = sc["grade"]
                    st.markdown(f"**Grade: {GRADE_EMOJI.get(grd,'')} {grd}**")
                    if sc["required_hits"]:
                        st.success(f"Found: {sc['required_hits']}")
                    if sc["required_missing"]:
                        st.error(f"Missing: {sc['required_missing']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Chunk Browser
# ══════════════════════════════════════════════════════════════════════════════


def tab_chunk_browser() -> None:
    st.caption(
        "Browse chunks from any built index. No API calls. "
        "Aggregate chunks (🔮) contain pre-computed summaries."
    )

    # ── Source selector ────────────────────────────────────────────────────────
    # Step 01 now uses format-aware chunking (the old naive baseline was merged in),
    # so the baseline and "format-aware" sources are the same SmartChunk stream.
    source_options: dict[str, str] = {"Step 01 (baseline + format-aware)": "step01"}
    history: list[dict] = st.session_state["exp_history"]
    for exp in history:
        source_options[exp["label"]] = f"exp:{exp['label']}"

    sel_source_label = st.selectbox("Source index", list(source_options.keys()), key="cb_source")
    source_key = source_options[sel_source_label]

    # ── Load chunks ────────────────────────────────────────────────────────────
    @st.cache_resource
    def _get_step01_chunks():
        from step_01_baseline_rag.implementation.chunker import load_and_chunk as _l01
        return _l01(CORPUS)

    with st.spinner("Loading chunks…"):
        if source_key == "step01":
            raw_chunks = _get_step01_chunks()
            def _ctype(c: Any) -> str:
                return str(getattr(c, "chunk_type", "prose"))
        else:
            # experiment — pull from ChromaDB
            exp_label = source_key.removeprefix("exp:")
            exp_data  = next((e for e in history if e["label"] == exp_label), None)
            if exp_data is None:
                st.error("Experiment not found in session history.")
                return
            db_p = Path(exp_data["db_path"])
            if not db_p.exists():
                st.error(f"ChromaDB path `{db_p}` does not exist.")
                return
            try:
                coll_br = _get_collection(str(db_p), exp_data["coll_name"])
            except Exception as e:
                st.error(f"Could not open collection: {e}")
                return
            # Get all docs from Chroma (limited to 2000 for performance)
            all_items = coll_br.get(
                limit=2000,
                include=["documents", "metadatas"],
            )
            exp_docs   = all_items.get("documents") or []
            exp_metas  = all_items.get("metadatas") or []

            # Wrap into simple objects for unified display below
            class _FakeChunk:
                def __init__(self, text: str, meta: dict[str, Any]) -> None:
                    self.text        = text
                    self.source      = str(meta.get("source", "—"))
                    self.chunk_type  = str(meta.get("chunk_type", "prose"))
                    self.format      = str(meta.get("format", "—"))
                    self.chunk_index = int(str(meta.get("chunk_index") or 0))
                    self.department  = str(meta.get("department", "—"))

            raw_chunks = [_FakeChunk(d, dict(m)) for d, m in zip(exp_docs, exp_metas)]

            def _ctype(c: Any) -> str:
                return str(getattr(c, "chunk_type", "prose"))

    # ── Stats ──────────────────────────────────────────────────────────────────
    st.markdown(f"**{len(raw_chunks)} chunks** from `{sel_source_label}`")

    type_counts: dict[str, int] = {}
    for c in raw_chunks:
        t = _ctype(c)
        type_counts[t] = type_counts.get(t, 0) + 1

    metric_cols = st.columns(max(len(type_counts), 1))
    for i, (t, n) in enumerate(sorted(type_counts.items())):
        icon = CHUNK_TYPE_ICON.get(t, "📄")
        metric_cols[i % len(metric_cols)].metric(f"{icon} {t}", n)

    # Type distribution
    fig_types = go.Figure(go.Bar(
        x=list(type_counts.keys()),
        y=list(type_counts.values()),
        marker_color=[CHUNK_TYPE_COLOR.get(t, "#555") for t in type_counts],
    ))
    fig_types.update_layout(height=220, margin=dict(t=10, b=10))
    st.plotly_chart(fig_types, use_container_width=True)

    # ── Filters ────────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    all_types_list = sorted(type_counts.keys())
    all_sources_list = sorted(set(c.source for c in raw_chunks))

    selected_types_cb = f1.multiselect(
        "Filter by type", all_types_list, default=all_types_list, key="cb_types"
    )
    selected_source_cb = f2.selectbox(
        "Filter by source file", ["(all)"] + all_sources_list, key="cb_src"
    )

    filtered = [
        c for c in raw_chunks
        if _ctype(c) in selected_types_cb
        and (selected_source_cb == "(all)" or c.source == selected_source_cb)
    ]
    st.caption(f"Showing {min(len(filtered), 100)} of {len(filtered)} matching chunks")

    # ── Chunk list ─────────────────────────────────────────────────────────────
    for c in filtered[:100]:
        ctype   = _ctype(c)
        icon    = CHUNK_TYPE_ICON.get(ctype, "📄")
        char_ct = len(c.text)
        label   = f"**[{ctype.upper()}]** `{c.source}` — {char_ct} chars"

        if ctype == "aggregate":
            with st.expander(f"🔮 {label}  ← pre-computed summary", expanded=False):
                st.code(c.text, language=None)
        else:
            with st.expander(f"{icon} {label}"):
                st.text(c.text[:800] + ("…" if char_ct > 800 else ""))

    if len(filtered) > 100:
        st.info(
            f"Showing first 100 of {len(filtered)} matching chunks. "
            "Narrow your filters to see more."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Step Progress
# ══════════════════════════════════════════════════════════════════════════════


def tab_step_progress() -> None:
    """Show accuracy progression from Step 01 → 08 with per-step annotations."""
    import pandas as pd

    STEP_META = [
        ("Step 01", "Baseline +\nformat-aware",    STEP01_EVAL, "MiniLM dense retrieval + format-aware chunks (CSV rows, aggregate pre-compute, section splits)"),
        ("Step 02", "CSV tool\ncalling",            STEP02_EVAL, "Pandas CSV tool for exact aggregates at query time (totals, breakdowns, period filters)"),
        ("Step 03", "BM25 hybrid\nretrieval",       STEP03_EVAL, "BM25 + dense RRF fusion — keyword recall meets semantic precision"),
        ("Step 04", "Knowledge graph\n+ Graph RAG", STEP04_EVAL, "Entity edges + alias resolution + BFS blast-radius expansion (graph_rag merged in)"),
        ("Step 05", "Multi-Agent\nSystem",          STEP05_EVAL, "QueryAnalyst → specialists (retrieval / graph / structured) → Critic → Synthesis"),
        ("Step 06", "Context engineering\n+ VSA",   STEP06_EVAL, "CrossEncoder rerank → dedup → compress → XML format + domain slicing (Finance/HR/Eng/General)"),
        ("Step 07", "Production\nhardening",        STEP07_EVAL, "Semantic cache + retry/backoff + confidence scoring + health monitor + graceful degradation"),
    ]

    evals = [(label, short, load_eval(path), note) for label, short, path, note in STEP_META]
    available = [
        (label, short, d, note) for label, short, d, note in evals
        if d is not None and sum(d.get("grade_counts", {}).values()) > 0
    ]

    if not available:
        st.warning("No eval results found. Run evaluation scripts first.")
        return

    # ── Progression line chart ─────────────────────────────────────────────────
    st.subheader("Accuracy Progression")

    x_labels  = [label  for label, _, _, _ in available]
    pass_rates = [d["pass_rate"] * 100 for _, _, d, _ in available]
    pp_rates   = [d["pass_or_partial_rate"] * 100 for _, _, d, _ in available]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels, y=pass_rates,
        mode="lines+markers+text",
        name="PASS %",
        line=dict(color="#2e7d32", width=3),
        marker=dict(size=10),
        text=[f"{v:.0f}%" for v in pass_rates],
        textposition="top center",
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=pp_rates,
        mode="lines+markers",
        name="PASS + PARTIAL %",
        line=dict(color="#1565c0", width=2, dash="dot"),
        marker=dict(size=7),
    ))
    fig.update_layout(
        yaxis=dict(title="Pass rate (%)", range=[0, 110], ticksuffix="%"),
        height=340,
        margin=dict(t=30, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Step cards ────────────────────────────────────────────────────────────
    st.subheader("What Each Step Added")
    cols = st.columns(min(len(available), 3))
    for i, (label, short, d, note) in enumerate(available):
        gc = d["grade_counts"]
        pct = d["pass_rate"] * 100
        delta = ""
        if i > 0:
            prev_pct = available[i - 1][2]["pass_rate"] * 100
            diff = pct - prev_pct
            delta = f"  **+{diff:.0f}pp**" if diff > 0 else ""
        with cols[i % 3]:
            st.markdown(
                f"**{label}** — {pct:.0f}%{delta}\n\n"
                f"✅ {gc['PASS']} &nbsp; ⚠️ {gc['PARTIAL']} &nbsp; ❌ {gc['FAIL']}\n\n"
                f"_{note}_"
            )

    # ── Grade heatmap across all steps ────────────────────────────────────────
    st.subheader("Per-question Grade Heatmap — All Steps")
    heatmap_runs = [(label, d) for label, _, d, _ in available]
    st.plotly_chart(_grade_heatmap(heatmap_runs), use_container_width=True)

    # ── Questions fixed per step ───────────────────────────────────────────────
    st.subheader("Questions Fixed at Each Step")
    if len(available) >= 2:
        step_labels = [label for label, _, _, _ in available]
        sel_step = st.selectbox("Show questions that changed vs previous step", step_labels[1:], key="sp_sel_step")
        sel_idx  = next(i for i, (l, _, _, _) in enumerate(available) if l == sel_step)
        prev_d   = available[sel_idx - 1][2]
        curr_d   = available[sel_idx][2]

        prev_grades = {r["id"]: r.get("grade", "FAIL") for r in prev_d.get("results", [])}
        curr_grades = {r["id"]: r.get("grade", "FAIL") for r in curr_d.get("results", [])}

        improved = [(qid, prev_grades[qid], curr_grades[qid])
                    for qid in prev_grades if curr_grades.get(qid, "FAIL") != prev_grades[qid]
                    and GRADE_NUM.get(curr_grades.get(qid, "FAIL"), 0) > GRADE_NUM.get(prev_grades[qid], 0)]
        regressed = [(qid, prev_grades[qid], curr_grades[qid])
                     for qid in prev_grades if curr_grades.get(qid, "FAIL") != prev_grades[qid]
                     and GRADE_NUM.get(curr_grades.get(qid, "FAIL"), 0) < GRADE_NUM.get(prev_grades[qid], 0)]

        c1, c2 = st.columns(2)
        with c1:
            if improved:
                st.success(f"**Improved ({len(improved)})**")
                for qid, before, after in improved:
                    st.markdown(f"- {qid}: {GRADE_EMOJI.get(before,'')} {before} → {GRADE_EMOJI.get(after,'')} {after}")
            else:
                st.info("No improvements vs previous step.")
        with c2:
            if regressed:
                st.error(f"**Regressed ({len(regressed)})**")
                for qid, before, after in regressed:
                    st.markdown(f"- {qid}: {GRADE_EMOJI.get(before,'')} {before} → {GRADE_EMOJI.get(after,'')} {after}")
            else:
                st.success("No regressions vs previous step.")

    # ── Provider breakdown (latest available step in 5/6/7) ───────────────────
    # Provider data lives in the per-step eval_results.json under each result's
    # "provider" field. Prefer the latest step that has results.
    s5_pb = load_eval(STEP05_EVAL)
    s6_pb = load_eval(STEP06_EVAL)
    s7_pb = load_eval(STEP07_EVAL)
    prov_step = None
    prov_label = ""
    if s7_pb and s7_pb.get("grade_counts", {}).get("PASS", 0) > 0:
        prov_step, prov_label = s7_pb, "Step 07 — Provider Breakdown (Gateway V2)"
    elif s6_pb and s6_pb.get("grade_counts", {}).get("PASS", 0) > 0:
        prov_step, prov_label = s6_pb, "Step 06 — Provider Breakdown (Gateway V2)"
    elif s5_pb and s5_pb.get("grade_counts", {}).get("PASS", 0) > 0:
        prov_step, prov_label = s5_pb, "Step 05 — Provider Breakdown (Gateway V2)"
    if prov_step:
        st.subheader(prov_label)
        providers: dict[str, int] = {}
        for r in prov_step.get("results", []):
            p = r.get("provider", "unknown")
            providers[p] = providers.get(p, 0) + 1
        if providers:
            fig_p = go.Figure(go.Bar(
                x=list(providers.keys()),
                y=list(providers.values()),
                marker_color="#1565c0",
            ))
            fig_p.update_layout(
                height=220,
                yaxis_title="Questions answered",
                margin=dict(t=10, b=20),
            )
            st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Live Compare
# ══════════════════════════════════════════════════════════════════════════════


# ── Cache helpers (declared at module level for @st.cache_resource) ────────────


# Single registry for the 6 step factories below. Adding a new step is one
# row here, not a new ~6-line @st.cache_resource block + a parallel call site.
_STEP_REGISTRY: dict[str, tuple[str, str, int]] = {
    "step_02_tools":                ("step_02_tools.implementation.pipeline",                "Step02ToolsRAG", 10),
    "step_03_hybrid_retrieval":     ("step_03_hybrid_retrieval.implementation.pipeline",     "Step03HybridRAG", 10),
    "step_04_knowledge_graph":      ("step_04_knowledge_graph.implementation.pipeline",      "Step04RAG",      10),
    "step_05_multi_agent":          ("step_05_multi_agent.implementation.pipeline",          "Step05RAG",      10),
    "step_06_context_engineering":  ("step_06_context_engineering.implementation.pipeline",  "Step06RAG",       5),
    "step_07_production":           ("step_07_production.implementation.pipeline",           "Step07RAG",       5),
}


@st.cache_resource
def _load_step_rag(step_key: str):
    """Lazy-load and build the RAG class for `step_key`. Each step's heavy
    init (Chroma index, BM25, graph, CrossEncoder, cache) is paid once per
    Streamlit session thanks to @st.cache_resource."""
    import importlib
    module_path, class_name, k = _STEP_REGISTRY[step_key]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(k=k).build()


# Thin per-step accessors kept for call-site readability and to preserve
# Streamlit's per-function cache identity.
def _load_step02_rag(): return _load_step_rag("step_02_tools")
def _load_step03_rag(): return _load_step_rag("step_03_hybrid_retrieval")
def _load_step04_rag(): return _load_step_rag("step_04_knowledge_graph")
def _load_step05_rag(): return _load_step_rag("step_05_multi_agent")
def _load_step06_rag(): return _load_step_rag("step_06_context_engineering")
def _load_step07_rag(): return _load_step_rag("step_07_production")


@st.cache_resource
def _load_graph():
    from step_04_knowledge_graph.implementation.graph_store import load_or_build
    return load_or_build(CORPUS, GRAPH_PATH)


@st.cache_data(ttl=30)
def _get_gateway_providers_cached():
    try:
        import httpx
        r = httpx.get("http://localhost:8100/v1/providers", timeout=3)
        return r.json().get("providers", [])
    except Exception:
        return []


def _gateway_generate(context: str, question: str, provider: str | None, system: str) -> tuple[str, str]:
    from llm_gatewayV2.client import LLM
    llm = LLM(base_url="http://localhost:8100", timeout=90)
    kwargs: dict = {"max_tokens": 512, "temperature": 0.0}
    if provider:
        kwargs["provider"] = provider
    result = llm.chat(
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
        system=system,
        **kwargs,
    )
    return result.get("text", ""), f"gateway:{result.get('provider', 'unknown')}"


def _run_live_compare_steps(question: str, provider: str | None, k: int) -> list[dict]:
    """Run all 7 pipeline steps sequentially; return list of result dicts."""
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context

    SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions about Vertexia Inc. "
        "Answer based only on the provided context. Be concise and precise."
    )

    results: list[dict] = []

    progress_bar = st.progress(0.0, text="Running Step 01…")

    # ── Step 01 — Baseline RAG + format-aware chunking ─────────────────────────
    step_label = "Step 01"
    step_name  = "Baseline + format-aware"
    try:
        t0 = time.perf_counter()
        coll01 = _get_collection(str(CHROMA_DB), CHROMA_COLLECTION)
        qvec = _embed_query_baseline(question)
        res01 = coll01.query(
            query_embeddings=[qvec],
            n_results=min(k, coll01.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs01  = (res01["documents"] or [[]])[0]
        metas01 = (res01["metadatas"] or [[]])[0]
        dists01 = (res01["distances"] or [[]])[0]
        chunks01 = [
            RetrievedChunk(
                text=d,
                source=str(m.get("source", "")),
                department=str(m.get("department", "")),
                format=str(m.get("format", "")),
                chunk_index=int(str(m.get("chunk_index") or 0)),
                distance=dist,
            )
            for d, m, dist in zip(docs01, metas01, dists01)
        ]
        context01 = format_context(chunks01)
        answer01, prov01 = _gateway_generate(context01, question, provider, SYSTEM_PROMPT)
        latency01 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": chunks01,
            "context_parts": {"Vector (MiniLM, format-aware chunks)": context01},
            "full_context": context01,
            "answer": answer01,
            "provider": prov01,
            "latency_ms": latency01,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(1 / 7, text="Running Step 02…")

    # ── Step 02 — CSV tool calling ─────────────────────────────────────────────
    step_label = "Step 02"
    step_name  = "CSV tool"
    try:
        t0 = time.perf_counter()
        rag02 = _load_step02_rag()
        result02 = rag02.query(question)
        latency02 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": list(getattr(result02, "retrieved_chunks", []) or []),
            "context_parts": {
                "CSV tool + vector": (
                    "Pandas CSV tool runs at query time for exact aggregates "
                    "(totals, breakdowns, period filters). Falls back to vector for prose."
                ),
                "Context sent": getattr(result02, "context_sent", "") or "",
            },
            "full_context": getattr(result02, "context_sent", "") or "",
            "answer": result02.answer,
            "provider": getattr(result02, "provider", "tools"),
            "latency_ms": latency02,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(2 / 7, text="Running Step 03…")

    # ── Step 03 — BM25 hybrid retrieval (BM25 + dense RRF) ─────────────────────
    step_label = "Step 03"
    step_name  = "BM25 hybrid"
    try:
        t0 = time.perf_counter()
        rag03 = _load_step03_rag()
        result03 = rag03.query(question)
        latency03 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": list(getattr(result03, "retrieved_chunks", []) or []),
            "context_parts": {
                "BM25 + Dense RRF": getattr(result03, "context_sent", "") or "",
            },
            "full_context": getattr(result03, "context_sent", "") or "",
            "answer": result03.answer,
            "provider": getattr(result03, "provider", "hybrid"),
            "latency_ms": latency03,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(3 / 7, text="Running Step 04…")

    # ── Step 04 — Knowledge graph + Graph RAG ──────────────────────────────────
    step_label = "Step 04"
    step_name  = "Knowledge graph + Graph RAG"
    try:
        t0 = time.perf_counter()
        rag04 = _load_step04_rag()
        result04 = rag04.query(question)
        latency04 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": list(getattr(result04, "retrieved_chunks", []) or []),
            "context_parts": {
                "Graph + BM25 + Dense": (
                    "Entity edges + alias resolution + BFS blast-radius expansion "
                    "(graph_rag merged into this step)."
                ),
                "Context sent": getattr(result04, "context_sent", "") or "",
            },
            "full_context": getattr(result04, "context_sent", "") or "",
            "answer": result04.answer,
            "provider": getattr(result04, "provider", "graph"),
            "latency_ms": latency04,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(4 / 7, text="Running Step 05…")

    # ── Step 05 — Multi-Agent system ───────────────────────────────────────────
    step_label = "Step 05"
    step_name  = "Multi-Agent"
    try:
        t0 = time.perf_counter()
        rag05 = _load_step05_rag()
        result05 = rag05.query(question)
        latency05 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": list(getattr(result05, "retrieved_chunks", []) or []),
            "context_parts": {
                "Multi-Agent Pipeline": (
                    "QueryAnalyst → RetrievalSpecialist (+ sub-question retrieval) → "
                    "GraphNavigator → StructuredData → Synthesis → Critic"
                ),
                "Context sent": getattr(result05, "context_sent", "") or "",
            },
            "full_context": getattr(result05, "context_sent", "") or "",
            "answer": result05.answer,
            "provider": getattr(result05, "provider", "multi-agent"),
            "latency_ms": latency05,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(5 / 7, text="Running Step 06…")

    # ── Step 06 — Context engineering + VSA ────────────────────────────────────
    step_label = "Step 06"
    step_name  = "Context eng. + VSA"
    try:
        t0 = time.perf_counter()
        rag06 = _load_step06_rag()
        ext06 = rag06.query_extended(question)
        result06 = ext06.rag_result
        ce06     = ext06.ce_metrics
        latency06 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": [],
            "context_parts": {
                "Routing (VSA)": (
                    f"Slice: {ext06.slice_name}  (confidence={ext06.router_confidence:.2f})\n"
                    f"Finance · HR · Engineering · General — domain-specific prompt + overrides"
                ),
                "CE Pipeline": (
                    f"Retrieve → CrossEncoder rerank → dedup → compress → XML format\n"
                    f"raw={ce06.get('raw_chars', 0):,} chars → "
                    f"engineered={ce06.get('engineered_chars', 0):,} chars "
                    f"({ce06.get('compression_ratio', 1.0):.0%}) | "
                    f"chunks {ce06.get('chunks_before', 0)}→"
                    f"{ce06.get('chunks_after_dedup', ce06.get('chunks_final', 0))}→"
                    f"{ce06.get('chunks_final', 0)}"
                ),
                "Engineered Context (XML)": result06.context_sent,
            },
            "full_context": result06.context_sent,
            "answer": result06.answer,
            "provider": getattr(result06, "provider", "ce+vsa"),
            "latency_ms": latency06,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(6 / 7, text="Running Step 07…")

    # ── Step 07 — Production hardening ─────────────────────────────────────────
    step_label = "Step 07"
    step_name  = "Production"
    try:
        t0 = time.perf_counter()
        rag07 = _load_step07_rag()
        ext07 = rag07.query_extended(question)
        result07 = ext07.rag_result
        ce07     = ext07.ce_metrics
        health07 = ext07.health_snapshot
        latency07 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": [],
            "context_parts": {
                "Routing": (
                    f"Slice: {ext07.slice_name}  (confidence={ext07.router_confidence:.2f})\n"
                    f"Cache: {'HIT' if ext07.from_cache else 'miss'} | "
                    f"Answer confidence: {ext07.confidence_label} ({ext07.confidence_score:.2f})"
                ),
                "CE metrics": (
                    f"raw={ce07.get('raw_chars', 0):,} chars → "
                    f"engineered={ce07.get('engineered_chars', 0):,} chars "
                    f"({ce07.get('compression_ratio', 1.0):.0%})"
                ),
                "Health": (
                    f"p50={health07.get('p50_latency_ms','?')}ms  "
                    f"p95={health07.get('p95_latency_ms','?')}ms  "
                    f"SLO={health07.get('slo_compliance',0):.0%}  "
                    f"status={health07.get('status','?')}"
                ),
            },
            "full_context": "",
            "answer": result07.answer,
            "provider": getattr(result07, "provider", "prod"),
            "latency_ms": latency07,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(1.0, text="All steps complete!")
    return results


def _grade_step_result(result: dict, golden_question: Any) -> dict:
    """Grade a step result against a golden question. Returns updated result dict."""
    if golden_question is None:
        return {**result, "grade": None, "required_hits": [], "required_missing": []}

    from step_01_baseline_rag.evaluation.run_eval import score as grade_fn
    from step_01_baseline_rag.implementation.pipeline import RAGResult
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk

    fake = RAGResult(
        question=golden_question.question,
        answer=result["answer"],
        provider=result["provider"],
        retrieved_chunks=result["chunks"],
        context_sent=result["full_context"],
        context_chars=len(result["full_context"]),
        retrieval_latency_ms=0.0,
        generation_latency_ms=result["latency_ms"],
    )
    scoring = grade_fn(fake, golden_question)
    return {
        **result,
        "grade": scoring["grade"],
        "required_hits": scoring["required_hits"],
        "required_missing": scoring["required_missing"],
    }


def tab_live_compare() -> None:
    """Live Compare tab: run all pipeline steps for one question."""
    from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS

    left_col, right_col = st.columns([35, 65])

    with left_col:
        st.markdown("#### Question")

        # Question selector
        question_options = ["✏️ Custom question"] + [
            f"{q.id} — {q.question[:65]}" for q in GOLDEN_QUESTIONS
        ]
        sel_question_opt = st.selectbox("Select question", question_options, key="lc_question_sel")

        if sel_question_opt == "✏️ Custom question":
            custom_q = st.text_area(
                "Your question",
                placeholder="Ask anything about Vertexia…",
                key="lc_custom_q",
                height=100,
            )
            active_question = (custom_q or "").strip()
            active_golden   = None
        else:
            q_id = sel_question_opt.split(" — ")[0]
            active_golden   = next((q for q in GOLDEN_QUESTIONS if q.id == q_id), None)
            active_question = active_golden.question if active_golden else ""

        st.divider()

        # Gateway provider selector
        st.markdown("#### Gateway model")
        gw_providers = _get_gateway_providers_cached()
        provider_options = ["⚡ Auto (gateway routes)"] + [p for p in gw_providers]
        sel_provider_opt = st.selectbox("Provider", provider_options, key="lc_provider_sel")
        active_provider: str | None = None if sel_provider_opt.startswith("⚡") else sel_provider_opt

        st.divider()

        # k slider
        lc_k = st.slider("Chunks to retrieve", 3, 15, 10, key="lc_k")

        st.divider()

        run_btn = st.button(
            "▶ Run All Steps",
            type="primary",
            use_container_width=True,
            disabled=not active_question,
            key="lc_run_btn",
        )

    with right_col:
        if "lc_results" not in st.session_state:
            # Show the static technique table before first run
            st.markdown("#### Pipeline techniques — what each step adds")
            st.markdown(
                """
| Step | Technique added |
|---|---|
| Step 01 | Baseline RAG: MiniLM dense retrieval + format-aware chunks (CSV rows + aggregate pre-compute + section splits) |
| Step 02 | CSV tool calling: Pandas at query time for exact aggregates |
| Step 03 | BM25 + Dense RRF fusion (keyword recall meets semantic precision) |
| Step 04 | Knowledge graph + Graph RAG: entity edges, alias resolution, BFS blast-radius |
| Step 05 | Multi-agent: QueryAnalyst → specialists (retrieval / graph / structured) → Critic → Synthesis |
| Step 06 | Context engineering + VSA: rerank → dedup → compress → XML + domain slicing (Finance/HR/Eng/General) |
| Step 07 | Production: semantic cache + retry/backoff + confidence scoring + health monitor + graceful degradation |
"""
            )

    # ── Run all steps ──────────────────────────────────────────────────────────
    if run_btn and active_question:
        with st.spinner("Initialising pipeline objects (cached after first run)…"):
            raw_results = _run_live_compare_steps(active_question, active_provider, lc_k)

        # Grade results against golden question (if applicable)
        graded: list[dict] = []
        for r in raw_results:
            graded.append(_grade_step_result(r, active_golden))

        st.session_state["lc_results"] = {
            "question":  active_question,
            "golden":    active_golden,
            "results":   graded,
        }

    # ── Display results ────────────────────────────────────────────────────────
    if "lc_results" in st.session_state:
        lc_data = st.session_state["lc_results"]
        lc_results: list[dict] = lc_data["results"]
        lc_golden  = lc_data["golden"]

        st.divider()
        st.markdown(f"**Question:** _{lc_data['question']}_")

        # 1. Summary scorecard ─────────────────────────────────────────────────
        st.markdown("#### Summary scorecard")
        score_cols = st.columns(len(lc_results))
        for i, r in enumerate(lc_results):
            with score_cols[i]:
                grade = r.get("grade")
                step_short = r["step"]
                if r["_error"]:
                    st.markdown(
                        f"<div style='text-align:center;padding:8px;background:#ffebee;"
                        f"border-radius:6px;font-size:0.8rem'>"
                        f"<b>{step_short}</b><br>⚠️ ERROR</div>",
                        unsafe_allow_html=True,
                    )
                elif grade is None:
                    st.markdown(
                        f"<div style='text-align:center;padding:8px;background:#f5f5f5;"
                        f"border-radius:6px;font-size:0.8rem'>"
                        f"<b>{step_short}</b><br>{r['label'][:20]}<br>"
                        f"<small>{r['latency_ms']:.0f}ms</small></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    bg = GRADE_COLOR.get(grade, "#555")
                    emoji = GRADE_EMOJI.get(grade, "")
                    st.markdown(
                        f"<div style='text-align:center;padding:8px;background:{bg}20;"
                        f"border:2px solid {bg};border-radius:6px;font-size:0.8rem'>"
                        f"<b>{step_short}</b><br>{emoji} <b>{grade}</b><br>"
                        f"<small>{r['latency_ms']:.0f}ms</small></div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

        # 2. Step expanders ────────────────────────────────────────────────────
        import pandas as pd

        for i, r in enumerate(lc_results):
            grade = r.get("grade")
            if r["_error"]:
                expander_title = f"⚠️ {r['step']} — {r['label']}  ·  ERROR"
            elif grade:
                grade_emoji = GRADE_EMOJI.get(grade, "")
                expander_title = (
                    f"{grade_emoji} {r['step']} — {r['label']}  ·  {grade}  ·  {r['latency_ms']:.0f}ms"
                )
            else:
                expander_title = f"🔵 {r['step']} — {r['label']}  ·  {r['latency_ms']:.0f}ms"

            with st.expander(expander_title, expanded=(i == 0)):
                if r["_error"]:
                    st.error(f"Step failed: {r['_error']}")
                    continue

                # Retrieved chunks
                if r["chunks"]:
                    st.markdown(f"**Retrieved Chunks** ({len(r['chunks'])})")
                    for rank, chunk in enumerate(r["chunks"], 1):
                        sim_val = getattr(chunk, "similarity", None)
                        sim_str = f"  ·  sim {sim_val:.3f}" if sim_val is not None else ""
                        with st.expander(f"#{rank}  `{chunk.source}`{sim_str}", expanded=False):
                            st.text(chunk.text)
                else:
                    st.caption("No retrieved chunks (agentic step uses tool calls internally).")

                # Context breakdown
                st.markdown("**Context Breakdown**")
                for ctx_key, ctx_val in r["context_parts"].items():
                    icon = "🟢" if "CSV" in ctx_key else "🔵" if "Graph" in ctx_key else "⚪"
                    with st.expander(f"{icon} {ctx_key}  ({len(ctx_val)} chars)"):
                        st.text(ctx_val)

                # Answer
                st.markdown("**Answer**")
                st.markdown(r["answer"])
                st.caption(f"Provider: `{r['provider']}`")

                # Grade (only if golden question)
                if lc_golden and grade:
                    st.markdown("**Grade**")
                    grade_bg  = GRADE_COLOR.get(grade, "#555")
                    grade_em  = GRADE_EMOJI.get(grade, "")
                    st.markdown(
                        f"<div style='display:inline-block;background:{grade_bg};"
                        f"color:white;padding:4px 12px;border-radius:4px;"
                        f"font-weight:bold'>{grade_em} {grade}</div>",
                        unsafe_allow_html=True,
                    )
                    if r.get("required_hits"):
                        st.success(f"Found: {r['required_hits']}")
                    if r.get("required_missing"):
                        st.error(f"Missing: {r['required_missing']}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    _sidebar()

    st.title("🔬 RAG Fusion Lab")
    st.caption(
        "Experiment with chunking, embedding, and retrieval — "
        "compare against fixed baselines in real time."
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Step Progress",
        "🔬 Live Compare",
        "🧪 Experiment Lab",
        "📊 Analysis",
        "🔍 Query Explorer",
        "🗂 Chunk Browser",
    ])

    with tab1:
        tab_step_progress()
    with tab2:
        tab_live_compare()
    with tab3:
        tab_experiment_lab()
    with tab4:
        tab_analysis()
    with tab5:
        tab_query_explorer()
    with tab6:
        tab_chunk_browser()


if __name__ == "__main__":
    main()
