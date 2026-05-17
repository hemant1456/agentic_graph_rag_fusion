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
import os
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
GRAPH_PATH  = ROOT / "step_05_knowledge_graph" / "results" / "graph.json"
STEP01_EVAL = ROOT / "step_01_baseline_rag" / "results" / "eval_results.json"
STEP04_EVAL = ROOT / "step_04_chunking" / "results" / "eval_results.json"
STEP05_EVAL = ROOT / "step_05_knowledge_graph" / "results" / "eval_results.json"
STEP06_EVAL = ROOT / "step_06_graph_rag" / "results" / "eval_results.json"
STEP07_EVAL = ROOT / "step_07_rag_fusion"   / "results" / "eval_results.json"
STEP08_EVAL = ROOT / "step_08_agentic_rag"  / "results" / "eval_results.json"
STEP09_EVAL = ROOT / "step_09_multi_agent"      / "results" / "eval_results.json"
STEP10_EVAL = ROOT / "step_10_context_engineering" / "results" / "eval_results.json"
STEP11_EVAL = ROOT / "step_11_vsa"                 / "results" / "eval_results.json"
STEP12_EVAL = ROOT / "step_12_production"          / "results" / "eval_results.json"
STEP01_DB   = ROOT / "step_01_baseline_rag" / "results" / "chroma_db"
STEP04_DB   = ROOT / "step_04_chunking" / "results" / "chroma_db"
CORPUS      = ROOT / "step_00_dataset" / "company_data"
EXP_DB_ROOT = ROOT / "step_04_chunking" / "results" / "chroma_experiments"

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


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai
    from step_01_baseline_rag.implementation.ingest import GEMINI_EMBED_MODEL
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=api_key)
    embeddings: list[list[float]] = []
    for i, txt in enumerate(texts):
        resp = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=txt)
        raw = (resp.embeddings or [None])[0]
        vec: list[float] = list(raw.values) if (raw is not None and raw.values is not None) else []
        embeddings.append(vec)
        if (i + 1) % 50 == 0:
            time.sleep(0.5)
    return embeddings


def _embed_query_gemini(query: str) -> list[float]:
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
    """Heatmap: rows = Q01..Q27, columns = run labels, colour = grade."""
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
    Full pipeline: rechunk corpus → embed → store in ChromaDB → run 27 golden questions.

    local_model: fastembed model name, or None to use Gemini API.
    strategy: "paragraph" | "fixed"
    include_csv_agg: whether to include aggregate chunks from CSV parser.
    """
    import chromadb
    from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
    from step_01_baseline_rag.evaluation.run_eval import score as grade_fn
    from step_01_baseline_rag.implementation.generate import generate_answer
    from step_01_baseline_rag.implementation.pipeline import RAGResult
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context

    embed_label = local_model or "gemini"
    model_slug  = (local_model or "gemini").replace("/", "_").replace("-", "_")

    # ── 1. Rechunk ────────────────────────────────────────────────────────────
    st.write("**1 / 4** — Chunking corpus…")
    text_chunks = _rechunk_corpus(chunk_size, overlap, strategy)

    # CSV aggregate chunks (optional)
    if include_csv_agg:
        from step_04_chunking.implementation.chunker import load_and_chunk as _smart_chunk
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
        from google import genai
        from step_01_baseline_rag.implementation.ingest import GEMINI_EMBED_MODEL
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key)
        embeddings: list[list[float]] = []
        for i, txt in enumerate(all_texts):
            resp = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=txt)
            raw = (resp.embeddings or [None])[0]
            vec: list[float] = list(raw.values) if (raw is not None and raw.values is not None) else []
            embeddings.append(vec)
            if i % 20 == 0:
                prog.progress((i + 1) / len(all_texts), text=f"Embedding {i+1}/{len(all_texts)}…")
            if (i + 1) % 50 == 0:
                time.sleep(0.5)
        prog.progress(1.0, text="Embedding complete.")

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
            qvec = _embed_query_gemini(gq.question)
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
    model_short = (local_model or "gemini").split("/")[-1] if local_model else "gemini"
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

        st.markdown("**Steps built**")
        steps = [
            ("✅", "Step 00", "Synthetic company dataset"),
            ("✅", "Step 01", "Baseline vector RAG (26%)"),
            ("✅", "Step 02", "Observability (Arize Phoenix)"),
            ("✅", "Step 03", "Evaluation framework"),
            ("✅", "Step 04", "Format-aware chunking (52%)"),
            ("✅", "Step 05", "Knowledge graph (78%)"),
            ("✅", "Step 06", "Graph RAG (85%)"),
            ("✅", "Step 07", "RAG Fusion + BM25 (89%)"),
            ("✅", "Step 08", "Agentic RAG / Gateway (85%)"),
            ("✅", "Step 09", "Multi-Agent System (93%)"),
            ("✅", "Step 10", "Context Engineering (85%)"),
            ("✅", "Step 11", "Vertical Slice Architecture (89%)"),
            ("✅", "Step 12", "Production Hardening (cache + retry + confidence)"),
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
            "Step 04's **aggregate chunks** pre-compute sums, breakdowns, and "
            "date-period totals at index time. This is a deliberate hack that "
            "proves why pure vector RAG fails on tabular data. "
            "The real fix (Step 07) runs a **structured query tool** (Pandas/SQL) "
            "at query time — not a pre-baked summary. Aggregate chunks are "
            "stale, inflexible, and don't generalise to filters you didn't predict."
        )
        st.divider()
        st.markdown("**Remaining failure modes**")
        st.markdown(
            "- Step 11 VSA routes each query to one of 4 domain slices (Finance / HR / Engineering / General)\n"
            "- Each slice has its own system prompt, retrieval overrides, and evaluation suite\n"
            "- Step 12: production hardening — reliability, cost control, SLOs"
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
        "**Step 01 (26%) and Step 04 (52%) are fixed baselines** — their numbers never change. "
        "Each time you click **Run**, you create a **new experiment** added to the history below. "
        "The comparison shows how your experiments stack up against the baselines.",
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
            ["🤗 Local (fastembed)", "☁️ Gemini API"],
        )
        use_local = embed_backend.startswith("🤗")

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
            if not os.environ.get("GOOGLE_API_KEY"):
                st.warning("Set `GOOGLE_API_KEY` to use Gemini embedding.")

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

    est_time = "~30 s (local CPU)" if use_local else "~3–8 min (API)"
    cost_note = "free" if use_local else "~$0.05"

    col_run, col_info = st.columns([1, 4])
    run_btn = col_run.button("▶ Run Experiment", type="primary", use_container_width=True)
    col_info.info(
        f"chunk_size={chunk_size}  overlap={overlap}  k={k_val}  "
        f"strategy={strategy_key}  agg={'yes' if include_csv_agg else 'no'}  "
        f"embed={'local' if use_local else 'gemini'}  |  "
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
        s4 = load_eval(STEP04_EVAL)
        s5 = load_eval(STEP05_EVAL)
        s6 = load_eval(STEP06_EVAL)
        s7 = load_eval(STEP07_EVAL)
        s8 = load_eval(STEP08_EVAL)
        s9  = load_eval(STEP09_EVAL)
        s10 = load_eval(STEP10_EVAL)
        s11 = load_eval(STEP11_EVAL)
        s12 = load_eval(STEP12_EVAL)
        all_runs: list[tuple[str, dict]] = []
        if s1:
            all_runs.append(("Step01 (baseline)", s1))
        if s4:
            all_runs.append(("Step04 (format-aware)", s4))
        if s5:
            all_runs.append(("Step05 (knowledge graph)", s5))
        if s6:
            all_runs.append(("Step06 (graph RAG)", s6))
        if s7:
            all_runs.append(("Step07 (RAG Fusion)", s7))
        if s8:
            all_runs.append(("Step08 (Agentic)", s8))
        if s9 and s9.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step09 (Multi-Agent)", s9))
        if s10 and s10.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step10 (Context Eng.)", s10))
        if s11 and s11.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step11 (VSA)", s11))
        if s12 and s12.get("grade_counts", {}).get("PASS", 0) > 0:
            all_runs.append(("Step12 (Production)", s12))
        for exp in history:
            all_runs.append((exp["label"], exp))

        st.markdown("#### Pass Rate Comparison")
        st.plotly_chart(_comparison_bar(all_runs), use_container_width=True)

        # Grade heatmap
        st.markdown("#### Grade Heatmap (Q01–Q27)")
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
    s4 = load_eval(STEP04_EVAL)
    s5 = load_eval(STEP05_EVAL)
    s6 = load_eval(STEP06_EVAL)
    s7 = load_eval(STEP07_EVAL)
    s8 = load_eval(STEP08_EVAL)
    s9  = load_eval(STEP09_EVAL)
    s10 = load_eval(STEP10_EVAL)
    s11 = load_eval(STEP11_EVAL)
    s12 = load_eval(STEP12_EVAL)

    if not any([s1, s4, s5, s6, s7, s8]):
        st.warning("No baseline eval results found. Run the evaluation scripts first.")
        st.code(
            "uv run python step_01_baseline_rag/evaluation/run_eval.py\n"
            "uv run python step_04_chunking/evaluation/run_eval.py\n"
            "uv run python step_05_knowledge_graph/evaluation/run_eval.py\n"
            "uv run python step_06_graph_rag/evaluation/run_eval.py\n"
            "uv run python step_07_rag_fusion/evaluation/run_eval.py\n"
            "uv run python -m step_08_agentic_rag.evaluation.run_eval\n"
            "uv run python step_09_multi_agent/evaluation/run_eval.py\n"
            "uv run python step_10_context_engineering/evaluation/run_eval.py\n"
            "uv run python step_11_vsa/evaluation/run_eval.py\n"
            "uv run python step_12_production/evaluation/run_eval.py"
        )
        return

    # Build run registry (baselines always present)
    base_runs: list[tuple[str, dict]] = []
    if s1:
        base_runs.append(("Step01 (26%)", s1))
    if s4:
        base_runs.append(("Step04 (52%)", s4))
    if s5:
        base_runs.append(("Step05 (78%)", s5))
    if s6:
        base_runs.append(("Step06 (85%)", s6))
    if s7:
        base_runs.append(("Step07 (89%)", s7))
    if s8:
        base_runs.append(("Step08 (85%)", s8))
    if s9 and s9.get("grade_counts", {}).get("PASS", 0) > 0:
        pct9 = round(s9["pass_rate"] * 100)
        base_runs.append((f"Step09 ({pct9}%)", s9))
    if s10 and s10.get("grade_counts", {}).get("PASS", 0) > 0:
        pct10 = round(s10["pass_rate"] * 100)
        base_runs.append((f"Step10 ({pct10}%)", s10))
    if s11 and s11.get("grade_counts", {}).get("PASS", 0) > 0:
        pct11 = round(s11["pass_rate"] * 100)
        base_runs.append((f"Step11 ({pct11}%)", s11))
    if s12 and s12.get("grade_counts", {}).get("PASS", 0) > 0:
        pct12 = round(s12["pass_rate"] * 100)
        base_runs.append((f"Step12 ({pct12}%)", s12))

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
        "For Step 01 and Step 04 only Gemini embedding will work "
        "(they were built with Gemini). Local models are only compatible with experiment indexes."
    )

    # ── Index selector ─────────────────────────────────────────────────────────
    history: list[dict] = st.session_state["exp_history"]
    index_options: dict[str, tuple[Path, str, str]] = {}  # label → (db_path, coll_name, built_with)

    if STEP01_DB.exists():
        index_options["Step 01 Baseline"] = (STEP01_DB, "vertexia_baseline", "gemini")
    if STEP04_DB.exists():
        index_options["Step 04 Format-aware"] = (STEP04_DB, "vertexia_step04", "gemini")
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
        st.warning(
            "This index was built with **Gemini embedding**. "
            "Querying with a local model will return garbage — use Gemini for correct results."
        )

    # ── Embedding mode ─────────────────────────────────────────────────────────
    embed_options = ["Same as index was built with (recommended)", "Gemini API"] + list(LOCAL_EMBED_MODELS.keys())
    embed_sel = st.selectbox("Query embedding mode", embed_options, key="qe_embed")

    if embed_sel == "Same as index was built with (recommended)":
        query_embed_mode = built_with
        query_local_model: str | None = None
    elif embed_sel == "Gemini API":
        query_embed_mode = "gemini"
        query_local_model = None
    else:
        query_embed_mode = "local"
        query_local_model = LOCAL_EMBED_MODELS[embed_sel]

    if query_embed_mode == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        st.error("Set `GOOGLE_API_KEY` to use Gemini embedding.")

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
                if query_embed_mode == "gemini" or (
                    query_embed_mode not in ("local",) and "gemini" in query_embed_mode.lower()
                ):
                    qvec = _embed_query_gemini(question)
                elif query_embed_mode == "local" and query_local_model:
                    qvec = _embed_local([question], query_local_model)[0]
                else:
                    qvec = _embed_query_gemini(question)
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
    source_options: dict[str, str] = {"Step 01 (baseline)": "step01", "Step 04 (format-aware)": "step04"}
    history: list[dict] = st.session_state["exp_history"]
    for exp in history:
        source_options[exp["label"]] = f"exp:{exp['label']}"

    sel_source_label = st.selectbox("Source index", list(source_options.keys()), key="cb_source")
    source_key = source_options[sel_source_label]

    # ── Load chunks ────────────────────────────────────────────────────────────
    @st.cache_resource
    def _get_step01_chunks():
        from step_01_baseline_rag.implementation.ingest import load_and_chunk as _l01
        return _l01(CORPUS)

    @st.cache_resource
    def _get_step04_chunks():
        from step_04_chunking.implementation.chunker import load_and_chunk as _l04
        return _l04(CORPUS)

    with st.spinner("Loading chunks…"):
        if source_key == "step01":
            raw_chunks = _get_step01_chunks()
            def _ctype(c: Any) -> str:
                return str(getattr(c, "format", "prose"))
        elif source_key == "step04":
            raw_chunks = _get_step04_chunks()
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
        ("Step 01", "Baseline\nvector RAG", STEP01_EVAL, "Naive 2k-char chunks + dense retrieval"),
        ("Step 04", "Format-aware\nchunking",   STEP04_EVAL, "CSV row chunks + aggregate pre-compute + section splitting"),
        ("Step 05", "Knowledge\ngraph",          STEP05_EVAL, "Entity extraction + graph traversal for org/product context"),
        ("Step 06", "Graph RAG\n+ aliases",      STEP06_EVAL, "Alias resolution (analytics dashboard → InsightLens) + dependency chains"),
        ("Step 07", "RAG Fusion\n+ BM25",        STEP07_EVAL, "BM25 + dense RRF merge + structured CSV query tool at query time"),
        ("Step 08", "Agentic RAG\n(Gateway)",    STEP08_EVAL, "Tool-calling agent via LLM Gateway V2 (Groq / Gemini / NVIDIA / Cerebras)"),
        ("Step 09", "Multi-Agent\nSystem",        STEP09_EVAL, "Orchestrator + 6 specialized subagents with typed contracts + Critic verification"),
        ("Step 10", "Context\nEngineering",       STEP10_EVAL, "CrossEncoder reranking → dedup → extractive compression → XML format → token budget"),
        ("Step 11", "Vertical Slice\nArchitecture", STEP11_EVAL, "4 domain slices (Finance / HR / Engineering / General) — each owns prompt, retrieval config, eval"),
        ("Step 12", "Production\nHardening",        STEP12_EVAL, "Semantic cache + retry/backoff + confidence scoring + health monitor + graceful degradation"),
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

    # ── Provider breakdown (Step 08 / 09) ─────────────────────────────────────
    s8 = load_eval(STEP08_EVAL)
    s9 = load_eval(STEP09_EVAL)
    s10 = load_eval(STEP10_EVAL)
    s11_pb = load_eval(STEP11_EVAL)
    s12_pb = load_eval(STEP12_EVAL)
    prov_step = None
    prov_label = ""
    if s12_pb and s12_pb.get("grade_counts", {}).get("PASS", 0) > 0:
        prov_step, prov_label = s12_pb, "Step 12 — Provider Breakdown (Gateway V2)"
    elif s11_pb and s11_pb.get("grade_counts", {}).get("PASS", 0) > 0:
        prov_step, prov_label = s11_pb, "Step 11 — Provider Breakdown (Gateway V2)"
    elif s10 and s10.get("grade_counts", {}).get("PASS", 0) > 0:
        prov_step, prov_label = s10, "Step 10 — Provider Breakdown (Gateway V2)"
    elif s9 and s9.get("grade_counts", {}).get("PASS", 0) > 0:
        prov_step, prov_label = s9, "Step 09 — Provider Breakdown (Gateway V2)"
    elif s8:
        prov_step, prov_label = s8, "Step 08 — Provider Breakdown (Gateway V2)"
    if prov_step:
        st.subheader(prov_label)
        s8 = prov_step  # reuse variable for the block below
    if s8:
        providers: dict[str, int] = {}
        for r in s8.get("results", []):
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

@st.cache_resource
def _load_step07_retriever():
    from step_07_rag_fusion.implementation.pipeline import Step07RAG
    return Step07RAG(k=10).build()


@st.cache_resource
def _load_step08_rag():
    from step_08_agentic_rag.implementation.pipeline import Step08RAG
    return Step08RAG(k=10).build()


@st.cache_resource
def _load_step09_rag():
    from step_09_multi_agent.implementation.pipeline import Step09RAG
    return Step09RAG(k=10).build()


@st.cache_resource
def _load_step10_rag():
    from step_10_context_engineering.implementation.pipeline import Step10RAG
    return Step10RAG(k=5, rerank_k=8, compress_ratio=0.60).build()


@st.cache_resource
def _load_step11_rag():
    from step_11_vsa.implementation.pipeline import Step11RAG
    return Step11RAG(k=5).build()


@st.cache_resource
def _load_step12_rag():
    from step_12_production.implementation.pipeline import Step12RAG
    return Step12RAG(k=5).build()


@st.cache_resource
def _load_graph():
    from step_05_knowledge_graph.implementation.graph_store import load_or_build
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
    """Run all 8 steps sequentially; return list of result dicts."""
    from step_01_baseline_rag.implementation.retrieve import RetrievedChunk, format_context

    SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions about Vertexia Inc. "
        "Answer based only on the provided context. Be concise and precise."
    )

    results: list[dict] = []

    progress_bar = st.progress(0.0, text="Running Step 01…")

    # ── Step 01 — Baseline Dense Vector ────────────────────────────────────────
    step_label = "Step 01"
    step_name  = "Baseline Dense Vector"
    try:
        t0 = time.perf_counter()
        coll01 = _get_collection(str(STEP01_DB), "vertexia_baseline")
        qvec = _embed_query_gemini(question)
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
            "context_parts": {"Vector": context01},
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

    progress_bar.progress(1 / 10, text="Running Step 04…")

    # ── Step 04 — Format-aware chunks ──────────────────────────────────────────
    step_label = "Step 04"
    step_name  = "Format-aware Chunks"
    try:
        t0 = time.perf_counter()
        coll04 = _get_collection(str(STEP04_DB), "vertexia_step04")
        qvec04 = _embed_query_gemini(question)
        res04 = coll04.query(
            query_embeddings=[qvec04],
            n_results=min(k, coll04.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs04  = (res04["documents"] or [[]])[0]
        metas04 = (res04["metadatas"] or [[]])[0]
        dists04 = (res04["distances"] or [[]])[0]
        chunks04 = [
            RetrievedChunk(
                text=d,
                source=str(m.get("source", "")),
                department=str(m.get("department", "")),
                format=str(m.get("format", "")),
                chunk_index=int(str(m.get("chunk_index") or 0)),
                distance=dist,
            )
            for d, m, dist in zip(docs04, metas04, dists04)
        ]
        context04 = format_context(chunks04)
        answer04, prov04 = _gateway_generate(context04, question, provider, SYSTEM_PROMPT)
        latency04 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": chunks04,
            "context_parts": {"Vector (format-aware)": context04},
            "full_context": context04,
            "answer": answer04,
            "provider": prov04,
            "latency_ms": latency04,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(2 / 10, text="Running Step 05…")

    # ── Step 05 — Knowledge Graph ───────────────────────────────────────────────
    step_label = "Step 05"
    step_name  = "Knowledge Graph"
    try:
        t0 = time.perf_counter()
        from step_05_knowledge_graph.implementation.query import expand_context, extract_entity_ids
        graph = _load_graph()
        coll05 = _get_collection(str(STEP04_DB), "vertexia_step04")
        qvec05 = _embed_query_gemini(question)
        res05 = coll05.query(
            query_embeddings=[qvec05],
            n_results=min(k, coll05.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs05  = (res05["documents"] or [[]])[0]
        metas05 = (res05["metadatas"] or [[]])[0]
        dists05 = (res05["distances"] or [[]])[0]
        chunks05 = [
            RetrievedChunk(
                text=d,
                source=str(m.get("source", "")),
                department=str(m.get("department", "")),
                format=str(m.get("format", "")),
                chunk_index=int(str(m.get("chunk_index") or 0)),
                distance=dist,
            )
            for d, m, dist in zip(docs05, metas05, dists05)
        ]
        vector_ctx05 = format_context(chunks05)
        entity_ids05 = extract_entity_ids([question] + [c.text for c in chunks05], graph)
        graph_ctx05  = expand_context(entity_ids05, graph)
        context05 = "\n\n".join(filter(None, [graph_ctx05, vector_ctx05]))
        answer05, prov05 = _gateway_generate(context05, question, provider, SYSTEM_PROMPT)
        latency05 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": chunks05,
            "context_parts": {"Graph": graph_ctx05, "Vector": vector_ctx05},
            "full_context": context05,
            "answer": answer05,
            "provider": prov05,
            "latency_ms": latency05,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(3 / 10, text="Running Step 06…")

    # ── Step 06 — Enhanced Graph RAG ────────────────────────────────────────────
    step_label = "Step 06"
    step_name  = "Enhanced Graph RAG"
    try:
        t0 = time.perf_counter()
        from step_06_graph_rag.implementation.graph_query import build_graph_context
        graph = _load_graph()
        coll06 = _get_collection(str(STEP04_DB), "vertexia_step04")
        qvec06 = _embed_query_gemini(question)
        res06 = coll06.query(
            query_embeddings=[qvec06],
            n_results=min(k, coll06.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs06  = (res06["documents"] or [[]])[0]
        metas06 = (res06["metadatas"] or [[]])[0]
        dists06 = (res06["distances"] or [[]])[0]
        chunks06 = [
            RetrievedChunk(
                text=d,
                source=str(m.get("source", "")),
                department=str(m.get("department", "")),
                format=str(m.get("format", "")),
                chunk_index=int(str(m.get("chunk_index") or 0)),
                distance=dist,
            )
            for d, m, dist in zip(docs06, metas06, dists06)
        ]
        vector_ctx06 = format_context(chunks06)
        graph_ctx06  = build_graph_context(question, [c.text for c in chunks06], graph)
        context06 = "\n\n".join(filter(None, [graph_ctx06, vector_ctx06]))
        answer06, prov06 = _gateway_generate(context06, question, provider, SYSTEM_PROMPT)
        latency06 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": chunks06,
            "context_parts": {"Graph (enhanced)": graph_ctx06, "Vector": vector_ctx06},
            "full_context": context06,
            "answer": answer06,
            "provider": prov06,
            "latency_ms": latency06,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(4 / 10, text="Running Step 07…")

    # ── Step 07 — RAG Fusion + CSV tool ─────────────────────────────────────────
    step_label = "Step 07"
    step_name  = "RAG Fusion + CSV Tool"
    try:
        t0 = time.perf_counter()
        from step_06_graph_rag.implementation.graph_query import build_graph_context as _bgc07
        from step_07_rag_fusion.implementation.csv_tool import detect_intent, run_query
        rag07 = _load_step07_retriever()
        chunks07 = rag07.retrieve(question, k=k)
        vector_ctx07 = format_context(chunks07)
        _graph07 = rag07.graph if rag07.graph is not None else _load_graph()
        graph_ctx07  = _bgc07(question, [c.text for c in chunks07], _graph07)
        csv_intent07 = detect_intent(question)
        csv_ctx07    = run_query(csv_intent07) if csv_intent07 else ""
        context07 = "\n\n".join(filter(None, [csv_ctx07, graph_ctx07, vector_ctx07]))
        answer07, prov07 = _gateway_generate(context07, question, provider, SYSTEM_PROMPT)
        latency07 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": chunks07,
            "context_parts": {
                "CSV Query": csv_ctx07,
                "Graph": graph_ctx07,
                "Vector (BM25+Dense)": vector_ctx07,
            },
            "full_context": context07,
            "answer": answer07,
            "provider": prov07,
            "latency_ms": latency07,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(5 / 10, text="Running Step 08…")

    # ── Step 08 — Agentic RAG ───────────────────────────────────────────────────
    step_label = "Step 08"
    step_name  = "Agentic RAG"
    try:
        t0 = time.perf_counter()
        rag08 = _load_step08_rag()
        result08 = rag08.query(question)
        answer08 = result08.answer
        prov08   = f"gateway:{getattr(result08, 'provider', 'unknown')}"
        latency08 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": [],
            "context_parts": {"Agentic (tool loop)": "Context-first agent, up to 3 tool rounds."},
            "full_context": "",
            "answer": answer08,
            "provider": prov08,
            "latency_ms": latency08,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(6 / 10, text="Running Step 09…")

    # ── Step 09 — Multi-Agent RAG ───────────────────────────────────────────────
    step_label = "Step 09"
    step_name  = "Multi-Agent"
    try:
        t0 = time.perf_counter()
        rag09 = _load_step09_rag()
        result09 = rag09.query(question)
        answer09 = result09.answer
        prov09   = getattr(result09, "provider", "multi-agent")
        latency09 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": [],
            "context_parts": {
                "Multi-Agent Pipeline": (
                    "QueryAnalyst → RetrievalSpecialist (+ sub-question retrieval) → "
                    "GraphNavigator → StructuredData → Synthesis → Critic"
                )
            },
            "full_context": "",
            "answer": answer09,
            "provider": prov09,
            "latency_ms": latency09,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(7 / 10, text="Running Step 10…")

    # ── Step 10 — Context Engineering ──────────────────────────────────────────
    step_label = "Step 10"
    step_name  = "Context Eng."
    try:
        t0 = time.perf_counter()
        rag10 = _load_step10_rag()
        ext10 = rag10.query_extended(question)
        result10 = ext10.rag_result
        ce10     = ext10.ce_metrics
        prov10   = getattr(result10, "provider", "ce")
        latency10 = (time.perf_counter() - t0) * 1000
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": [],
            "context_parts": {
                "CE Pipeline": (
                    f"Retrieve k=20 → CrossEncoder rerank → dedup → compress (60%) → XML format\n"
                    f"raw={ce10['raw_chars']:,} chars → engineered={ce10['engineered_chars']:,} chars "
                    f"({ce10['compression_ratio']:.0%}) | "
                    f"chunks {ce10['chunks_before']}→{ce10['chunks_after_dedup']}→{ce10['chunks_final']}"
                ),
                "Engineered Context (XML)": result10.context_sent,
            },
            "full_context": result10.context_sent,
            "answer": result10.answer,
            "provider": prov10,
            "latency_ms": latency10,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(8 / 10, text="Running Step 11…")

    # ── Step 11 — VSA ──────────────────────────────────────────────────────────
    step_label = "Step 11"
    step_name  = "VSA"
    try:
        t0 = time.perf_counter()
        rag11 = _load_step11_rag()
        ext11 = rag11.query_extended(question)
        result11 = ext11.rag_result
        prov11   = getattr(result11, "provider", "vsa")
        latency11 = (time.perf_counter() - t0) * 1000
        ce11     = ext11.ce_metrics
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": [],
            "context_parts": {
                "Routing": (
                    f"Slice: {ext11.slice_name}  (confidence={ext11.router_confidence:.2f})\n"
                    f"Finance · HR · Engineering · General — domain-specific prompt + overrides"
                ),
                "CE metrics": (
                    f"raw={ce11.get('raw_chars', 0):,} chars → "
                    f"engineered={ce11.get('engineered_chars', 0):,} chars "
                    f"({ce11.get('compression_ratio', 1.0):.0%}) | "
                    f"chunks {ce11.get('chunks_before', 0)}→{ce11.get('chunks_final', 0)}"
                ),
            },
            "full_context": "",
            "answer": result11.answer,
            "provider": prov11,
            "latency_ms": latency11,
            "_error": None,
        })
    except Exception as exc:
        results.append({
            "step": step_label, "label": step_name, "chunks": [], "context_parts": {},
            "full_context": "", "answer": "", "provider": "", "latency_ms": 0.0, "_error": str(exc),
        })

    progress_bar.progress(9 / 10, text="Running Step 12…")

    # ── Step 12 — Production Hardening ─────────────────────────────────────────
    step_label = "Step 12"
    step_name  = "Production"
    try:
        t0 = time.perf_counter()
        rag12 = _load_step12_rag()
        ext12 = rag12.query_extended(question)
        result12 = ext12.rag_result
        prov12   = getattr(result12, "provider", "prod")
        latency12 = (time.perf_counter() - t0) * 1000
        ce12     = ext12.ce_metrics
        health12 = ext12.health_snapshot
        results.append({
            "step": step_label,
            "label": step_name,
            "chunks": [],
            "context_parts": {
                "Routing": (
                    f"Slice: {ext12.slice_name}  (confidence={ext12.router_confidence:.2f})\n"
                    f"Cache: {'HIT' if ext12.from_cache else 'miss'} | "
                    f"Answer confidence: {ext12.confidence_label} ({ext12.confidence_score:.2f})"
                ),
                "CE metrics": (
                    f"raw={ce12.get('raw_chars', 0):,} chars → "
                    f"engineered={ce12.get('engineered_chars', 0):,} chars "
                    f"({ce12.get('compression_ratio', 1.0):.0%})"
                ),
                "Health": (
                    f"p50={health12.get('p50_latency_ms','?')}ms  "
                    f"p95={health12.get('p95_latency_ms','?')}ms  "
                    f"SLO={health12.get('slo_compliance',0):.0%}  "
                    f"status={health12.get('status','?')}"
                ),
            },
            "full_context": "",
            "answer": result12.answer,
            "provider": prov12,
            "latency_ms": latency12,
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
| Step 01 | Dense vector only, naive 2k-char chunks |
| Step 04 | Format-aware chunks: CSV rows + aggregate + section split |
| Step 05 | Knowledge graph: entity context + org/product relationships |
| Step 06 | Graph RAG: alias resolution + dependency chain traversal |
| Step 07 | BM25 + Dense RRF fusion + structured CSV query tool |
| Step 08 | Agentic loop: tool-calling LLM via Gateway V2 |
| Step 09 | Multi-agent: orchestrator + 6 specialized subagents + Critic |
| Step 10 | Context engineering: CrossEncoder rerank → dedup → compress → XML |
| Step 11 | VSA: route to Finance / HR / Engineering / General slice by domain |
| Step 12 | Production: semantic cache + retry/backoff + confidence scoring + health monitor |
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
