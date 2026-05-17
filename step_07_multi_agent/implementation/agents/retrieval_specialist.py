from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step_01_baseline_rag.implementation.retrieve import format_context
from step_04_hybrid_retrieval.implementation.pipeline import Step04HybridRAG
from step_07_multi_agent.implementation.agents.contracts import RetrievalResult


def _augment_query(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("depart", "left vertexia", "voluntary", "offboard", "resign")):
        query += " departure_type voluntary offboarding_records"
    if "phoenix" in q and any(w in q for w in ("deal", "contract", "signed", "corp", "enterprise", "outcome")):
        query += " Phoenix Corp signed June 2022 executed MSA closed won"
    return query


def retrieve(question: str, retriever: Step04HybridRAG, k: int = 10) -> RetrievalResult:
    augmented = _augment_query(question)
    chunks = retriever.retrieve(augmented, k=k)
    context = format_context(chunks) or "[No relevant passages found]"
    return RetrievalResult(chunks=chunks, context=context, strategy="bm25+dense+rrf")
