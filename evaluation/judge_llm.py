"""LLM judge for RAGAS — direct OpenAI caller using OPENAI_API_KEY from .env.

Using gpt-5.4-mini (fast OpenAI model) so RAGAS evaluation completes quickly.
Swap to NVIDIA NIM (free, slower) by changing JUDGE_MODEL / JUDGE_BASE_URL below.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai")  # "openai" | "nvidia"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.4-mini")


def build_judge_llm(temperature: float = 0.0, max_tokens: int = 1024) -> ChatOpenAI:
    """Return a LangChain ChatOpenAI configured for the chosen judge backend.

    RAGAS wraps this with LangchainLLMWrapper internally.
    """
    if JUDGE_PROVIDER == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set in .env")
        return ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            model=os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
            max_retries=2,
        )

    # default: OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return ChatOpenAI(
        api_key=api_key,
        model=JUDGE_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
        max_retries=2,
    )
