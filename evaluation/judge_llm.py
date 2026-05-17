"""LLM judge for RAGAS via llm_gatewayV2 (free-tier provider routing).

The gateway (port 8100) routes each call among gemini / nvidia / groq / cerebras
based on RPM/RPD/cooldown. With max_workers=1 in the RAGAS RunConfig, calls are
sequential so the router cleanly fails over when one provider hits its cap.

A direct OpenAI fallback is available via JUDGE_PROVIDER=openai for cases when
the gateway is down or you want a fast paid run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from llm_gatewayV2.client import LLM as GatewayClient

load_dotenv(ROOT / ".env")

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gateway")  # "gateway" | "openai" | "ollama"
GATEWAY_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.2:latest")


def _msg_role(m: BaseMessage) -> str:
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, AIMessage):
        return "assistant"
    return "user"


class GatewayChat(BaseChatModel):
    """LangChain ChatModel backed by llm_gatewayV2.

    Provider selection — empirical findings on this gateway (2026-05-17):
      - groq (llama-3.3-70b):    ~200 ms/call, 30 RPM, 1k RPD — fast, but RPD
                                  exhausts after ~70 min of full eval runs
      - cerebras (qwen3-235b):   ~500 ms-2 s/call, 30 RPM, 9999 RPD — fast,
                                  high daily cap, occasional empty responses
      - gemini (3.1-flash-lite): ~900 ms/call, 15 RPM, 1k RPD — reliable
      - nvidia (deepseek-v4):    ~30 s/call — too slow for batch eval

    Default order: cerebras → gemini → groq.  Each provider is tried in
    sequence; empty text or RPM/RPD errors fall through to the next.
    Override via JUDGE_PROVIDERS env var (comma-separated).
    """

    base_url: str = GATEWAY_URL
    temperature: float = 0.0
    max_tokens: int = 1024
    providers: list[str] = []  # set in model_post_init

    def model_post_init(self, __context: Any) -> None:
        # gemini RPD typically exhausts mid-way through a full --all run, so
        # keep it but list cerebras + groq first; nvidia is slow but reliable
        # as a last resort.
        env_order = os.getenv("JUDGE_PROVIDERS", "cerebras,groq,gemini,nvidia")
        self.providers = [p.strip() for p in env_order.split(",") if p.strip()]

    @property
    def _llm_type(self) -> str:
        return "llm_gateway_v2"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = GatewayClient(base_url=self.base_url)
        system_msgs = [m.content for m in messages if isinstance(m, SystemMessage)]
        chat_msgs = [
            {"role": _msg_role(m), "content": m.content}
            for m in messages if not isinstance(m, SystemMessage)
        ]
        last_err: Exception | None = None
        for provider_name in self.providers:
            try:
                result = client.chat(
                    messages=chat_msgs,
                    system=system_msgs if system_msgs else None,
                    provider=provider_name,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                text = result.get("text", "") or ""
                if not text:
                    last_err = RuntimeError(f"{provider_name} returned empty text")
                    continue
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"All judge providers failed: {last_err}")


def build_judge_llm(temperature: float = 0.0, max_tokens: int = 1024):
    """Return a LangChain chat model for RAGAS.

    JUDGE_PROVIDER env var picks the backend:
      - "ollama"  → local Ollama via its OpenAI-compatible endpoint (no rate limits).
                    Model defaults to deepseek-r1:1.5b; override via OLLAMA_JUDGE_MODEL.
      - "openai"  → real OpenAI API (paid, very fast).
      - "gateway" → llm_gatewayV2 free-tier rotation (default, rate-limit prone).
    """
    if JUDGE_PROVIDER == "ollama":
        return ChatOpenAI(
            base_url=f"{OLLAMA_URL}/v1",
            api_key="ollama",  # any non-empty string works
            model=OLLAMA_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
            max_retries=2,
        )

    if JUDGE_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (JUDGE_PROVIDER=openai)")
        return ChatOpenAI(
            api_key=api_key,
            model=os.getenv("JUDGE_MODEL", "gpt-5.4-mini"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=2,
        )

    # default: route via llm_gatewayV2 (free-tier providers)
    return GatewayChat(temperature=temperature, max_tokens=max_tokens)
