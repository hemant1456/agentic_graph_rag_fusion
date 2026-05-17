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

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gateway")  # "gateway" | "openai"
GATEWAY_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")


def _msg_role(m: BaseMessage) -> str:
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, AIMessage):
        return "assistant"
    return "user"


class GatewayChat(BaseChatModel):
    """LangChain ChatModel backed by llm_gatewayV2.

    Provider selection — empirical findings on this gateway:
      - groq (llama-3.3-70b):  ~200ms/call, 30 RPM — fastest, reliable
      - gemini (3.1-flash):    ~900ms/call, 15 RPM — reliable fallback
      - nvidia (deepseek-v4):  ~30s/call,   40 RPM — too slow for batch eval
      - cerebras (qwen3-235b): empty responses     — broken, skip

    Prefer groq, fall back to gemini on RPM exhaustion or transient failure.
    Override via JUDGE_PROVIDERS env var (comma-separated, in priority order).
    """

    base_url: str = GATEWAY_URL
    temperature: float = 0.0
    max_tokens: int = 1024
    providers: list[str] = []  # set in model_post_init

    def model_post_init(self, __context: Any) -> None:
        env_order = os.getenv("JUDGE_PROVIDERS", "groq,gemini")
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
    """Return a LangChain chat model for RAGAS. Defaults to the free-tier gateway."""
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
