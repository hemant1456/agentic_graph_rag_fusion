"""Python client for LLM Gateway V2. Backward-compatible kwargs from V1; new
kwargs (tools=, cache_system=, reasoning=, response_format=) are opt-in.

HTTP-level retry lives here so callers don't need a separate retry wrapper.
The gateway server already does multi-provider fallback for upstream provider
errors; client-side retry handles transient *transport* problems (connection
refused, read timeout) and gateway 5xx that propagated through. We do not
retry 4xx — those are caller errors.
"""
import os, json, time
import httpx
from typing import Any, Optional

DEFAULT_URL = os.getenv("LLM_GATEWAY_V2_URL", "http://localhost:8100")

# Transport-level retry. The gateway handles provider fallback internally.
_MAX_ATTEMPTS = int(os.getenv("LLM_CLIENT_RETRIES", "3"))
_BASE_DELAY = 0.5  # seconds; doubles each attempt, capped at 8s
_RETRY_TRANSPORT = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


def _should_retry_status(status_code: int) -> bool:
    """Retry on gateway-side server errors only. 4xx is the caller's fault."""
    return 500 <= status_code < 600


class LLM:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_with_retry(self, path: str, body: dict) -> dict:
        delay = _BASE_DELAY
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                r = httpx.post(f"{self.base_url}{path}", json=body, timeout=self.timeout)
                if _should_retry_status(r.status_code) and attempt < _MAX_ATTEMPTS:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                r.raise_for_status()
                return r.json()
            except _RETRY_TRANSPORT as e:
                last_exc = e
                if attempt == _MAX_ATTEMPTS:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
        # Should be unreachable, but satisfies type checker
        raise last_exc or RuntimeError("gateway retry loop exited without result")

    def chat(self, prompt: str = None, *,
             messages: Optional[list] = None,
             system: Any = None,
             provider: str = None, model: str = None,
             max_tokens: int = 2048, temperature: float = 0.7,
             tools: Optional[list] = None,
             tool_choice: Any = None,
             cache_system: Optional[bool] = None,
             reasoning: Optional[str] = None,
             response_format: Any = None) -> dict:
        body = {
            "prompt": prompt, "messages": messages, "system": system,
            "provider": provider, "model": model,
            "max_tokens": max_tokens, "temperature": temperature, "stream": False,
            "tools": tools, "tool_choice": tool_choice,
            "cache_system": cache_system, "reasoning": reasoning,
            "response_format": response_format,
        }
        body = {k: v for k, v in body.items() if v is not None}
        return self._post_with_retry("/v1/chat", body)

    def stream(self, prompt: str = None, *, messages=None, system=None,
               provider: str = None, model: str = None,
               max_tokens: int = 2048, temperature: float = 0.7,
               tools=None, tool_choice=None,
               cache_system=None, reasoning=None, response_format=None):
        body = {
            "prompt": prompt, "messages": messages, "system": system,
            "provider": provider, "model": model,
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "tools": tools, "tool_choice": tool_choice,
            "cache_system": cache_system, "reasoning": reasoning,
            "response_format": response_format,
        }
        body = {k: v for k, v in body.items() if v is not None}
        with httpx.stream("POST", f"{self.base_url}/v1/chat", json=body, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                if "delta" in d:
                    yield d["delta"]
                if d.get("done") or d.get("error"):
                    return

    def capabilities(self):
        return httpx.get(f"{self.base_url}/v1/capabilities", timeout=30).json()


def ask(prompt: str, provider: str = None, **kw) -> str:
    return LLM().chat(prompt, provider=provider, **kw)["text"]


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    print(ask("Say hello in one short line.", provider=p))
