"""
Step 03 — Evaluation Framework: LLM-as-judge.

Makes a single structured LLM call and returns a parsed JSON dict.
Used by every metric function in metrics.py.

The judge uses the same LLM providers as the RAG pipeline (Gemini primary,
Anthropic fallback) but with a lower token budget — evaluation prompts ask
for a score + one-sentence reasoning, not a full answer.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

JUDGE_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
JUDGE_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MAX_TOKENS = 512


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(text)


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=JUDGE_GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            max_output_tokens=JUDGE_MAX_TOKENS,
            temperature=0.0,
        ),
    )
    return response.text


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=JUDGE_ANTHROPIC_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def judge(prompt: str, retries: int = 2) -> dict[str, Any]:
    """
    Call the LLM judge and return a parsed JSON dict.

    Tries Gemini first, falls back to Anthropic.
    On JSON parse failure, retries up to `retries` times.
    On total failure, returns a safe default {"score": 0.5, "reasoning": "..."}.
    """
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(2 ** (attempt - 1))

        try:
            raw = _call_gemini(prompt)
        except Exception as gemini_err:
            try:
                raw = _call_anthropic(prompt)
            except Exception as anthropic_err:
                last_error = anthropic_err
                continue

        try:
            return _extract_json(raw)
        except (json.JSONDecodeError, ValueError, IndexError) as parse_err:
            last_error = parse_err
            continue

    return {
        "score": 0.5,
        "reasoning": f"Judge failed after {retries + 1} attempts: {last_error}",
    }
