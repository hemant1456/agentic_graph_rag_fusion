"""
Step 02 — Observability: generate_answer variant that also returns token usage.

Extends Step 01's generate_answer() by capturing the usage metadata that the
Gemini and Anthropic SDKs return in the response object. Step 01 code is
unchanged — this is an additive Step 02 module.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from step_01_baseline_rag.implementation.generate import (
    ANTHROPIC_MODEL,
    GEMINI_MODEL,
    SYSTEM_PROMPT,
)

load_dotenv()


@dataclass
class GenerationOutput:
    answer: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int


def generate_with_usage(context: str, question: str) -> GenerationOutput:
    """
    Generate an answer and capture token counts from the API response.

    Gemini: response.usage_metadata.{prompt_token_count, candidates_token_count}
    Anthropic: message.usage.{input_tokens, output_tokens}
    """
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    # ── Gemini (primary) ──────────────────────────────────────────────────────
    try:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=512,
                temperature=0.0,
            ),
        )
        usage = response.usage_metadata
        return GenerationOutput(
            answer=response.text,
            provider="gemini",
            model=GEMINI_MODEL,
            prompt_tokens=usage.prompt_token_count or 0,
            completion_tokens=usage.candidates_token_count or 0,
        )
    except Exception:
        pass

    # ── Anthropic (fallback) ──────────────────────────────────────────────────
    ac = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = ac.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return GenerationOutput(
        answer=message.content[0].text,
        provider="anthropic",
        model=ANTHROPIC_MODEL,
        prompt_tokens=message.usage.input_tokens,
        completion_tokens=message.usage.output_tokens,
    )
