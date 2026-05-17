import os

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about Vertexia Inc. using \
provided context documents. Answer based only on the context. If the context \
does not contain enough information to answer fully, say so explicitly. \
Do not hallucinate facts not present in the context.\
"""


def _generate_gemini(context: str, question: str) -> str:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=512,
            temperature=0.0,   # deterministic for evaluation
        ),
    )
    return response.text


def _generate_anthropic(context: str, question: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return next(block.text for block in message.content if hasattr(block, "text"))


def generate_answer(context: str, question: str) -> tuple[str, str]:
    """
    Generate an answer given formatted context and a question.
    Returns (answer, provider) where provider is 'gemini' or 'anthropic'.
    """
    try:
        return _generate_gemini(context, question), "gemini"
    except Exception as e:
        print(f"  [Gemini failed: {e}] — falling back to Anthropic")
        return _generate_anthropic(context, question), "anthropic"
