from llm_gatewayV2.client import LLM

SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about Vertexia Inc. using \
provided context documents. Answer based only on the context. If the context \
does not contain enough information to answer fully, say so explicitly. \
Do not hallucinate facts not present in the context.\
"""

_client = LLM()


def generate_answer(context: str, question: str) -> tuple[str, str]:
    """Generate an answer given formatted context and a question.

    Routes through llm_gatewayV2 which handles provider selection, fallback
    (cerebras → groq → gemini → nvidia), retries, and rate-limit cooldowns.

    Returns (answer, provider) where provider is the gateway-selected name.
    """
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    resp = _client.chat(
        messages=[{"role": "user", "content": user_message}],
        system=SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.0,
    )
    return resp["text"], resp["provider"]
