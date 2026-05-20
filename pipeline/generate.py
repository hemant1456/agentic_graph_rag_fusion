from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
load_dotenv()

_client : genai.Client | None = None
def _get_client():
    global _client 
    if _client is None:
        _client = genai.Client()
    return _client

MODEL_NAME = 'gemini-3.1-flash-lite-preview'
SYSTEM_PROMPT = "you are a helpful assistant who answer the question asked in query based only on the context provided, if you don't find the needed info in context say so"


def generate(client,query, context):
    response =  client.models.generate_content(
        model = MODEL_NAME,
        contents = f"please answer this question: {query} based on given context: {context}",
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=500
        )
    )
    return response