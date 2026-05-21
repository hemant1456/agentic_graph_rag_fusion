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

SYSTEM_PROMPT = """\
You are an assistant answering questions about Vertexia using:
  1. Retrieved context documents (for facts, narratives, qualitative info).
  2. Tools (for deterministic aggregates: sums, counts, totals over structured data).

It you want to do some analysis over CSV data:
  1. First call list_csvs to see what tables are available.
  2. then Call csv_info on the relevant table to learn its columns and value formats.
  3. and then Call query_csv with the right filters and aggregation.
Do not skip step 2 if you don't know the column names.

Otherwise, answer from the provided context. \
If neither context nor tools can answer, say so explicitly. Do not hallucinate. Please try available tools once before concluding"""


def generate(client,query, context, tools = None):
    contents = [genai_types.Content(
        role = "user",
        parts= [genai_types.Part(f"please answer this question: {query} based on given context: {context}")]
    )]
    config_kwargs = {
            "system_instruction":SYSTEM_PROMPT,
            "temperature":0,
            "max_output_tokens":500,
            "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(disable=True)
        }
    _tool_map = {}
    if tools is not None:
        config_kwargs["tools"] = tools
        _tool_map = {tool.__name__:tool for tool in tools}

    
    for _ in range(10):
        response =  client.models.generate_content(
        
        model = MODEL_NAME,
        contents = contents,
        config = genai_types.GenerateContentConfig(**config_kwargs))

        if not response.function_calls:
            return response
        contents.append(response.candidates[0].content)

        function_result_parts = []

        for fnc in response.function_calls:
            try:
                result = _tool_map[fnc.name](**fnc.args)
            except Exception as e:
                result = {'error':str(e)}
            function_result_parts.append(genai_types.Part.from_function_response(
                name = fnc.name,
                response={'result': result}
            ))
            print(f'called function {fnc.name} with arguments {fnc.args}')
        contents.append(genai_types.Content(
            role= 'user',
            parts = function_result_parts
        ))
    contents.append(genai_types.Content(
    role="user",
    parts=[genai_types.Part(text="You have used your full tool budget. Based on what you have learned so far, write your final answer now. Do not request any more tool calls.")],
))
    config_kwargs.pop('tools',None)
    response =  client.models.generate_content(
        
    model = MODEL_NAME,
    contents = contents,
    config = genai_types.GenerateContentConfig(**config_kwargs))

    return response

