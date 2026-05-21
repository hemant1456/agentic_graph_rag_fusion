import argparse
import sys
from pathlib import Path
from pydantic import BaseModel, Field

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
from pipeline.pipeline import BaseLineRag, RAGWithTools, HybridRAG
from google import genai
from google.genai import types as genai_type
from typing import Literal
from collections import defaultdict
load_dotenv()

class JudgeVerdict(BaseModel):
    verdict: Literal["pass","fail","partial"] = Field(description="pass means answered correctly, fail means it could answer only some part and partial means that it could answer mostly but didn't get all or exactly right")
    correctness: float = Field(description="how much the answer by agent and reference are aligned, score between 0 to 1")
    faithfullness: float = Field(description="how much does llm follow the retrieved chunks to generate answer and didn't add things on its own, score between 0 to 1")
    recall: float = Field(description="how much of required context does the retrieved chunks were able to capture, score between 0 to 1")
    precision: float = Field(description="how much of retrieved chunks were helpful, score between 0 to 1")


MODEL_NAME = 'gemini-3.1-flash-lite-preview'

def judge(client, system_prompt, question, reference, answer, context):
    verdict = client.models.generate_content(
        model = MODEL_NAME,
        contents = f"""judge the accuracy of answer: {answer} 
        based on following details:
        question: {question}
        context: {context}
        reference: {reference}
        """,
        config = genai_type.GenerateContentConfig(
            system_instruction = system_prompt,
            temperature = 0, 
            max_output_tokens = 2000,
            response_schema=JudgeVerdict
        )
    )
    return verdict



def main():
    parser = argparse.ArgumentParser(description="Run the 14-question golden eval with LLM-as-judge.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Splitter chunk size in chars")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Splitter overlap in chars")
    parser.add_argument("--reset", action = "store_true")
    args = parser.parse_args()


    print(f"Config: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, reset={args.reset}")
    print()

    # rag = BaseLineRag().build(
    #     reset=args.reset,
    #     chunk_size=args.chunk_size,
    #     chunk_overlap=args.chunk_overlap,
    # )

    # rag = RAGWithTools().build(
    #     reset=args.reset,
    #     chunk_size=args.chunk_size,
    #     chunk_overlap=args.chunk_overlap,
    # )
    rag = HybridRAG().build(
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    
    client = genai.Client()
    
    SYSTEM_PROMPT = "you are a judge, who will give score based on the provided context "
    overall_verdict = defaultdict(int)
    correctness = 0
    number_of_question = 14

    for idx,q in enumerate(GOLDEN_QUESTIONS[:number_of_question],start=1):
        result = rag.query(q.question)
        response = judge(client, SYSTEM_PROMPT,q.question, q.reference_answer, result.answer, result.context)
        verdict = JudgeVerdict.model_validate_json(response.text)
        print(f"Number: {idx} question {q.question}")
        print(verdict.verdict)
        print(f"\n\n")
        overall_verdict[verdict.verdict]+=1
        correctness+= verdict.correctness
    print(f"overall score {overall_verdict}")
    print(f"average_correctness {correctness/number_of_question}")




if __name__ == "__main__":
    main()
