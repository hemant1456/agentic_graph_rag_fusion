
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: str
    department: str


class RAGResult(BaseModel):
    question: str 
    answer: str 
    retrieved_chunks: list[RetrievedChunk]
    context: str
    retrieval_latency_ms: float  = Field(description="retrieval latency in milli seconds")
    generation_latency_ms: float = Field(description="generation latency in milli seconds")

class SmartChunk(BaseModel):
    text: str
    source: str
    department: str
    chunk_idx: int 
    format: str