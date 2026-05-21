import argparse
from retrieve import _get_embedder
from ingest import load_vector_store
from retrieve import get_chunks, hybrid_retrieve
from generate import generate, _get_client
import time
from types_data import RAGResult
from dotenv import load_dotenv
from ingest import load_and_chunk, build_vector_store
load_dotenv()
from pathlib import Path
import sys
import argparse
from tools import query_csv, list_csvs, csv_info
from rank_bm25 import BM25Okapi


def format_context(chunks):
    return "\n\n".join([f"{chunk.source} \n {chunk.text}" for chunk in chunks])



class BaseLineRag():
    def __init__(self,client=None, embedder = None):
        self.vs = None
        if client is None:
            self.client = _get_client()
        else:
            self.client = client
        if embedder is None:
            self.embedder = _get_embedder()
        else:
            self.embedder = embedder
    
    def build(self,reset = False, docs_dir = None, chunk_size = 1000, chunk_overlap = 100):
        if not reset:
            try:
                self.vs = load_vector_store()
                if self.vs.count()>0:
                    return self
            except:
                pass
        chunks = load_and_chunk(docs_dir, chunk_size, chunk_overlap)
        vs = build_vector_store(chunks,reset = reset)
        self.vs = vs

        return self
    
    def query(self,question):
        retrieval_start = time.perf_counter()
        rel_chunks = get_chunks(question,self.vs,self.embedder)
        retrieval_end = time.perf_counter()
        retrieval_latency = (retrieval_end-retrieval_start)* 1000

        context = format_context(rel_chunks)

        generation_start = time.perf_counter()
        answer = generate(self.client,question,context)
        generation_end = time.perf_counter()
        generation_latency = (generation_end-generation_start)* 1000

        result = RAGResult(question=question, answer=answer.text, retrieved_chunks=rel_chunks, context=context, retrieval_latency_ms=retrieval_latency, generation_latency_ms=generation_latency)

        return result

class RAGWithTools(BaseLineRag):
    def __init__(self, client=None, embedder=None):
        super().__init__(client=client, embedder=embedder)
        self.tools = [query_csv, list_csvs,csv_info]
    def query(self,question):
        retrieval_start = time.perf_counter()
        rel_chunks = get_chunks(question,self.vs,self.embedder)
        retrieval_end = time.perf_counter()
        retrieval_latency = (retrieval_end-retrieval_start)* 1000

        context = format_context(rel_chunks)

        generation_start = time.perf_counter()
        answer = generate(client = self.client,query = question,context = context,tools = self.tools)
        generation_end = time.perf_counter()
        generation_latency = (generation_end-generation_start)* 1000

        result = RAGResult(question=question, answer=answer.text, retrieved_chunks=rel_chunks, context=context, retrieval_latency_ms=retrieval_latency, generation_latency_ms=generation_latency)

        return result
    

class HybridRAG(RAGWithTools):
    def build(self,**kwargs):
        super().build(**kwargs)
        all_data = self.vs.get(include=["documents","metadatas"])
        self.chunk_ids = all_data['ids']
        self.id_to_idx = {chunk_id:i for i,chunk_id in enumerate(self.chunk_ids)}
        self.chunk_texts = all_data["documents"]
        self.chunk_metas = all_data["metadatas"]
        split_text = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(split_text)

        return self
    def query(self,question):
        retrieval_start = time.perf_counter()
        rel_chunks = hybrid_retrieve(self,question)
        retrieval_end = time.perf_counter()
        retrieval_latency = (retrieval_end-retrieval_start)* 1000

        context = format_context(rel_chunks)

        generation_start = time.perf_counter()
        answer = generate(client = self.client,query = question,context = context,tools = self.tools)
        generation_end = time.perf_counter()
        generation_latency = (generation_end-generation_start)* 1000

        result = RAGResult(question=question, answer=answer.text, retrieved_chunks=rel_chunks, context=context, retrieval_latency_ms=retrieval_latency, generation_latency_ms=generation_latency)

        return result

if __name__=='__main__':
    questions = ["What is the name of the CEO?","Who was on-call during the August 2023 NexusFlow outage?","What is the overall total revenue in 2023?"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset',action="store_true")
    parser.add_argument('--chunk_size',default=1000,type = int,required=False)
    parser.add_argument('--chunk_overlap',default=100, type= int,required=False)
    args = parser.parse_args()
    #rag_with_tool = RAGWithTools().build(reset= args.reset, chunk_size = args.chunk_size, chunk_overlap = args.chunk_overlap)
    hybrid_rag = HybridRAG().build(reset= args.reset, chunk_size = args.chunk_size, chunk_overlap = args.chunk_overlap)
    
    for question in questions:
        result = hybrid_rag.query(question)
        print(f"Question: {question}")
        print(f"Answer: {result.answer}")
        print(f"retrieval_latency_ms: {result.retrieval_latency_ms}")
        print(f"generation_latency_ms: {result.generation_latency_ms}\n\n")