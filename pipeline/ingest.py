from pathlib import Path
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
import chromadb
import time
from types_data import SmartChunk

CHROMA_DIR = Path(__file__).parent/"chroma"
DOCS_DIR = Path(__file__).parent.parent/"dataset"/"company_data"
COLLECTION_NAME = "langchain"

_embedder : SentenceTransformer | None = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


_client = None
def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path = str(CHROMA_DIR))
    return _client



def load_and_chunk(docs_dir, chunk_size = 1000, chunk_overlap= 100):
    if docs_dir is None:
        docs_dir = DOCS_DIR
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap= chunk_overlap)
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),('##','h2'),("###",'h3')])
    supported_formats = ['.txt','.csv','.md']

    chunks : list[SmartChunk] = []
    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix not in supported_formats:
            continue
        if file_path.suffix=='.md':
            with open(file_path,'r') as f:
                content = f.read()
                chunk_texts = markdown_splitter.split_text(content)
                new_chunks = [SmartChunk(text = chunk.page_content, 
                source = str(file_path.relative_to(docs_dir)),
                department = file_path.parent.name,
                chunk_idx = i,
                format = file_path.suffix)
                for i,chunk in enumerate(chunk_texts)]
                chunks.extend(new_chunks)
        elif file_path.suffix=='.txt':
            with open(file_path,'r') as f:
                content = f.read()
                chunk_texts = text_splitter.split_text(content)
                new_chunks = [SmartChunk(text = chunk, 
                source = str(file_path.relative_to(docs_dir)),
                department = file_path.parent.name,
                chunk_idx = i,
                format = file_path.suffix)
                for i,chunk in enumerate(chunk_texts)]
                chunks.extend(new_chunks)

        elif file_path.suffix=='.csv':
            with open(file_path,'r') as f:
                data = csv.DictReader(f)
                new_chunks = [SmartChunk(text = ' | '.join([f"{k}:{v}" for k,v in row.items()]), 
                source = str(file_path.relative_to(docs_dir)),
                department = file_path.parent.name,
                chunk_idx = i,
                format = file_path.suffix)
                for i,row in enumerate(data)]
                chunks.extend(new_chunks)
    return chunks
   

def build_vector_store(chunks,name = COLLECTION_NAME,reset = False):
    client = _get_client() 
    if reset:
        try:
            client.delete_collection(name)
        except:
            pass
    start_time = time.perf_counter()
    
    collection = client.get_or_create_collection(
            name = name,
            metadata = {'hnsw:space':'cosine'}
        )
    embedder = _get_embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts).tolist()
    metadata = [{"source":chunk.source,"idx":chunk.chunk_idx,"department":chunk.department,"format":chunk.format} for chunk in chunks]
    ids = [f"{chunk.source}:{chunk.chunk_idx}" for chunk in chunks]
    collection.upsert(
        ids  = ids,
        documents = texts,
        metadatas = metadata,
        embeddings = embeddings
    )
    end_time = time.perf_counter()
    time_taken = (end_time-start_time)
    print(f"number of vector embeddings: {collection.count()}")
    print(f"time taken to build vector databasae {time_taken} seconds")
    return collection

def load_vector_store(name=COLLECTION_NAME):
    client = _get_client()
    return client.get_collection(name=name)
    
