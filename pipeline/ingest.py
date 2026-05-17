from pathlib import Path
from langchain_core.documents import Document
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

DOCS_DIR = Path(__file__).parent.parent/"step_00_dataset"/"company_data"

def load_and_chunk(docs_dir):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 100)
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),('##','h2'),("###",'h3')])

    chunks = []
    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix not in ['.csv','.md']:
            with open(file_path,'r') as f:
                content = f.read()
                metadata = {'source':f'{file_path.relative_to(docs_dir)}','department':f'{file_path.parent.name}'}
                document = Document(page_content=content, metadata=metadata)
                new_chunks = text_splitter.split_documents([document])
                for idx,chunk in enumerate(new_chunks):
                    chunk.metadata.update({'chunk_index':idx})
                chunks.extend(new_chunks)
        elif file_path.suffix=='.md':
            with open(file_path,'r') as f:
                content = f.read()
                metadata = {'source':f'{file_path.relative_to(docs_dir)}','department':f'{file_path.parent.name}'}
                
                new_chunks = markdown_splitter.split_text(content)
                for idx,chunk in enumerate(new_chunks):
                    chunk.metadata.update({'chunk_index':idx,'source':str(file_path.relative_to(docs_dir)),'department':str(file_path.parent.name)})
                chunks.extend(new_chunks)
        elif file_path.suffix=='.csv':
            with open(file_path,'r') as f:
                data = csv.DictReader(f)
                for i, row in enumerate(data):
                    text = ' | '.join([f"{k}:{v}" for k,v in row.items()])
                    metadata = {'source':str(file_path.relative_to(docs_dir)),'department':file_path.parent.name,'chunk_index':i}
                    chunks.append(Document(page_content=text,metadata = metadata))
    return chunks

CHROMA_DIR = Path(__file__).parent/"chroma"
def build_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks,embedding=embedding_model,persist_directory=str(CHROMA_DIR))
    return vector_store

def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(embedding_function=embedding_model,persist_directory=str(CHROMA_DIR))
    return vector_store

if __name__=='__main__':
    if CHROMA_DIR.exists():
        vector_stores = load_vector_store()
    else:
        chunks = load_and_chunk(DOCS_DIR)
        vector_stores = build_vector_store(chunks)
    print(f"number of vector embeddings: {vector_stores._collection.count()}")
