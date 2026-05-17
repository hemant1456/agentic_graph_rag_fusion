from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from ingest import load_vector_store
from retrieve import get_chunks




def format_context(chunks):
    return "\n\n".join([f"{chunk.metadata["source"]} \n {chunk.page_content}" for chunk,score in chunks])

prompt = ChatPromptTemplate.from_messages([
    ('system','answer the question asked in query based only on the context provided'),
    ('human','Query: {query} \n\n Context: {context}')])
llm = ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite-preview', temperature=0)
chain = prompt | llm 

def generate(query,rel_chunks):
    
    
    context = format_context(rel_chunks)
    result = chain.invoke({'query': query, 'context':context})
    if isinstance(result.content, str):
        return result.content
    else:
        return result.content[0]["text"]

if __name__=='__main__':
    vs = load_vector_store()
    questions = ["What is the name of the CEO?","Who was on-call during the August 2023 NexusFlow outage?","What was the total Q3 2023 revenue?"]
    for query in questions:
        print(f"Question: {query}")
        rel_chunks = get_chunks(query,vs)
        
        result = generate(query,rel_chunks)
        
        print(f"Annwer: {result}\n\n")