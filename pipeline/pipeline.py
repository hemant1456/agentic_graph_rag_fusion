from ingest import load_vector_store
from retrieve import get_chunks
from generate import generate


def ask(query,vs):
    rel_chunks = get_chunks(query,vs)
    result = generate(query,rel_chunks)
    return result

if __name__=='__main__':
    vs = load_vector_store()
    questions = ["What is the name of the CEO?","Who was on-call during the August 2023 NexusFlow outage?","What was the total Q3 2023 revenue?"]
    for query in questions:
        result = ask(query, vs)
        print(f"Question: {query}")
        print(f"Annwer: {result}\n\n")