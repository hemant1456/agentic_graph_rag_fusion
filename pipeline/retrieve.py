from ingest import load_vector_store

def get_chunks(query,vs,num_chunks=5):
    relevant_chunks = vs.similarity_search_with_relevance_scores(query, k = num_chunks)
    return relevant_chunks


if __name__=='__main__':
    vs = load_vector_store()
    relevant_chunks = get_chunks("Who is the CEO of Vertexia?", vs)
    for chunk,score in relevant_chunks:
        print('\n\n')
        print(f"Score: {score:.3f} metadata: {chunk.metadata}\n")
        print(f"{chunk.page_content[:150]}\n")