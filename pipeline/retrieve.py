from ingest import load_vector_store
from types_data import RetrievedChunk
from ingest import _get_embedder




def get_chunks(query,vs,embedder,num_chunks=5):
    query_vec = embedder.encode(query)
    results = vs.query(
        query_embeddings = query_vec,
        n_results = num_chunks,
        include = ['documents','metadatas','distances']
    )
    chunks = [RetrievedChunk(text=document,score = 1- distance/2, source= metadata.get('source',''),department= metadata.get('department','')) for document,metadata,distance in zip(results['documents'][0],results['metadatas'][0],results['distances'][0])]

    return chunks



if __name__=='__main__':
    vs = load_vector_store()
    embedder = _get_embedder()
    relevant_chunks = get_chunks("Who is the CEO of Vertexia?", vs, embedder)
    for chunk in relevant_chunks:
        print('\n\n')
        print(f"Score: {chunk.score:.3f} metadata: {chunk.department}\n")
        print(f"{chunk.text[:150]}\n")