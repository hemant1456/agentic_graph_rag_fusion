from ingest import load_vector_store
from types_data import RetrievedChunk
from ingest import _get_embedder
from collections import defaultdict



def get_chunks(query,vs,embedder,num_chunks=5):
    query_vec = embedder.encode(query)
    results = vs.query(
        query_embeddings = query_vec,
        n_results = num_chunks,
        include = ['documents','metadatas','distances']
    )
    chunks = [RetrievedChunk(text=document,score = 1- distance/2, source= metadata.get('source',''),department= metadata.get('department','')) for document,metadata,distance in zip(results['documents'][0],results['metadatas'][0],results['distances'][0])]

    return chunks

def rrf_fuse(rankings,k =60, top_k= 5):
    scores = defaultdict(float)
    for ranking in rankings:
        for i,chunk_id in enumerate(ranking,start=1):
            scores[chunk_id] += 1/(k+i)
    top_k_scores = sorted(scores, key = lambda x: scores.get(x), reverse = True)[:top_k]
    return top_k_scores

def bm25_retrieve(hybrid_rag,query,num_chunks = 10):
    scores = hybrid_rag.bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(hybrid_rag.chunk_ids)),key=lambda i: scores[i], reverse=True)[:num_chunks]
    chunk_ids = [hybrid_rag.chunk_ids[idx] for idx in top_idx]
    return chunk_ids

def hybrid_retrieve(hybrid_rag,query):
    query_vec = hybrid_rag.embedder.encode(query)
    dense_chunks = hybrid_rag.vs.query(
        query_embeddings = query_vec,
        n_results = 20,
        include = ['documents','metadatas','distances']
    )

    dense_chunk_ids = dense_chunks['ids'][0]
    bm25_chunk_ids = bm25_retrieve(hybrid_rag,query,num_chunks=10)
    top_k_ids = rrf_fuse([dense_chunk_ids,bm25_chunk_ids])
    rel_chunks = []
    for chunk_id in top_k_ids:
        idx = hybrid_rag.id_to_idx[chunk_id]
        rel_chunks.append(RetrievedChunk(text=hybrid_rag.chunk_texts[idx],
        score = 1,
        source= hybrid_rag.chunk_metas[idx]['source'],
        department= hybrid_rag.chunk_metas[idx]['department']))
    return rel_chunks


if __name__=='__main__':
    vs = load_vector_store()
    embedder = _get_embedder()
    relevant_chunks = get_chunks("Who is the CEO of Vertexia?", vs, embedder)
    for chunk in relevant_chunks:
        print('\n\n')
        print(f"Score: {chunk.score:.3f} metadata: {chunk.department}\n")
        print(f"{chunk.text[:150]}\n")