from typing import List, Tuple, Dict


def retrieve(col, embedder, query: str, k: int = 4) -> List[Tuple[str, Dict]]:
    """
    Retrieve top-K documents from the vector store.
    Returns list of (chunk_text, metadata)
    """
    q_vec = embedder.encode([query]).tolist()
    res = col.query(query_embeddings=q_vec, n_results=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    return list(zip(docs, metas))
