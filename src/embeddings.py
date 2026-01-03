from sentence_transformers import SentenceTransformer
from typing import List


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> List[list]:
    """
    Returns embeddings as python lists (serializable).
    """
    return embedder.encode(texts).tolist()
