from typing import Tuple, Optional, List, Dict

import chromadb
from chromadb.config import Settings

from src.loader import load_documents
from src.chunking import chunk_text, bad_chunk_text
from src.embeddings import get_embedder, embed_texts


def build_vector_store(
    knowledge_dir: str,
    use_bad_chunking: bool = False,
    collection_name: str = "rag_demo_internal_docs",
    persist_dir: Optional[str] = None,
    embed_model: str = "all-MiniLM-L6-v2",
) -> Tuple[object, object]:
    """
    Build a Chroma collection + embedder.

    persist_dir:
      - None = in-memory
      - "chroma_db" = persisted to disk (recommended if large docs)
    """
    docs = load_documents(knowledge_dir)

    embedder = get_embedder(embed_model)

    if persist_dir:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
    else:
        client = chromadb.Client(Settings(anonymized_telemetry=False))

    # recreate collection each run for clean demo
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)

    chunker = bad_chunk_text if use_bad_chunking else chunk_text

    all_items: List[Dict] = []
    idx = 0
    for d in docs:
        for ch in chunker(d["text"]):
            all_items.append(
                {
                    "id": f"c{idx}",
                    "chunk": ch,
                    "meta": {"path": d["path"]},
                }
            )
            idx += 1

    texts = [x["chunk"] for x in all_items]
    vectors = embed_texts(embedder, texts)
    ids = [x["id"] for x in all_items]
    metas = [x["meta"] for x in all_items]

    col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    return col, embedder
