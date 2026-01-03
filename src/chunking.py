from typing import List


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple sliding-window chunker (character-based).
    Great for demos. For production consider structural chunking.
    """
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks


def bad_chunk_text(text: str) -> List[str]:
    """
    Intentionally bad chunking to demonstrate failure:
    returns a single giant chunk (no separation).
    """
    return [text]
