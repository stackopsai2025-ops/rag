from pathlib import Path
from typing import List, Dict, Set


def load_documents(knowledge_dir: str, exts: Set[str] = {".md", ".txt", ".py"}) -> List[Dict]:
    """
    Load knowledge files (documents) from a directory.

    Returns list of dicts:
      {"path": "...", "text": "..."}
    """
    base = Path(knowledge_dir)
    if not base.exists():
        raise FileNotFoundError(f"Knowledge folder not found: {knowledge_dir}")

    docs = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            docs.append({"path": str(p), "text": p.read_text(errors="ignore")})
    return docs
