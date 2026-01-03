import os
from groq import Groq

from src.vector_store import build_vector_store
from src.retrieval import retrieve
from src.rag_chain import answer_with_rag


MODEL = "llama-3.3-70b-versatile"
KNOWLEDGE_DIR = "./internal_docs"


def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def run_query(client: Groq, col, embedder, query: str, k: int, label: str, strict: bool = True):
    print_section(label)
    print(f"Question: {query}\n")

    contexts = retrieve(col, embedder, query, k=k)

    print("Retrieved chunks (top-K):")
    if not contexts:
        print("  (none)")
    for i, (txt, meta) in enumerate(contexts, start=1):
        preview = txt.replace("\n", " ")[:180] + ("..." if len(txt) > 180 else "")
        print(f"  {i}) {meta.get('path','?')}  |  {preview}")

    print("\nModel answer:")
    print(answer_with_rag(client, query, contexts, model=MODEL, strict=strict))


def main():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY first: export GROQ_API_KEY=...")

    client = Groq(api_key=GROQ_API_KEY)

    # GOOD index
    col_good, embedder_good = build_vector_store(
        knowledge_dir=KNOWLEDGE_DIR,
        use_bad_chunking=False,
        collection_name="rag_demo_internal_docs_good",
        persist_dir=None,  # set to "chroma_db" if you want persistence
    )

    # ✅ SUCCESS
    # run_query(
    #     client, col_good, embedder_good,
    #     query="How does authentication work in this system?",
    #     k=4,
    #     label="SUCCESS: Good retrieval + grounded answer",
    #     strict=True,
    # )

    # ❌ FAILURE (not in docs)
    # run_query(
    #     client, col_good, embedder_good,
    #     query="Does this system support OAuth or SSO login?",
    #     k=4,
    #     label="FAILURE (Correct): Not in context → model should refuse",
    #     strict=True,
    # )

    # # ⚠️ Retrieval failure: top_k too low
    # run_query(
    #     client, col_good, embedder_good,
    #     query="Which environment variable controls token expiry?",
    #     k=1,
    #     label="FAILURE (Retrieval): top_k too low → missing context",
    #     strict=True,
    # )

    # # ⚠️ Design failure: bad chunking
    col_bad, embedder_bad = build_vector_store(
        knowledge_dir=KNOWLEDGE_DIR,
        use_bad_chunking=True,
        collection_name="rag_demo_internal_docs_bad",
        persist_dir=None,
    )
    run_query(
        client, col_bad, embedder_bad,
        query="Where is token verification performed?",
        k=2,
        label="FAILURE (Design): bad chunking → noisy retrieval",
        strict=True,
    )


if __name__ == "__main__":
    main()
