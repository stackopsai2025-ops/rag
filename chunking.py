

import os
import re
import json
import numpy as np

import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Create a key at https://console.groq.com/keys and export GROQ_API_KEY."
    )


# ----------------------------
# Groq + model
# ----------------------------
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  
client = Groq(api_key=GROQ_API_KEY)  


# ----------------------------
# Token counting (tiktoken)
# ----------------------------
enc = tiktoken.get_encoding("o200k_base")


def tok_len(text: str) -> int:
    return len(enc.encode(text))


def decode_tokens(tokens) -> str:
    return enc.decode(tokens)


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------
# Example doc (has headings)
# ----------------------------
DOC = """
# RAG Overview
Retrieval-Augmented Generation (RAG) improves answer quality by retrieving relevant information at query time and providing it to the language model as context.

# Context Windows
A model can only process a fixed number of tokens in its context window. That window must include instructions, the user query, retrieved passages, and the model’s own output.

# Chunking
Chunking splits documents into retrievable units. Too small loses meaning. Too large becomes noisy.

# Overlap
Overlap repeats a tail of text into the next chunk to reduce boundary loss. But overlap increases duplicated tokens, cost, and sometimes retrieval noise.

# Practical Guidance
Production systems often combine approaches: structural splits first, then token-bounded chunking, and optionally semantic or LLM-based refinement for messy sources.
""".strip()

QUESTION = "What are the trade-offs of overlap in RAG chunking?"


# ----------------------------
# Chunking methods
# ----------------------------
def fixed_chunk(text: str, chunk_tokens: int):
    tokens = enc.encode(text)
    out = []
    i = 0
    while i < len(tokens):
        out.append(decode_tokens(tokens[i : i + chunk_tokens]))
        i += chunk_tokens
    return out


def fixed_chunk_overlap(text: str, chunk_tokens: int, overlap_tokens: int):
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be < chunk_tokens")

    tokens = enc.encode(text)
    out = []
    step = chunk_tokens - overlap_tokens
    i = 0
    while i < len(tokens):
        out.append(decode_tokens(tokens[i : i + chunk_tokens]))
        i += step
    return out


def split_sentences(text: str):
    text = re.sub(r"\n+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z#])", text)
    return [p.strip() for p in parts if p.strip()]


def dynamic_sentence_chunk(text: str, max_tokens: int):
    sents = split_sentences(text)
    out, buf, buf_t = [], [], 0
    for s in sents:
        t = tok_len(s)
        if buf and buf_t + t > max_tokens:
            out.append(" ".join(buf))
            buf, buf_t = [], 0
        buf.append(s)
        buf_t += t
    if buf:
        out.append(" ".join(buf))
    return out


def structural_heading_chunk(text: str, max_tokens: int):
    lines = text.splitlines()
    sections = []
    title = "Intro"
    body = []

    for line in lines:
        if line.startswith("#"):
            if body:
                sections.append((title, " ".join(body).strip()))
            title = line.lstrip("#").strip()
            body = []
        else:
            if line.strip():
                body.append(line.strip())

    if body:
        sections.append((title, " ".join(body).strip()))

    out = []
    for (t, b) in sections:
        combined = f"{t}: {b}".strip()
        if tok_len(combined) <= max_tokens:
            out.append(combined)
        else:
            out.extend(dynamic_sentence_chunk(combined, max_tokens=max_tokens))
    return out


def semantic_tfidf_grouping(text: str, max_tokens: int, sim_threshold: float = 0.20):
    """
    Very simple 'semantic-ish' grouping:
    - Start from paragraphs
    - Merge adjacent paragraphs if TF-IDF cosine similarity is high
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        return paras

    vec = TfidfVectorizer(stop_words="english").fit_transform(paras)
    sims = cosine_similarity(vec)

    out = []
    i = 0
    while i < len(paras):
        cur = paras[i]
        j = i + 1
        while j < len(paras):
            merged = cur + "\n\n" + paras[j]
            if tok_len(merged) > max_tokens:
                break
            if sims[j - 1, j] < sim_threshold:
                break
            cur = merged
            j += 1
        out.append(cur)
        i = j
    return out


def llm_based_chunking_groq(text: str):
  
    prompt = f"""
Split the document into 4 to 6 coherent chunks for RAG retrieval.

Rules:
- Each chunk must be self-contained and focused on one idea.
- Prefer splitting at topic boundaries.
- Add a short title for each chunk.
Return ONLY valid JSON array in this format:
[{{"title":"...","chunk":"..."}}, ...]

DOCUMENT:
{text}
""".strip()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return strictly valid JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )  # Groq chat completions :contentReference[oaicite:3]{index=3}

    raw = resp.choices[0].message.content.strip()
    items = json.loads(raw)
    return [f"{it.get('title','Chunk')}: {it.get('chunk','')}".strip() for it in items]


# ----------------------------
# Tiny retrieval + answer (kept intentionally simple)
# ----------------------------
def keyword_score(q: str, chunk: str) -> int:
    q_words = set(re.findall(r"[a-zA-Z']+", q.lower()))
    c_words = set(re.findall(r"[a-zA-Z']+", chunk.lower()))
    return len(q_words & c_words)


def top_k_chunks(question: str, chunks, k=3):
    ranked = sorted(chunks, key=lambda c: keyword_score(question, c), reverse=True)
    return ranked[:k]


def ask_groq_with_context(question: str, chunks):
    context = "\n\n".join([f"- {clean_ws(c)}" for c in chunks])

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context. If missing, say what is missing."},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
        ],
        temperature=0.2,
        max_completion_tokens=220,
    )
    return resp.choices[0].message.content.strip()


def show(name: str, chunks):
    print("\n" + "=" * 90)
    print("STRATEGY:", name)
    print("=" * 90)
    print("chunks:", len(chunks))
    print("first chunk tokens:", tok_len(chunks[0]) if chunks else 0)
    print("first chunk preview:", clean_ws(chunks[0])[:180] + ("..." if len(chunks[0]) > 180 else ""))

    retrieved = top_k_chunks(QUESTION, chunks, k=3)
    print("\nretrieved top-3 scores:", [keyword_score(QUESTION, c) for c in retrieved])

    answer = ask_groq_with_context(QUESTION, retrieved)
    print("\nGroq answer:\n", answer)


if __name__ == "__main__":
    max_t = 110

    chunks_fixed = fixed_chunk(DOC, chunk_tokens=max_t)
    chunks_fixed_ov = fixed_chunk_overlap(DOC, chunk_tokens=max_t, overlap_tokens=25)
    chunks_dyn = dynamic_sentence_chunk(DOC, max_tokens=max_t)
    chunks_struct = structural_heading_chunk(DOC, max_tokens=max_t)
    chunks_sem = semantic_tfidf_grouping(DOC, max_tokens=max_t, sim_threshold=0.18)
    chunks_llm = llm_based_chunking_groq(DOC)

    show("Fixed-size", chunks_fixed)
    show("Fixed-size + overlap", chunks_fixed_ov)
    show("Dynamic (sentence-bounded)", chunks_dyn)
    show("Structural (headings first)", chunks_struct)
    show("Semantic (TF-IDF similarity grouping)", chunks_sem)
    show("LLM-based (Groq Llama splits boundaries)", chunks_llm)
