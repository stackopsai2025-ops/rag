import re
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# -----------------------------
# 1) Demo Data (Documents + Metadata)
# -----------------------------
@dataclass
class Doc:
    doc_id: str
    title: str
    service: str
    version: str
    doc_type: str
    updated: str
    text: str


DOCS = [
    Doc(
        doc_id="D1",
        title="IAM Basics: Identity vs Resource Policies",
        service="IAM",
        version="latest",
        doc_type="guide",
        updated="2025-11-10",
        text="""
IAM policies define permissions. Resource-based policies can be attached to resources like S3 buckets.
Cross-account access typically requires granting the principal permissions to call relevant actions.
Best practice: least privilege, explicit deny where appropriate.
""".strip()
    ),
    Doc(
        doc_id="D2",
        title="S3 Cross-Account Access (Recommended)",
        service="S3",
        version="latest",
        doc_type="guide",
        updated="2025-12-05",
        text="""
To enable S3 cross-account access, you generally need BOTH:
1) An identity-based policy on the caller (user/role) allowing S3 actions, and
2) A bucket policy (resource-based policy) on the target bucket allowing that principal.

Common required S3 permissions depend on the use case:
- Read objects: s3:GetObject
- List bucket: s3:ListBucket
- Write objects: s3:PutObject
In many cases you also need:
- s3:GetBucketLocation (some SDK flows)
Use explicit ARN scoping:
- Bucket: arn:aws:s3:::my-bucket
- Objects: arn:aws:s3:::my-bucket/*

If access is via AssumeRole:
- Caller needs sts:AssumeRole on the target role
- Target role trust policy must allow the caller account/role principal
""".strip()
    ),
    Doc(
        doc_id="D3",
        title="S3 Policy Example (Deprecated - Avoid)",
        service="S3",
        version="v0.11",
        doc_type="deprecated",
        updated="2020-03-02",
        text="""
[DEPRECATED] Example bucket policy:
{
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:*",
    "Resource": "arn:aws:s3:::my-bucket/*"
  }]
}
This example is convenient for testing but should not be used in production.
""".strip()
    ),
    Doc(
        doc_id="D4",
        title="Terraform S3 Module Notes",
        service="S3",
        version="v1.5",
        doc_type="config",
        updated="2025-08-18",
        text="""
Module supports bucket policy templates for cross-account access.
Inputs:
- allowed_principals
- allowed_actions
- prefix_restrictions
Note: Prefer restricting actions to GetObject/ListBucket/PutObject as needed.
""".strip()
    ),
    Doc(
        doc_id="D5",
        title="S3 Overview",
        service="S3",
        version="latest",
        doc_type="guide",
        updated="2025-10-22",
        text="""
Amazon S3 is an object storage service.
Access is controlled via IAM policies, bucket policies, ACLs, and other controls.
For cross-account patterns, bucket policies are commonly used.
""".strip()
    ),
]


# -----------------------------
# 2) Chunking (simple paragraph chunker)
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    service: str
    version: str
    doc_type: str
    updated: str
    text: str

def chunk_docs(docs: List[Doc]) -> List[Chunk]:
    chunks = []
    for d in docs:
        paras = [p.strip() for p in d.text.split("\n") if p.strip()]
        for i, p in enumerate(paras):
            chunks.append(
                Chunk(
                    chunk_id=f"{d.doc_id}#P{i+1}",
                    doc_id=d.doc_id,
                    title=d.title,
                    service=d.service,
                    version=d.version,
                    doc_type=d.doc_type,
                    updated=d.updated,
                    text=p
                )
            )
    return chunks

CHUNKS = chunk_docs(DOCS)


# -----------------------------
# 3) Dense + Sparse Indices
#    - Sparse: TF-IDF
#    - "Dense": LSA (TF-IDF -> SVD) to mimic semantic embeddings without external models
# -----------------------------
def build_indices(chunks: List[Chunk], n_components: int = 64):
    texts = [c.text for c in chunks]

    # Sparse
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X_sparse = tfidf.fit_transform(texts)

    # Dense-ish: LSA
    svd = TruncatedSVD(n_components=min(n_components, X_sparse.shape[1]-1))
    X_dense = svd.fit_transform(X_sparse)
    X_dense = normalize(X_dense)

    return tfidf, X_sparse, svd, X_dense

TFIDF, X_SPARSE, SVD, X_DENSE = build_indices(CHUNKS)


# -----------------------------
# 4) Similarity + Retrieval Utilities
# -----------------------------
def cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # mat assumed normalized if dense
    q = query_vec.reshape(1, -1)
    # cosine for normalized dense: dot product
    return (mat @ q.T).reshape(-1)

def dense_retrieve(query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
    q_sparse = TFIDF.transform([query])
    q_dense = normalize(SVD.transform(q_sparse))
    sims = cosine_sim_matrix(q_dense[0], X_DENSE)
    idx = np.argsort(-sims)[:k]
    return [(CHUNKS[i], float(sims[i])) for i in idx]

def sparse_retrieve(query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
    q = TFIDF.transform([query])  # shape (1, vocab)

    # --- Dot products: (n_docs x vocab) @ (vocab x 1) => (n_docs x 1)
    dots = (X_SPARSE @ q.T).toarray().ravel()  # force ndarray (n_docs,)

    # --- Norms
    X_norm = np.sqrt(X_SPARSE.multiply(X_SPARSE).sum(axis=1))
    X_norm = np.asarray(X_norm).ravel()  # force ndarray (n_docs,)

    q_norm = float(np.sqrt(q.multiply(q).sum()))  # force Python float

    # --- Cosine similarity
    denom = (X_norm * q_norm) + 1e-9
    sims = dots / denom  # ndarray (n_docs,)

    idx = np.argsort(-sims)[:k]
    return [(CHUNKS[i], float(sims[i])) for i in idx]



# -----------------------------
# 5) Metadata Filtering
# -----------------------------
def metadata_filter(
    items: List[Tuple[Chunk, float]],
    service: str = None,
    allowed_doc_types: List[str] = None,
    exclude_doc_types: List[str] = None,
    min_year: int = None,
) -> List[Tuple[Chunk, float]]:
    out = []
    for c, s in items:
        if service and c.service != service:
            continue
        if allowed_doc_types and c.doc_type not in allowed_doc_types:
            continue
        if exclude_doc_types and c.doc_type in exclude_doc_types:
            continue
        if min_year:
            y = int(c.updated.split("-")[0])
            if y < min_year:
                continue
        out.append((c, s))
    return out


# -----------------------------
# 6) Simple Re-ranker (cross-encoder-like heuristic)
#    We score query-chunk relevance using:
#    - sparse cosine
#    - keyword overlap bonus for critical tokens
#    - penalty for "deprecated"/"testing"/wildcards
# -----------------------------
CRITICAL_TERMS = {
    "s3": 0.8,
    "bucket": 0.4,
    "policy": 0.6,
    "cross-account": 1.0,
    "assumerole": 0.8,
    "trust": 0.8,
    "sts:assumerole": 1.0,
    "s3:getobject": 1.0,
    "s3:listbucket": 1.0,
    "s3:putobject": 1.0,
}

NEGATIVE_TERMS = {
    "deprecated": 1.0,
    "testing": 0.6,
    "\"principal\": \"*\"": 0.9,
    "s3:*": 0.8,
    "allow\": \"*\"": 0.8
}

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())

def rerank(query: str, candidates: List[Chunk], top_n: int = 5) -> List[Tuple[Chunk, float]]:
    qn = normalize_text(query)

    # baseline sparse score
    base = dict((c.chunk_id, score) for c, score in sparse_retrieve(query, k=len(CHUNKS)))

    scored = []
    for c in candidates:
        textn = normalize_text(c.text)

        score = base.get(c.chunk_id, 0.0)

        # critical token bonuses
        for term, w in CRITICAL_TERMS.items():
            if term in qn and term in textn:
                score += w
            # bonus if term in chunk even if not in query (helps S3 action mentions)
            if term in textn and term.startswith("s3:"):
                score += 1.0

        # light overlap score (word intersection)
        q_words = set(re.findall(r"[a-z0-9:\-]+", qn))
        t_words = set(re.findall(r"[a-z0-9:\-]+", textn))
        overlap = len(q_words & t_words) / (len(q_words) + 1e-9)
        score += 0.6 * overlap

        # negative penalties
        for term, p in NEGATIVE_TERMS.items():
            if term in textn:
                score -= p

        # metadata bonus
        if c.doc_type == "guide":
            score += 0.2
        if c.doc_type == "deprecated":
            score -= 0.7

        scored.append((c, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# -----------------------------
# 7) Pipeline Variants
# -----------------------------
def basic_rag_pipeline(query: str, k: int = 5):
    # "Basic": dense-only top-k, no filtering, no rerank
    return dense_retrieve(query, k=k)

def advanced_rag_pipeline(query: str, k_dense=15, k_sparse=15, top_n=5):
    # 1) retrieve candidates via dense + sparse
    dense_hits = dense_retrieve(query, k=k_dense)
    sparse_hits = sparse_retrieve(query, k=k_sparse)

    # merge candidates (dedupe by chunk_id)
    merged: Dict[str, Chunk] = {}
    for c, _ in dense_hits + sparse_hits:
        merged[c.chunk_id] = c
    candidates = list(merged.values())

    # 2) metadata filtering (keep S3 guides/config; exclude deprecated)
    filtered = [c for c in candidates if (
        c.service == "S3"
        and c.doc_type in ["guide", "config"]
        and c.doc_type != "deprecated"
    )]

    # 3) rerank
    reranked = rerank(query, filtered, top_n=top_n)
    return reranked


# -----------------------------
# 8) Display Helpers (for your screen share)
# -----------------------------
def print_results(title: str, results: List[Tuple[Chunk, float]]):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    for rank, (c, s) in enumerate(results, 1):
        print(f"{rank:02d}. {c.chunk_id} | score={s:.4f}")
        print(f"    {c.title}  [{c.service} | {c.doc_type} | {c.version} | {c.updated}]")
        print(f"    {c.text}\n")

def naive_answer_from_context(query: str, ctx: List[Chunk]) -> str:
   
    key_terms = ["s3:", "bucket policy", "trust policy", "sts:assumerole", "cross-account", "s3 getobject", "s3 listbucket", "s3 putobject"]
    collected = []
    for c in ctx:
        for sent in re.split(r"(?<=[.!?])\s+", c.text):
            sn = sent.lower()
            if any(k in sn for k in key_terms):
                collected.append(sent.strip())
    if not collected:
        # fallback: show the chunks as context summary
        collected = [c.text.strip() for c in ctx[:3]]
    return " ".join(collected)


# -----------------------------
# 9) Run Demo
# -----------------------------
if __name__ == "__main__":
    query = "What permissions are required for S3 cross-account access?"

    # basic = basic_rag_pipeline(query, k=5)
    # print_results("BASIC RAG (dense-only, no filters, no reranking)", basic)

    # basic_ctx = [c for c, _ in basic]
    # print("BASIC 'ANSWER' (context-derived):")
    # print(naive_answer_from_context(query, basic_ctx))

    adv = advanced_rag_pipeline(query, k_dense=15, k_sparse=15, top_n=5)
    print_results("ADVANCED RAG (hybrid + metadata filter + rerank)", adv)

    adv_ctx = [c for c, _ in adv]
    print("\nADVANCED 'ANSWER' (context-derived):")
    print(naive_answer_from_context(query, adv_ctx))
