"""
Hybrid retrieval: combines Chroma's semantic (embedding) search with BM25
keyword search, merged via Reciprocal Rank Fusion (RRF).

Why: pure semantic search sometimes misses exact matches -- e.g. someone
asks about a specific function name or error string that's a precise
keyword match but not necessarily the closest embedding. BM25 catches
that; RRF blends the two rankings without needing to normalize scores
that live on completely different scales.
"""
import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

from ingest import get_chroma_client, get_embedder

# In-memory BM25 index per repo, rebuilt every time a repo is (re-)indexed.
_bm25_indexes: Dict[str, Dict] = {}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)]


def build_bm25_index(repo_id: str, documents: List[str], metadatas: List[Dict]) -> None:
    """Called once at the end of indexing, alongside the Chroma writes."""
    if not documents:
        _bm25_indexes[repo_id] = {"bm25": None, "documents": [], "metadatas": []}
        return
    tokenized = [_tokenize(doc) for doc in documents]
    _bm25_indexes[repo_id] = {
        "bm25": BM25Okapi(tokenized),
        "documents": documents,
        "metadatas": metadatas,
    }


def _semantic_search(repo_id: str, query: str, n: int) -> List[Dict]:
    client = get_chroma_client()
    collection = client.get_collection(repo_id)
    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"text": d, **m} for d, m in zip(docs, metas)]


def _keyword_search(repo_id: str, query: str, n: int) -> List[Dict]:
    index = _bm25_indexes.get(repo_id)
    if not index or index["bm25"] is None:
        return []
    scores = index["bm25"].get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [{"text": index["documents"][i], **index["metadatas"][i]} for i in ranked if scores[i] > 0]


def hybrid_search(repo_id: str, query: str, top_k: int, pool_size: int = 20) -> List[Dict]:
    """Reciprocal Rank Fusion of semantic + keyword candidate lists."""
    semantic = _semantic_search(repo_id, query, pool_size)
    keyword = _keyword_search(repo_id, query, pool_size)

    def dedup_key(item: Dict):
        return (item.get("file"), item.get("chunk_index"))

    rrf_k = 60  # standard RRF smoothing constant
    scores: Dict[tuple, float] = {}
    lookup: Dict[tuple, Dict] = {}

    for rank, item in enumerate(semantic):
        key = dedup_key(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        lookup[key] = item

    for rank, item in enumerate(keyword):
        key = dedup_key(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        lookup.setdefault(key, item)

    ranked_keys = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [lookup[key] for key, _ in ranked_keys]
