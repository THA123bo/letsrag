"""Hybrid search: Query Expansion + HyDE + BM25 + Semantic + RRF + Reranking."""

import re
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import spacy
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from chonkie.handshakes import ChromaHandshake

from rag_studio.llm import llm_chat
from rag_studio.config import Config


# Singleton instances (expensive to load, so loaded once)
_reranker: CrossEncoder | None = None
_nlp = spacy.load("es_core_news_sm")


def get_reranker() -> CrossEncoder:
    """Returns the reranking model, initializing it on the first call."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(Config.RERANK_MODEL)
    return _reranker


# BM25 index cache
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenization with spaCy lemmatization for Spanish.

    Lemmatizes words, removes stopwords and non-alphabetic tokens.
    """
    doc = _nlp(text.lower())
    return [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]


def _get_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Builds or returns a cached BM25 index over all ChromaDB chunks."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus

    handshake = ChromaHandshake(
        path=str(Config.CHROMA_DIR),
        collection_name=Config.COLLECTION_NAME,
        embedding_model=Config.EMBEDDING_MODEL,
    )
    
    collection = handshake.collection
    all_docs = collection.get(include=["documents", "metadatas"])

    _bm25_corpus = [
        {"text": text, "id": doc_id, "metadata": meta or {}}
        for text, doc_id, meta in zip(all_docs["documents"], all_docs["ids"], all_docs["metadatas"])
    ]
    
    _bm25_index = BM25Okapi([_tokenize(doc["text"]) for doc in _bm25_corpus])
    return _bm25_index, _bm25_corpus


def invalidate_bm25_cache() -> None:
    """Invalidates the BM25 cache (call after re-ingestion)."""
    global _bm25_index, _bm25_corpus
    _bm25_index = None
    _bm25_corpus = None


def expand_query(question: str, n_variants: int = 3) -> list[str]:
    """Generates question variants using an LLM to improve recall.

    Args:
        question (str): Original user question.
        n_variants (int): Number of variants to generate.

    Returns:
        list[str]: Original question plus variants.
    """
    prompt = (
        f"Generate exactly {n_variants} distinct reformulations of this question. "
        "Each reformulation must use different vocabulary for searching in documents. "
        "Reply ONLY with the reformulations, one per line, without numbers or dashes.\n\n"
        f"Question: {question}"
    )
    raw = llm_chat([{"role": "user", "content": prompt}])

    variants = [
        re.sub(r"^[\d\-\.\)\*]+\s*", "", line).strip()
        for line in raw.split("\n")
    ]
    variants = [v for v in variants if len(v) > 10]

    return [question] + variants[:n_variants]


def hyde_expand(question: str) -> str:
    """Generates a hypothetical document answering the question (HyDE).

    Args:
        question (str): The user question.

    Returns:
        str: Hypothetical document text.
    """
    prompt = (
        "Generate a brief technical paragraph (3-5 sentences) answering "
        "this question as if it were an official document excerpt. "
        "Use precise terminology and specific data.\n\n"
        f"Question: {question}"
    )
    return llm_chat([{"role": "user", "content": prompt}])


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = Config.RRF_K,
    weights: list[float] | None = None,
) -> list[dict]:
    """Merges multiple rankings using Reciprocal Rank Fusion with optional weights.

    Args:
        ranked_lists (list[list[dict]]): List of rankings ordered by relevance.
        k (int): Smoothing constant (typically 60).
        weights (list[float] | None): Weight per list. Defaults to equal weight.

    Returns:
        list[dict]: Merged list ordered by descending RRF score.
    """
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}

    for idx, ranking in enumerate(ranked_lists):
        w = weights[idx] if idx < len(weights) else 1.0
        for rank, doc in enumerate(ranking):
            doc_id = doc.get("id", doc["text"][:100])
            scores[doc_id] += w / (k + rank + 1)
            doc_map.setdefault(doc_id, doc)

    result = []
    for doc_id in sorted(scores, key=scores.get, reverse=True):
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        result.append(doc)

    return result


def bm25_search(query: str, limit: int = 10) -> list[dict]:
    """Keyword search using BM25.

    Args:
        query (str): Search query.
        limit (int): Maximum results.

    Returns:
        list[dict]: Results ordered by BM25 score.
    """
    index, corpus = _get_bm25_index()
    scores = index.get_scores(_tokenize(query))

    scored = sorted([(scores[i], corpus[i]) for i in range(len(corpus))], key=lambda x: x[0], reverse=True)
    
    return [
        {**doc, "bm25_score": float(score)}
        for score, doc in scored[:limit] if score > 0
    ]


def semantic_search(query: str, limit: int = 10) -> list[dict]:
    """Semantic search using ChromaDB embeddings.

    Args:
        query (str): Search query.
        limit (int): Maximum results.

    Returns:
        list[dict]: Results ordered by similarity.
    """
    handshake = ChromaHandshake(
        path=str(Config.CHROMA_DIR),
        collection_name=Config.COLLECTION_NAME,
        embedding_model=Config.EMBEDDING_MODEL,
    )
    return handshake.search(query=query, limit=limit) or []


def search_hybrid(
    question: str,
    limit: int = Config.RETRIEVAL_LIMIT,
    use_expansion: bool = True,
    use_hyde: bool = True,
) -> list[dict]:
    """Full hybrid search pipeline.

    Args:
        question (str): User question.
        limit (int): Maximum final results.
        use_expansion (bool): Whether to generate query variants.
        use_hyde (bool): Whether to generate a hypothetical document for semantic search.

    Returns:
        list[dict]: Ranked results.
    """
    logger = logging.getLogger("rag_studio")
    queries = [question]
    hypo_doc = None
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_expand = executor.submit(expand_query, question) if use_expansion else None
        f_hyde = executor.submit(hyde_expand, question) if use_hyde else None
            
        if f_expand:
            queries = f_expand.result()
            logger.info(f"Expansion complete: {len(queries)} queries")
            
        if f_hyde:
            try:
                hypo_doc = f_hyde.result()
                logger.info("HyDE complete")
            except Exception as e:
                logger.warning(f"HyDE error: {e}")

    sem_rankings, bm25_rankings = [], []

    for q in queries:
        if sem_res := semantic_search(q, limit=Config.RETRIEVAL_TOP_K):
            sem_rankings.append(sem_res)
        if bm25_res := bm25_search(q, limit=Config.RETRIEVAL_TOP_K):
            bm25_rankings.append(bm25_res)

    if use_hyde and hypo_doc:
        if hyde_res := semantic_search(hypo_doc, limit=Config.RETRIEVAL_TOP_K):
            sem_rankings.append(hyde_res)

    all_rankings = sem_rankings + bm25_rankings
    weights = (
        [Config.RRF_WEIGHT_SEMANTIC] * len(sem_rankings) +
        [Config.RRF_WEIGHT_BM25] * len(bm25_rankings)
    )

    if not (fused := reciprocal_rank_fusion(all_rankings, weights=weights)):
        return []

    candidates = fused[: Config.RETRIEVAL_TOP_K * 2]
    scores = get_reranker().predict([[question, doc["text"]] for doc in candidates])

    for i, doc in enumerate(candidates):
        doc["rerank_score"] = float(scores[i])

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:limit]
